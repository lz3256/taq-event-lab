"""Warm-start T2 end-to-end updates from the existing converged frozen heads."""
from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import core
from legacy.common import read_json,write_json,digest,file_hash

HERE=Path(__file__).resolve().parent


def initial_head(root,plan,seed,mode):
    method='random' if mode=='scratch' else 'pretrained'
    name=f"{plan['task']}_{method}_n{plan['label_budget']}_seed{seed}"
    info=read_json(root/'probe_jobs'/f'{name}.json')
    parameters=core.verified_npz(root/'probe_jobs'/f'{name}_head.npz',info['head_sha256'])
    return info,parameters


def model_from_probe(source,head,seed):
    torch.manual_seed(seed)
    n=13
    model=core.ClockClassifier(source['checkpoint']['config'],None,
        (head['mean'][:n],head['scale'][:n]),(head['mean'][n:],head['scale'][n:]))
    with torch.no_grad():
        model.classifier.weight.copy_(torch.as_tensor(head['weight'],dtype=torch.float32))
        model.classifier.bias.copy_(torch.as_tensor(head['bias'],dtype=torch.float32))
    return model


def make_model(root,plan,source,seed,mode):
    info,head=initial_head(root,plan,seed,mode)
    model=model_from_probe(source,head,seed)
    if mode=='pretrained':
        model.encoder.load_state_dict({k:v for k,v in source['checkpoint']['model'].items() if k!='head.weight'},strict=True)
    return model,info


def train_job(root,plan,source,data,seed,mode,lr,signature):
    folder=HERE/'warm_runs'/f'{mode}_seed{seed}_lr{lr:g}'
    folder.mkdir(parents=True,exist_ok=True)
    job_signature=digest({'study':signature,'seed':seed,'mode':mode,'lr':lr})
    if (folder/'summary.json').exists():
        saved=read_json(folder/'summary.json')
        if saved['signature']!=job_signature or file_hash(folder/'best.pt')!=saved['checkpoint_sha256']:
            raise ValueError('Completed warm job mismatch')
        return saved
    model,info=make_model(root,plan,source,seed,mode)
    selected=data[f'selection_{seed}'][:plan['label_budget']]
    assert info['selected_sha256']==digest(selected.tolist())
    x=data['train_x'][selected];z=data['train_numeric'][selected];y=data['train_y'][selected,1]
    stream=core.batches(len(selected),plan['steps'],plan['batch_size'],seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    strength=info['selected_lambda']
    best=float('inf');best_step=0;history=[];elapsed=0.;start_step=1
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu')
        if saved['signature']!=job_signature:raise ValueError('Warm resume signature differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];best_step=saved['best_step'];history=saved['history'];elapsed=saved['elapsed'];start_step=saved['step']+1
    wall=time.perf_counter()
    def evaluate(step,loss=None):
        nonlocal best,best_step
        p=core.predict(model,data['val_x'],data['val_numeric'],plan['evaluation_batch_size'])
        score=core.scores(data['val_y'][:,1],p)
        if step==0:
            method='random' if mode=='scratch' else 'pretrained'
            reference=core.baseline_predictions(root,data,source,plan,seed,method)['val']
            np.testing.assert_allclose(p,reference,atol=2e-6,rtol=2e-6)
        history.append({'step':step,'train_objective':loss,'train_seconds':elapsed,**score})
        # Protect step-zero selection against insignificant float32 reevaluation differences.
        if score['nll']<best-1e-10:
            best=score['nll'];best_step=step
            torch.save({'model':model.state_dict(),'signature':job_signature,'step':step},folder/'best.tmp.pt')
            (folder/'best.tmp.pt').replace(folder/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'signature':job_signature,
          'step':step,'best':best,'best_step':best_step,'history':history,'elapsed':elapsed},folder/'last.tmp.pt')
        (folder/'last.tmp.pt').replace(folder/'last.pt')
        write_json(folder/'history.json',history)
        print(f"{folder.name} {step}/{plan['steps']} val={score['nll']:.6f}",flush=True)
    if start_step==1:evaluate(0)
    for step in range(start_step,plan['steps']+1):
        tick=time.perf_counter();model.train();model.encoder.eval() # dropout off, gradients remain enabled
        batch=stream[step-1]
        optimizer.zero_grad(set_to_none=True)
        logits=model(torch.as_tensor(x[batch].astype(np.int64)),torch.as_tensor(z[batch],dtype=torch.float32))
        loss=F.cross_entropy(logits,torch.as_tensor(y[batch],dtype=torch.long))+strength*model.classifier.weight.square().sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite warm objective')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();elapsed+=time.perf_counter()-tick
        if step%plan['eval_every']==0 or step==plan['steps']:evaluate(step,float(loss.detach()))
    saved={'name':folder.name,'signature':job_signature,'seed':seed,'mode':mode,'lr':lr,
      'steps':plan['steps'],'best_step':best_step,'best_val_nll':best,'selected_lambda':strength,
      'train_seconds':elapsed,'wall_seconds':time.perf_counter()-wall,
      'selected_sha256':digest(selected.tolist()),'batch_stream_sha256':digest(stream.tolist()),
      'initial_head_sha256':info['head_sha256'],'checkpoint_sha256':file_hash(folder/'best.pt')}
    write_json(folder/'summary.json',saved);return saved


def save_tables(data,examples,predictions,prefix,comparisons):
    rows=[];sessions=[]
    for (seed,method),values in predictions.items():
        for split,p in values.items():
            y=data[f'{split}_y'][:,1]
            rows.append({'seed':seed,'method':method,'split':split,**core.scores(y,p)})
            detail=examples[examples.split==split].reset_index(drop=True)
            for (symbol,date),part in detail.groupby(['symbol','date']):
                idx=part.index.to_numpy()
                sessions.append({'seed':seed,'method':method,'split':split,'symbol':symbol,'date':date,**core.scores(y[idx],p[idx])})
    frame=pd.DataFrame(rows);frame.to_csv(HERE/f'{prefix}_metrics.csv',index=False)
    pd.DataFrame(sessions).to_csv(HERE/f'{prefix}_session_metrics.csv',index=False)
    summary=frame.groupby(['split','method'],sort=False).agg(nll=('nll','mean'),seed_sd=('nll','std'),balanced_accuracy=('balanced_accuracy','mean')).reset_index()
    summary.to_csv(HERE/f'{prefix}_summary.csv',index=False)
    paired=[]
    for split in core.SPLITS:
        pivot=frame[frame.split==split].pivot(index='seed',columns='method',values='nll')
        for lhs,rhs in comparisons:
            delta=pivot[lhs]-pivot[rhs]
            paired.append({'split':split,'comparison':lhs+' - '+rhs,'mean_delta':delta.mean(),
              'wins':int((delta<-1e-7).sum()),'ties':int((delta.abs()<=1e-7).sum()),
              **{f'seed_{s}':float(delta.loc[s]) for s in pivot.index}})
    pd.DataFrame(paired).to_csv(HERE/f'{prefix}_paired.csv',index=False)
    np.savez_compressed(HERE/f'{prefix}_predictions.npz',**{f'{m}_seed{s}_{split}':p for (s,m),v in predictions.items() for split,p in v.items()})
    print(summary.to_string(index=False),flush=True);print(pd.DataFrame(paired).to_string(index=False),flush=True)


def main():
    plan=read_json(HERE/'plan.json');torch.set_num_threads(plan['threads'])
    root,meta,data,examples,sources=core.resources(plan)
    source_hashes={name:file_hash(HERE/name) for name in ['warm_start.py','core.py','legacy/model.py','legacy/common.py']}
    signature=digest({'plan':plan,'task':file_hash(root/'tasks.npz'),'sources':meta['sources'],
       'heads':{f'{s}_{m}':initial_head(root,plan,s,m)[0]['head_sha256'] for s in plan['seeds'] for m in ('scratch','pretrained')},
       'code':source_hashes,'torch':torch.__version__})
    if (HERE/'warm_provenance.json').exists() and read_json(HERE/'warm_provenance.json')['signature']!=signature:
        raise ValueError('Warm code/plan/input changed')
    write_json(HERE/'warm_provenance.json',{'signature':signature,'plan':plan,'code':source_hashes,
       'task_sha256':file_hash(root/'tasks.npz'),'runtime':torch.__version__})
    jobs=[]
    for seed in plan['seeds']:
        for lr in plan['learning_rates']:
            for mode in ('scratch','pretrained'):jobs.append(train_job(root,plan,sources[seed],data,seed,mode,lr,signature))
    pd.DataFrame(jobs).to_csv(HERE/'warm_candidates.csv',index=False)
    predictions={};selected=[]
    for seed in plan['seeds']:
        for method in ('frequency','stats','random','pretrained'):
            predictions[(seed,'frozen_'+method if method in ('random','pretrained') else method)]=core.baseline_predictions(root,data,sources[seed],plan,seed,method)
        for mode in ('scratch','pretrained'):
            choice=min([j for j in jobs if j['seed']==seed and j['mode']==mode],key=lambda x:(x['best_val_nll'],x['lr']))
            selected.append(choice)
            model,_=make_model(root,plan,sources[seed],seed,mode)
            state=torch.load(HERE/'warm_runs'/choice['name']/'best.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(state['model'])
            if choice['best_step']==0:
                # Keep exact old float64 probe probabilities when no update was selected.
                method='random' if mode=='scratch' else 'pretrained'
                predictions[(seed,'warm_'+mode)]=core.baseline_predictions(root,data,sources[seed],plan,seed,method)
            else:predictions[(seed,'warm_'+mode)]={s:core.predict(model,data[f'{s}_x'],data[f'{s}_numeric'],plan['evaluation_batch_size']) for s in core.SPLITS}
    pd.DataFrame(selected).to_csv(HERE/'warm_selected.csv',index=False)
    save_tables(data,examples,predictions,'warm',[('warm_pretrained','frozen_pretrained'),('warm_pretrained','stats'),('warm_pretrained','warm_scratch')])
    write_json(HERE/'warm_complete.json',{'signature':signature,'completed':True,'candidate_jobs':len(jobs),
       'outputs':{p.name:file_hash(p) for p in HERE.glob('warm_*.csv')},'predictions_sha256':file_hash(HERE/'warm_predictions.npz')})


if __name__=='__main__':main()
