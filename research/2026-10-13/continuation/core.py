"""Continue the existing clock-task study with paired end-to-end fine-tuning."""
from __future__ import annotations
import argparse
import copy
import html
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from legacy.common import digest, file_hash, read_json, write_json
from legacy.model import EventTransformer

HERE = Path(__file__).resolve().parent
SPLITS = ('val', 'test', 'heldout')


def scale_fit(values):
    return values.mean(0), np.maximum(values.std(0), 1e-6)


class ClockClassifier(nn.Module):
    def __init__(self, cfg, initial_state, numeric, hidden):
        super().__init__()
        self.encoder = EventTransformer(cfg, 'joint')
        self.encoder.head = nn.Identity()
        if initial_state is not None:
            self.encoder.load_state_dict({k: v for k, v in initial_state.items() if k != 'head.weight'}, strict=True)
        for name, array in [('numeric_mean', numeric[0]), ('numeric_scale', numeric[1]),
                            ('hidden_mean', hidden[0]), ('hidden_scale', hidden[1])]:
            self.register_buffer(name, torch.as_tensor(array, dtype=torch.float32))
        self.classifier = nn.Linear(cfg['model']['d_model'] + len(numeric[0]), 3)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, events, numeric):
        if events.shape[1:] != (self.encoder.context, 3):
            raise ValueError('Only the fixed observed-event history is accepted')
        hidden = self.encoder(self.encoder.encode(events))[:, -1]
        features = torch.cat([(numeric-self.numeric_mean)/self.numeric_scale,
                              (hidden-self.hidden_mean)/self.hidden_scale], dim=1)
        return self.classifier(features)


def scores(y, p):
    y, p = np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1., rtol=1e-5, atol=1e-6)
    predicted = p.argmax(1)
    c = np.bincount(3*y+predicted, minlength=9).reshape(3,3)
    support, guessed, tp = c.sum(1), c.sum(0), np.diag(c)
    recall = np.divide(tp, support, out=np.zeros(3), where=support>0)
    f1 = np.divide(2*tp, support+guessed, out=np.zeros(3), where=(support+guessed)>0)
    return {'nll': float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            'balanced_accuracy': float(recall.mean()), 'macro_f1': float(f1.mean()),
            'accuracy': float((predicted==y).mean()), 'examples': len(y)}


@torch.no_grad()
def predict(model, x, numeric, batch=64):
    model.eval()
    values=[]
    for start in range(0,len(x),batch):
        e=torch.as_tensor(x[start:start+batch].astype(np.int64))
        z=torch.as_tensor(numeric[start:start+batch],dtype=torch.float32)
        values.append(model(e,z).softmax(-1).numpy())
    return np.concatenate(values)


def batches(n, steps, batch, seed):
    generator=torch.Generator().manual_seed(seed+6000)
    permutation=torch.randperm(n,generator=generator)
    cursor=0
    result=[]
    for _ in range(steps):
        pieces=[]
        needed=batch
        while needed:
            take=min(needed,n-cursor)
            pieces.append(permutation[cursor:cursor+take])
            cursor+=take
            needed-=take
            if cursor==n:
                permutation=torch.randperm(n,generator=generator)
                cursor=0
        result.append(torch.cat(pieces).numpy())
    return np.stack(result)


def verified_npz(path, expected):
    if file_hash(path)!=expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return dict(np.load(path,allow_pickle=False))


def resources(plan):
    original=Path(plan['original_project'])
    root=original/'runs/clock_probes'
    meta=read_json(root/'task_metadata.json')
    data=verified_npz(root/'tasks.npz',meta['artifacts']['tasks.npz'])
    if file_hash(root/'examples.csv')!=meta['artifacts']['examples.csv']:
        raise ValueError('Example identity changed')
    examples=pd.read_csv(root/'examples.csv')
    for split in ('train',)+SPLITS:
        rows=examples[examples.split==split]
        np.testing.assert_array_equal(rows[f"{plan['task']}_label"],data[f'{split}_y'][:,int(plan['task'][1:])-1])
    sources={}
    for seed in plan['seeds']:
        source=next(s for s in meta['sources'] if s['seed']==seed)
        cp=Path(source['checkpoint'])
        if file_hash(cp)!=source['sha256']:
            raise ValueError('Pretraining checkpoint changed')
        checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
        embeddings={}
        for mode in ('random','pretrained'):
            path=root/'embeddings'/f'{mode}_seed{seed}'
            info=read_json(path.with_suffix('.json'))
            if mode=='pretrained' and info['checkpoint_sha256']!=source['sha256']:
                raise ValueError('Embedding/checkpoint mismatch')
            embeddings[mode]=verified_npz(path.with_suffix('.npz'),info['sha256'])
        sources[seed]={'source':source,'checkpoint':checkpoint,'embeddings':embeddings}
    return root,meta,data,examples,sources


def make_model(plan,source,data,seed,mode,selected):
    cfg=source['checkpoint']['config']
    numeric=scale_fit(data['train_numeric'][selected])
    hidden=scale_fit(source['embeddings']['random' if mode=='scratch' else 'pretrained']['train'][selected])
    torch.manual_seed(seed)
    return ClockClassifier(cfg, source['checkpoint']['model'] if mode=='pretrained' else None,numeric,hidden)


def train_candidate(plan,source,data,seed,mode,lr,signature):
    name=f'{mode}_seed{seed}_lr{lr:g}'
    folder=HERE/'runs'/name
    folder.mkdir(parents=True,exist_ok=True)
    job_signature=digest({'study':signature,'seed':seed,'mode':mode,'lr':lr})
    summary=folder/'summary.json'
    if summary.exists():
        saved=read_json(summary)
        if saved['signature']!=job_signature or file_hash(folder/'best.pt')!=saved['checkpoint_sha256']:
            raise ValueError('Completed job mismatch')
        print('Reuse',name,flush=True)
        return saved
    n=plan['label_budget']
    selected=data[f'selection_{seed}'][:n]
    model=make_model(plan,source,data,seed,mode,selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=plan['weight_decay'])
    stream=batches(n,plan['steps'],plan['batch_size'],seed)
    j=int(plan['task'][1:])-1
    y=data['train_y'][selected,j]
    x=data['train_x'][selected]
    numeric=data['train_numeric'][selected]
    best=float('inf'); best_step=None; training_seconds=0.; history=[]; first_step=1
    resumed=folder/'last.pt'
    if resumed.exists():
        last=torch.load(resumed,map_location='cpu',weights_only=True)
        if last['signature']!=job_signature:
            raise ValueError('Resume configuration/data/code changed')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        first_step=last['step']+1;history=last['history'];best=last['best'];best_step=last['best_step']
        training_seconds=last['training_seconds']
    wall_start=time.perf_counter()
    def validate(step,loss=None):
        nonlocal best,best_step
        p=predict(model,data['val_x'],data['val_numeric'],plan['evaluation_batch_size'])
        result=scores(data['val_y'][:,j],p)
        history.append({'step':step,'train_nll':loss,'train_seconds':training_seconds,**result})
        if result['nll']<best:
            best=result['nll'];best_step=step
            torch.save({'model':model.state_dict(),'signature':job_signature,'step':step},folder/'best.tmp.pt')
            (folder/'best.tmp.pt').replace(folder/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'signature':job_signature,'history':history,'best':best,'best_step':best_step,
                    'training_seconds':training_seconds},folder/'last.tmp.pt')
        (folder/'last.tmp.pt').replace(folder/'last.pt')
        write_json(folder/'history.json',history)
        print(f'{name} step {step}/{plan["steps"]}: val NLL={result["nll"]:.6f}',flush=True)
    if first_step==1:
        validate(0)
    for step in range(first_step,plan['steps']+1):
        tick=time.perf_counter();model.train();torch.manual_seed(seed+7000+step)
        idx=stream[step-1]
        optimizer.zero_grad(set_to_none=True)
        logits=model(torch.as_tensor(x[idx].astype(np.int64)),torch.as_tensor(numeric[idx],dtype=torch.float32))
        loss=F.cross_entropy(logits,torch.as_tensor(y[idx],dtype=torch.long))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();training_seconds+=time.perf_counter()-tick
        if step%plan['eval_every']==0 or step==plan['steps']:
            validate(step,float(loss.detach()))
    saved={'signature':job_signature,'name':name,'seed':seed,'mode':mode,'lr':lr,'steps':plan['steps'],
           'best_step':best_step,'best_val_nll':best,'train_seconds':training_seconds,
           'invocation_wall_seconds':time.perf_counter()-wall_start,'label_budget':n,
           'labeled_exposures':plan['steps']*plan['batch_size'],
           'selected_sha256':digest(selected.tolist()),'batch_stream_sha256':digest(stream.tolist()),
           'checkpoint_sha256':file_hash(folder/'best.pt')}
    write_json(summary,saved)
    return saved


def baseline_predictions(root,data,source,plan,seed,method):
    name=f'{plan["task"]}_{method}_n{plan["label_budget"]}_seed{seed}'
    info=read_json(root/'probe_jobs'/f'{name}.json')
    selected=data[f'selection_{seed}'][:plan['label_budget']]
    if info['selected_sha256']!=digest(selected.tolist()):
        raise ValueError('Baseline label subset differs')
    p=verified_npz(root/'probe_jobs'/f'{name}.npz',info['predictions_sha256'])
    if method!='frequency':
        head=verified_npz(root/'probe_jobs'/f'{name}_head.npz',info['head_sha256'])
        for split in SPLITS:
            x=data[f'{split}_numeric']
            if method!='stats':x=np.concatenate([x,source['embeddings'][method][split]],axis=1)
            z=(x-head['mean'])/head['scale']
            logits=z@head['weight'].T+head['bias'];logits-=logits.max(1,keepdims=True)
            q=np.exp(logits);q/=q.sum(1,keepdims=True)
            np.testing.assert_allclose(q,p[split],rtol=1e-10,atol=1e-12)
    return p


def report(plan,data,examples,sources,root,candidates,signature):
    rows=[];sessions=[];selection=[];all_predictions={}
    j=int(plan['task'][1:])-1
    for seed in plan['seeds']:
        predictions={}
        for method in ('frequency','stats','random','pretrained'):
            predictions['frozen_'+method if method in ('random','pretrained') else method]=baseline_predictions(root,data,sources[seed],plan,seed,method)
        for mode in ('scratch','pretrained'):
            options=[c for c in candidates if c['seed']==seed and c['mode']==mode]
            best=min(options,key=lambda c:(c['best_val_nll'],c['lr']))
            selection.append(best)
            model=make_model(plan,sources[seed],data,seed,mode,data[f'selection_{seed}'][:plan['label_budget']])
            state=torch.load(HERE/'runs'/best['name']/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['model'])
            predictions['finetuned_'+mode]={split:predict(model,data[f'{split}_x'],data[f'{split}_numeric'],plan['evaluation_batch_size']) for split in SPLITS}
        for method,values in predictions.items():
            for split,p in values.items():
                all_predictions[f'{method}_seed{seed}_{split}']=p
                y=data[f'{split}_y'][:,j]
                rows.append({'method':method,'seed':seed,'split':split,**scores(y,p)})
                details=examples[examples.split==split].reset_index(drop=True)
                for (symbol,day),part in details.groupby(['symbol','date']):
                    idx=part.index.to_numpy()
                    sessions.append({'method':method,'seed':seed,'split':split,'symbol':symbol,'date':day,**scores(y[idx],p[idx])})
    frame=pd.DataFrame(rows);frame.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(sessions).to_csv(HERE/'session_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(HERE/'selected_candidates.csv',index=False)
    np.savez_compressed(HERE/'predictions.npz',**all_predictions)
    summary=frame.groupby(['split','method'],sort=False).agg(nll=('nll','mean'),seed_sd=('nll','std'),balanced_accuracy=('balanced_accuracy','mean')).reset_index()
    summary.to_csv(HERE/'summary.csv',index=False)
    paired=[]
    for split in SPLITS:
        local=frame[frame.split==split].pivot(index='seed',columns='method',values='nll')
        for lhs,rhs in [('finetuned_pretrained','finetuned_scratch'),('finetuned_pretrained','stats'),('finetuned_pretrained','frozen_pretrained')]:
            delta=local[lhs]-local[rhs]
            paired.append({'split':split,'comparison':lhs+' - '+rhs,'mean_delta':float(delta.mean()),'wins':int((delta<0).sum()),'seeds':len(delta),**{f'seed_{s}':float(delta.loc[s]) for s in plan['seeds']}})
    pairs=pd.DataFrame(paired);pairs.to_csv(HERE/'paired_differences.csv',index=False)
    content='<!doctype html><html lang="zh"><meta charset="utf-8"><title>TAQ 预训练迁移：端到端微调</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17222b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d9d9d9;padding:8px;text-align:left}th{background:#edf2f4}code{background:#eee;padding:2px 4px}</style><body><h1>TAQ 预训练迁移：端到端微调</h1>'
    content+='<p>2026-10-13 · 使用真实 TAQ 的已有开发数据。任务是未来 10 秒成交速率相对过去 300 秒速率的三分类。没有新的独立测试期或交易收益结果。</p>'
    content+='<p>三种子；每种子相同 1,024 个拟合标签；随机初始化与预训练各获得两个学习率、128 次更新。验证集独立选择学习率和训练步数。新头一致，历史统计与初始表征仅用拟合子集标准化。预训练总成本另计。</p>'
    content+='<h2>结果</h2><p>NLL 越低越好；种子标准差只反映优化差异。</p>'+summary.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    content+='<h2>配对比较</h2><p>负差值表示左侧方法更好。胜出种子数不表示市场统计显著性。</p>'+pairs.to_html(index=False,float_format=lambda v:f'{v:.6f}')
    content+='<h2>验证集选出的候选</h2>'+pd.DataFrame(selection)[['seed','mode','lr','best_step','best_val_nll','train_seconds']].to_html(index=False,float_format=lambda v:f'{v:.6g}')
    content+='<h2>解释边界</h2><p>仅一个任务、一个标签预算和四轮数据曝光。没有穷尽优化方法。拟合标签预算不包括之前的任务分位数校准标签及验证标签。未来日期与 NVDA 各只有两个旧交易日，历史窗口重叠，股票受共同市场因素影响。</p>'
    content+='<p>旧冻结探针概率已从保存的回归参数独立重建核对。所有测试预测在候选训练与验证选择结束后生成。文件中 test 与 heldout 是历史字段名，均为开发集。</p></body></html>'
    (HERE/'report.html').write_text(content,encoding='utf-8')
    write_json(HERE/'completion.json',{'signature':signature,'candidate_jobs':len(candidates),'completed':True,'plan':plan,
          'outputs':{f:file_hash(HERE/f) for f in ['metrics.csv','summary.csv','session_metrics.csv','selected_candidates.csv','predictions.npz','paired_differences.csv','report.html']}})
    print(summary.to_string(index=False),flush=True)
    print(pairs.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    plan=read_json(HERE/'plan.json');torch.set_num_threads(plan['threads'])
    root,meta,data,examples,sources=resources(plan)
    original=Path(plan['original_project'])
    if file_hash(HERE/'legacy/model.py')!=file_hash(original/'taq_lab/model.py'):
        raise ValueError('Vendored encoder differs from original')
    signature=digest({'plan':plan,'task_sha256':file_hash(root/'tasks.npz'),'task_signature':meta['signature'],
        'checkpoints':{str(s):sources[s]['source']['sha256'] for s in plan['seeds']},
        'code':file_hash(Path(__file__)),'encoder':file_hash(HERE/'legacy/model.py'),'runtime':torch.__version__})
    if args.benchmark:
        seed=plan['seeds'][0];idx=data[f'selection_{seed}'][:plan['label_budget']]
        model=make_model(plan,sources[seed],data,seed,'pretrained',idx)
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rates'][0])
        tick=time.perf_counter()
        for _ in range(5):
            model.train();optimizer.zero_grad()
            b=idx[:plan['batch_size']]
            loss=F.cross_entropy(model(torch.as_tensor(data['train_x'][b].astype(np.int64)),torch.as_tensor(data['train_numeric'][b],dtype=torch.float32)),torch.as_tensor(data['train_y'][b,0],dtype=torch.long))
            loss.backward();optimizer.step()
        result={'seconds_per_step':(time.perf_counter()-tick)/5,'parameters':sum(p.numel() for p in model.parameters())}
        write_json(HERE/'benchmark.json',result);print(result);return
    existing=HERE/'provenance.json'
    if existing.exists() and read_json(existing)['signature']!=signature:
        raise ValueError('Study configuration/data/source/runtime changed')
    write_json(existing,{'signature':signature,'torch':torch.__version__,'python':platform.python_version(),
          'task_metadata_sha256':file_hash(root/'task_metadata.json'),'original_project':str(original),'plan':plan,
          'code_sha256':file_hash(Path(__file__)),'task_archive_sha256':file_hash(root/'tasks.npz')})
    candidates=[]
    for seed in plan['seeds']:
        for lr in plan['learning_rates']:
            for mode in ('scratch','pretrained'):
                candidates.append(train_candidate(plan,sources[seed],data,seed,mode,lr,signature))
    pd.DataFrame(candidates).to_csv(HERE/'candidate_jobs.csv',index=False)
    report(plan,data,examples,sources,root,candidates,signature)


if __name__=='__main__':main()
