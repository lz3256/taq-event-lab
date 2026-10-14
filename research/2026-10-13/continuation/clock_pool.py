"""Equal-view frozen probes: clock-aligned versus event-aligned past endpoints."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
import core
from legacy.common import read_json,write_json,digest,file_hash
from legacy.model import EventTransformer
from legacy.tokenization import EventBinner
from warm_start import save_tables

HERE=Path(__file__).resolve().parent


def recover_nanoseconds(features,rows):
    if features.dtype!=np.float64:raise ValueError('Require original float64 interval features')
    delta=np.rint(np.expm1(features[:,0])*1e9).astype(np.int64)
    if np.any(delta<0):raise ValueError('Negative intervals')
    cumulative=np.r_[np.int64(0),np.cumsum(delta,dtype=np.int64)]
    r=rows.last_raw_trade_index.to_numpy(dtype=np.int64)
    exact=np.rint(rows.last_trade_seconds.to_numpy()*1e9).astype(np.int64)
    offsets=exact-cumulative[r]
    if not np.all(offsets==offsets[0]):raise ValueError('Recovered timestamps disagree with saved anchor nanoseconds')
    result=cumulative+offsets[0]
    np.testing.assert_array_equal(result[r],exact)
    return result


def older_ends(times,anchors,current_raw,mode,context=128,seconds=20,event_offset=128):
    if mode=='clock':
        cutoff=np.asarray(anchors,dtype=np.int64)*1_000_000_000-int(seconds)*1_000_000_000
        r=np.searchsorted(times,cutoff,side='right')-1
        if (r<0).any() or np.any(times[r]>cutoff):raise ValueError('Clock-window endpoint leakage')
    elif mode=='event':r=np.asarray(current_raw,dtype=np.int64)-event_offset
    else:raise ValueError('Unknown window policy')
    if np.any(r<context) or np.any(r>=current_raw):raise ValueError('Insufficient past context')
    return r


def prepare_windows(plan,meta,data,examples,sources):
    output=HERE/'pool_windows.npz';metadata=HERE/'pool_window_metadata.json'
    cfg=sources[plan['seeds'][0]]['checkpoint']['config']
    prepared=Path(cfg['data']['prepared_dir']);manifest=read_json(prepared/'manifest.json')
    hashes={s['id']:{'codes':file_hash(prepared/s['codes']),'features':file_hash(prepared/s['features'])} for s in manifest['sessions']}
    for s in manifest['sessions']:
        if hashes[s['id']]['codes']!=s['codes_sha256']:raise ValueError('Original codes changed')
    signature=digest({'task':meta['signature'],'pooling':plan['pooling'],'session_hashes':hashes,
      'code':file_hash(Path(__file__)),'tokenizer':file_hash(HERE/'legacy/tokenization.py')})
    if metadata.exists():
        saved=read_json(metadata)
        if saved['signature']!=signature:raise ValueError('Window code/settings/features changed')
        return core.verified_npz(output,saved['sha256']),saved
    binner=EventBinner.from_dict(manifest['binner'])
    arrays={f'{s}_{m}':np.empty_like(data[f'{s}_x']) for s in ('train',)+core.SPLITS for m in ('clock','event')}
    audit=[]
    for session in manifest['sessions']:
        rows=examples[examples.session==session['id']].copy()
        if not len(rows):raise ValueError('Missing session anchors')
        features=np.load(prepared/session['features'],mmap_mode='r',allow_pickle=False)
        codes=np.load(prepared/session['codes'],mmap_mode='r',allow_pickle=False)
        if len(features)!=len(codes):raise ValueError('Feature/code length differs')
        for start in range(0,len(features),250000):
            np.testing.assert_array_equal(binner.transform(features[start:start+250000]),codes[start:start+250000])
        times=recover_nanoseconds(features,rows)
        split=session['split']
        detail=examples[examples.split==split].reset_index(drop=True)
        idx=detail.index[detail.session==session['id']].to_numpy()
        current=rows.last_raw_trade_index.to_numpy(dtype=np.int64)
        context=cfg['model']['context_events']
        indices=current[:,None]-np.arange(context,0,-1)
        np.testing.assert_array_equal(codes[indices],data[f'{split}_x'][idx])
        for mode in ('clock','event'):
            ends=older_ends(times,rows.anchor_seconds.to_numpy(),current,mode,context,
               plan['pooling']['clock_offset_seconds'],plan['pooling']['event_offset'])
            indexes=ends[:,None]-np.arange(context,0,-1)
            arrays[f'{split}_{mode}'][idx]=codes[indexes]
            for position,old_end,start_raw in zip(idx,ends,ends-context+1):
                anchor=int(detail.loc[position,'anchor_seconds'])
                audit.append({'split':split,'symbol':session['symbol'],'date':session['date'],'example_index':int(position),
                  'policy':mode,'anchor_seconds':anchor,'older_last_raw_trade_index':int(old_end),
                  'older_endpoint_seconds':float(times[old_end]/1e9),
                  'older_endpoint_age_seconds':float(anchor-times[old_end]/1e9),
                  'two_view_total_span_seconds':float(anchor-times[start_raw]/1e9),
                  'windows_overlap_events':int(max(0,context-(current[np.where(idx==position)[0][0]]-old_end)))})
        print('Prepared two-view windows',session['symbol'],session['date'],flush=True)
    np.savez_compressed(output,**arrays)
    pd.DataFrame(audit).to_csv(HERE/'pool_window_audit.csv',index=False)
    saved={'signature':signature,'sha256':file_hash(output),'session_hashes':hashes,
      'exact_saved_anchor_nanoseconds':True,'all_current_tokens_match':True,'all_feature_bins_match':True,
      'audit_sha256':file_hash(HERE/'pool_window_audit.csv'),'anchors':len(examples),'policies':2}
    write_json(metadata,saved);return arrays,saved


@torch.no_grad()
def extra_embeddings(source,mode,policy,windows,plan,signature):
    folder=HERE/'pool_embeddings';folder.mkdir(exist_ok=True)
    name=f"{policy}_{mode}_seed{source['source']['seed']}"
    path=folder/f'{name}.npz';info=folder/f'{name}.json'
    local=digest({'study':signature,'seed':source['source']['seed'],'mode':mode,'policy':policy})
    if info.exists():
        saved=read_json(info)
        if saved['signature']!=local:raise ValueError('Embedding cache changed')
        return core.verified_npz(path,saved['sha256'])
    torch.manual_seed(source['source']['seed'])
    model=EventTransformer(source['checkpoint']['config'],'joint')
    if mode=='pretrained':model.load_state_dict(source['checkpoint']['model'])
    model.head=nn.Identity();model.eval().requires_grad_(False)
    result={};started=time.perf_counter()
    for split in ('train',)+core.SPLITS:
        x=windows[f'{split}_{policy}'];values=[]
        for start in range(0,len(x),plan['evaluation_batch_size']):
            events=torch.as_tensor(x[start:start+plan['evaluation_batch_size']].astype(np.int64))
            values.append(model(model.encode(events))[:,-1].numpy())
        result[split]=np.concatenate(values)
        if not np.isfinite(result[split]).all():raise ValueError('Nonfinite extra embeddings')
        print(name,split,'embedded',flush=True)
    np.savez_compressed(path,**result)
    write_json(info,{'signature':local,'sha256':file_hash(path),'seconds':time.perf_counter()-started})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    plan=read_json(HERE/'plan.json');torch.set_num_threads(plan['threads'])
    root,meta,data,examples,sources=core.resources(plan)
    windows,window_meta=prepare_windows(plan,meta,data,examples,sources)
    if args.prepare_only:return
    original=Path(plan['original_project']);sys.path.insert(0,str(original))
    from taq_lab.clock_probes import fit_head,head_predict
    code={f:file_hash(HERE/f) for f in ['clock_pool.py','warm_start.py','core.py','legacy/model.py','legacy/tokenization.py']}
    signature=digest({'plan':plan,'windows':window_meta['signature'],'task':meta['signature'],
      'checkpoints':meta['sources'],'code':code,'head_code':file_hash(original/'taq_lab/clock_probes.py'),'runtime':torch.__version__})
    if (HERE/'pool_provenance.json').exists() and read_json(HERE/'pool_provenance.json')['signature']!=signature:
        raise ValueError('Pooling run changed')
    write_json(HERE/'pool_provenance.json',{'signature':signature,'plan':plan,'code':code,'window_signature':window_meta['signature']})
    jobs=HERE/'pool_heads';jobs.mkdir(exist_ok=True);predictions={};searches=[];selections=[]
    for seed in plan['seeds']:
        selected=data[f'selection_{seed}'][:plan['label_budget']]
        for method in ('frequency','stats','random','pretrained'):
            predictions[(seed,'single_'+method if method in ('random','pretrained') else method)]=core.baseline_predictions(root,data,sources[seed],plan,seed,method)
        for policy in ('event','clock'):
            for mode in ('random','pretrained'):
                method=f'{policy}_{mode}'
                extra=extra_embeddings(sources[seed],mode,policy,windows,plan,signature)
                features={s:np.concatenate([data[f'{s}_numeric'],(sources[seed]['embeddings'][mode][s]+extra[s])/2],axis=1) for s in ('train',)+core.SPLITS}
                best=float('inf');head=None;strength=None
                for regularization in plan['pooling']['l2_grid']:
                    params,diag=fit_head(features['train'][selected],data['train_y'][selected,1],regularization,
                        plan['pooling']['max_iter'],plan['pooling']['gradient_tolerance'])
                    val=core.scores(data['val_y'][:,1],head_predict(params,features['val']))['nll']
                    searches.append({'seed':seed,'method':method,'val_nll':val,**diag})
                    if val<best:best=val;head=params;strength=regularization
                predictions[(seed,method)]={s:head_predict(head,features[s]) for s in core.SPLITS}
                name=f'{method}_seed{seed}';np.savez_compressed(jobs/f'{name}.npz',**head)
                selections.append({'seed':seed,'method':method,'lambda':strength,'val_nll':best,
                    'selected_sha256':digest(selected.tolist()),'head_sha256':file_hash(jobs/f'{name}.npz')})
                print('Pool probe fitted',name,'val',best,flush=True)
    pd.DataFrame(searches).to_csv(HERE/'pool_search.csv',index=False)
    pd.DataFrame(selections).to_csv(HERE/'pool_selected.csv',index=False)
    save_tables(data,examples,predictions,'pool',[('clock_pretrained','event_pretrained'),('clock_pretrained','single_pretrained'),
        ('clock_pretrained','stats'),('clock_pretrained','clock_random'),('event_pretrained','single_pretrained')])
    write_json(HERE/'pool_complete.json',{'completed':True,'signature':signature,'head_candidates':len(searches),
        'outputs':{p.name:file_hash(p) for p in HERE.glob('pool_*.csv')},'predictions_sha256':file_hash(HERE/'pool_predictions.npz')})


if __name__=='__main__':main()
