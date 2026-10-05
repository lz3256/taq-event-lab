"""Independent checks of saved task indices, fitted scalers, predictions and old NLL."""
from pathlib import Path

import numpy as np
import pandas as pd

from .clock_data import TASKS, fit_thresholds, apply_thresholds
from .common import digest, file_hash, read_json, write_json
from .downstream import metrics


def audit(root):
    root = Path(root)
    meta = read_json(root / 'task_metadata.json')
    for filename in ('task_metadata.json', 'probe_metadata.json', 'field_metadata.json'):
        for name, sha in read_json(root / filename)['artifacts'].items():
            if file_hash(root / name) != sha:
                raise ValueError('Audit found modified artifact')
    data = dict(np.load(root / 'tasks.npz', allow_pickle=False))
    example = pd.read_csv(root / 'examples.csv')
    q = fit_thresholds(data['train_targets'])
    np.testing.assert_array_equal(q, meta['thresholds'])
    for split in ('train','val','test','heldout'):
        details = example[example.split==split].reset_index(drop=True)
        np.testing.assert_array_equal(apply_thresholds(data[f'{split}_targets'],q), data[f'{split}_y'])
        if not (details.last_trade_seconds <= details.anchor_seconds).all():
            raise ValueError('Future trade in input')
        if not (details.future_end_seconds <= 23400).all():
            raise ValueError('Cross-session target')
        for _,v in details.groupby('session'):
            if np.any(np.diff(v.anchor_seconds) < 35):
                raise ValueError('Overlapping future windows')
    results = pd.read_csv(root / 'probe_metrics.csv',dtype={'budget':str})
    cached, checked_jobs, checked_scores, max_probability_error = {}, 0, 0, 0.
    for path in sorted((root/'probe_jobs').glob('*.json')):
        info = read_json(path)
        name=path.stem
        seed,task,method,budget = (info[k] for k in ('seed','task','method','budget'))
        selected=data[f'selection_{seed}'][:info['n']]
        if info['selected_sha256'] != digest(selected.tolist()):
            raise ValueError('Unpaired/nonnested training subset')
        predpath=root/'probe_jobs'/f'{name}.npz'
        if file_hash(predpath)!=info['predictions_sha256']:
            raise ValueError('Predictions changed')
        predictions=dict(np.load(predpath,allow_pickle=False))
        if method != 'frequency':
            if not all(c['converged'] for c in info['search']):
                raise ValueError('Unconverged candidate')
            best=min(info['search'],key=lambda c:c['val_nll'])
            if best['lambda']!=info['selected_lambda']:
                raise ValueError('Regularization not selected on validation')
            headpath=root/'probe_jobs'/f'{name}_head.npz'
            if file_hash(headpath)!=info['head_sha256']:
                raise ValueError('Saved probe head changed')
            head=dict(np.load(headpath,allow_pickle=False))
            train=data['train_numeric']
            if method!='stats':
                key=(method,seed)
                if key not in cached:
                    archive=root/'embeddings'/f'{method}_seed{seed}.npz'
                    saved=read_json(archive.with_suffix('.json'))
                    if file_hash(archive)!=saved['sha256']:
                        raise ValueError('Embedding archive changed')
                    cached[key]=dict(np.load(archive,allow_pickle=False))
                train=np.column_stack([train,cached[key]['train']])
            np.testing.assert_allclose(head['mean'],train[selected].mean(0),atol=1e-12)
            np.testing.assert_allclose(head['scale'],np.maximum(train[selected].std(0),1e-6),atol=1e-12)
        for split,p in predictions.items():
            y=data[f'{split}_y'][:,TASKS.index(task)]
            saved=results[(results.task==task)&(results.method==method)&(results.seed==seed)&
                          (results.budget==budget)&(results.split==split)]
            if len(saved)!=1:
                raise ValueError('Missing/duplicate result')
            recomputed=metrics(y,p)
            for key in ('nll','balanced_accuracy','macro_f1','brier'):
                np.testing.assert_allclose(recomputed[key],saved.iloc[0][key],rtol=1e-10,atol=1e-12)
            if method=='frequency':
                count=np.bincount(data['train_y'][selected,TASKS.index(task)],minlength=3)
                expected=np.tile((count+1)/(len(selected)+3),(len(p),1))
            else:
                x=data[f'{split}_numeric']
                if method!='stats':
                    x=np.column_stack([x,cached[(method,seed)][split]])
                logits=((x-head['mean'])/head['scale'])@head['weight'].T+head['bias']
                logits-=logits.max(1,keepdims=True)
                expected=np.exp(logits)
                expected/=expected.sum(1,keepdims=True)
            max_probability_error=max(max_probability_error,float(np.abs(expected-p).max()))
            np.testing.assert_allclose(p,expected,rtol=1e-10,atol=1e-12)
            checked_scores+=1
        checked_jobs+=1
    fields=pd.read_csv(root/'field_nll.csv')
    original=pd.read_csv(root.parent/'dense_stability'/'metrics.csv')
    errors=[]
    for (variant,seed,split),part in fields.groupby(['variant','seed','split'],dropna=False):
        old_variant='seq_tpv' if variant=='sequential' else variant
        ref=original[(original.variant==old_variant)&(original.split==split)]
        ref=ref[ref.seed.isna()] if pd.isna(seed) else ref[ref.seed==seed]
        if len(ref)!=1:
            raise ValueError('No unique original NLL reference')
        error=abs(part.nll.sum()-ref.iloc[0].nll_per_event)
        errors.append(float(error))
        if error>1e-6:
            raise ValueError('Field sum disagrees with original saved NLL')
    result={'checked_jobs':checked_jobs,'checked_metric_rows':checked_scores,
            'max_reconstructed_probability_error':max_probability_error,
            'max_original_nll_error':max(errors),'task_signature':meta['signature'],
            'checks':['train-only thresholds','causal anchors','disjoint future supports','nested paired subsets',
                      'subset-only fitted scalers','validation-only regularization selection','converged heads',
                      'head-to-prediction reconstruction','recomputed metrics','original upstream NLL recovery']}
    write_json(root/'audit_checks.json',result)
    print(result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    audit(parser.parse_args().root)
