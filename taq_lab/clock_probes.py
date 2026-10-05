"""Cached frozen encoders and paired, validation-selected multinomial probes."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .clock_data import TASKS, SPLITS, prepare_tasks
from .common import digest, file_hash, read_json, write_json
from .downstream import metrics
from .engine import load_checkpoint
from .model import EventTransformer


def code_signature():
    return {n: file_hash(Path(__file__).with_name(n)) for n in ('clock_probes.py', 'model.py', 'downstream.py')}


@torch.no_grad()
def frozen_embeddings(source, mode, plan, root, meta, data):
    if mode not in ('random', 'pretrained'):
        raise ValueError('Unknown encoder initialization')
    cache = root / 'embeddings' / f"{mode}_seed{source['seed']}"
    cache.parent.mkdir(exist_ok=True)
    signature = digest({'source': source, 'mode': mode, 'task': meta['signature'], 'code': code_signature()})
    metadata = cache.with_suffix('.json')
    archive = cache.with_suffix('.npz')
    if metadata.exists():
        saved = read_json(metadata)
        if saved['signature'] != signature or file_hash(archive) != saved['sha256']:
            raise ValueError('Embedding cache mismatch')
        return dict(np.load(archive, allow_pickle=False))
    started = time.perf_counter()
    torch.manual_seed(source['seed'])
    model = EventTransformer(source['cfg'], 'joint')
    if mode == 'pretrained':
        model.load_state_dict(load_checkpoint(source['checkpoint'])['model'])
    model.head = nn.Identity()
    model.eval().requires_grad_(False)
    result = {}
    for split in SPLITS:
        values = []
        x = data[f'{split}_x']
        for start in range(0, len(x), plan['batch_size']):
            events = torch.as_tensor(x[start:start + plan['batch_size']].astype(np.int64))
            values.append(model(model.encode(events))[:, -1].numpy())
        result[split] = np.concatenate(values)
        if not np.isfinite(result[split]).all():
            raise ValueError('Nonfinite frozen embeddings')
        print(f"Embedding {mode} seed{source['seed']} {split}: {len(x)}", flush=True)
    temp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **result)
    temp.replace(archive)
    write_json(metadata, {'signature': signature, 'sha256': file_hash(archive), 'mode': mode,
                         'seed': source['seed'], 'seconds': time.perf_counter() - started,
                         'pooling': 'last observed event hidden state', 'dropout': 'disabled',
                         'checkpoint_sha256': source['sha256'] if mode == 'pretrained' else None})
    return result


def fit_head(x, y, strength, max_iter=500, gradient_tolerance=1e-5):
    """Convex multinomial regression with subset-only scaling, deterministic zero init."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if not len(x) or set(np.unique(y)) != {0, 1, 2}:
        raise ValueError('Probe training subset must contain all three classes')
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = torch.as_tensor((x - mean) / scale, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.long)
    head = nn.Linear(x.shape[1], 3, dtype=torch.float64)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=max_iter, max_eval=max_iter * 2,
                                  tolerance_grad=gradient_tolerance * .01, tolerance_change=1e-13,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(head(z), target) + strength * head.weight.square().sum()
        loss.backward()
        return loss
    start = time.perf_counter()
    optimizer.step(closure)
    objective = float(closure().detach())
    gradient = max(float(p.grad.abs().max()) for p in head.parameters())
    converged = gradient <= gradient_tolerance
    if not converged:
        raise RuntimeError(f'Probe did not converge: gradient={gradient}, lambda={strength}')
    parameters = {'mean': mean, 'scale': scale, 'weight': head.weight.detach().numpy(),
                  'bias': head.bias.detach().numpy()}
    diagnostics = {'lambda': strength, 'objective': objective, 'gradient_max': gradient,
                   'converged': converged, 'iterations': optimizer.state[head.weight]['n_iter'],
                   'seconds': time.perf_counter() - start}
    return parameters, diagnostics


def head_predict(parameters, x):
    z = (np.asarray(x, dtype=np.float64) - parameters['mean']) / parameters['scale']
    logits = z @ parameters['weight'].T + parameters['bias']
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def select_head(features, y, selected, strength_grid, settings):
    candidates = []
    best, best_score = None, float('inf')
    for strength in strength_grid:
        parameters, diagnostic = fit_head(features['train'][selected], y['train'][selected], strength,
                                           settings['probe_max_iter'], settings['probe_gradient_tolerance'])
        score = metrics(y['val'], head_predict(parameters, features['val']))['nll']
        candidates.append({**diagnostic, 'val_nll': score})
        if score < best_score:
            best, best_score = parameters, score
            selected_strength = strength
    return best, selected_strength, candidates


def run_probes(path):
    plan, sources, root, meta, data = prepare_tasks(path)
    torch.set_num_threads(plan['threads'])
    signature = digest({'task': meta['signature'], 'plan': plan, 'code': code_signature()})
    progress = root / 'probe_progress.json'
    if progress.exists() and read_json(progress)['signature'] != signature:
        raise ValueError('Probe code/plan changed; use new study directory')
    jobs = root / 'probe_jobs'
    jobs.mkdir(exist_ok=True)
    stats = {s: data[f'{s}_numeric'] for s in SPLITS}
    examples = pd.read_csv(root / 'examples.csv')
    rows, session_rows, searches, counts = [], [], [], []
    completed = 0
    start = time.perf_counter()
    for source in sources:
        seed = source['seed']
        embeddings = {m: frozen_embeddings(source, m, plan, root, meta, data) for m in ('random', 'pretrained')}
        features = {'stats': stats, **{m: {s: np.concatenate([stats[s], embeddings[m][s]], 1) for s in SPLITS}
                                      for m in embeddings}}
        for j, task in enumerate(TASKS):
            y = {s: data[f'{s}_y'][:, j] for s in SPLITS}
            for budget in plan['budgets']:
                n = len(y['train']) if budget == 'all' else budget
                selected = data[f'selection_{seed}'][:n]
                selected_sha = digest(selected.tolist())
                class_counts = np.bincount(y['train'][selected], minlength=3)
                counts.append({'task': task, 'seed': seed, 'budget': str(budget), 'n': n,
                               'selected_sha256': selected_sha, **{f'class_{k}': int(class_counts[k]) for k in range(3)}})
                for method in ('frequency', 'stats', 'random', 'pretrained'):
                    name = f'{task}_{method}_n{budget}_seed{seed}'
                    jobfile = jobs / f'{name}.json'
                    predfile = jobs / f'{name}.npz'
                    if jobfile.exists():
                        result = read_json(jobfile)
                        if result['signature'] != signature or file_hash(predfile) != result['predictions_sha256']:
                            raise ValueError('Probe result cache mismatch')
                        if result.get('head_sha256') and file_hash(jobs / f'{name}_head.npz') != result['head_sha256']:
                            raise ValueError('Probe head cache mismatch')
                        predictions = dict(np.load(predfile, allow_pickle=False))
                    else:
                        head_sha = None
                        if method == 'frequency':
                            prior = (class_counts + 1.) / (n + 3.)
                            predictions = {s: np.tile(prior, (len(y[s]), 1)) for s in ('val', 'test', 'heldout')}
                            strength, candidates = None, []
                        else:
                            head, strength, candidates = select_head(features[method], y, selected, plan['l2_grid'], plan)
                            predictions = {s: head_predict(head, features[method][s]) for s in ('val', 'test', 'heldout')}
                            np.savez_compressed(jobs / f'{name}_head.npz', **head)
                            head_sha = file_hash(jobs / f'{name}_head.npz')
                        np.savez_compressed(predfile, **predictions)
                        result = {'signature': signature, 'task': task, 'method': method, 'budget': str(budget),
                                  'seed': seed, 'n': n, 'selected_sha256': selected_sha,
                                  'selected_lambda': strength, 'search': candidates, 'head_sha256': head_sha,
                                  'predictions_sha256': file_hash(predfile)}
                        write_json(jobfile, result)
                    key = {k: result[k] for k in ('task', 'method', 'budget', 'seed', 'n', 'selected_lambda')}
                    for candidate in result['search']:
                        searches.append({**key, **candidate})
                    for split, probability in predictions.items():
                        score = metrics(y[split], probability)
                        rows.append({**key, 'split': split, **{k: v for k, v in score.items() if not isinstance(v, list)}})
                        details = examples[examples.split == split].reset_index(drop=True)
                        np.testing.assert_array_equal(details[f'{task}_label'], y[split])
                        for (symbol, date), part in details.groupby(['symbol', 'date']):
                            idx = part.index.to_numpy()
                            local = metrics(y[split][idx], probability[idx])
                            session_rows.append({**key, 'split': split, 'symbol': symbol, 'date': date,
                                                 **{k: v for k, v in local.items() if not isinstance(v, list)}})
                    completed += 1
                    write_json(progress, {'signature': signature, 'complete': completed,
                                          'total': len(sources)*len(TASKS)*len(plan['budgets'])*4,
                                          'last_job': name, 'current_invocation_seconds': time.perf_counter()-start})
                print(f'Probes complete: {task} n{budget} seed{seed}', flush=True)
    outputs = {'probe_metrics.csv': rows, 'probe_session_metrics.csv': session_rows,
               'probe_search.csv': searches, 'probe_training_counts.csv': counts}
    for name, values in outputs.items():
        pd.DataFrame(values).to_csv(root / name, index=False)
    write_json(root / 'probe_metadata.json', {'signature': signature, 'task_signature': meta['signature'],
               'code_sha256': code_signature(), 'completed_jobs': completed,
               'artifacts': {n: file_hash(root / n) for n in outputs},
               'interpretation': 'Frozen probes only. No fine-tuning evidence or untouched final test.'})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fields', 'probes', 'report', 'run'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_tasks(args.plan)
    elif args.command == 'fields':
        from .clock_data import load_plan
        from .nll_fields import decompose
        plan, _, _, root, _ = load_plan(args.plan)
        decompose(Path(args.plan).resolve().parent / plan['dense_study_plan'], root)
    elif args.command == 'probes':
        run_probes(args.plan)
    elif args.command == 'report':
        from .clock_report import make_report
        from .clock_data import load_plan
        make_report(load_plan(args.plan)[3])
    else:
        from .clock_data import load_plan
        from .nll_fields import decompose
        from .clock_report import make_report
        plan, _, _, root, _ = load_plan(args.plan)
        decompose(Path(args.plan).resolve().parent / plan['dense_study_plan'], root)
        run_probes(args.plan)
        make_report(root)


if __name__ == '__main__':
    main()
