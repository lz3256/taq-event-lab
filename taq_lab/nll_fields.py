"""Chain-rule field NLL: identical TPV conditioning for neural and Markov models."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .common import digest, file_hash, read_json, write_json
from .engine import load_checkpoint, training_signature, windows
from .model import EventTransformer
from .study import load_plan
from .study_report import prepare_baselines
from .tokenization import joint_encode

FIELDS = ('interval', 'price', 'size')


def joint_field_nll(log_probabilities, targets, bins):
    """Log joint probabilities -> N x 3 TPV conditional negative log likelihoods."""
    lp = log_probabilities.reshape(-1, bins, bins, bins)
    i = torch.arange(len(lp), device=lp.device)
    dt, price, size = targets.unbind(1)
    log_dt = torch.logsumexp(lp, dim=(2, 3))[i, dt]
    log_dt_price = torch.logsumexp(lp, dim=3)[i, dt, price]
    log_joint = lp[i, dt, price, size]
    return torch.stack([-log_dt, log_dt - log_dt_price, log_dt_price - log_joint], 1)


def model_field_nll(model, events):
    if events.shape[1] != model.context + 1 or model.order != (0, 1, 2):
        raise ValueError('Require full event windows and TPV order')
    tokens = model.encode(events)
    logits = model(tokens[:, :-1]).float()
    if model.kind == 'joint':
        return joint_field_nll(logits[:, -1].log_softmax(-1), events[:, -1], model.bins)
    return torch.stack([F.cross_entropy(logits[:, 3 * model.context - 1 + j, j * model.bins:(j + 1) * model.bins],
                                        events[:, -1, j], reduction='none') for j in range(3)], 1)


@torch.no_grad()
def decompose(study_plan, output):
    _, jobs, manifest, _, study_signature = load_plan(study_plan)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = jobs[0]['cfg']
    torch.set_num_threads(cfg['train']['threads'])
    probabilities, params = prepare_baselines(study_plan)
    sources = []
    for job in jobs:
        path = Path(job['cfg']['output_dir']) / job['kind'] / 'best.pt'
        sources.append({'variant': job['kind'], 'seed': job['cfg']['train']['seed'],
                        'checkpoint': str(path), 'sha256': file_hash(path), 'cfg': job['cfg']})
    signature = digest({'study': study_signature, 'sources': sources, 'code': file_hash(Path(__file__)),
                        'baseline': params['probabilities_sha256']})
    done = root / 'field_metadata.json'
    if done.exists():
        old = read_json(done)
        if old['signature'] != signature:
            raise ValueError('Field decomposition signature changed')
        for name, sha in old['artifacts'].items():
            if file_hash(root / name) != sha:
                raise ValueError('Field decomposition artifact changed')
        return root
    datasets = {s: windows(cfg, manifest, s) for s in ('val', 'test', 'heldout')}
    models = [('markov_full', None, None, None)] + [(s['variant'], s['seed'], s, j) for s, j in zip(sources, jobs)]
    metrics, sessions = [], []
    index_hashes, alignment_errors = {}, []
    for kind, seed, source, job in models:
        model = None
        if source:
            cp = load_checkpoint(source['checkpoint'])
            if cp['signature'] != training_signature(source['cfg'], manifest, kind):
                raise ValueError('Field model checkpoint mismatch')
            model = EventTransformer(source['cfg'], kind).eval()
            model.load_state_dict(cp['model'])
        for split, dataset in datasets.items():
            indices = dataset.evaluation_indices(cfg['evaluation']['max_windows'])
            index_hashes[split] = digest(indices.tolist())
            values, group = [], []
            for start in range(0, len(indices), cfg['evaluation']['batch_size']):
                ids = indices[start:start + cfg['evaluation']['batch_size']]
                events = torch.as_tensor(dataset.batch(ids), dtype=torch.long)
                if model is None:
                    previous = joint_encode(events[:, -2].numpy(), cfg['tokenizer']['bins'])
                    lp = torch.as_tensor(np.log(probabilities['markov_full'][previous]), dtype=torch.float64)
                    losses = joint_field_nll(lp, events[:, -1], cfg['tokenizer']['bins'])
                    targets = joint_encode(events[:, -1].numpy(), cfg['tokenizer']['bins'])
                    reference = -lp[torch.arange(len(ids)), torch.as_tensor(targets)]
                else:
                    losses = model_field_nll(model, events)
                    # Independently recompute the existing public event loss on each batch.
                    reference = model.event_nll(events)
                error = float((losses.sum(1) - reference).abs().max())
                if error > 2e-5:
                    raise ValueError('Fields do not sum to original event NLL')
                alignment_errors.append(error)
                values.append(losses.numpy())
                group.extend(dataset.sessions[dataset.locate(int(i))[0]]['id'] for i in ids)
            values = np.concatenate(values)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite field NLL')
            for j, field in enumerate(FIELDS):
                metrics.append({'variant': kind, 'seed': seed, 'split': split, 'field': field,
                                'nll': float(values[:, j].mean()), 'events': len(values)})
            details = pd.DataFrame(values, columns=FIELDS)
            details['session'] = group
            for sid, part in details.groupby('session'):
                s = next(s for s in dataset.sessions if s['id'] == sid)
                for field in FIELDS:
                    sessions.append({'variant': kind, 'seed': seed, 'split': split, 'session': sid,
                                     'symbol': s['symbol'], 'date': s['date'], 'field': field,
                                     'nll': float(part[field].mean()), 'events': len(part)})
        print(f'Field NLL complete: {kind} seed {seed}', flush=True)
    pd.DataFrame(metrics).to_csv(root / 'field_nll.csv', index=False)
    pd.DataFrame(sessions).to_csv(root / 'field_session_nll.csv', index=False)
    write_json(done, {'signature': signature, 'data_fingerprint': manifest['fingerprint'],
                     'sources': sources, 'evaluation_index_sha256': index_hashes,
                     'max_sum_alignment_error': max(alignment_errors),
                     'conditioning': 'TPV chain; price sees target interval; size sees target interval and price.',
                     'policy': 'All existing dates are development data.',
                     'artifacts': {n: file_hash(root / n) for n in ('field_nll.csv', 'field_session_nll.csv')}})
    return root
