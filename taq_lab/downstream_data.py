"""Causal labels and a frozen task dataset for small-label transfer experiments."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, file_hash, load_config, read_json, write_json
from .data import load_prepared
from .engine import load_checkpoint, training_signature


def load_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    root = (path.parent / plan['output_dir']).resolve()
    sources = []
    reference = None
    for source in plan['sources']:
        cfg = load_config(path.parent / source['config'])
        comparable = {**cfg, 'output_dir': None, 'train': {**cfg['train'], 'seed': None}}
        if reference is None:
            reference = comparable
            manifest = load_prepared(cfg)
        elif comparable != reference:
            raise ValueError('Pretraining configurations differ beyond seed/output')
        cp_path = Path(cfg['output_dir']) / 'sequential' / 'best.pt'
        cp = load_checkpoint(cp_path)
        if cp['signature'] != training_signature(cfg, manifest, 'sequential'):
            raise ValueError('Pretraining checkpoint/config mismatch')
        sources.append({'seed': cfg['train']['seed'], 'cfg': cfg, 'checkpoint': str(cp_path),
                        'sha256': file_hash(cp_path)})
    seeds = [s['seed'] for s in sources]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Require unique pretraining seeds')
    if plan['budgets'] != sorted(set(plan['budgets'])) or min(plan['budgets']) < 3:
        raise ValueError('Require unique increasing label budgets >= 3')
    if plan['horizon_events'] < 1 or plan['neutral_band_bps'] < 0 or plan['evaluation_examples'] < 1:
        raise ValueError('Invalid task settings')
    t = plan['train']
    if any(t[k] < 1 for k in ('steps', 'batch_size', 'eval_every', 'threads')):
        raise ValueError('Invalid training counts')
    if t['learning_rate'] <= 0 or t['weight_decay'] < 0 or not 0 <= t['warmup_steps'] <= t['steps']:
        raise ValueError('Invalid optimizer settings')
    fp = digest({'plan': plan, 'sources': sources, 'prepared_fingerprint': manifest['fingerprint']})
    return plan, sources, manifest, root, fp


def future_labels(features, anchors, horizon, band):
    """Feature return at j is log(P[j+1]/P[j]); future starts AFTER anchor."""
    cumulative = np.r_[0., np.cumsum(features[:, 1], dtype=np.float64)]
    returns = cumulative[anchors + horizon + 1] - cumulative[anchors + 1]
    labels = np.where(returns < -band, 0, np.where(returns > band, 2, 1)).astype(np.int64)
    return labels, returns


def engineered_features(past):
    """Only observed history: last fields and trailing return/time/size summaries."""
    columns = [past[:, -1, j] for j in range(3)]
    for width in (5, 20, past.shape[1]):
        block = past[:, -min(width, past.shape[1]):]
        r = block[:, :, 1]
        columns.extend([r.sum(1), r.std(1), np.abs(r).mean(1), block[:, :, 0].mean(1),
                        block[:, :, 2].mean(1), block[:, :, 2].std(1)])
    return np.column_stack(columns).astype(np.float32)


def prepare_task(path):
    plan, sources, manifest, root, fingerprint = load_plan(path)
    root.mkdir(parents=True, exist_ok=True)
    cfg = sources[0]['cfg']
    prepared = Path(cfg['data']['prepared_dir'])
    # Continuous features define labels and baselines, so freeze their hashes too.
    feature_hashes = {s['id']: file_hash(prepared / s['features']) for s in manifest['sessions']}
    signature = digest({'plan_fingerprint': fingerprint, 'features': feature_hashes,
                        'data_code_sha256': file_hash(Path(__file__))})
    metadata_path = root / 'dataset_metadata.json'
    if metadata_path.exists():
        meta = read_json(metadata_path)
        if meta['signature'] != signature:
            raise ValueError('Task dataset changed; use a new output directory')
        for name, sha in meta['artifacts'].items():
            if file_hash(root / name) != sha:
                raise ValueError(f'Task artifact checksum mismatch: {name}')
        return plan, sources, root, meta
    context, horizon = cfg['model']['context_events'], plan['horizon_events']
    arrays, records, sizes = {}, [], {}
    for split in ('train', 'val', 'test', 'heldout'):
        sessions = [s for s in manifest['sessions'] if s['split'] == split]
        eligible = [(s, np.arange(context - 1, s['events'] - horizon, context + horizon, dtype=np.int64))
                    for s in sessions]
        count = sum(len(a) for _, a in eligible)
        if not count:
            raise ValueError(f'No downstream examples in {split}')
        ids = np.arange(count) if split == 'train' else np.linspace(0, count - 1, min(count, plan['evaluation_examples']), dtype=np.int64)
        tokens, labels, numeric = [], [], []
        offset = 0
        for session, anchors in eligible:
            local = ids[(ids >= offset) & (ids < offset + len(anchors))] - offset
            offset += len(anchors)
            anchors = anchors[local]
            if not len(anchors):
                continue
            codes = np.load(prepared / session['codes'], mmap_mode='r', allow_pickle=False)
            features = np.load(prepared / session['features'], mmap_mode='r', allow_pickle=False)
            if len(codes) != len(features) or len(codes) != session['events'] or not np.isfinite(features).all():
                raise ValueError('Invalid feature/code arrays')
            indexes = anchors[:, None] - np.arange(context - 1, -1, -1)
            past = np.asarray(features[indexes])
            y, returns = future_labels(features, anchors, horizon, plan['neutral_band_bps'])
            dt = np.r_[0., np.cumsum(np.expm1(features[:, 0]), dtype=np.float64)]
            seconds = dt[anchors + horizon + 1] - dt[anchors + 1]
            tokens.append(np.asarray(codes[indexes], dtype=np.uint8))
            labels.append(y)
            numeric.append(engineered_features(past))
            for anchor, target, value, elapsed in zip(anchors, y, returns, seconds):
                records.append({'split': split, 'session': session['id'], 'symbol': session['symbol'],
                                'date': session['date'], 'input_start': int(anchor - context + 1),
                                'anchor': int(anchor), 'label_end': int(anchor + horizon), 'label': int(target),
                                'future_log_return_bps': float(value), 'horizon_seconds': float(elapsed)})
        arrays[f'{split}_x'] = np.concatenate(tokens)
        arrays[f'{split}_y'] = np.concatenate(labels)
        arrays[f'{split}_numeric'] = np.concatenate(numeric)
        sizes[split] = len(arrays[f'{split}_y'])
    if sizes['train'] < max(plan['budgets']):
        raise ValueError('Insufficient nonoverlapping training examples')
    frame = pd.DataFrame(records)
    frame['example_index'] = frame.groupby('split', sort=False).cumcount()
    frame.to_csv(root / 'examples.csv', index=False)
    # Common samples within each seed, nested across label budgets; labels never select samples.
    for source in sources:
        indices = np.random.default_rng(source['seed'] + 4000).permutation(sizes['train'])[:max(plan['budgets'])]
        arrays[f"train_selection_seed{source['seed']}"] = indices
    temp = root / 'dataset.tmp.npz'
    np.savez_compressed(temp, **arrays)
    temp.replace(root / 'dataset.npz')
    meta = {'signature': signature, 'plan_fingerprint': fingerprint, 'plan': plan,
            'prepared_fingerprint': manifest['fingerprint'], 'provenance': manifest['provenance'],
            'feature_sha256': feature_hashes, 'sizes': sizes, 'context_events': context,
            'split_spec': manifest['split_spec'], 'label_names': ['down', 'flat', 'up'],
            'warning': 'Exploratory reuse of previously inspected dates; not a new untouched confirmation set.',
            'artifacts': {name: file_hash(root / name) for name in ('dataset.npz', 'examples.csv')}}
    write_json(metadata_path, meta)
    return plan, sources, root, meta
