"""Auditable causal clock-time tasks, kept separate from the historical event task."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, file_hash, load_config, read_json, write_json
from .data import clean_inputs, load_prepared
from .engine import load_checkpoint, training_signature
from .tokenization import EventBinner

TASKS = ('T1', 'T2', 'T3')
SPLITS = ('train', 'val', 'test', 'heldout')
FEATURES = ('log_rate_300', 'log_rate_ratio_5', 'log_rate_ratio_30', 'log_vol_300',
            'vol_ratio_30', 'return_z_5', 'return_z_30', 'return_z_300',
            'log_mean_size_300', 'last_price_vs_vwap_z', 'trade_age_seconds',
            'clock_sin', 'clock_cos')


def load_plan(path):
    path = Path(path).resolve()
    plan = read_json(path)
    if plan['anchor_seconds'] < 35 or plan['history_seconds'] != 300:
        raise ValueError('Require nonoverlapping 35-second targets and 300-second history')
    if plan['past_rv_floor'] <= 0 or plan['max_trade_age_seconds'] <= 0:
        raise ValueError('Invalid exclusion rule')
    if any(x <= 0 for x in plan['l2_grid']):
        raise ValueError('L2 strengths must be positive')
    sources, reference, manifest = [], None, None
    for name in plan['sources']:
        cfg = load_config(path.parent / name)
        comparable = copy.deepcopy(cfg)
        comparable.pop('output_dir')
        comparable['train'].pop('seed')
        if reference is None:
            reference = comparable
            manifest = load_prepared(cfg)
        elif comparable != reference:
            raise ValueError('Sources must differ only in seed and output directory')
        if cfg['train'].get('supervision') != 'dense' or cfg['train']['max_steps'] != 1000:
            raise ValueError('Main experiment requires dense 1000-step checkpoints')
        cp_path = Path(cfg['output_dir']) / 'joint' / 'best.pt'
        cp = load_checkpoint(cp_path)
        if cp['signature'] != training_signature(cfg, manifest, 'joint'):
            raise ValueError('Pretraining checkpoint signature mismatch')
        last = load_checkpoint(cp_path.with_name('last.pt'))
        if last['signature'] != cp['signature'] or last['step'] != cfg['train']['max_steps']:
            raise ValueError('Incomplete pretraining')
        sources.append({'cfg': cfg, 'seed': cfg['train']['seed'], 'checkpoint': str(cp_path),
                        'sha256': file_hash(cp_path), 'best_step': cp['step']})
    if len({s['seed'] for s in sources}) != len(sources) or not sources:
        raise ValueError('Duplicate or empty seeds')
    root = (path.parent / plan['output_dir']).resolve()
    signature = digest({'plan': plan, 'sources': sources, 'data': manifest['fingerprint'],
                        'task_code': file_hash(Path(__file__)),
                        'raw_cleaner': file_hash(Path(__file__).with_name('data.py'))})
    return plan, sources, manifest, root, signature


def clock_session(t, p, size, codes, settings, context, session_seconds=23400):
    """t is sorted elapsed seconds from local session open; codes[r-1] is trade r."""
    t, p, size = [np.asarray(v, dtype=np.float64) for v in (t, p, size)]
    if len(codes) != len(t) - 1 or len(p) != len(t) or len(size) != len(t):
        raise ValueError('Trade/token alignment mismatch')
    if not np.isfinite(np.concatenate([t, p, size])).all() or np.any(np.diff(t) < 0):
        raise ValueError('Invalid raw session')
    if np.any(p <= 0) or np.any(size <= 0) or np.any(t < 0) or np.any(t >= session_seconds):
        raise ValueError('Trades outside session or nonpositive price/size')
    grid = np.arange(session_seconds + 1)
    gi = np.searchsorted(t, grid, side='right') - 1
    grid_log = np.log(p[np.maximum(gi, 0)])
    squared = np.r_[0., np.diff(grid_log) ** 2]
    rv = np.cumsum(squared)
    cumulative_size = np.r_[0., np.cumsum(size)]
    cumulative_value = np.r_[0., np.cumsum(p * size)]
    rows, windows, numeric, targets = [], [], [], []
    excluded = Counter()
    anchors = np.arange(300, session_seconds - 35 + 1, settings['anchor_seconds'], dtype=int)
    for a in anchors:
        end = np.searchsorted(t, a, side='right')
        start = np.searchsorted(t, a - 300, side='right')
        if end - 1 < context:
            excluded['short_event_context'] += 1
            continue
        if gi[a - 300] < 0 or end == start:
            excluded['incomplete_history'] += 1
            continue
        age = a - t[end - 1]
        if age > settings['max_trade_age_seconds']:
            excluded['stale_last_trade'] += 1
            continue
        past_rv = rv[a] - rv[a - 300]
        if past_rv <= settings['past_rv_floor']:
            excluded['zero_past_volatility'] += 1
            continue
        left = np.searchsorted(t, a + 25, side='right')
        right = np.searchsorted(t, a + 35, side='right')
        if left == right:
            excluded['empty_future_vwap'] += 1
            continue
        sigma30 = np.sqrt(.1 * past_rv)
        past_count = end - start
        future_count = np.searchsorted(t, a + 10, side='right') - end
        past_vwap = (cumulative_value[end] - cumulative_value[end - 10]) / (cumulative_size[end] - cumulative_size[end - 10])
        future_vwap = (cumulative_value[right] - cumulative_value[left]) / (cumulative_size[right] - cumulative_size[left])
        target = [future_count * 30. / past_count,
                  np.sqrt(max(0., rv[a + 30] - rv[a]) / (.1 * past_rv)),
                  np.log(future_vwap / past_vwap) / sigma30]
        rate = past_count / 300.
        rate5 = (end - np.searchsorted(t, a - 5, side='right')) / 5.
        rate30 = (end - np.searchsorted(t, a - 30, side='right')) / 30.
        features = [np.log1p(rate), np.log((rate5 + 1 / 300) / (rate + 1 / 300)),
                    np.log((rate30 + 1 / 300) / (rate + 1 / 300)), np.log(np.sqrt(past_rv / 300)),
                    np.sqrt(max(0., rv[a] - rv[a - 30]) / (.1 * past_rv)),
                    *[(grid_log[a] - grid_log[a - h]) / np.sqrt(past_rv * h / 300) for h in (5, 30, 300)],
                    np.log1p((cumulative_size[end] - cumulative_size[start]) / past_count),
                    np.log(p[end - 1] / past_vwap) / sigma30, age,
                    np.sin(2 * np.pi * a / session_seconds), np.cos(2 * np.pi * a / session_seconds)]
        # Raw last trade r=end-1 corresponds to code r-1. End-exclusive slice is r.
        r = end - 1
        windows.append(codes[r - context:r])
        numeric.append(features)
        targets.append(target)
        rows.append({'anchor_seconds': int(a), 'last_raw_trade_index': int(r),
                     'last_code_index': int(r - 1), 'last_trade_seconds': float(t[r]),
                     'future_end_seconds': int(a + 35), 'past_trade_count': int(past_count),
                     'future_10s_count': int(future_count), 'future_vwap_count': int(right - left),
                     'context_seconds': float(t[r] - t[r - context + 1])})
    return (np.asarray(windows, dtype=np.uint8).reshape(-1, context, 3),
            np.asarray(numeric, dtype=np.float64).reshape(-1, len(FEATURES)),
            np.asarray(targets, dtype=np.float64).reshape(-1, 3), rows,
            {'candidate_anchors': len(anchors), 'retained': len(rows), 'excluded': dict(excluded)})


def fit_thresholds(targets):
    thresholds = np.quantile(targets, [1 / 3, 2 / 3], axis=0).T
    if not np.isfinite(thresholds).all() or np.any(thresholds[:, 0] >= thresholds[:, 1]):
        raise ValueError('Degenerate tertiles; do not force balanced labels')
    return thresholds


def apply_thresholds(targets, thresholds):
    return np.column_stack([np.searchsorted(thresholds[j], targets[:, j], side='right') for j in range(3)])


def prepare_tasks(path):
    plan, sources, manifest, root, signature = load_plan(path)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / 'task_metadata.json'
    if metadata.exists():
        meta = read_json(metadata)
        if meta['signature'] != signature:
            raise ValueError('Task configuration/code changed; use a new output directory')
        for name, sha in meta['artifacts'].items():
            if file_hash(root / name) != sha:
                raise ValueError(f'Task cache checksum mismatch: {name}')
        return plan, sources, root, meta, dict(np.load(root / 'tasks.npz', allow_pickle=False))
    cfg = sources[0]['cfg']
    paths = [s['path'] for s in manifest['source_files']]
    for source in manifest['source_files']:
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('Raw input no longer matches pretraining manifest')
    frame, audit = clean_inputs(paths, cfg)
    sessions = {(s['symbol'], s['date']): s for s in manifest['sessions']}
    binner = EventBinner.from_dict(manifest['binner'])
    parts = {s: [] for s in SPLITS}
    examples, session_audit = [], []
    prepared = Path(cfg['data']['prepared_dir'])
    for key, group in frame.groupby(['symbol', 'date'], sort=True):
        if key not in sessions:
            continue
        session = sessions[key]
        day_open = pd.Timestamp(f"{key[1]} {cfg['data']['session_start']}", tz=cfg['data']['timezone'])
        day_close = pd.Timestamp(f"{key[1]} {cfg['data']['session_end']}", tz=cfg['data']['timezone'])
        ns = group.timestamp.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').astype('int64').to_numpy()
        t = (ns - day_open.value).astype(np.float64) / 1e9
        p, size = group.price.to_numpy(dtype=float), group['size'].to_numpy(dtype=float)
        features = np.column_stack([np.log1p(np.diff(ns).astype(float) / 1e9), np.log(p[1:] / p[:-1]) * 10000, np.log1p(size[1:])])
        codes = np.load(prepared / session['codes'], allow_pickle=False)
        if not np.array_equal(binner.transform(features), codes):
            raise ValueError(f"Rebuilt events disagree with pretraining: {session['id']}")
        x, numeric, target, rows, stats = clock_session(t, p, size, codes, plan, cfg['model']['context_events'], int((day_close - day_open).total_seconds()))
        parts[session['split']].append((x, numeric, target))
        for row in rows:
            examples.append({'split': session['split'], 'session': session['id'], 'symbol': key[0], 'date': key[1], **row})
        session_audit.append({'session': session['id'], 'symbol': key[0], 'date': key[1], 'split': session['split'], **stats})
        print(f"Clock tasks {key}: {len(x)} anchors", flush=True)
    data = {}
    for split, batches in parts.items():
        if not batches:
            raise ValueError(f'Empty split: {split}')
        for j, name in enumerate(('x', 'numeric', 'targets')):
            data[f'{split}_{name}'] = np.concatenate([v[j] for v in batches])
            if not np.isfinite(data[f'{split}_{name}']).all():
                raise ValueError('Nonfinite task data')
    thresholds = fit_thresholds(data['train_targets'])
    distribution, frame_rows = [], []
    for split in SPLITS:
        data[f'{split}_y'] = apply_thresholds(data[f'{split}_targets'], thresholds)
        details = pd.DataFrame([r for r in examples if r['split'] == split])
        for j, task in enumerate(TASKS):
            details[f'{task}_target'] = data[f'{split}_targets'][:, j]
            details[f'{task}_label'] = data[f'{split}_y'][:, j]
            for (symbol, day), sub in details.groupby(['symbol', 'date']):
                counts = np.bincount(sub[f'{task}_label'], minlength=3)
                distribution.append({'task': task, 'split': split, 'symbol': symbol, 'date': day,
                                     'n': len(sub), **{f'class_{k}': int(counts[k]) for k in range(3)}})
        frame_rows.append(details)
    n = len(data['train_y'])
    for budget in plan['budgets']:
        if budget != 'all' and (not isinstance(budget, int) or not 0 < budget <= n):
            raise ValueError('Label budget exceeds available anchors')
    for source in sources:
        data[f"selection_{source['seed']}"] = np.random.default_rng(source['seed'] + 4000).permutation(n)
    temp = root / 'tasks.tmp.npz'
    np.savez_compressed(temp, **data)
    temp.replace(root / 'tasks.npz')
    pd.concat(frame_rows, ignore_index=True).to_csv(root / 'examples.csv', index=False)
    pd.DataFrame(distribution).to_csv(root / 'class_distribution.csv', index=False)
    meta = {'signature': signature, 'data_fingerprint': manifest['fingerprint'], 'plan': plan,
            'sources': [{k: v for k, v in s.items() if k != 'cfg'} for s in sources],
            'feature_names': FEATURES, 'thresholds': thresholds.tolist(), 'threshold_fit_split': 'train',
            'label_budget_scope': 'Model fitting labels only; shared full-train threshold calibration and validation labels are additional.',
            'evaluation_policy': 'All historical evaluation dates are DEVELOPMENT, not final confirmation.',
            'session_audit': session_audit, 'raw_audit': audit,
            'split_sizes': {s: len(data[f'{s}_y']) for s in SPLITS},
            'artifacts': {f: file_hash(root / f) for f in ('tasks.npz', 'examples.csv', 'class_distribution.csv')}}
    write_json(metadata, meta)
    return plan, sources, root, meta, data
