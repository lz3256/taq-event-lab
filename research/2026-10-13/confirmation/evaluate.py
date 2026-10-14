"""Locked-model confirmation. Prepare labels once, predict without fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
ORIGINAL = HERE.parents[2]
PREVIOUS = HERE.parent / 'continuation'
WORK = HERE / 'work' / 'taq_confirmation'
sys.path.insert(0, str(ORIGINAL))
from taq_lab.clock_data import clock_session, apply_thresholds
from taq_lab.clock_probes import head_predict
from taq_lab.data import clean_inputs
from taq_lab.tokenization import EventBinner
from taq_lab.model import EventTransformer
from taq_lab.common import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def resources():
    for name, expected in PLAN['artifacts_sha256'].items():
        assert file_hash(name) == expected, name
    meta = read(ORIGINAL / 'runs/clock_probes/task_metadata.json')
    sources = {}
    for s in meta['sources']:
        assert file_hash(s['checkpoint']) == s['sha256']
        sources[s['seed']] = torch.load(s['checkpoint'], map_location='cpu', weights_only=True)
    return meta, sources


def older_windows(t, codes, details, policy):
    r = details.last_raw_trade_index.to_numpy(dtype=np.int64)
    if policy == 'event':
        ends = r - 128
    elif policy == 'clock':
        cutoff = details.anchor_seconds.to_numpy() - 20
        ends = np.searchsorted(t, cutoff, side='right') - 1
        assert np.all(t[ends] <= cutoff)
    else:
        raise ValueError(policy)
    assert np.all(ends >= 128) and np.all(ends < r)
    idx = ends[:, None] - np.arange(128, 0, -1)
    assert np.all(t[idx + 1] <= details.anchor_seconds.to_numpy()[:, None])
    audit = pd.DataFrame({'policy': policy, 'anchor_seconds': details.anchor_seconds,
                         'older_endpoint_seconds': t[ends],
                         'total_span_seconds': details.anchor_seconds.to_numpy() - t[ends - 127],
                         'overlap_events': np.maximum(0, 128 - (r - ends))})
    return codes[idx], audit


def prepare():
    meta, sources = resources()
    extraction = read(HERE / 'extraction_manifest.json')
    assert extraction['completed_dates'] == len(PLAN['dates'])
    assert extraction['plan_sha256'] == file_hash(HERE / 'plan.json')
    cfg = copy.deepcopy(sources[42]['config'])
    cfg['split'] = {'seen_symbols': PLAN['seen_symbols'] + PLAN['unseen_symbols'],
                    'heldout_symbols': [], 'train_dates': [], 'val_dates': [], 'test_dates': PLAN['dates']}
    manifest = read(Path(cfg['data']['prepared_dir']) / 'manifest.json')
    binner = EventBinner.from_dict(manifest['binner'])
    audit, examples, spans = [], [], []
    parts = {k: [] for k in ('current', 'event', 'clock', 'numeric', 'targets', 'y')}
    expected = set(cfg['split']['seen_symbols'])
    for day in PLAN['dates']:
        raw = WORK / 'raw' / (day + '.csv.gz')
        evidence = next(x for x in extraction['completed'] if x['date'] == day)
        assert file_hash(raw) == evidence['sha256']
        frame, cleaning = clean_inputs([raw], cfg)
        assert set(frame.symbol) == expected and set(frame.date) == {day}
        for symbol, group in frame.groupby('symbol', sort=True):
            ns = group.timestamp.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[ns]').astype('int64').to_numpy()
            opening = pd.Timestamp(day + ' 09:30:00', tz='America/New_York').value
            t = (ns - opening).astype(np.float64) / 1e9
            price, size = group.price.to_numpy(float), group['size'].to_numpy(float)
            f = np.column_stack([np.log1p(np.diff(ns).astype(float) / 1e9),
                                 np.log(price[1:] / price[:-1]) * 10000, np.log1p(size[1:])])
            codes = binner.transform(f)
            current, numeric, target, rows, quality = clock_session(t, price, size, codes, meta['plan'], 128)
            assert len(rows) > 0
            detail = pd.DataFrame(rows)
            # All policies use exactly the same anchors, targets and historical statistics.
            raw_r = detail.last_raw_trade_index.to_numpy(dtype=np.int64)
            np.testing.assert_array_equal(current, codes[raw_r[:, None] - np.arange(128, 0, -1)])
            assert np.all(t[raw_r] <= detail.anchor_seconds.to_numpy())
            local = {'current': current, 'numeric': numeric, 'targets': target,
                     'y': apply_thresholds(target, np.asarray(meta['thresholds']))[:, 1]}
            for policy in ('event', 'clock'):
                local[policy], a = older_windows(t, codes, detail, policy)
                a.insert(0, 'symbol', symbol); a.insert(0, 'date', day)
                spans.append(a)
            detail.insert(0, 'symbol', symbol); detail.insert(0, 'date', day)
            detail['T2_target'] = target[:, 1]; detail['T2_label'] = local['y']
            examples.append(detail)
            for k in parts:
                parts[k].append(local[k])
            audit.append({'date': day, 'symbol': symbol, 'trades': len(t), **quality})
        print('Prepared locked confirmation', day, flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    assert all(np.isfinite(a).all() for a in arrays.values())
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / 'tasks.npz', **arrays)
    pd.concat(examples, ignore_index=True).to_csv(HERE / 'new_examples.csv', index=False)
    pd.concat(spans, ignore_index=True).to_csv(HERE / 'new_window_audit.csv', index=False)
    pd.DataFrame(audit).to_csv(HERE / 'new_session_audit.csv', index=False)
    write(HERE / 'new_data_metadata.json', {'plan_sha256': file_hash(HERE / 'plan.json'),
          'tasks_sha256': file_hash(WORK / 'tasks.npz'), 'examples_sha256': file_hash(HERE / 'new_examples.csv'),
          'thresholds': meta['thresholds'], 'sessions': len(audit), 'anchors': len(arrays['y']),
          'server_filter_scope': extraction['completed'][0]['extraction'],
          'source_time': 'nanoseconds preserved; new ties sorted by source, sequence, exchange and trade ID',
          'original_shared_task_future_availability_screen_retained': True})


def model_for(cp, seed, mode):
    torch.manual_seed(seed)
    model = EventTransformer(cp['config'], 'joint')
    if mode == 'pretrained':
        model.load_state_dict(cp['model'])
    model.head = nn.Identity()
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def embed(model, x):
    result = []
    for start in range(0, len(x), PLAN['batch_size']):
        e = torch.as_tensor(x[start:start + PLAN['batch_size']].astype(np.int64))
        result.append(model(model.encode(e))[:, -1].numpy())
    return np.concatenate(result)


def replay():
    """Check new inference against old independent saved arrays before fresh scoring."""
    meta, sources = resources()
    old = np.load(ORIGINAL / 'runs/clock_probes/tasks.npz', allow_pickle=False)
    windows = np.load(PREVIOUS / 'pool_windows.npz', allow_pickle=False)
    predictions = np.load(PREVIOUS / 'pool_predictions.npz', allow_pickle=False)
    selected = pd.read_csv(PREVIOUS / 'pool_selected.csv')
    errors = []
    for seed in PLAN['seeds']:
        for mode in ('random', 'pretrained'):
            model = model_for(sources[seed], seed, mode)
            current = embed(model, old['val_x'][:32])
            saved = np.load(ORIGINAL / f'runs/clock_probes/embeddings/{mode}_seed{seed}.npz')
            np.testing.assert_allclose(current, saved['val'][:32], atol=1e-6, rtol=1e-5)
            for policy in ('event', 'clock'):
                extra = embed(model, windows[f'val_{policy}'][:32])
                method = policy + '_' + mode
                row = selected[(selected.seed == seed) & (selected.method == method)].iloc[0]
                path = PREVIOUS / 'pool_heads' / f'{method}_seed{seed}.npz'
                assert file_hash(path) == row.head_sha256
                head = dict(np.load(path, allow_pickle=False))
                x = np.concatenate([old['val_numeric'][:32], (current + extra) / 2], axis=1)
                p = head_predict(head, x)
                expected = predictions[f'{method}_seed{seed}_val'][:32]
                error = float(np.max(abs(p - expected)))
                assert error < 1e-6
                errors.append({'seed': seed, 'method': method, 'max_probability_error': error})
    # Synthetic future-price intervention must leave the first anchor's inputs unchanged.
    t = np.arange(0, 700, .5)
    p = 100 * np.exp(.001 * np.sin(t / 7))
    size = np.ones(len(t)) * 100
    codes = np.zeros((len(t) - 1, 3), dtype=np.uint8)
    a = clock_session(t, p, size, codes, meta['plan'], 128)
    p2 = p.copy(); p2[t > 300] *= np.exp(.002 * np.sin(t[t > 300]))
    b = clock_session(t, p2, size, codes, meta['plan'], 128)
    np.testing.assert_array_equal(a[0][0], b[0][0])
    np.testing.assert_array_equal(a[1][0], b[1][0])
    assert a[2][0, 1] != b[2][0, 1]
    write(HERE / 'replay_audit.json', {'passed': True, 'reconstructed_old_predictions': errors,
          'synthetic_future_intervention': 'historical input unchanged; future T2 target changed'})
    print('Old inference replay and future-intervention check passed', flush=True)


def predict():
    assert read(HERE / 'replay_audit.json')['passed']
    meta, sources = resources()
    metadata = read(HERE / 'new_data_metadata.json')
    assert metadata['plan_sha256'] == file_hash(HERE / 'plan.json')
    assert metadata['tasks_sha256'] == file_hash(WORK / 'tasks.npz')
    data = dict(np.load(WORK / 'tasks.npz', allow_pickle=False))
    selections = pd.read_csv(PREVIOUS / 'pool_selected.csv')
    result = {}
    start = time.perf_counter()
    for seed in PLAN['seeds']:
        root = ORIGINAL / 'runs/clock_probes/probe_jobs'
        name = f'T2_stats_n1024_seed{seed}'
        info = read(root / (name + '.json'))
        path = root / (name + '_head.npz')
        assert file_hash(path) == info['head_sha256']
        result[f'stats_seed{seed}'] = head_predict(dict(np.load(path)), data['numeric'])
        for mode in ('random', 'pretrained'):
            model = model_for(sources[seed], seed, mode)
            current = embed(model, data['current'])
            for policy in ('event', 'clock'):
                method = policy + '_' + mode
                extra = embed(model, data[policy])
                x = np.concatenate([data['numeric'], (current + extra) / 2], axis=1)
                row = selections[(selections.seed == seed) & (selections.method == method)].iloc[0]
                path = PREVIOUS / 'pool_heads' / f'{method}_seed{seed}.npz'
                assert file_hash(path) == row.head_sha256
                result[f'{method}_seed{seed}'] = head_predict(dict(np.load(path)), x)
                print('Predicted frozen', method, seed, len(x), 'elapsed', round(time.perf_counter() - start, 1), flush=True)
        # Save only complete seed checkpoints; no outcome metrics examined during prediction.
        np.savez_compressed(WORK / f'predictions_seed{seed}.npz',
                            **{k: v for k, v in result.items() if k.endswith(f'_seed{seed}')})
        write(HERE / 'prediction_progress.json', {'completed_seed': seed,
              'elapsed_seconds': time.perf_counter() - start, 'outcomes_scored': False})
    np.savez_compressed(HERE / 'new_predictions.npz', **result)
    write(HERE / 'prediction_complete.json', {'complete': True, 'methods': len(result),
          'plan_sha256': file_hash(HERE / 'plan.json'), 'data_sha256': metadata['tasks_sha256'],
          'predictions_sha256': file_hash(HERE / 'new_predictions.npz'),
          'seconds': time.perf_counter() - start, 'fitting_steps': 0})


if __name__ == '__main__':
    torch.set_num_threads(PLAN['threads'])
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'replay', 'predict'])
    args = parser.parse_args()
    globals()[args.command]()
