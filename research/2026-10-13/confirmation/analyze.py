"""Paired loss interactions and date-block uncertainty, with no model selection."""
import json
from pathlib import Path
import sys
import html
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
ORIGINAL = HERE.parents[2]
PREVIOUS = HERE.parent / 'continuation'
sys.path.insert(0, str(ORIGINAL))
from taq_lab.common import file_hash
from taq_lab.downstream import metrics

METHODS = PLAN['methods']
CONTRASTS = list(PLAN['primary_contrasts'])


def loss(y, p):
    assert p.shape == (len(y), 3) and np.isfinite(p).all() and np.all(p >= 0)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))


def contrasts(v):
    result = pd.DataFrame(index=v.index)
    result['clock_policy'] = v.clock_pretrained - v.event_pretrained
    result['interaction'] = result.clock_policy - (v.clock_random - v.event_random)
    result['incremental_over_statistics'] = v.clock_pretrained - v.stats
    return result


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def bootstrap(values, block, replicates=10000, seed=20261013):
    """Rows are dates, columns are paired contrasts; jointly resample columns."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        raise ValueError('Too few dates for the locked uncertainty calculation')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (replicates, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(replicates, -1)[:, :n]
    draws = values[idx].mean(1)
    theta = values.mean(0)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    p = (1 + (abs(draws - theta) >= abs(theta)).sum(0)) / (replicates + 1)
    return theta, lower, upper, p


def session_metrics(examples, y, predictions, stage):
    sessions, pointwise = [], []
    for seed in PLAN['seeds']:
        for method in METHODS:
            suffix = '_test' if stage == 'development_same_stock' else '_heldout' if stage == 'development_nvda' else ''
            p = predictions[f'{method}_seed{seed}{suffix}']
            losses = loss(y, p)
            details = examples[['date', 'symbol']].copy()
            details['loss'] = losses
            # Labels are evaluated for all methods on identical indices.
            for (day, symbol), group in details.groupby(['date', 'symbol'], sort=True):
                idx = group.index.to_numpy()
                score = metrics(y[idx], p[idx])
                sessions.append({'stage': stage, 'date': day, 'symbol': symbol,
                                 'seed': seed, 'method': method,
                                 **{k: v for k, v in score.items() if not isinstance(v, list)}})
            pointwise.append(losses)
    return pd.DataFrame(sessions), np.asarray(pointwise)


def summarize(sessions, stage, cohorts):
    aggregate, paired, seed_rows, daily_rows, loo_rows = [], [], [], [], []
    for cohort, symbols in cohorts.items():
        sub = sessions[sessions.symbol.isin(symbols)].copy()
        assert len(sub)
        # Seed-average within stock-date, then equal-weight symbols within date.
        means = sub.groupby(['date', 'symbol', 'method']).nll.mean().unstack('method')
        assert not means.isna().any().any()
        daily = means.groupby(level='date').mean().sort_index()
        d = contrasts(daily)
        for method in METHODS:
            aggregate.append({'stage': stage, 'cohort': cohort, 'method': method,
                              'nll': daily[method].mean(), 'dates': len(daily), 'symbols': len(symbols)})
        for name in CONTRASTS:
            daily_rows.extend({'stage': stage, 'cohort': cohort, 'date': day,
                               'comparison': name, 'delta_nll': value} for day, value in d[name].items())
        seed_deltas = {}
        for seed in PLAN['seeds']:
            local = sub[sub.seed == seed].pivot(index=['date', 'symbol'], columns='method', values='nll')
            delta = contrasts(local.groupby(level='date').mean()).mean()
            seed_deltas[seed] = delta
            seed_rows.extend({'stage': stage, 'cohort': cohort, 'seed': seed,
                              'comparison': k, 'delta_nll': delta[k]} for k in CONTRASTS)
        inference = stage == 'new_confirmation'
        for block in ([2, 1] if inference else [None]):
            if inference:
                theta, lo, hi, p = bootstrap(d.to_numpy(), block, PLAN['inference']['replicates'], PLAN['inference']['seed'])
                adj = holm(p) if cohort == 'unseen_symbols' and block == 2 else [np.nan] * 3
            else:
                theta, lo, hi, p, adj = d.mean().to_numpy(), [np.nan] * 3, [np.nan] * 3, [np.nan] * 3, [np.nan] * 3
            for j, name in enumerate(CONTRASTS):
                wins = sum(seed_deltas[s][name] < -1e-7 for s in PLAN['seeds'])
                paired.append({'stage': stage, 'cohort': cohort, 'comparison': name,
                               'delta_nll': theta[j], 'ci95_low': lo[j], 'ci95_high': hi[j],
                               'approx_p_two_sided': p[j], 'holm_primary_p': adj[j],
                               'block_days': block, 'dates': len(daily), 'improving_seeds': wins})
        if inference:
            for day in d.index:
                dropped = d.drop(day).mean()
                loo_rows.extend({'cohort': cohort, 'omitted_date': day, 'comparison': name,
                                 'delta_nll': dropped[name]} for name in CONTRASTS)
    return [pd.DataFrame(x) for x in (aggregate, paired, seed_rows, daily_rows, loo_rows)]


def main():
    output = {k: [] for k in ['summary', 'contrasts', 'seed_contrasts', 'daily_contrasts', 'leave_one_date_out']}
    old_data = np.load(ORIGINAL / 'runs/clock_probes/tasks.npz', allow_pickle=False)
    old_examples = pd.read_csv(ORIGINAL / 'runs/clock_probes/examples.csv')
    old_pred = np.load(PREVIOUS / 'pool_predictions.npz', allow_pickle=False)
    stages = []
    for split, stage in [('test', 'development_same_stock'), ('heldout', 'development_nvda')]:
        ex = old_examples[old_examples.split == split].reset_index(drop=True)
        sessions, _ = session_metrics(ex, old_data[f'{split}_y'][:, 1], old_pred, stage)
        stages.append(sessions)
        tables = summarize(sessions, stage, {'all': sorted(ex.symbol.unique())})
        for key, table in zip(output, tables):
            output[key].append(table)
    done = json.loads((HERE / 'prediction_complete.json').read_text())
    assert done['complete'] and done['fitting_steps'] == 0
    assert file_hash(HERE / 'new_predictions.npz') == done['predictions_sha256']
    ex = pd.read_csv(HERE / 'new_examples.csv')
    assert sorted(ex.date.unique()) == PLAN['dates']
    assert ex.groupby('date').symbol.nunique().eq(6).all()
    pred = np.load(HERE / 'new_predictions.npz', allow_pickle=False)
    y = ex.T2_label.to_numpy(dtype=int)
    sessions, losses = session_metrics(ex, y, pred, 'new_confirmation')
    stages.append(sessions)
    cohorts = {k: PLAN[k] for k in ('seen_symbols', 'unseen_symbols', 'new_symbols')}
    cohorts.update({symbol: [symbol] for symbol in PLAN['seen_symbols'] + PLAN['unseen_symbols']})
    tables = summarize(sessions, 'new_confirmation', cohorts)
    for key, table in zip(output, tables):
        output[key].append(table)
    for key, tables in output.items():
        pd.concat(tables, ignore_index=True).to_csv(HERE / (key + '.csv'), index=False)
    pd.concat(stages, ignore_index=True).to_csv(HERE / 'session_metrics.csv', index=False)
    np.savez_compressed(HERE / 'new_per_anchor_losses.npz', losses=losses,
                        methods=np.asarray([f'{m}_seed{s}' for s in PLAN['seeds'] for m in METHODS]))
    counts = ex.groupby(['date', 'symbol', 'T2_label']).size().unstack(fill_value=0)
    counts.to_csv(HERE / 'class_distribution.csv')
    print('Scored all locked comparisons; date-block inference completed.', flush=True)


if __name__ == '__main__':
    main()
