import copy

import pandas as pd
import pytest

from taq_lab.common import write_json
from taq_lab.followup_report import run


def test_followup_report_checks_shared_windows_and_paired_arithmetic(cfg, tmp_path):
    roots = {s: tmp_path / s for s in ('sparse', 'dense', 'timed')}
    for stage, root in roots.items():
        (root / 'evaluation').mkdir(parents=True)
        seeds = [42] if stage == 'timed' else [42, 43]
        variants = ['joint', 'sequential'] if stage == 'timed' else ['joint', 'seq_tpv']
        jobs, configs, rows = [], [], []
        for seed in seeds:
            for variant in variants:
                c = copy.deepcopy(cfg)
                c['train']['seed'] = seed
                if stage != 'sparse':
                    c['train']['supervision'] = 'dense'
                if stage == 'timed':
                    c['train']['time_budget_seconds'] = 2.5
                jobs.append({'variant': variant})
                configs.append(c)
                loss = {'sparse': {'joint': 5., 'seq_tpv': 4.8},
                        'dense': {'joint': 4.6, 'seq_tpv': 4.5},
                        'timed': {'joint': 4.4, 'sequential': 4.45}}[stage][variant]
                for split in ('val', 'test', 'heldout'):
                    score = loss + (seed - 42) * .01
                    rows.append({'variant': variant, 'seed': seed, 'split': split, 'nll_per_event': score})
                    pd.DataFrame({'window_index': [0, 1], 'session': ['a', 'a'], 'symbol': ['X', 'X'],
                                  'date': ['2025-01-01'] * 2, 'target_event_index': [4, 5],
                                  'nll_per_event': [score, score]}).to_csv(
                                      root / 'evaluation' / f'{variant}_seed{seed}_{split}.csv', index=False)
        rows.extend({'variant': 'markov_full', 'seed': None, 'split': s, 'nll_per_event': 4.7}
                    for s in ('val', 'test', 'heldout'))
        pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
        write_json(root / 'evaluation_metadata.json', {'data_fingerprint': 'same', 'seeds': seeds,
                   'configs': configs, 'plan': {'jobs': jobs}, 'provenance': 'synthetic'})
    pd.DataFrame({'variant': ['joint', 'sequential'], 'step': [3, 3],
                  'checkpoint_sha256': ['x', 'y'], 'last_checkpoint_sha256': ['a', 'b']}).to_csv(
                      roots['timed'] / 'training_runs.csv', index=False)
    output = run(roots['sparse'], roots['dense'], roots['timed'], tmp_path / 'report')
    assert 'SYNTHETIC DATA' in output.read_text()
    paired = pd.read_csv(output.parent / 'paired_differences.csv')
    assert paired.dense_joint_minus_sequential.mean() == pytest.approx(.1)
    assert paired.sparse_joint_minus_sequential.mean() == pytest.approx(.2)
    assert paired.joint_nll_improvement.mean() == pytest.approx(.4)
    bad = roots['dense'] / 'evaluation' / 'joint_seed43_test.csv'
    frame = pd.read_csv(bad)
    frame.loc[0, 'target_event_index'] += 1
    frame.to_csv(bad, index=False)
    with pytest.raises(AssertionError):
        run(roots['sparse'], roots['dense'], roots['timed'], tmp_path / 'bad-report')
