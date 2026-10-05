import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taq_lab.common import read_json, write_json
from taq_lab.data import prepare
from taq_lab.engine import train, windows
from taq_lab.study import load_plan, train_study
from taq_lab.study_report import sampled_transition_counts, evaluate_study
from taq_lab.tokenization import joint_encode


@pytest.mark.parametrize('supervision', ['last_event', 'dense'])
def test_sampled_markov_uses_actual_neural_training_indices(cfg, monkeypatch, supervision):
    import taq_lab.engine as engine
    cfg['train']['supervision'] = supervision
    manifest = prepare(cfg)
    observed = []
    original = engine.tensor_batch

    def capture(data, indices, device):
        if data.sessions[0]['split'] == 'train':
            observed.extend(int(i) for i in indices)
        return original(data, indices, device)

    monkeypatch.setattr(engine, 'tensor_batch', capture)
    train(cfg, 'joint')
    data = windows(cfg, manifest, 'train', True)
    actual = sampled_transition_counts(data, 8, cfg['train']['seed'], cfg['train']['max_steps'], cfg['train']['batch_size'], supervision)
    expected = np.zeros((512, 512), dtype=np.int64)
    for index in observed:
        codes = joint_encode(data[index] if supervision == 'dense' else data[index][-2:], 8)
        for previous, target in zip(codes[:-1], codes[1:]):
            expected[previous, target] += 1
    np.testing.assert_array_equal(actual, expected)
    assert actual.sum() == cfg['train']['max_steps'] * cfg['train']['batch_size'] * (data.context if supervision == 'dense' else 1)


@pytest.mark.parametrize('include_alternative_order', [True, False])
def test_study_reuses_checkpoints_and_scores_identical_windows(cfg, tmp_path, include_alternative_order):
    prepare(cfg)
    jobs = []
    cases = [('joint', 'joint', [0, 1, 2]), ('seq_tpv', 'sequential', [0, 1, 2])]
    if include_alternative_order:
        cases.append(('seq_ptv', 'sequential', [1, 0, 2]))
    for variant, kind, order in cases:
        c = copy.deepcopy(cfg)
        c['output_dir'] = str(tmp_path / variant)
        c['tokenizer']['field_order'] = order
        path = tmp_path / f'{variant}.json'
        write_json(path, c)
        train(c, kind)
        jobs.append({'variant': variant, 'kind': kind, 'config': path.name})
    path = tmp_path / 'study.json'
    write_json(path, {'seeds': [cfg['train']['seed']], 'output_dir': 'study',
                      'markov_strengths': [1., 10.], 'jobs': jobs})
    root = train_study(path, workers=1)
    assert set(read_json(root / 'progress.json')['jobs'].values()) == {'complete_reused'}
    output = evaluate_study(path)
    assert 'SYNTHETIC DATA' in output.read_text()
    metrics = pd.read_csv(root / 'metrics.csv')
    assert len(metrics) == 3 * (len(cases) + 3)
    assert np.isfinite(metrics.nll_per_event).all()
    for split in ('val', 'test', 'heldout'):
        frames = [pd.read_csv(p)[['window_index', 'session', 'target_event_index']]
                  for p in (root / 'evaluation').glob(f'*_{split}.csv')]
        assert len(frames) == len(cases) + 3
        assert all(frames[0].equals(other) for other in frames[1:])
    params = read_json(root / 'baseline_parameters.json')
    assert params['models']['markov_matched_seed42']['training_target_exposures'] == 16
    # A model-width change must not be silently mixed into a tokenization study.
    last_config = tmp_path / f'{cases[-1][0]}.json'
    c = read_json(last_config)
    c['model']['d_model'] *= 2
    write_json(last_config, c)
    with pytest.raises(ValueError, match='more than seed'):
        load_plan(path)
