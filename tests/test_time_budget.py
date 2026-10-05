import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from taq_lab.common import validate_config
from taq_lab.data import prepare
from taq_lab.engine import load_checkpoint, time_learning_rate_factor, train


def test_time_schedule_has_shared_budget_phase():
    assert time_learning_rate_factor(0, 100) == .001
    assert time_learning_rate_factor(5, 100) == .5
    assert time_learning_rate_factor(10, 100) == 1.
    assert time_learning_rate_factor(100, 100) == pytest.approx(.1)
    assert time_learning_rate_factor(200, 100) == pytest.approx(.1)
    assert time_learning_rate_factor(55, 100) == time_learning_rate_factor(110, 200)


@pytest.mark.parametrize('kind', ['joint', 'sequential'])
def test_time_budget_stops_and_resumes_with_mock_clock(cfg, tmp_path, monkeypatch, kind):
    # Each training update consumes exactly one simulated second; validation,
    # checkpointing and wall-clock queries must not consume the training budget.
    import taq_lab.engine as engine
    clock = iter(range(10000))
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: float(next(clock)))
    cfg['train'].update(supervision='dense', time_budget_seconds=2.5, max_steps=20)
    prepare(cfg)
    summary = train(cfg, kind)
    assert summary['complete'] and summary['stopping_reason'] == 'time_budget'
    assert summary['step'] == 3
    assert summary['train_seconds'] == 3.
    assert summary['budget_overshoot_seconds'] == .5
    assert summary['budget_overshoot_seconds'] <= summary['final_update_seconds']
    assert summary['target_events_seen'] == 3 * 4 * 4
    uninterrupted = load_checkpoint(Path(cfg['output_dir']) / kind / 'last.pt')
    assert uninterrupted['history'][-1]['val_nll_per_event'] is not None
    c = copy.deepcopy(cfg)
    c['output_dir'] = str(tmp_path / 'resumed')
    partial = train(c, kind, stop_after=1)
    assert not partial['complete']
    resumed = train(c, kind, resume=True)
    assert resumed['step'] == 3 and resumed['train_seconds'] == 3.
    restored = load_checkpoint(Path(c['output_dir']) / kind / 'last.pt')
    for key, value in uninterrupted['model'].items():
        torch.testing.assert_close(value, restored['model'][key], rtol=0, atol=0)
    assert [r['learning_rate'] for r in restored['history']] == [r['learning_rate'] for r in uninterrupted['history']]
    again = train(c, kind, resume=True)
    assert again['step'] == 3 and again['final_update_seconds'] == 1.


def test_step_safety_cap_does_not_claim_time_completion(cfg, monkeypatch):
    import taq_lab.engine as engine
    clock = iter(range(10000))
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: float(next(clock)))
    cfg['train'].update(time_budget_seconds=100.)
    prepare(cfg)
    summary = train(cfg, 'joint')
    assert not summary['complete']
    assert summary['stopping_reason'] == 'step_limit'
    assert summary['train_seconds'] == cfg['train']['max_steps']


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf'), True])
def test_invalid_time_budget_rejected(cfg, budget):
    cfg['train']['time_budget_seconds'] = budget
    with pytest.raises(ValueError, match='time_budget_seconds'):
        validate_config(cfg)
