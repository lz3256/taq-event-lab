from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from taq_lab.common import read_json, write_json
from taq_lab.compute_study import evaluate_compute, load_compute_plan
from taq_lab.data import prepare
from taq_lab.engine import train


def test_equal_time_report_has_identical_windows_and_budget_audit(cfg, tmp_path, monkeypatch):
    import taq_lab.engine as engine
    clock = iter(range(10000))
    monkeypatch.setattr(engine.time, 'perf_counter', lambda: float(next(clock)))
    cfg['train'].update(supervision='dense', time_budget_seconds=2.5, max_steps=20)
    prepare(cfg)
    write_json(tmp_path/'timed.json', cfg)
    plan_path = tmp_path/'time-plan.json'
    write_json(plan_path, {'output_dir':'time-study','markov_strengths':[1.,10.],
                          'jobs':[{'kind':k,'config':'timed.json'} for k in ['sequential','joint']]})
    for kind in ['sequential','joint']:
        train(cfg, kind)
    output = evaluate_compute(plan_path)
    assert 'SYNTHETIC DATA' in output.read_text()
    metrics = pd.read_csv(output.parent/'metrics.csv')
    assert len(metrics) == 12 and np.isfinite(metrics.nll_per_event).all()
    for split in ['val','test','heldout']:
        frames = [pd.read_csv(p)[['window_index','session','symbol','date','target_event_index']]
                  for p in (output.parent/'evaluation').glob(f'*_{split}.csv')]
        assert len(frames) == 4 and all(f.equals(frames[0]) for f in frames)
    budgets = pd.read_csv(output.parent/'training_runs.csv')
    assert (budgets.step == 3).all()
    assert (budgets.overshoot_seconds <= budgets.final_update_seconds).all()
    assert (budgets.unique_targets <= budgets.target_exposures).all()
    summary_path = Path(cfg['output_dir'])/'joint'/'training_summary.json'
    summary = read_json(summary_path)
    summary['complete'] = False
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match='not completed'):
        evaluate_compute(plan_path)


def test_compute_plan_rejects_mismatched_settings(cfg, tmp_path):
    cfg['train'].update(supervision='dense', time_budget_seconds=1.)
    prepare(cfg)
    write_json(tmp_path/'a.json', cfg)
    cfg['train']['seed'] += 1
    write_json(tmp_path/'b.json', cfg)
    plan_path = tmp_path/'plan.json'
    write_json(plan_path, {'output_dir':'study','markov_strengths':[1.],
                          'jobs':[{'kind':'joint','config':'a.json'}, {'kind':'sequential','config':'b.json'}]})
    with pytest.raises(ValueError, match='identical settings'):
        load_compute_plan(plan_path)
