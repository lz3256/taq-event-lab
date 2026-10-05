import math

import pytest
import torch
from torch.nn import functional as F

from taq_lab.common import validate_config
from taq_lab.data import prepare
from taq_lab.engine import train
from taq_lab.model import EventTransformer


@pytest.mark.parametrize('kind,order', [('joint', [0, 1, 2]), ('sequential', [0, 1, 2]),
                                      ('sequential', [1, 0, 2]), ('sequential', [1, 2, 0])])
def test_dense_matches_independent_prefix_probabilities(cfg, kind, order):
    torch.set_num_threads(1)
    cfg['tokenizer']['field_order'] = order
    model = EventTransformer(cfg, kind).eval()
    events = torch.randint(8, (2, model.context + 1, 3))
    expected = []
    with torch.no_grad():
        for target_index in range(1, model.context + 1):
            prefix = model.encode(events[:, :target_index])
            if kind == 'joint':
                target = model.encode(events[:, target_index:target_index + 1])[:, 0]
                loss = F.cross_entropy(model(prefix)[:, -1], target, reduction='none')
            else:
                loss = torch.zeros(len(events))
                for field in order:
                    logits = model(prefix)[:, -1, field * 8:(field + 1) * 8]
                    target = events[:, target_index, field]
                    loss += F.cross_entropy(logits, target, reduction='none')
                    prefix = torch.cat([prefix, (target + 8 * field)[:, None]], dim=1)
            expected.append(loss)
        actual = model.dense_event_nll(events)
        torch.testing.assert_close(actual, torch.stack(expected, dim=1))
        torch.testing.assert_close(actual[:, -1], model.event_nll(events))
        changed = events.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 8
        torch.testing.assert_close(actual[:, :2], model.dense_event_nll(changed)[:, :2], rtol=0, atol=0)
        for parameter in model.parameters():
            parameter.zero_()
        torch.testing.assert_close(model.dense_event_nll(events), torch.full_like(actual, 3 * math.log(8)))


@pytest.mark.parametrize('kind', ['joint', 'sequential'])
def test_dense_supervises_all_complete_event_positions(cfg, kind, monkeypatch):
    model = EventTransformer(cfg, kind)
    events = torch.randint(8, (2, model.context + 1, 3))
    captured = []
    original = model.forward
    def forward(tokens):
        logits = original(tokens)
        logits.retain_grad()
        captured.append(logits)
        return logits
    monkeypatch.setattr(model, 'forward', forward)
    model.training_nll(events, 'dense').mean().backward()
    norm = captured[0].grad.abs().sum((0, 2))
    if kind == 'joint':
        assert (norm > 0).all()
    else:
        assert (norm[:2] == 0).all()  # no incomplete target in event zero
        assert (norm[2:] > 0).all()


def test_dense_config_extra_evaluation_and_accounting(cfg):
    cfg['train'].update(supervision='dense', extra_eval_steps=[1])
    validate_config(cfg)
    prepare(cfg)
    summary = train(cfg, 'joint')
    assert summary['target_events_seen'] == 4 * 4 * cfg['model']['context_events']
    from pathlib import Path
    from taq_lab.engine import load_checkpoint
    cp = load_checkpoint(Path(cfg['output_dir']) / 'joint/last.pt')
    assert cp['history'][0]['step'] == 1
    assert cp['history'][0]['target_events_seen'] == 4 * cfg['model']['context_events']
    cfg['train']['supervision'] = 'unknown'
    with pytest.raises(ValueError, match='supervision'):
        validate_config(cfg)


def test_unique_coverage_counts_positions_not_repeated_exposures():
    import numpy as np
    from taq_lab.dense_report import coverage
    class OneWindow:
        context = 3
        arrays = [np.zeros((4, 3))]
        def __len__(self):
            return 1
        def locate(self, index):
            return 0, 0
    frame = coverage(OneWindow(), 42, 4, 2, {1, 4}).set_index(['supervision', 'step'])
    assert frame.loc[('sparse', 4), 'target_exposures'] == 8
    assert frame.loc[('sparse', 4), 'unique_targets'] == 1
    assert frame.loc[('dense', 4), 'target_exposures'] == 24
    assert frame.loc[('dense', 4), 'unique_targets'] == 3


def test_dense_comparison_pipeline(cfg, tmp_path):
    import copy
    import pandas as pd
    from taq_lab.common import write_json, read_json
    from taq_lab.engine import evaluate
    from taq_lab.dense_report import run
    prepare(cfg)
    for mode in ('sparse', 'dense'):
        c = copy.deepcopy(cfg)
        c['output_dir'] = str(tmp_path / mode)
        if mode == 'dense':
            c['train'].update(supervision='dense', extra_eval_steps=[1])
        write_json(tmp_path / f'{mode}.json', c)
        for kind in ('joint', 'sequential'):
            train(c, kind)
        evaluate(c)
    plan = {'dense_config': 'dense.json', 'reference_config': 'sparse.json',
            'output_dir': 'comparison', 'markov_strengths': [1., 10.]}
    write_json(tmp_path / 'plan.json', plan)
    run(tmp_path / 'plan.json')
    root = tmp_path / 'comparison'
    metrics = pd.read_csv(root / 'metrics.csv')
    assert len(metrics) == 24
    assert 'SYNTHETIC DATA' in (root / 'report.html').read_text()
    params = read_json(root / 'baseline_parameters.json')
    assert params['markov_matched_dense']['target_exposures'] == 64
    assert params['markov_matched_sparse']['target_exposures'] == 16
    curves = pd.read_csv(root / 'learning_curves.csv')
    assert (curves.unique_targets <= curves.target_events_seen).all()
