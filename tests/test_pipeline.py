import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from taq_lab.data import prepare
from taq_lab.engine import evaluate, generate, load_checkpoint, train
from taq_lab.report import report


@pytest.mark.parametrize("kind", ["joint", "sequential"])
@pytest.mark.parametrize("supervision", ["last_event", "dense"])
def test_checkpoint_resume_matches_uninterrupted_training(cfg, kind, supervision):
    if supervision == 'dense':
        cfg['train']['supervision'] = 'dense'
    prepare(cfg)
    uninterrupted = copy.deepcopy(cfg)
    uninterrupted["output_dir"] += "_full"
    train(uninterrupted, kind)
    train(cfg, kind, stop_after=2)
    train(cfg, kind, resume=True)
    a = load_checkpoint(Path(uninterrupted["output_dir"]) / kind / "last.pt")
    b = load_checkpoint(Path(cfg["output_dir"]) / kind / "last.pt")
    assert a["step"] == b["step"] == 4
    from taq_lab.common import read_json
    summary = read_json(Path(cfg['output_dir']) / kind / 'training_summary.json')
    assert summary['sampled_windows'] == 16
    assert summary['target_events_seen'] == 16 * (cfg['model']['context_events'] if supervision == 'dense' else 1)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    changed = copy.deepcopy(cfg)
    changed["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="Resume rejected"):
        train(changed, kind, resume=True)


def test_full_pipeline_and_same_evaluation_targets(cfg):
    prepare(cfg)
    for kind in ("joint", "sequential"):
        train(cfg, kind)
    rows = evaluate(cfg)
    assert len(rows) == 9
    assert all(np.isfinite(r["nll_per_event"]) for r in rows)
    root = Path(cfg["output_dir"])
    for split in ("val", "test", "heldout"):
        ids = [pd.read_csv(root / "evaluation" / f"{kind}_{split}_events.csv").window_index for kind in ("frequency", "joint", "sequential")]
        assert ids[0].equals(ids[1]) and ids[1].equals(ids[2])
    for kind in ("joint", "sequential"):
        generated = pd.read_csv(generate(cfg, kind, steps=3))
        assert len(generated) == 3
        assert (generated.approx_interarrival_seconds >= 0).all()
    path = report(cfg)
    assert "SYNTHETIC DATA" in path.read_text()
    assert (root / "validation_by_events.png").stat().st_size > 1000
    assert (root / "summary.csv").exists()
