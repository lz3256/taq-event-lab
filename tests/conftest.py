import json
from pathlib import Path

import pytest

from taq_lab.common import load_config, write_json
from taq_lab.data import make_synthetic


@pytest.fixture
def cfg(tmp_path):
    template = Path(__file__).resolve().parents[1] / "configs" / "taq.json"
    c = json.loads(template.read_text())
    c["data"].update(input_glob="raw/*.csv", prepared_dir="prepared", provenance="synthetic")
    c["model"].update(context_events=4, d_model=16, heads=2, layers=1, dropout=.1)
    c["train"].update(device="cpu", threads=1, batch_size=4, max_steps=4, warmup_steps=1,
                      eval_every=2, log_every=1, learning_rate=.001)
    c["evaluation"].update(batch_size=8, stride=4, max_windows=12, benchmark_repeats=2)
    c["output_dir"] = "run"
    make_synthetic(tmp_path / "raw", events_per_session=24)
    write_json(tmp_path / "config.json", c)
    return load_config(tmp_path / "config.json")
