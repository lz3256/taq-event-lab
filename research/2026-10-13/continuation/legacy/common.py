from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = read_json(path)
    required = {"data", "split", "tokenizer", "model", "train", "evaluation", "output_dir"}
    if set(cfg) != required:
        raise ValueError(f"Config keys must be {sorted(required)}; got {sorted(cfg)}")
    for key in ("input_glob", "prepared_dir"):
        cfg["data"][key] = str((path.parent / cfg["data"][key]).resolve())
    cfg["output_dir"] = str((path.parent / cfg["output_dir"]).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    s = cfg["split"]
    seen, held = s["seen_symbols"], s["heldout_symbols"]
    if not seen or not held or set(seen) & set(held):
        raise ValueError("Nonempty, disjoint seen_symbols and heldout_symbols required")
    if len(set(seen)) != len(seen) or len(set(held)) != len(held):
        raise ValueError("Duplicate symbols")
    dates = [s[k] for k in ("train_dates", "val_dates", "test_dates")]
    if any(not x or x != sorted(set(x)) for x in dates):
        raise ValueError("Each date list must be nonempty, unique, sorted ISO dates")
    import datetime
    for x in sum(dates, []):
        datetime.date.fromisoformat(x)
    if not (max(dates[0]) < min(dates[1]) and max(dates[1]) < min(dates[2])):
        raise ValueError("Require train dates < validation dates < test dates")
    b = cfg["tokenizer"]["bins"]
    if not 2 <= b <= 32:
        raise ValueError("bins must be between 2 and 32")
    if sorted(cfg["tokenizer"]["field_order"]) != [0, 1, 2]:
        raise ValueError("field_order must be a permutation of [0, 1, 2]")
    m, t, e = cfg["model"], cfg["train"], cfg["evaluation"]
    if m["d_model"] % m["heads"] or m["layers"] < 1 or m["context_events"] < 1:
        raise ValueError("Invalid model dimensions")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    for key in ("batch_size", "max_steps", "eval_every", "log_every", "stride", "threads"):
        if t[key] < 1:
            raise ValueError(f"train.{key} must be positive")
    if not 0 <= t["warmup_steps"] <= t["max_steps"]:
        raise ValueError("warmup_steps must be in [0,max_steps]")
    if t["learning_rate"] <= 0 or t["gradient_clip"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer configuration")
    if t.get("supervision", "last_event") not in ("last_event", "dense"):
        raise ValueError("train.supervision must be last_event or dense")
    if "time_budget_seconds" in t:
        budget = t["time_budget_seconds"]
        warmup_fraction = t.get("time_warmup_fraction", .1)
        if isinstance(budget, bool) or not isinstance(budget, (float, int)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("train.time_budget_seconds must be finite and positive")
        if not isinstance(warmup_fraction, (float, int)) or not 0 < warmup_fraction < 1:
            raise ValueError("train.time_warmup_fraction must be in (0,1)")
    elif "time_warmup_fraction" in t:
        raise ValueError("time_warmup_fraction requires time_budget_seconds")
    extra_steps = t.get("extra_eval_steps", [])
    if not isinstance(extra_steps, list) or extra_steps != sorted(set(extra_steps)) or any(
            not isinstance(x, int) or isinstance(x, bool) or not 1 <= x <= t["max_steps"] for x in extra_steps):
        raise ValueError("train.extra_eval_steps must be unique sorted steps within the training budget")
    for key in ("batch_size", "stride", "max_windows", "benchmark_repeats"):
        if e[key] < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    if cfg["data"]["provenance"] not in ("synthetic", "real_taq"):
        raise ValueError("data.provenance must be synthetic or real_taq")
    if cfg["data"]["missing_policy"] not in ("error", "skip"):
        raise ValueError("missing_policy must be error or skip")
