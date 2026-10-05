from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import digest, read_json, write_json
from .data import EventWindows, SPLITS, load_prepared
from .model import EventTransformer
from .tokenization import EventBinner, joint_encode


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def amp_context(cfg, device):
    precision = cfg["train"]["precision"]
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError("Supported precision: fp32 anywhere, bf16 on supported CUDA GPUs")


def environment(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or str(device)}


def windows(cfg, manifest, split, training=False):
    return EventWindows(cfg["data"]["prepared_dir"], manifest, split, cfg["model"]["context_events"],
                        cfg["train"]["stride"] if training else cfg["evaluation"]["stride"])


def tensor_batch(data, indices, device):
    return torch.from_numpy(data.batch(indices)).to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate_model(model, data, cfg, device):
    model.eval()
    ids = data.evaluation_indices(cfg["evaluation"]["max_windows"])
    results = []
    for start in range(0, len(ids), cfg["evaluation"]["batch_size"]):
        selected = ids[start:start + cfg["evaluation"]["batch_size"]]
        with amp_context(cfg, device):
            loss = model.event_nll(tensor_batch(data, selected, device))
        results.extend(loss.float().cpu().tolist())
    values = np.asarray(results)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite evaluation NLL")
    return ids, values


def rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # All checkpoints contain only tensors and ordinary Python primitives.
    return torch.load(path, map_location="cpu", weights_only=True)


def training_signature(cfg, manifest, kind):
    # Device/threads affect reproducibility, so they are included, too. Keep config fixed on resume.
    return digest({"config": cfg, "data": manifest["fingerprint"], "kind": kind})


def time_learning_rate_factor(elapsed, budget, warmup_fraction=.1):
    """Time-budget experiments share the same schedule in accumulated training seconds.

    Use elapsed time BEFORE the next update. The first update uses a small,
    nonzero multiplier; no prediction of the next step's duration is required.
    """
    phase = min(1., max(0., elapsed / budget))
    if phase <= warmup_fraction:
        return max(.001, phase / warmup_fraction)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * (phase - warmup_fraction) / (1 - warmup_fraction)))


def train(cfg, kind, resume=False, stop_after=None):
    manifest = load_prepared(cfg)
    device = device_for(cfg["train"]["device"])
    torch.set_num_threads(cfg["train"]["threads"])
    torch.manual_seed(cfg["train"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["train"]["seed"])
    model = EventTransformer(cfg, kind).to(device)
    # Validate requested AMP before spending any training time.
    with amp_context(cfg, device):
        pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    train_data, val_data = windows(cfg, manifest, "train", True), windows(cfg, manifest, "val")
    sampler = torch.Generator().manual_seed(cfg["train"]["seed"] + 1000)
    run = Path(cfg["output_dir"]) / kind
    run.mkdir(parents=True, exist_ok=True)
    signature = training_signature(cfg, manifest, kind)
    step, best, train_seconds, wall_seconds = 0, float("inf"), 0., 0.
    history = []
    if (run / "last.pt").exists():
        if not resume:
            raise FileExistsError(f"{run}/last.pt exists; use --resume or a new output_dir")
        cp = load_checkpoint(run / "last.pt")
        if cp["signature"] != signature:
            raise ValueError("Resume rejected: configuration/data fingerprint changed")
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        sampler.set_state(cp["sampler_rng"])
        restore_rng(cp["rng"])
        step, best = cp["step"], cp["best_val_nll"]
        train_seconds, wall_seconds = cp["train_seconds"], cp["wall_seconds"]
        history = cp["history"]
        # Checkpoint is authoritative if interrupted between file writes.
        _write_history(run / "history.jsonl", history)
    elif resume:
        raise FileNotFoundError(f"No checkpoint to resume in {run}")
    elif (run / "history.jsonl").exists():
        raise FileExistsError("History exists without checkpoint. Use a fresh output_dir.")
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", environment(device))
    write_json(run / "parameters.json", model.parameter_counts())
    max_steps = cfg["train"]["max_steps"]
    time_budget = cfg["train"].get("time_budget_seconds")
    supervision = cfg["train"].get("supervision", "last_event")
    targets_per_window = cfg["model"]["context_events"] if supervision == "dense" else 1
    extra_eval_steps = set(cfg["train"].get("extra_eval_steps", []))
    end_step = min(max_steps, stop_after) if stop_after is not None else max_steps
    if end_step < step:
        raise ValueError("stop_after is an absolute step and cannot precede the checkpoint")
    origin = time.perf_counter()
    starting_wall = wall_seconds
    last_loss = None
    # Sample with replacement. Identical dedicated generator gives identical target-event streams.
    while step < end_step and (time_budget is None or train_seconds < time_budget):
        model.train()
        sync(device)
        t0 = time.perf_counter()
        selected = torch.randint(len(train_data), (cfg["train"]["batch_size"],), generator=sampler).numpy()
        batch = tensor_batch(train_data, selected, device)
        warmup = cfg["train"]["warmup_steps"]
        next_step = step + 1
        factor = next_step / warmup if warmup and next_step <= warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (next_step - warmup) / max(1, max_steps - warmup)))
        if time_budget is not None:
            factor = time_learning_rate_factor(train_seconds, time_budget, cfg["train"].get("time_warmup_fraction", .1))
        for group in optimizer.param_groups:
            group["lr"] = cfg["train"]["learning_rate"] * factor
        optimizer.zero_grad(set_to_none=True)
        with amp_context(cfg, device):
            loss = model.training_nll(batch, supervision).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        previous_train_seconds = train_seconds
        train_seconds += time.perf_counter() - t0
        step = next_step
        last_loss = float(loss.detach().cpu())
        validation = step % cfg["train"]["eval_every"] == 0 or step == end_step or step in extra_eval_steps
        if time_budget is not None:
            # Equal validation opportunities: 10%, 20%, ... of the time budget.
            # Overshoot is bounded by the final indivisible optimizer update.
            validation = (int(train_seconds / time_budget * 10) > int(previous_train_seconds / time_budget * 10)
                          or train_seconds >= time_budget or step == end_step)
        if validation or step % cfg["train"]["log_every"] == 0:
            val_loss = None
            if validation:
                _, losses = evaluate_model(model, val_data, cfg, device)
                val_loss = float(losses.mean())
            record = {"step": step, "target_events_seen": step * cfg["train"]["batch_size"] * targets_per_window,
                      "sampled_windows": step * cfg["train"]["batch_size"], "supervision": supervision,
                      "targets_per_window": targets_per_window,
                      "train_nll_per_event": last_loss, "val_nll_per_event": val_loss,
                      "train_seconds": train_seconds, "wall_seconds": starting_wall + time.perf_counter() - origin,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            if time_budget is not None:
                record["update_seconds"] = train_seconds - previous_train_seconds
            history.append(record)
            is_best = validation and val_loss < best
            if is_best:
                best = val_loss
            cp = {"schema_version": 1, "kind": kind, "signature": signature, "data_fingerprint": manifest["fingerprint"],
                  "config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                  "best_val_nll": best, "train_seconds": train_seconds,
                  "wall_seconds": starting_wall + time.perf_counter() - origin,
                  "sampler_rng": sampler.get_state(), "rng": rng_state(), "history": history}
            if is_best:
                save_checkpoint(run / "best.pt", cp)
            save_checkpoint(run / "last.pt", cp)
            _write_history(run / "history.jsonl", history)
            print(f"{kind} step={step}/{max_steps} train_event_nll={last_loss:.4f}" + (f" val_event_nll={val_loss:.4f}" if val_loss is not None else ""), flush=True)
    time_complete = time_budget is not None and train_seconds >= time_budget
    summary = {"kind": kind, "step": step, "complete": time_complete if time_budget is not None else step == max_steps, "best_val_nll": best,
               "supervision": supervision, "targets_per_window": targets_per_window,
               "sampled_windows": step * cfg["train"]["batch_size"],
               "target_events_seen": step * cfg["train"]["batch_size"] * targets_per_window, "train_seconds": train_seconds,
               "train_target_events_per_second": step * cfg["train"]["batch_size"] * targets_per_window / max(train_seconds, 1e-12),
               "history_event_occurrences": step * cfg["train"]["batch_size"] * cfg["model"]["context_events"],
               "parameters": model.parameter_counts(), "provenance": manifest["provenance"]}
    if time_budget is not None:
        summary.update(time_budget_seconds=time_budget, budget_overshoot_seconds=max(0., train_seconds-time_budget),
                       final_update_seconds=history[-1]["update_seconds"] if history else None,
                       stopping_reason="time_budget" if time_complete else "step_limit",
                       schedule="accumulated training time; validation/checkpoint time excluded")
    write_json(run / "training_summary.json", summary)
    return summary


def _write_history(path, history):
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in history:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(tmp, path)


def fit_baseline(cfg, manifest, smoothing=1.0):
    data = windows(cfg, manifest, "train", True)
    counts = np.full(cfg["tokenizer"]["bins"] ** 3, smoothing, dtype=np.float64)
    # Count exactly the eligible training targets, excluding per-session context warm-up.
    for a in data.arrays:
        target = np.asarray(a[data.context::data.stride])
        counts += np.bincount(joint_encode(target, cfg["tokenizer"]["bins"]), minlength=len(counts))
    return counts / counts.sum()


@torch.no_grad()
def benchmark(model, data, cfg, device):
    model.eval()
    history = tensor_batch(data, [0], device)[:, :-1]
    for _ in range(3):
        with amp_context(cfg, device):
            model.next_event(history)
    durations = []
    for _ in range(cfg["evaluation"]["benchmark_repeats"]):
        sync(device)
        start = time.perf_counter()
        with amp_context(cfg, device):
            model.next_event(history)
        sync(device)
        durations.append((time.perf_counter() - start) * 1000)
    return {"greedy_ms_per_event_p50": float(np.median(durations)),
            "greedy_ms_per_event_p95": float(np.quantile(durations, .95)),
            "repeats": len(durations), "batch_size": 1, "kv_cache": False,
            "scope": "model-only complete-event greedy generation; synchronized device; not exchange latency"}


def evaluate(cfg):
    manifest = load_prepared(cfg)
    device = device_for(cfg["train"]["device"])
    torch.set_num_threads(cfg["train"]["threads"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    probabilities = fit_baseline(cfg, manifest)
    write_json(out / "frequency_baseline.json", {"smoothing": 1.0, "probabilities": probabilities.tolist(), "data_fingerprint": manifest["fingerprint"]})
    models = {}
    for kind in ("joint", "sequential"):
        cp = load_checkpoint(out / kind / "best.pt")
        if cp["signature"] != training_signature(cfg, manifest, kind):
            raise ValueError(f"{kind}: checkpoint/config mismatch")
        model = EventTransformer(cfg, kind).to(device)
        model.load_state_dict(cp["model"])
        model.eval()
        models[kind] = (model, cp["step"])
    rows, session_rows = [], []
    details = out / "evaluation"
    details.mkdir(exist_ok=True)
    for split in ("val", "test", "heldout"):
        data = windows(cfg, manifest, split)
        ids = data.evaluation_indices(cfg["evaluation"]["max_windows"])
        targets = np.stack([data[int(i)][-1] for i in ids])
        all_losses = {"frequency": -np.log(probabilities[joint_encode(targets, cfg["tokenizer"]["bins"])])}
        for kind, (model, _) in models.items():
            eval_ids, losses = evaluate_model(model, data, cfg, device)
            if not np.array_equal(ids, eval_ids):
                raise RuntimeError("Evaluation targets differ across models")
            all_losses[kind] = losses
        locations = [data.locate(int(i)) for i in ids]
        for name, losses in all_losses.items():
            per_event = []
            for i, (sid, start), nll in zip(ids, locations, losses):
                s = data.sessions[sid]
                per_event.append({"window_index": int(i), "session": s["id"], "symbol": s["symbol"], "date": s["date"],
                                  "target_event_index": start + data.context, "nll_per_event": float(nll)})
            pf = pd.DataFrame(per_event)
            pf.to_csv(details / f"{name}_{split}_events.csv", index=False)
            grouped = pf.groupby(["session", "symbol", "date"])["nll_per_event"].agg(["mean", "count"]).reset_index()
            for record in grouped.to_dict("records"):
                session_rows.append({"model": name, "split": split, **record})
            rows.append({"model": name, "split": split, "nll_per_event": float(np.mean(losses)),
                         "session_macro_nll": float(grouped["mean"].mean()), "evaluated_events": len(ids),
                         "available_windows": len(data), "selected_checkpoint_step": models[name][1] if name in models else 0,
                         "provenance": manifest["provenance"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(session_rows).to_csv(out / "session_metrics.csv", index=False)
    timings = {kind: benchmark(model, windows(cfg, manifest, "val"), cfg, device) for kind, (model, _) in models.items()}
    write_json(out / "benchmark.json", {"environment": environment(device), "models": timings})
    write_json(out / "evaluation_metadata.json", {"data_fingerprint": manifest["fingerprint"], "config_fingerprint": digest(cfg),
                "config": cfg, "provenance": manifest["provenance"], "event_nll_unit": "nats per complete discretized event",
                "selection": "best validation NLL; test and heldout are not used for model selection",
                "subsampling": "equally spaced deterministic window indices per split up to max_windows",
                "warning": "Overlapping windows are dependent. No IID-event significance claims. Synthetic results are pipeline checks only."})
    return rows


@torch.no_grad()
def generate(cfg, kind, steps=10, sample=False):
    if steps < 1:
        raise ValueError("steps must be positive")
    manifest = load_prepared(cfg)
    cp = load_checkpoint(Path(cfg["output_dir"]) / kind / "best.pt")
    if cp["signature"] != training_signature(cfg, manifest, kind):
        raise ValueError("Checkpoint/config mismatch")
    device = device_for(cfg["train"]["device"])
    torch.manual_seed(cfg["train"]["seed"])
    model = EventTransformer(cfg, kind).to(device)
    model.load_state_dict(cp["model"])
    model.eval()
    data = windows(cfg, manifest, "heldout")
    history = tensor_batch(data, [0], device)[:, :-1]
    output = []
    for _ in range(steps):
        with amp_context(cfg, device):
            event = model.next_event(history, sample=sample)
        output.append(event.cpu().numpy()[0])
        history = torch.cat([history[:, 1:], event[:, None]], dim=1)
    codes = np.asarray(output)
    decoded = EventBinner.from_dict(manifest["binner"]).inverse(codes)
    frame = pd.DataFrame({"step": np.arange(steps), "dt_bin": codes[:, 0], "return_bin": codes[:, 1], "size_bin": codes[:, 2],
                          "approx_interarrival_seconds": np.expm1(decoded[:, 0]), "approx_log_return_bps": decoded[:, 1],
                          "approx_size": np.expm1(decoded[:, 2])})
    path = Path(cfg["output_dir"]) / f"{kind}_generated_events.csv"
    frame.to_csv(path, index=False)
    write_json(path.with_suffix(".metadata.json"), {"synthetic_generated_sequence": True, "seed": cfg["train"]["seed"],
                "sampling": "multinomial" if sample else "greedy", "data_provenance": manifest["provenance"],
                "note": "Bin-center approximations. No order book, quote feed, execution model or trading P&L."})
    return path
