from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, write_json
from .data import make_synthetic, prepare
from .engine import evaluate, generate, train
from .report import report


def demo(directory, steps=12):
    root = Path(directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Demo destination must be empty; use a fresh --directory")
    root.mkdir(parents=True, exist_ok=True)
    make_synthetic(root / "raw", events_per_session=128)
    template = Path(__file__).resolve().with_name("default_config.json")
    cfg = json.loads(template.read_text())
    cfg["data"].update({"input_glob": "raw/*.csv", "prepared_dir": "prepared", "provenance": "synthetic"})
    cfg["model"].update({"context_events": 16, "d_model": 32, "heads": 2, "layers": 1, "dropout": 0.0})
    cfg["train"].update({"device": "cpu", "threads": 2, "batch_size": 8, "max_steps": steps,
                         "warmup_steps": min(2, steps), "eval_every": max(1, steps // 3), "log_every": 2,
                         "learning_rate": .001})
    cfg["evaluation"].update({"batch_size": 32, "stride": 16, "max_windows": 128, "benchmark_repeats": 5})
    cfg["output_dir"] = "results"
    write_json(root / "config.json", cfg)
    loaded = load_config(root / "config.json")
    prepare(loaded)
    for kind in ("joint", "sequential"):
        train(loaded, kind)
    evaluate(loaded)
    generate(loaded, "joint", steps=10)
    generate(loaded, "sequential", steps=10)
    return report(loaded)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TAQ Event Lab — trade-only controlled tokenization experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run complete CPU pipeline with clearly marked synthetic data")
    p.add_argument("--directory", required=True)
    p.add_argument("--steps", type=int, default=12)
    for command in ("prepare", "train", "evaluate", "report", "run", "generate", "download-wrds"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "prepare":
            p.add_argument("--overwrite", action="store_true")
        if command in ("train", "generate"):
            p.add_argument("--model", choices=("joint", "sequential"), required=True)
        if command in ("train", "run"):
            p.add_argument("--resume", action="store_true")
        if command == "train":
            p.add_argument("--stop-after", type=int, help="Pause at absolute step, preserving the original LR schedule")
        if command == "generate":
            p.add_argument("--steps", type=int, default=10)
            p.add_argument("--sample", action="store_true")
        if command == "download-wrds":
            p.add_argument("--output", required=True)
            p.add_argument("--username")
            p.add_argument("--library", default="taqmsec")
    args = parser.parse_args(argv)
    if args.command == "demo":
        if args.steps < 1:
            parser.error("--steps must be positive")
        demo(args.directory, args.steps)
        return
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare(cfg, overwrite=args.overwrite)
    elif args.command == "train":
        train(cfg, args.model, args.resume, args.stop_after)
    elif args.command == "evaluate":
        evaluate(cfg)
    elif args.command == "report":
        report(cfg)
    elif args.command == "generate":
        print(generate(cfg, args.model, args.steps, args.sample))
    elif args.command == "download-wrds":
        from .wrds_download import download
        download(cfg, args.output, args.username, args.library)
    elif args.command == "run":
        if not (Path(cfg["data"]["prepared_dir"]) / "manifest.json").exists():
            prepare(cfg)
        for kind in ("joint", "sequential"):
            existing = (Path(cfg["output_dir"]) / kind / "last.pt").exists()
            summary = train(cfg, kind, resume=args.resume and existing)
            if not summary['complete']:
                raise RuntimeError(f"{kind} did not complete its training budget; evaluation was not started")
        evaluate(cfg)
        report(cfg)


if __name__ == "__main__":
    main()
