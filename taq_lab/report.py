from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "taq-event-lab-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "taq-event-lab-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import digest, read_json
from .data import load_prepared


def report(cfg):
    root = Path(cfg["output_dir"])
    meta = read_json(root / "evaluation_metadata.json")
    manifest = load_prepared(cfg)
    if meta["config_fingerprint"] != digest(cfg) or meta["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Report metadata mismatch. Run evaluate with the intended config.")
    metrics = pd.read_csv(root / "metrics.csv")
    timing = read_json(root / "benchmark.json")
    synthetic = meta["provenance"] == "synthetic"
    banner = "SYNTHETIC DATA — PIPELINE VALIDATION ONLY" if synthetic else "REAL TAQ — EXPLORATORY RESEARCH"
    colors = {"joint": "#167d8d", "sequential": "#bb5b37"}
    summaries = {kind: read_json(root / kind / "training_summary.json") for kind in ("joint", "sequential")}
    short_run = any(s["step"] <= 20 for s in summaries.values())
    if short_run and not synthetic:
        banner = "REAL TAQ — SHORT RUN / PIPELINE VALIDATION ONLY"
    for xkey, xlabel, filename in [("target_events_seen", "Sampled target-event exposures", "validation_by_events.png"),
                                   ("train_seconds", "Training seconds (excludes validation)", "validation_by_time.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for kind in ("joint", "sequential"):
            with (root / kind / "history.jsonl").open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            rows = [r for r in rows if r["val_nll_per_event"] is not None]
            ax.plot([r[xkey] for r in rows], [r["val_nll_per_event"] for r in rows], marker="o", color=colors[kind], label=kind)
            summaries[kind] = read_json(root / kind / "training_summary.json")
        ax.axhline(float(metrics[(metrics.model == "frequency") & (metrics.split == "val")].nll_per_event.iloc[0]),
                   color="#68737d", linestyle="--", label="frequency baseline")
        ax.set(xlabel=xlabel, ylabel="NLL / complete event (nats)", title=banner)
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(root / filename, dpi=160)
        plt.close(fig)
    overview = []
    for kind in ("frequency", "joint", "sequential"):
        row = {"model": kind}
        for split in ("val", "test", "heldout"):
            record = metrics[(metrics.model == kind) & (metrics.split == split)].iloc[0]
            row[f"{split}_nll"] = float(record.nll_per_event)
        if kind != "frequency":
            row.update({"parameters": summaries[kind]["parameters"]["total"],
                        "backbone_parameters": summaries[kind]["parameters"]["backbone"],
                        "train_target_events_per_second": summaries[kind]["train_target_events_per_second"],
                        "greedy_ms_per_event_p50": timing["models"][kind]["greedy_ms_per_event_p50"],
                        "finished_steps": summaries[kind]["step"]})
        overview.append(row)
    table = pd.DataFrame(overview)
    table.to_csv(root / "summary.csv", index=False)
    limitations = [
        "This is a trade-only TAQ adaptation, not an official TradeFM/LOBS5 reproduction. No quote/order-book state is modeled.",
        "Identical field bins, history-event count and sampled target stream; different vocabularies and token sequence lengths. Total parameters and compute need not match.",
        "Sequential NLL sums conditional field losses, using valid-field normalization. Scores are comparable only on this shared discretization.",
        "Targets are sampled with replacement during training; exposures are not unique events. Evaluation uses fixed deterministic subsampling when capped.",
        "Best checkpoint is selected by validation NLL. Future-date and held-out-stock tests are separate. Held-out stocks never calibrate the tokenizer.",
        "Performance curves are one-run observations, not confidence intervals. Overlapping windows are dependent. Repeat seeds and assess uncertainty by sessions/days before generalizing.",
        "Generation latency is batch-1 model-only greedy decoding without a KV cache, not exchange/network latency; sequential generation recomputes prefixes.",
        "Timestamp ties retain file/row order. Corrections and trade conditions use the configured strict allowlist, not a universal TAQ cleaning standard.",
        "No transaction-cost backtest, profitable alpha claim, downstream fine-tuning, multi-GPU benchmark, or production trading claim is supported by this report.",
    ]
    if cfg['train'].get('supervision', 'last_event') == 'dense':
        limitations.insert(0, 'Dense training averages complete-event NLL across all next-event positions; evaluation still scores only the last complete event. Early training positions have shorter histories. Target exposures count repeated supervision, not unique events or epochs.')
    if synthetic:
        limitations.insert(0, "All numbers below come from fabricated data and test the software only. They are not market results.")
    elif short_run:
        limitations.insert(0, "At least one model ran 20 steps or fewer. This report checks the real-data pipeline; it cannot establish model quality or a tokenization winner.")
    if not all(s["complete"] for s in summaries.values()):
        limitations.insert(0, "At least one run stopped before max_steps; this is a partial-run report.")
    warnings = manifest["audit"]["warnings"]
    table_html = table.to_html(index=False, float_format=lambda x: f"{x:.4f}", na_rep="—", border=0)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAQ Event Lab — Experiment Report</title><style>
body{{font:16px/1.65 system-ui,sans-serif;background:#f4f6f7;color:#203039;max-width:1100px;margin:40px auto;padding:0 24px}}
h1{{line-height:1.2}} .banner{{padding:16px;background:{'#fff0da' if synthetic else '#deeff0'};border-left:5px solid #bb5b37;font-weight:700}}
section{{background:white;padding:24px;margin:22px 0;border-radius:12px}} table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:9px;text-align:right;border-bottom:1px solid #ddd}} td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow-x:auto}}img{{width:100%;max-width:850px}} code{{overflow-wrap:anywhere}}li{{margin:8px 0}}
</style><h1>Joint vs. Sequential Event Tokenization</h1><p class="banner">{banner}</p>
<section><h2>Results</h2><p>Lower event NLL is better. Same raw events and quantization; validation selects the checkpoint.</p><div class="scroll">{table_html}</div></section>
<section><h2>Learning curves</h2><img src="validation_by_events.png" alt="Validation NLL by sampled target events"><img src="validation_by_time.png" alt="Validation NLL by training time"></section>
<section><h2>Scope and interpretation</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in limitations)}</ul></section>
<section><h2>Data audit</h2><p>{len(manifest['sessions'])} sessions; {sum(s['events'] for s in manifest['sessions']):,} retained feature events.</p>
<pre>{html.escape(json.dumps(manifest['audit']['filters'], indent=2))}</pre><ul>{''.join('<li>'+html.escape(w)+'</li>' for w in warnings)}</ul>
<p>Data fingerprint: <code>{meta['data_fingerprint']}</code></p><p>Full details: metrics.csv, session_metrics.csv, evaluation/*_events.csv, evaluation_metadata.json, and each model's config/history/checkpoints.</p></section>
<section><h2>References</h2><p><a href="https://arxiv.org/abs/2602.23784">TradeFM</a> · <a href="https://arxiv.org/abs/2309.00638">LOBS5</a> · <a href="https://www.nyse.com/data-products/catalog/daily-taq">NYSE Daily TAQ</a></p></section></html>'''
    (root / "report.html").write_text(document, encoding="utf-8")
    (root / "report.md").write_text("# TAQ Event Lab\n\n" + banner + "\n\n```text\n" + table.to_string(index=False) + "\n```\n\n" +
                                      "\n".join("- " + x for x in limitations) + "\n\n![Validation by events](validation_by_events.png)\n\n![Validation by time](validation_by_time.png)\n", encoding="utf-8")
    print(f"Report: {root / 'report.html'}", flush=True)
    return root / "report.html"
