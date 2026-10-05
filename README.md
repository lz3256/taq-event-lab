<img src="assets/social-preview.jpg" width="560" alt="TAQ Event Lab cover">

# TAQ Event Lab

**How does event tokenization change a small Transformer's behavior on financial trade sequences?**

This research lab compares joint event tokens with sequential field tokens for TAQ trade data. It includes frequency baselines, cleaning audits, train-only quantization, unseen-stock evaluation, event generation, downstream direction tasks and frozen-representation clock probes.

## Work completed

The implementation covers two trade-event tokenizations, small Transformer baselines, TAQ cleaning audits, and controlled comparisons under event-exposure and training-time budgets. The experiments extend to repeated seeds, downstream direction tasks and frozen-representation clock probes, showing where the ranking changes with the comparison budget.

## Research highlights

- From-scratch small Transformers with two event encodings.
- Controlled target-event exposure, three-seed replication and separate equal-time experiments.
- Per-session evaluation and explicit held-out stock/date splits.
- Checkpoint selection, resumable runs and synthetic end-to-end examples.
- A documented tradeoff: better likelihood at fixed event exposure does not imply better compute efficiency.

## What the recorded experiments show

Sequential field tokens lead in the dense-supervision experiments at equal target-event exposure. In the recorded single-seed, equal-CPU-training-time comparison, joint tokens instead lead on both test sets. These are distinct comparisons, and neither result establishes a generally superior encoding.

This is a small-scale adaptation inspired by TradeFM and LOBS5, not an official reproduction. Trade likelihood and direction classification are not trading-profit evidence. The current implementation does not include a full order book, market-making simulator or transaction-cost backtest.

## Quick start

Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m taq_lab demo --directory runs/demo --steps 18
```

Choose a new output directory for each demo. It creates synthetic trade events, trains both models and writes an HTML report to `runs/demo/results/report.html`. Synthetic demo metrics only validate the workflow.

Run the automated checks:

```bash
python -m pytest
```

For real data, supply your own authorized TAQ exports and update a configuration. The public repository excludes raw/prepared market records, row-level predictions and trained checkpoints. See [data import instructions](DATA_IMPORT.md) and [detailed project guide](GUIDE.zh.md).

## Experiment reports

- [First controlled experiment](FIRST_RESULTS.md)
- [Dense supervision](DENSE_RESULTS.md)
- [Three-seed dense replication](DENSE_STABILITY_RESULTS.md)
- [Compute-efficiency comparison](EFFICIENCY_RESULTS.md)
- [Downstream tasks](DOWNSTREAM_RESULTS.md)
- [Clock probes and frozen representations](CLOCK_PROBE_RESULTS.md)

## Structure

`taq_lab/` contains the implementation, `configs/` the experiment settings, `tests/` validation, and `runs/` selected aggregate results from previously recorded experiments. Large datasets, trained weights and local environments are excluded.

## Credits

Project owner: [Linfeng Zhao (@lz3256)](https://github.com/lz3256).

Developed with substantial assistance from OpenAI Codex for implementation, analysis, debugging and documentation. Recorded experiments and limitations are described in the linked reports.

The research is inspired by [TradeFM](https://arxiv.org/abs/2602.23784) and [LOBS5](https://arxiv.org/abs/2309.00638). This repository contains a small-scale adaptation and controlled experiments, not an official implementation or reproduction. Real TAQ records are not distributed in this repository.
