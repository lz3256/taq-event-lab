<img src="assets/social-preview.jpg" width="560" alt="TAQ Event Lab cover">

# TAQ Event Lab

**When do next-event pretraining gains transfer to volatility classification across stocks?**

This research lab compares joint event tokens with sequential field tokens for TAQ trade data. It includes frequency baselines, cleaning audits, train-only quantization, unseen-stock evaluation, event generation, downstream direction tasks and frozen-representation clock probes.

## Latest research: October 13, 2026

**[Self-Supervised Trade-Event Representations: Pretraining vs. Transfer](research/2026-10-13/README.md)** follows a ~2M-parameter joint-event Transformer pretrained on 7.9M TAQ training events into a frozen-model volatility-classification study.

- Next-event NLL improved over Markov by **0.114 / 0.177 nats per event** on the original future-date / unseen-stock splits.
- In a locked evaluation over **six stocks, ten additional dates and 32,162 anchors**, clock-based history lowered pretrained unseen-stock NLL: **paired change −0.028**, 95% CI **[−0.041, −0.015]**.
- A handcrafted-statistics logistic benchmark was severely miscalibrated on JPM/XOM: **91–94% confidence versus 23–27% accuracy**, below uniform-random chance.
- **Cross-stock transfer remained unestablished:** pretrained representations still trailed random encoders and uniform forecasts in the unseen-stock group.

Window improvement is a useful result without implying a pretraining-transfer benefit. The update publishes method code, aggregate results, design records and explicit statistical limits. Earlier tokenization/exposure/compute comparisons remain separately documented below.

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

- [October 2026 pretraining-versus-transfer update](research/2026-10-13/README.md)

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
