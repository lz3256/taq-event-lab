# Self-Supervised Trade-Event Representations: Pretraining vs. Transfer

Research update recorded October 13, 2026. The original tokenization studies remain in the repository; this update follows one joint-event model into a downstream volatility task.

## Pretraining

A roughly 2M-parameter joint-event Transformer was pretrained on 7.9M TAQ training events. On the original future-date and unseen-stock (NVDA) splits, next-event NLL improved over Markov by **0.1137 / 0.1774 nats per event**. The larger 0.16/0.25 figures in other project records belong to the sequential-field model and are not substituted here.

## Transfer experiment

The downstream task is three-class short-horizon relative volatility. Thresholds and selected models were frozen before ten additional historical trading dates, covering six stocks and **32,162 evaluation anchors**. Seen stocks are AAPL, AMZN and MSFT; the unseen-stock group is NVDA, JPM and XOM. These anchors are not 32,162 independent observations; inference resamples dates jointly across stocks and seeds.

Random and pretrained encoders use comparable event- and clock-based history policies. The clock policy moves the older history endpoint to a fixed past clock cutoff; it is not a claim that every input becomes a regular-time bar. Forecasts combine the same handcrafted statistics with frozen event representations. The statistics-only reference uses regularized multinomial logistic regression on 13 features, including past volatility, trade activity, normalized returns, size, trade age and intraday clock features.

## Recorded findings

| New-date cohort | Event/random NLL | Clock/random NLL | Event/pretrained NLL | Clock/pretrained NLL |
|---|---:|---:|---:|---:|
| Seen stocks | 1.0804 | 1.0822 | 1.0312 | 1.0316 |
| Unseen stocks | 1.2614 | 1.3368 | 1.4215 | 1.3932 |

On unseen stocks, switching the pretrained model from event to clock history lowered mean NLL: **paired change −0.0282**, with date-block **95% CI [−0.0413, −0.0153]** and Holm-adjusted p≈0.0003 in the three-comparison primary family. The effect improves the window policy; it does not establish a pretraining advantage. The pretrained clock model still trails random encoders and uniform probabilities (NLL ln(3)≈1.0986) in the unseen-stock group.

The handcrafted-statistics benchmark was severely miscalibrated on JPM/XOM: average confidence about **91–94%**, versus **23–27% accuracy**, below the **33.3% uniform-random chance level**. The diagnostic was added after outcomes. This is the project's own reference model; it does not establish a defect in unrelated published models. Beating this miscalibrated reference alone is not evidence of successful transfer.

Seen-stock pretrained encoders have lower average NLL than random encoders, but that direct contrast was not one of the three pre-specified confirmation tests. Any added significance comparison must be labeled supplemental/post hoc. Clock-policy improvement occurred in two of three seed averages; it should not be described as seed-universal.

## Inference and scope

The three primary comparisons are clock-policy effect, its interaction with pretraining, and incremental performance over the statistics benchmark. They use seed-averaged losses, equal-weight stocks within dates, and equal-weight dates. Paired circular date-block bootstrap uses block length two and 10,000 draws; Holm adjustment applies to the three unseen-stock primary p-values. The displayed percentile intervals are pointwise, not simultaneous Holm-adjusted intervals.

The study covers ten additional historical dates and conditions on frozen fitted models. It does not resample training, establish transfer to all stocks, or demonstrate trading profitability. Original pretraining success and downstream cross-stock failure are different empirical conclusions.

## Public package

`continuation/` preserves the numerical window-selection, warm-start and representation-comparison implementations plus four substantive data-independent tests. `confirmation/` preserves evaluation, paired inference, probability diagnostics and raw-label audit source. `protocols/` records the historical design and reporting amendment with private host paths and artifact hashes removed. `results/` includes aggregate tables only; no per-anchor probabilities, trade data or trained weights.

From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s research/2026-10-13/continuation -p 'test_*.py' -v
```

Full historical reruns require your authorized TAQ records, the original trained checkpoints, feature/threshold metadata, probe heads and saved-prediction artifacts. These are excluded from the public repository. The confirmation scripts document methods and expected inputs; they are not a one-command empirical reproduction from public files alone. For a new run, create an ignored local `confirmation/plan.json` from the public protocol and establish new input hashes before observing outcomes. Existing model metadata and relative prepared-data paths must likewise be supplied for that run.

Developed by Linfeng Zhao with substantial OpenAI Codex assistance. WRDS TAQ is the market-data source. The project builds on the earlier TradeFM/LOBS5-inspired lab; it is a small controlled adaptation, not an official reproduction.
