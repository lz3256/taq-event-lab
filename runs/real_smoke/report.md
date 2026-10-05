# TAQ Event Lab

REAL TAQ — SHORT RUN / PIPELINE VALIDATION ONLY

```text
     model  val_nll  test_nll  heldout_nll  parameters  backbone_parameters  train_target_events_per_second  greedy_ms_per_event_p50  finished_steps
 frequency 3.986940  3.740435     4.540833         NaN                  NaN                             NaN                      NaN             NaN
     joint 6.061369  5.950664     6.025491   2001024.0            1779840.0                       89.987655                 2.640375            10.0
sequential 4.853028  4.755945     4.947094   1863168.0            1779840.0                       31.677463                12.663459            10.0
```

- At least one model ran 20 steps or fewer. This report checks the real-data pipeline; it cannot establish model quality or a tokenization winner.
- This is a trade-only TAQ adaptation, not an official TradeFM/LOBS5 reproduction. No quote/order-book state is modeled.
- Identical field bins, history-event count and sampled target stream; different vocabularies and token sequence lengths. Total parameters and compute need not match.
- Sequential NLL sums conditional field losses, using valid-field normalization. Scores are comparable only on this shared discretization.
- Targets are sampled with replacement during training; exposures are not unique events. Evaluation uses fixed deterministic subsampling when capped.
- Best checkpoint is selected by validation NLL. Future-date and held-out-stock tests are separate. Held-out stocks never calibrate the tokenizer.
- Performance curves are one-run observations, not confidence intervals. Overlapping windows are dependent. Repeat seeds and assess uncertainty by sessions/days before generalizing.
- Generation latency is batch-1 model-only greedy decoding without a KV cache, not exchange/network latency; sequential generation recomputes prefixes.
- Timestamp ties retain file/row order. Corrections and trade conditions use the configured strict allowlist, not a universal TAQ cleaning standard.
- No transaction-cost backtest, profitable alpha claim, downstream fine-tuning, multi-GPU benchmark, or production trading claim is supported by this report.

![Validation by events](validation_by_events.png)

![Validation by time](validation_by_time.png)
