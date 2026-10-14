"""Post-hoc probability-level diagnosis; no fitting or additional hypothesis tests."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    examples = pd.read_csv(HERE / 'new_examples.csv')
    predictions = np.load(HERE / 'new_predictions.npz', allow_pickle=False)
    rows = []
    for symbol, group in examples.groupby('symbol'):
        idx = group.index.to_numpy()
        y = group.T2_label.to_numpy()
        observed = np.eye(3)[y]
        for method in ['stats', 'event_random', 'clock_random', 'event_pretrained', 'clock_pretrained']:
            for seed in [42, 43, 44]:
                q = predictions[f'{method}_seed{seed}'][idx]
                rows.append({'symbol': symbol, 'method': method, 'seed': seed,
                             'brier': ((q - observed) ** 2).sum(1).mean(),
                             'accuracy': (q.argmax(1) == y).mean(),
                             'mean_confidence': q.max(1).mean(),
                             **{f'predicted_class{k}': q[:, k].mean() for k in range(3)},
                             **{f'observed_class{k}': observed[:, k].mean() for k in range(3)}})
    pd.DataFrame(rows).to_csv(HERE / 'posthoc_probability_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
