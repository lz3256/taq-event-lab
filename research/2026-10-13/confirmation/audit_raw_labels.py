"""Spot-check new T2 targets directly against raw prices, without clock_session."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE / 'work' / 'taq_confirmation'


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    examples = pd.read_csv(HERE / 'new_examples.csv')
    evidence = []
    for day in [plan['dates'][0], plan['dates'][-1]]:
        parts = []
        for chunk in pd.read_csv(WORK / 'raw' / (day + '.csv.gz'),
                                 usecols=['sym_root', 'time_m', 'price'], chunksize=250000,
                                 keep_default_na=False):
            parts.append(chunk)
        frame = pd.concat(parts, ignore_index=True)
        for symbol, group in frame.groupby('sym_root', sort=True):
            # Parse exact nanosecond strings independently of the original cleaner.
            ns = pd.to_datetime(day + ' ' + group.time_m, format='mixed').astype('datetime64[ns]').astype('int64').to_numpy()
            opening = pd.Timestamp(day + ' 09:30:00').value
            times = (ns - opening) / 1e9
            price = group.price.to_numpy(dtype=float)
            assert np.all(np.diff(times) >= 0)
            detail = examples[(examples.date == day) & (examples.symbol == symbol)]
            # Deterministic first/middle/last available anchor; no outcome-based selection.
            for position in sorted(set([0, len(detail) // 2, len(detail) - 1])):
                row = detail.iloc[position]
                anchor = int(row.anchor_seconds)
                grids = [np.arange(anchor - 300, anchor + 1), np.arange(anchor, anchor + 31)]
                rv = []
                for grid in grids:
                    indices = np.searchsorted(times, grid, side='right') - 1
                    assert np.all(indices >= 0)
                    rv.append(float(np.sum(np.diff(np.log(price[indices])) ** 2)))
                target = float(np.sqrt(rv[1] / (.1 * rv[0])))
                np.testing.assert_allclose(target, row.T2_target, rtol=1e-7, atol=1e-9)
                evidence.append({'date': day, 'symbol': symbol, 'anchor': anchor,
                                 'absolute_target_error': abs(target - row.T2_target)})
    result = {'passed': True, 'raw_price_target_checks': len(evidence),
              'sample_policy': 'first/middle/last retained anchor on first/last locked dates for all six stocks',
              'checks': evidence}
    (HERE / 'raw_label_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Independent raw-price target checks passed:', len(evidence))


if __name__ == '__main__':
    main()
