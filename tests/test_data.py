import copy

import numpy as np
import pandas as pd
import pytest

from taq_lab.common import validate_config
from taq_lab.data import EventWindows, load_prepared, normalize_and_clean, prepare
from taq_lab.tokenization import EventBinner, joint_decode, joint_encode, sequential_decode, sequential_encode


def test_encoding_bijections_and_field_order():
    events = np.indices((8, 8, 8)).reshape(3, -1).T
    assert len(np.unique(joint_encode(events, 8))) == 512
    np.testing.assert_array_equal(joint_decode(joint_encode(events, 8), 8), events)
    for order in [(0, 1, 2), (1, 2, 0)]:
        np.testing.assert_array_equal(sequential_decode(sequential_encode(events, 8, order), 8, order), events)


def test_quality_timezones_and_nanosecond_precision(cfg):
    raw = pd.DataFrame({"symbol": ["AAPL"] * 6,
                        "timestamp": ["2025-02-03T14:30:00.000000001Z", "2025-02-03T14:30:00.000000003Z",
                                      "2025-02-03T14:31:00Z", "2025-02-03T14:31:01Z", "2025-02-03T21:00:00Z", "bad"],
                        "price": [100] * 6, "size": [100] * 6, "tr_corr": ["00", "00", "01", "00", "00", "00"],
                        "tr_scond": ["@", "@", "@", "T", "@", "@"]})
    clean, audit = normalize_and_clean(raw, cfg)
    assert len(clean) == 2
    assert (clean.timestamp.iloc[1] - clean.timestamp.iloc[0]).value == 2
    assert sum(audit["filters"].values()) + audit["retained_rows"] == audit["input_rows"]
    assert audit["filters"]["excluded_correction"] == 1
    assert audit["filters"]["excluded_condition"] == 1


def test_missing_quality_fails(cfg):
    raw = pd.DataFrame({"symbol": ["AAPL"], "timestamp": ["2025-02-03 09:31:00"], "price": [100], "size": [10]})
    with pytest.raises(ValueError, match="Missing tr_corr"):
        normalize_and_clean(raw, cfg)


def test_wrds_columns_suffix_and_split(cfg):
    raw = pd.DataFrame({"sym_root": ["AAPL", "AAPL"], "sym_suffix": ["", "A"], "date": ["2025-02-03"] * 2,
                        "time_m": ["09:31:00.000000001", "09:31:00.000000003"], "price": [100, 101], "size": [10, 20],
                        "tr_corr": [0, 0], "tr_scond": ["@", "@"]})
    clean, _ = normalize_and_clean(raw, cfg)
    assert clean.symbol.tolist() == ["AAPL"]


def test_preparation_no_leakage_or_cross_session_windows(cfg):
    m = prepare(cfg)
    root = cfg["data"]["prepared_dir"]
    train_values = np.concatenate([np.load(f"{root}/{s['features']}") for s in m["sessions"] if s["split"] == "train"])
    expected = EventBinner(8).fit(train_values)
    np.testing.assert_array_equal(expected.edges, m["binner"]["edges"])
    assert all(s["symbol"] in cfg["split"]["seen_symbols"] for s in m["sessions"] if s["split"] == "train")
    data = EventWindows(root, m, "test", 4)
    for i in [0, int(data.ends[0]) - 1, int(data.ends[0]), len(data) - 1]:
        sid, start = data.locate(i)
        np.testing.assert_array_equal(data[i], data.arrays[sid][start:start + 5])
        assert data[i].shape == (5, 3)
    # Change held-out raw features and re-prepare: training quantiles must remain identical.
    from pathlib import Path
    source = next(Path(cfg["data"]["input_glob"]).parent.glob("*.csv"))
    frame = pd.read_csv(source)
    mask = frame.symbol.isin(cfg["split"]["heldout_symbols"])
    frame.loc[mask, "size"] = 1e9
    frame.to_csv(source, index=False)
    changed = prepare(cfg, overwrite=True)
    np.testing.assert_array_equal(m["binner"]["edges"], changed["binner"]["edges"])


def test_fingerprint_tampering_detected(cfg):
    m = prepare(cfg)
    path = f"{cfg['data']['prepared_dir']}/{m['sessions'][0]['codes']}"
    a = np.load(path)
    a[0, 0] = (a[0, 0] + 1) % 8
    np.save(path, a)
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(cfg)


def test_overlapping_splits_fail(cfg):
    broken = copy.deepcopy(cfg)
    broken["split"]["val_dates"] = broken["split"]["train_dates"]
    with pytest.raises(ValueError, match="train dates"):
        validate_config(broken)


def test_duplicate_quantiles_finite():
    binner = EventBinner(8).fit(np.zeros((100, 3)))
    codes = binner.transform(np.array([[0., 0., 0.], [1e6, -1e6, 1e6]]))
    assert ((codes >= 0) & (codes < 8)).all()
    assert np.isfinite(binner.inverse(codes)).all()


def test_zip_chunks_match_full_cleaning(cfg, tmp_path):
    from zipfile import ZipFile
    from taq_lab.data import clean_inputs
    raw = pd.DataFrame({
        "SYM_ROOT": ["AAPL"] * 6, "DATE": ["2025-02-03"] * 6,
        "TIME_M": ["09:30:00.000000003", "09:30:00.000000001",
                   "09:30:00.000000001", "16:00:00", "09:31:00", "09:32:00"],
        "PRICE": [100] * 6, "SIZE": [10] * 6,
        "TR_CORR": ["00"] * 5 + ["01"], "TR_SCOND": ["@"] * 4 + ["T", "@"],
    })
    archive = tmp_path / "export.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("export.csv", raw.to_csv(index=False))
    expected, reference = normalize_and_clean(raw, cfg)
    actual, audit = clean_inputs([archive], cfg, chunksize=2)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    for key in ("input_rows", "retained_rows", "filters", "same_symbol_timestamp_ties", "identical_print_rows", "correction_counts", "condition_counts"):
        assert audit[key] == reference[key]
    assert audit["raw_sessions"] == {"AAPL 2025-02-03": 6}
