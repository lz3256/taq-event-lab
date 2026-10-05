from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .common import digest, file_hash, read_json, write_json
from .tokenization import EventBinner, FIELD_NAMES

SPLITS = ("train", "val", "test", "heldout")


def split_for(symbol, date, spec):
    if symbol in spec["seen_symbols"]:
        for name in ("train", "val", "test"):
            if date in spec[name + "_dates"]:
                return name
    if symbol in spec["heldout_symbols"] and date in spec["test_dates"]:
        return "heldout"
    return None


def load_raw(paths):
    """Small research dataset loader; inputs must fit RAM. No silent row truncation."""
    frames = []
    for path in paths:
        if str(path).endswith(".parquet"):
            frame = pd.read_parquet(path)
        elif str(path).endswith((".csv", ".csv.gz")):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported input: {path}; use csv, csv.gz or parquet")
        frame.columns = [str(c).lower() for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate case-insensitive columns in {path}")
        frame["_input_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def iter_raw(paths, chunksize=250_000):
    """Read CSV/ZIP/Parquet in bounded raw batches; never extract ZIP paths."""
    for path in paths:
        path = Path(path)
        if path.suffix == ".zip":
            with ZipFile(path) as archive:
                members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/"))
                if not members:
                    raise ValueError(f"No CSV members in {path}")
                for member in members:
                    with archive.open(member) as source:
                        yield from pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=chunksize)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize):
                yield batch.to_pandas()
        elif str(path).endswith((".csv", ".csv.gz")):
            yield from pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize)
        else:
            raise ValueError(f"Unsupported input: {path}")


def clean_inputs(paths, cfg, chunksize=250_000):
    """Stream raw input, retaining only cleaned rows in RAM for global sorting."""
    frames, warnings = [], set()
    audit = {"input_rows": 0, "filters": {}, "warnings": []}
    counts = {key: Counter() for key in ("filters", "correction_counts", "condition_counts", "raw_sessions")}
    for i, raw in enumerate(iter_raw(paths, chunksize)):
        raw = raw.reset_index(drop=True)
        raw.columns = [str(c).lower() for c in raw.columns]
        if raw.columns.duplicated().any():
            raise ValueError("Duplicate case-insensitive input columns")
        dc, sc = cfg["data"]["date_column"].lower(), cfg["data"]["symbol_column"].lower()
        if dc in raw and (sc in raw or "sym_root" in raw):
            symbols = raw[sc].astype(str) if sc in raw else raw.sym_root.astype(str)
            if sc not in raw and "sym_suffix" in raw:
                symbols = symbols + raw.sym_suffix.fillna("").astype(str).str.strip().map(lambda s: "." + s if s else "")
            counts["raw_sessions"].update((symbols + " " + raw[dc].astype(str)).value_counts().to_dict())
        clean, part = normalize_and_clean(raw, cfg)
        clean["_row"] += audit["input_rows"]
        audit["input_rows"] += part["input_rows"]
        for key in ("filters", "correction_counts", "condition_counts"):
            counts[key].update(part.get(key, {}))
        warnings.update(part["warnings"])
        frames.append(clean)
        if (i + 1) % 4 == 0:
            print(f"Scanned {audit['input_rows']:,} raw rows", flush=True)
    if not frames:
        raise ValueError("No input rows")
    frame = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date", "timestamp", "_row"], kind="stable")
    audit.update({k: dict(v) for k, v in counts.items()})
    audit["retained_rows"] = len(frame)
    audit["same_symbol_timestamp_ties"] = int(frame.duplicated(["symbol", "timestamp"]).sum())
    audit["identical_print_rows"] = int(frame.duplicated(["symbol", "timestamp", "price", "size"]).sum())
    if audit["same_symbol_timestamp_ties"]:
        warnings.add("Equal timestamps are retained in stable input-file/row order; this is not an inferred exchange-wide causal order.")
    if audit["identical_print_rows"]:
        warnings.add("Identical prints retained: without reliable trade IDs they may be distinct executions. Avoid overlapping input exports.")
    audit["warnings"] = sorted(warnings)
    return frame, audit


def normalize_and_clean(raw, cfg):
    d = cfg["data"]
    raw = raw.copy()
    raw.columns = [str(c).lower() for c in raw.columns]
    audit = {"input_rows": len(raw), "filters": {}, "warnings": []}
    # A WRDS stock identifier includes suffix: never silently merge share classes.
    symbol_col = d["symbol_column"].lower()
    if symbol_col not in raw and symbol_col == "symbol" and "sym_root" in raw:
        suffix = raw["sym_suffix"].fillna("").astype(str).str.strip() if "sym_suffix" in raw else pd.Series("", index=raw.index)
        raw["symbol"] = raw["sym_root"].astype(str).str.strip() + suffix.map(lambda x: "." + x if x else "")
    required = [symbol_col, d["price_column"].lower(), d["size_column"].lower()]
    for key in required:
        if key not in raw:
            raise ValueError(f"Missing required column: {key}")
    quality = [("correction", d["correction_column"].lower(), d["allowed_corrections"]),
               ("condition", d["condition_column"].lower(), d["allowed_sale_conditions"])]
    for label, col, allowed in quality:
        if col not in raw:
            if d["require_quality_fields"]:
                raise ValueError(f"Missing {col}; do not bypass TAQ quality fields without auditing the source")
            audit["warnings"].append(f"{label} filtering unavailable: {col} missing")
        else:
            value = raw[col].fillna("").astype(str).str.strip()
            if label == "correction":
                value = value.str.replace(r"\.0$", "", regex=True)
            raw["_" + label] = value
            audit[label + "_counts"] = {str(k): int(v) for k, v in value.value_counts().items()}

    tc = d["timestamp_column"].lower()
    if tc in raw:
        if pd.api.types.is_numeric_dtype(raw[tc]):
            raise ValueError("Numeric timestamps are ambiguous; provide ISO datetime strings or date + time_m")
        ts = pd.to_datetime(raw[tc], errors="coerce", format="mixed")
    else:
        dc, clock = d["date_column"].lower(), d["time_column"].lower()
        if dc not in raw or clock not in raw:
            raise ValueError(f"Need {tc} or both {dc} and {clock}")
        dates = pd.to_datetime(raw[dc].astype(str), errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
        ts = pd.to_datetime(dates + " " + raw[clock].astype(str).str.strip(), errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError("Mixed timestamp timezones; normalize the input to one timezone first")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(d["timezone"], ambiguous="NaT", nonexistent="NaT")
    else:
        ts = ts.dt.tz_convert(d["timezone"])
    frame = pd.DataFrame({
        "symbol": raw[symbol_col].astype(str).str.strip(), "timestamp": ts,
        "price": pd.to_numeric(raw[d["price_column"].lower()], errors="coerce"),
        "size": pd.to_numeric(raw[d["size_column"].lower()], errors="coerce"),
        "_row": np.arange(len(raw)),
    })
    alive = pd.Series(True, index=raw.index)

    def keep(label, mask):
        nonlocal alive
        mask = mask.fillna(False)
        audit["filters"][label] = int((alive & ~mask).sum())
        alive &= mask

    keep("invalid_timestamp", frame.timestamp.notna())
    keep("invalid_price_or_size", np.isfinite(frame.price) & np.isfinite(frame["size"]) & (frame.price > 0) & (frame["size"] > 0))
    for label, col, allowed in quality:
        if "_" + label in raw:
            keep("excluded_" + label, raw["_" + label].isin(allowed))
    local_time = frame.timestamp.dt.tz_localize(None)
    frame["date"] = local_time.dt.strftime("%Y-%m-%d")
    # Half-open session. Excludes the close auction; early closes require an adjusted config.
    clock = local_time.dt.hour * 3600 + local_time.dt.minute * 60 + local_time.dt.second
    def seconds(value):
        h, m, s = map(int, value.split(":"))
        return h * 3600 + m * 60 + s
    keep("outside_session", (clock >= seconds(d["session_start"])) & (clock < seconds(d["session_end"])))
    frame["split"] = [split_for(str(s), str(day), cfg["split"]) for s, day in zip(frame.symbol, frame.date)]
    keep("outside_fixed_split", frame.split.notna())
    frame = frame.loc[alive].copy()
    # No guessed global ordering from tape-local sequence numbers. Stable file order breaks ties.
    frame = frame.sort_values(["symbol", "date", "timestamp", "_row"], kind="stable")
    audit["retained_rows"] = len(frame)
    audit["same_symbol_timestamp_ties"] = int(frame.duplicated(["symbol", "timestamp"]).sum())
    audit["identical_print_rows"] = int(frame.duplicated(["symbol", "timestamp", "price", "size"]).sum())
    if audit["same_symbol_timestamp_ties"]:
        audit["warnings"].append("Equal timestamps are retained in stable input-file/row order; this is not an inferred exchange-wide causal order.")
    if audit["identical_print_rows"]:
        audit["warnings"].append("Identical prints retained: without reliable trade IDs they may be distinct executions. Avoid overlapping input exports.")
    return frame, audit


def prepare(cfg, overwrite=False):
    root = Path(cfg["data"]["prepared_dir"])
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; use --overwrite to rebuild deliberately")
    paths = sorted(glob.glob(cfg["data"]["input_glob"]))
    if not paths:
        raise FileNotFoundError(f"No inputs match {cfg['data']['input_glob']}")
    # Fail rather than silently claiming demo data are real.
    for path in paths:
        marker = Path(path).parent / "SYNTHETIC_DATA.json"
        if marker.exists() and cfg["data"]["provenance"] != "synthetic":
            raise ValueError("Synthetic input marker conflicts with real_taq provenance")
    frame, audit = clean_inputs(paths, cfg)
    sessions, values = [], []
    context = cfg["model"]["context_events"]
    observed = set()
    for (symbol, date), group in frame.groupby(["symbol", "date"], sort=True):
        # .as_unit('ns') avoids pandas version-dependent datetime integer resolutions.
        ns = group.timestamp.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
        p = group.price.to_numpy(dtype=np.float64)
        size = group["size"].to_numpy(dtype=np.float64)
        delta = np.diff(ns).astype(np.float64) / 1e9
        features = np.column_stack([np.log1p(delta), np.log(p[1:] / p[:-1]) * 10000, np.log1p(size[1:])])
        if len(features) <= context:
            audit["warnings"].append(f"Skipped short session {symbol} {date}: {len(features)} events")
            continue
        sid = f"session_{len(sessions):04d}"
        sessions.append({"id": sid, "symbol": str(symbol), "date": str(date), "split": str(group.split.iloc[0]),
                         "events": len(features), "codes": f"sessions/{sid}.npy", "features": f"features/{sid}.npy"})
        values.append(features)
        observed.add((symbol, date))
    expected = {(symbol, day) for symbol in cfg["split"]["seen_symbols"] for k in ("train_dates", "val_dates", "test_dates") for day in cfg["split"][k]}
    expected |= {(symbol, day) for symbol in cfg["split"]["heldout_symbols"] for day in cfg["split"]["test_dates"]}
    missing = sorted(expected - observed)
    audit["missing_or_short_sessions"] = [list(x) for x in missing]
    if missing and cfg["data"]["missing_policy"] == "error":
        raise ValueError(f"Missing/short sessions: {missing}. Inspect inputs or deliberately choose missing_policy=skip.")
    counts = Counter(s["split"] for s in sessions)
    if any(counts[x] == 0 for x in SPLITS):
        raise ValueError(f"All four splits require at least one usable session; got {counts}")
    training = np.concatenate([v for s, v in zip(sessions, values) if s["split"] == "train"])
    binner = EventBinner(cfg["tokenizer"]["bins"]).fit(training)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(exist_ok=True)
    for session, features in zip(sessions, values):
        codes = binner.transform(features)
        np.save(root / session["codes"], codes, allow_pickle=False)
        np.save(root / session["features"], features, allow_pickle=False)
        session["codes_sha256"] = file_hash(root / session["codes"])
        session["outside_training_range_fraction"] = ((features < binner.train_min) | (features > binner.train_max)).mean(0).tolist()
    manifest = {"schema_version": 1, "provenance": cfg["data"]["provenance"],
                "feature_names": FIELD_NAMES, "sessions": sessions, "split_spec": cfg["split"],
                "prepare_spec": {"data": cfg["data"], "split": cfg["split"], "bins": cfg["tokenizer"]["bins"], "context_events": context},
                "source_files": [{"path": str(p), "sha256": file_hash(p)} for p in paths],
                "binner": binner.to_dict(), "audit": audit}
    manifest["fingerprint"] = digest(manifest)
    write_json(manifest_path, manifest)
    pd.DataFrame([{k: v for k, v in s.items() if k != "outside_training_range_fraction"} for s in sessions]).to_csv(root / "sessions.csv", index=False)
    write_json(root / "audit.json", audit)
    print(f"Prepared {len(sessions)} sessions / {sum(s['events'] for s in sessions):,} events ({manifest['provenance']})", flush=True)
    return manifest


def load_prepared(cfg, verify_files=True):
    root = Path(cfg["data"]["prepared_dir"])
    m = read_json(root / "manifest.json")
    fp = m["fingerprint"]
    if digest({k: v for k, v in m.items() if k != "fingerprint"}) != fp:
        raise ValueError("Prepared manifest checksum mismatch")
    spec = {"data": cfg["data"], "split": cfg["split"], "bins": cfg["tokenizer"]["bins"], "context_events": cfg["model"]["context_events"]}
    if spec != m["prepare_spec"]:
        raise ValueError("Prepared data/config mismatch. Re-run prepare with the intended data/splits/bins/context.")
    if verify_files:
        for s in m["sessions"]:
            if file_hash(root / s["codes"]) != s["codes_sha256"]:
                raise ValueError(f"Prepared session checksum mismatch: {s['id']}")
    return m


class EventWindows:
    """Lazy windows backed by per-session .npy files; one full event target per example."""

    def __init__(self, root, manifest, split, context, stride=1):
        self.root, self.context = Path(root), context
        self.sessions = [s for s in manifest["sessions"] if s["split"] == split]
        self.stride = stride
        self.counts = np.array([max(0, (s["events"] - context - 1) // stride + 1) for s in self.sessions], dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self.arrays = [np.load(self.root / s["codes"], mmap_mode="r", allow_pickle=False) for s in self.sessions]
        if not len(self):
            raise ValueError(f"No windows in split {split}")

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def locate(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        sid = int(np.searchsorted(self.ends, i, side="right"))
        local = i - (int(self.ends[sid - 1]) if sid else 0)
        return sid, int(local) * self.stride

    def __getitem__(self, i):
        sid, start = self.locate(int(i))
        return np.asarray(self.arrays[sid][start:start + self.context + 1], dtype=np.int64)

    def batch(self, indices):
        return np.stack([self[int(i)] for i in indices])

    def evaluation_indices(self, maximum):
        return np.linspace(0, len(self) - 1, min(len(self), maximum), dtype=np.int64)


def make_synthetic(directory, events_per_session=256, seed=7):
    """Deliberately learnable toy process; NEVER evidence about market performance."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob("*.csv")) or list(root.glob("*.parquet")):
        raise FileExistsError(f"Refusing to mix demo data with existing files in {root}")
    rng = np.random.default_rng(seed)
    symbols = ["AAPL", "MSFT", "AMZN", "NVDA", "META"]
    days = pd.bdate_range("2025-02-03", periods=10)
    rows = []
    for si, symbol in enumerate(symbols):
        for day in days:
            ns = (day + pd.Timedelta(hours=9, minutes=30)).value
            price, drift = 100 + 20 * si, 0.0
            for _ in range(events_per_session):
                dt = float(rng.choice([.02, .08, .2, .8]))
                ns += int(dt * 1e9)
                drift = .85 * drift + rng.normal(0, .15)
                price *= np.exp((drift + rng.normal(0, .08)) / 10000)
                size = int(rng.choice([10, 50, 100, 200]) * (1 + (abs(drift) > .3)))
                rows.append((symbol, pd.Timestamp(ns).isoformat(), price, size, "00", "@"))
    frame = pd.DataFrame(rows, columns=["symbol", "timestamp", "price", "size", "tr_corr", "tr_scond"])
    frame.to_csv(root / "synthetic_trades.csv", index=False)
    write_json(root / "SYNTHETIC_DATA.json", {"synthetic": True, "seed": seed, "events_per_session": events_per_session,
                                               "warning": "Fabricated data for pipeline tests, not trading research evidence."})
    return root / "synthetic_trades.csv"
