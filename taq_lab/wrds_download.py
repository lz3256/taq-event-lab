"""Optional, read-only WRDS access. No credentials are accepted as CLI arguments."""
from __future__ import annotations

import re
from pathlib import Path

from .common import write_json


def download(cfg, output, username=None, library="taqmsec"):
    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError('Install optional dependency: pip install -e ".[wrds]"') from exc
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", library):
        raise ValueError("Invalid WRDS library identifier")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    split = cfg["split"]
    days = sorted(set(split["train_dates"] + split["val_dates"] + split["test_dates"]))
    symbols = tuple(sorted(set(split["seen_symbols"] + split["heldout_symbols"])))
    if any("." in s for s in symbols):
        raise ValueError("Downloader currently supports root symbols without suffixes. Export share classes manually.")
    collisions = [root / f"ctm_{day.replace('-', '')}.parquet" for day in days if (root / f"ctm_{day.replace('-', '')}.parquet").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite exports: {collisions}")
    db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
    exported = []
    try:
        for day in days:
            table = "ctm_" + day.replace("-", "")
            # Identifiers validated locally; all user data passed as SQL parameters.
            sql = f'''select date, time_m, ex, sym_root, sym_suffix, tr_scond,
                             size, price, tr_corr, tr_seqnum, tr_source
                      from {library}.{table}
                      where sym_root in %(symbols)s
                        and (sym_suffix is null or trim(sym_suffix) = '')
                        and time_m >= cast(%(start)s as time)
                        and time_m < cast(%(end)s as time)
                      order by sym_root, time_m, tr_source, tr_seqnum'''
            # WRDS raw_sql can return chunks. Write each day without collecting the whole export.
            import pyarrow as pa
            import pyarrow.parquet as pq
            path = root / f"{table}.parquet"
            temp = path.with_suffix(".parquet.tmp")
            writer, nrows = None, 0
            try:
                chunks = db.raw_sql(sql, params={"symbols": symbols, "start": cfg["data"]["session_start"], "end": cfg["data"]["session_end"]},
                                    chunksize=100_000, return_iter=True)
                for frame in chunks:
                    if frame.empty:
                        continue
                    # Preserve date/time as text, including fractional seconds; no float timestamps.
                    for col in ("date", "time_m", "ex", "sym_root", "sym_suffix", "tr_scond", "tr_corr", "tr_seqnum", "tr_source"):
                        frame[col] = frame[col].fillna("").astype(str)
                    frame["price"] = frame["price"].astype("float64")
                    frame["size"] = frame["size"].astype("float64")
                    tab = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temp, tab.schema)
                    writer.write_table(tab)
                    nrows += len(frame)
                if writer is not None:
                    writer.close()
                    writer = None
                    temp.replace(path)
                else:
                    raise ValueError(f"WRDS returned no rows for {day}")
            finally:
                if writer is not None:
                    writer.close()
                if temp.exists():
                    temp.unlink()
            exported.append({"date": day, "file": str(path), "rows": nrows})
            write_json(root / "download_manifest.json", {"source": "WRDS", "library": library, "exports": exported})
            print(f"Downloaded {day}: {nrows:,} rows", flush=True)
    finally:
        db.close()
    return exported

