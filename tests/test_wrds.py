import sys
import types

import pandas as pd
import pytest

from taq_lab.wrds_download import download


def test_wrds_chunked_export_and_parameterized_query(cfg, tmp_path, monkeypatch):
    calls = []

    class MockConnection:
        def __init__(self, **kwargs):
            self.closed = False
            calls.append(self)

        def raw_sql(self, sql, params, chunksize, return_iter):
            assert "%(symbols)s" in sql
            assert "AAPL" not in sql
            assert chunksize == 100_000 and return_iter
            assert params["symbols"] == ("AAPL", "AMZN", "MSFT", "NVDA")
            self.query = sql
            return iter([pd.DataFrame({"date": ["2025-02-03"], "time_m": ["09:30:00.123456789"], "ex": ["N"],
                                      "sym_root": ["AAPL"], "sym_suffix": [None], "tr_scond": ["@"],
                                      "price": [100.], "size": [10], "tr_corr": ["00"], "tr_seqnum": [1], "tr_source": ["C"]})])

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "wrds", types.SimpleNamespace(Connection=MockConnection))
    out = tmp_path / "wrds"
    rows = download(cfg, out, username="test_user")
    assert len(rows) == 10
    assert calls[0].closed
    frame = pd.read_parquet(rows[0]["file"])
    assert frame.time_m.iloc[0] == "09:30:00.123456789"
    assert frame.sym_suffix.iloc[0] == ""
    with pytest.raises(FileExistsError, match="overwrite"):
        download(cfg, out)
    with pytest.raises(ValueError, match="identifier"):
        download(cfg, tmp_path / "other", library="taqmsec;drop table anything")


def test_wrds_connection_closed_on_query_failure(cfg, tmp_path, monkeypatch):
    calls = []

    class MockConnection:
        def __init__(self):
            self.closed = False
            calls.append(self)

        def raw_sql(self, *args, **kwargs):
            raise RuntimeError("permission denied in mock")

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "wrds", types.SimpleNamespace(Connection=MockConnection))
    with pytest.raises(RuntimeError, match="permission denied"):
        download(cfg, tmp_path / "failed")
    assert calls[0].closed
    assert not list((tmp_path / "failed").glob("*.tmp"))
