"""Research data store (SQLite now; schema is PostgreSQL/TimescaleDB-compatible - see docs/quantx_schema.sql).

Tables
- instruments      symbol metadata per source
- ohlcv            bars, keyed by (source, symbol, timeframe, ts_utc); upserted
- ingestion_runs   provenance of every fetch (rows, range, hash, quality, warnings)
- analysis_runs    every analysis: request, data ids, code version, full result JSON
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
  source TEXT NOT NULL, symbol TEXT NOT NULL, asset_class TEXT, price_type TEXT, volume_type TEXT,
  calendar TEXT, first_seen_utc TEXT, PRIMARY KEY (source, symbol));
CREATE TABLE IF NOT EXISTS ohlcv (
  source TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL, ts_utc TEXT NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume REAL, spread REAL, ingestion_run TEXT,
  PRIMARY KEY (source, symbol, timeframe, ts_utc));
CREATE TABLE IF NOT EXISTS ingestion_runs (
  run_id TEXT PRIMARY KEY, source TEXT, server TEXT, symbol TEXT, timeframe TEXT, retrieved_at_utc TEXT,
  rows INTEGER, first_ts TEXT, last_ts TEXT, data_sha256 TEXT, quality REAL, warnings_json TEXT, meta_json TEXT);
CREATE TABLE IF NOT EXISTS analysis_runs (
  run_id TEXT PRIMARY KEY, created_at_utc TEXT, kind TEXT, request_json TEXT, data_ids_json TEXT,
  code_version TEXT, result_json TEXT);
"""


def data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df[["open", "high", "low", "close"]]).values.tobytes()).hexdigest()[:16]


class Store:
    def __init__(self, path: str | Path = "data/quantx.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def save_bars(self, df: pd.DataFrame, meta: dict, quality: float, warnings: list[str]) -> str:
        run = uuid.uuid4().hex[:12]
        sha = data_hash(df)
        src = meta["source"]
        self.db.execute("INSERT OR IGNORE INTO instruments VALUES (?,?,?,?,?,?,?)",
                        (src, meta["symbol"], meta.get("asset_class"), meta.get("price_type"),
                         meta.get("volume_type"), meta.get("calendar"), meta["retrieved_at_utc"]))
        has_spread = "spread" in df.columns
        rows = [(src, meta["symbol"], meta["timeframe"], ts.isoformat(), float(r.open), float(r.high),
                 float(r.low), float(r.close), None if pd.isna(r.volume) else float(r.volume),
                 float(r.spread) if has_spread and not pd.isna(r.spread) else None, run)
                for ts, r in df.iterrows()]
        self.db.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.db.execute("INSERT INTO ingestion_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run, src, meta.get("server"), meta["symbol"], meta["timeframe"], meta["retrieved_at_utc"],
                         len(df), str(df.index[0]) if len(df) else None, str(df.index[-1]) if len(df) else None,
                         sha, quality, json.dumps(warnings), json.dumps(meta)))
        self.db.commit()
        return run

    def load_bars(self, source: str, symbol: str, timeframe: str) -> pd.DataFrame:
        q = "SELECT ts_utc, open, high, low, close, volume FROM ohlcv WHERE source=? AND symbol=? AND timeframe=? ORDER BY ts_utc"
        df = pd.read_sql_query(q, self.db, params=(source, symbol, timeframe))
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        return df.set_index("ts_utc")

    def save_analysis(self, kind: str, request: dict, data_ids: dict, code_version: str, result: dict) -> str:
        run = uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO analysis_runs VALUES (?,?,?,?,?,?,?)",
                        (run, datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, json.dumps(request),
                         json.dumps(data_ids), code_version, json.dumps(result, default=str)))
        self.db.commit()
        return run

    def close(self):
        self.db.close()
