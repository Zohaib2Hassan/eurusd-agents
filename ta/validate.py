"""Validation layer: checks bars BEFORE any calculation (spec 17.C / 17.D step 3).

Returns cleaned completed bars + a list of warnings + a 0..1 quality score.
FX convention: market closed roughly Fri 22:00 UTC -> Sun 21:00 UTC; gaps over the weekend are expected.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _weekend_gap(t0: pd.Timestamp, t1: pd.Timestamp) -> bool:
    """True if the interval (t0, t1) spans the FX weekend close."""
    days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
    return any(d.weekday() == 5 for d in days)  # interval contains a Saturday


def market_open(now: datetime) -> bool:
    wd, h = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 4 and h >= 22:
        return False
    if wd == 6 and h < 21:
        return False
    return True


def validate(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> tuple[pd.DataFrame, list[str], float, dict]:
    now = now or datetime.now(timezone.utc)
    warn: list[str] = []
    score = 1.0
    tf = pd.Timedelta(minutes=TF_MIN[timeframe])
    meta = {"timeframe": timeframe, "bars_in": int(len(df))}

    if df.empty:
        return df, ["no data"], 0.0, meta
    df = df.sort_index()

    # duplicates
    dup = df.index.duplicated(keep="last")
    if dup.any():
        warn.append(f"{int(dup.sum())} duplicate timestamps removed")
        df = df[~dup]; score -= 0.05

    # OHLC consistency / non-positive
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1)) | \
          (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        warn.append(f"{int(bad.sum())} bars with inconsistent OHLC removed")
        df = df[~bad]; score -= 0.1

    # incomplete (still forming) last bar - MT5 position 0 is the live candle
    if len(df) and df.index[-1] + tf > pd.Timestamp(now):
        meta["dropped_forming_bar"] = str(df.index[-1])
        df = df.iloc[:-1]

    # gaps (excluding expected FX weekend)
    if len(df) > 2:
        d = df.index.to_series().diff().iloc[1:]
        big = d[d > tf * (1.5 if timeframe != "D1" else 3.5)]
        unexpected = [(i, g) for i, g in big.items() if not _weekend_gap(i - g, i)]
        if unexpected:
            warn.append(f"{len(unexpected)} unexpected gaps (largest {max(g for _, g in unexpected)})")
            score -= min(0.2, 0.02 * len(unexpected))

    # outliers: |log return| > 8 robust sigmas
    if len(df) > 30:
        lr = np.log(df["close"]).diff().dropna()
        mad = np.median(np.abs(lr - lr.median())) * 1.4826
        if mad > 0:
            n_out = int((np.abs(lr - lr.median()) > 8 * mad).sum())
            if n_out:
                warn.append(f"{n_out} extreme return bars (>8 robust sigma) - check data or news spikes")
                score -= min(0.15, 0.03 * n_out)

    # staleness
    if len(df):
        age = pd.Timestamp(now) - (df.index[-1] + tf)
        meta["last_completed_bar"] = str(df.index[-1])
        meta["age_minutes"] = round(age.total_seconds() / 60, 1)
        if market_open(now) and age > tf * 2 + pd.Timedelta(minutes=5):
            warn.append(f"stale data: last completed bar is {age} old while market is open")
            score -= 0.3

    meta["bars_out"] = int(len(df))
    return df, warn, max(0.0, round(score, 2)), meta
