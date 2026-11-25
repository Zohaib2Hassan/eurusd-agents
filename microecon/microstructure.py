"""Market microstructure measures from data you can actually get (bars + broker spread).

- Quoted spread (MT5): broker's recorded spread per bar, in points/pips and bps.
- Roll (1984): effective spread from negative serial covariance of price changes:
      S = 2 * sqrt(-Cov(dP_t, dP_{t-1}))   (undefined when the covariance is positive)
- Corwin & Schultz (2012): spread from daily high/low ranges over 2-day windows:
      beta = E[ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2],  gamma = ln(max(H_t,H_{t+1})/min(L_t,L_{t+1}))^2
      alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
      S = 2 (e^alpha - 1)/(1 + e^alpha)   (negative two-day estimates set to 0, as in the paper)
- Amihud (2002) illiquidity: mean(|r_t| / dollar volume_t) - needs real traded volume (exchanges, not OTC FX).
All are ESTIMATES with known biases (e.g. Roll assumes no autocorrelation in the efficient price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _f(x, d=6):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def quoted_spread(df: pd.DataFrame, point: float, pip: float) -> dict:
    if "spread" not in df or df["spread"].isna().all():
        return {"error": "source has no spread column"}
    sp = df["spread"].astype(float)
    bps = sp * point / df["close"] * 1e4
    return {"median_points": _f(sp.median(), 1), "p90_points": _f(sp.quantile(.9), 1), "max_points": _f(sp.max(), 1),
            "median_pips": _f(sp.median() * point / pip, 2), "median_bps": _f(bps.median(), 3),
            "note": "MT5 bar 'spread' is the broker's recorded spread for the bar (often the minimum), not a volume-weighted average"}


def roll(close: pd.Series) -> dict:
    d = close.diff().dropna()
    cov = np.cov(d.iloc[1:], d.iloc[:-1])[0, 1]
    if cov >= 0:
        return {"estimate": None, "serial_cov": _f(cov, 12), "note": "covariance >= 0: Roll estimator undefined for this sample"}
    s = 2 * np.sqrt(-cov)
    return {"estimate_price": _f(s, 8), "estimate_bps": _f(s / close.mean() * 1e4, 3), "serial_cov": _f(cov, 12)}


def corwin_schultz(df: pd.DataFrame) -> dict:
    h, l = np.log(df["high"].to_numpy(float)), np.log(df["low"].to_numpy(float))
    hl2 = (h - l) ** 2
    beta = hl2[1:] + hl2[:-1]
    gamma = (np.maximum(h[1:], h[:-1]) - np.minimum(l[1:], l[:-1])) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = np.where(s < 0, 0.0, s)
    return {"mean_bps": _f(np.mean(s) * 1e4, 3), "median_bps": _f(np.median(s) * 1e4, 3),
            "share_zero_windows": _f((s == 0).mean(), 3),
            "note": "high-low estimator; biased upward by overnight/weekend gaps and by volatility within the bar"}


def amihud(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume") -> dict:
    r = np.log(df[price_col]).diff().abs()
    dollar_vol = df[vol_col] * df[price_col]
    x = (r / dollar_vol.replace(0, np.nan)).dropna()
    return {"illiq_per_billion_usd": _f(x.mean() * 1e9, 6), "median_dollar_volume": _f(dollar_vol.median(), 0),
            "interpretation": "average absolute return per $1B traded; higher = less liquid (more price impact)"}


def liquidity_profile(df: pd.DataFrame, meta: dict, point: float | None = None, pip: float | None = None) -> dict:
    out = {"bars": len(df), "start_utc": str(df.index[0]), "end_utc": str(df.index[-1]),
           "roll": roll(df["close"]), "corwin_schultz": corwin_schultz(df)}
    if point and pip and "spread" in df:
        out["quoted_spread"] = quoted_spread(df, point, pip)
    vt = str(meta.get("volume_type", "")).lower()
    if "volume" in df and ("binance" in vt or "exchange" in vt or meta.get("asset_class") == "crypto"):
        out["amihud"] = amihud(df)
    else:
        out["amihud"] = {"skipped": "OTC FX tick volume is not traded volume - Amihud needs real dollar volume"}
    return out
