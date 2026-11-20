"""Deterministic calculation service (spec sections 6-7).

Every number the TA agent reports comes from these functions - never from the language model.

Conventions (CALC_VERSION applies to all):
- Input: pandas DataFrame indexed by UTC timestamp, columns open/high/low/close[/tick_volume],
  COMPLETED bars only, ascending. Prices are arithmetic (not log) unless stated.
- Output at index t uses only bars <= t (no look-ahead). Warm-up values are NaN.
- EMA: alpha = 2/(n+1), seeded with the SMA of the first n values ("sma_seed").
- Wilder smoothing (RMA): alpha = 1/n, seeded with the SMA of the first n values.
- Standard deviation in bands: population (ddof=0), as in Bollinger's convention.
- Volatility estimators: sample variance (ddof=1) where an estimator is defined that way;
  annualised with periods_per_year (FX: 260 trading days; intraday = 260*24*60/minutes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALC_VERSION = "ta-calc/1.0.0"


# ----------------------------------------------------------------- helpers
def _seeded(x: pd.Series, n: int, alpha: float) -> pd.Series:
    """Recursive smoother y_t = alpha*x_t + (1-alpha)*y_{t-1}, seeded by SMA of first n valid x."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.flatnonzero(~np.isnan(v))
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    start = valid[0]
    seed_end = start + n - 1
    if np.isnan(v[start:seed_end + 1]).any():
        # gaps inside warm-up: fall back to first n valid values
        seed_end = valid[n - 1]
    out[seed_end] = np.nanmean(v[start:seed_end + 1])
    for i in range(seed_end + 1, len(v)):
        xi = v[i]
        out[i] = out[i - 1] if np.isnan(xi) else alpha * xi + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=x.index)


def periods_per_year(timeframe: str) -> float:
    minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[timeframe]
    return 260.0 if timeframe == "D1" else 260.0 * 24 * 60 / minutes


# ----------------------------------------------------------------- moving averages
def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return _seeded(x, n, 2.0 / (n + 1))


def rma(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. SMMA / RMA)."""
    return _seeded(x, n, 1.0 / n)


def wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return x.rolling(n, min_periods=n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)


def dema(x: pd.Series, n: int) -> pd.Series:
    e1 = ema(x, n)
    return 2 * e1 - ema(e1, n)


def tema(x: pd.Series, n: int) -> pd.Series:
    e1 = ema(x, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


# ----------------------------------------------------------------- momentum
def momentum(x: pd.Series, n: int = 10) -> pd.Series:
    return x - x.shift(n)


def roc(x: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change in percent: 100*(P_t/P_{t-n} - 1)."""
    return 100 * (x / x.shift(n) - 1)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. First value at index n (needs n price changes)."""
    d = close.diff()
    g = rma(d.clip(lower=0), n)
    l = rma((-d).clip(lower=0), n)
    rs = g / l
    out = 100 - 100 / (1 + rs)
    out[(l == 0) & (g > 0)] = 100.0
    out[(l == 0) & (g == 0)] = 50.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(df: pd.DataFrame, k: int = 14, smooth_k: int = 3, d: int = 3):
    """Full stochastic. smooth_k=1 gives 'fast' %K; smooth_k=3 gives 'slow'."""
    ll = df["low"].rolling(k, min_periods=k).min()
    hh = df["high"].rolling(k, min_periods=k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    pk = sma(raw, smooth_k) if smooth_k > 1 else raw
    return pk, sma(pk, d)


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int = 20, c: float = 0.015) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = sma(tp, n)
    mad = tp.rolling(n, min_periods=n).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    return (tp - m) / (c * mad.replace(0, np.nan))


def dpo(close: pd.Series, n: int = 20) -> pd.Series:
    """Non-centred (no look-ahead) DPO: P_{t-(n/2+1)} - SMA_n(t)."""
    return close.shift(n // 2 + 1) - sma(close, n)


# ----------------------------------------------------------------- trend / range
def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]  # first bar: no previous close
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def dmi(df: pd.DataFrame, n: int = 14):
    """Wilder +DI, -DI, ADX."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pdm.iloc[0] = mdm.iloc[0] = np.nan
    tr = true_range(df); tr.iloc[0] = np.nan
    a = rma(tr, n)
    pdi = 100 * rma(pdm, n) / a
    mdi = 100 * rma(mdm, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, rma(dx, n)


def aroon(df: pd.DataFrame, n: int = 25):
    w = n + 1
    up = df["high"].rolling(w, min_periods=w).apply(lambda a: 100 * (n - (n - int(np.argmax(a)))) / n, raw=True)
    dn = df["low"].rolling(w, min_periods=w).apply(lambda a: 100 * (n - (n - int(np.argmin(a)))) / n, raw=True)
    return up, dn


def psar(df: pd.DataFrame, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    up = h[1] >= h[0]
    sar, ep, af = (l[0], h[0], step) if up else (h[0], l[0], step)
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_af)
        out[i] = sar
    return pd.Series(out, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan=9, kijun=26, senkou=52, disp=26) -> dict:
    """Values KNOWN at bar t: the cloud 'under' bar t was computed disp bars earlier."""
    mid = lambda n: (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    span_a_now = ((t + k) / 2).shift(disp)
    span_b_now = mid(senkou).shift(disp)
    return {"tenkan": t, "kijun": k, "span_a": span_a_now, "span_b": span_b_now,
            "chikou_vs_price": df["close"] - df["close"].shift(disp)}


def linreg(x: pd.Series, n: int = 50, log: bool = True):
    """Rolling OLS on (log) price vs time: slope per bar and R^2."""
    y = np.log(x) if log else x
    t = np.arange(n, dtype=float)
    tc = t - t.mean()

    def slope(a):
        return np.dot(tc, a - a.mean()) / np.dot(tc, tc)

    def r2(a):
        b = slope(a)
        resid = a - (a.mean() + b * tc)
        sst = np.sum((a - a.mean()) ** 2)
        return 1 - np.sum(resid ** 2) / sst if sst > 0 else 0.0

    return y.rolling(n, min_periods=n).apply(slope, raw=True), y.rolling(n, min_periods=n).apply(r2, raw=True)


# ----------------------------------------------------------------- bands / channels
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return {"upper": up, "mid": mid, "lower": lo,
            "pct_b": (close - lo) / (up - lo).replace(0, np.nan),
            "bandwidth": (up - lo) / mid}


def keltner(df: pd.DataFrame, n: int = 20, atr_n: int = 10, k: float = 2.0):
    mid = ema(df["close"], n)
    a = atr(df, atr_n)
    return {"upper": mid + k * a, "mid": mid, "lower": mid - k * a}


def donchian(df: pd.DataFrame, n: int = 20):
    """Includes the current bar. For breakout signals compare close_t with the channel at t-1."""
    up = df["high"].rolling(n, min_periods=n).max()
    lo = df["low"].rolling(n, min_periods=n).min()
    return {"upper": up, "mid": (up + lo) / 2, "lower": lo}


# ----------------------------------------------------------------- volatility estimators
def hist_vol(close: pd.Series, n: int, ppy: float, log: bool = True) -> pd.Series:
    r = np.log(close / close.shift()) if log else close.pct_change()
    return r.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(ppy)


def parkinson(df, n, ppy):
    x = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(x.rolling(n, min_periods=n).mean() / (4 * np.log(2)) * ppy)


def garman_klass(df, n, ppy):
    x = 0.5 * np.log(df["high"] / df["low"]) ** 2 - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    return np.sqrt(x.rolling(n, min_periods=n).mean().clip(lower=0) * ppy)


def rogers_satchell(df, n, ppy):
    h, l, o, c = (np.log(df[k]) for k in ("high", "low", "open", "close"))
    x = (h - c) * (h - o) + (l - c) * (l - o)
    return np.sqrt(x.rolling(n, min_periods=n).mean().clip(lower=0) * ppy)


def yang_zhang(df, n, ppy):
    o, c = np.log(df["open"]), np.log(df["close"])
    ov = (o - c.shift()).rolling(n, min_periods=n).var(ddof=1)        # overnight / gap
    cv = (c - o).rolling(n, min_periods=n).var(ddof=1)                # open-to-close
    h, l = np.log(df["high"]), np.log(df["low"])
    rs = ((h - c) * (h - o) + (l - c) * (l - o)).rolling(n, min_periods=n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((ov + k * cv + (1 - k) * rs).clip(lower=0) * ppy)


# ----------------------------------------------------------------- volume (tick volume for OTC FX)
def obv(df: pd.DataFrame, vol_col: str = "tick_volume") -> pd.Series:
    """For OTC FX this uses BROKER TICK VOLUME, not consolidated market volume."""
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df[vol_col]).cumsum()


# ----------------------------------------------------------------- pivots (from previous COMPLETED period)
def pivots(h: float, l: float, c: float, o_next: float | None = None) -> dict:
    r = h - l
    p = (h + l + c) / 3
    pw = (h + l + 2 * c) / 4
    return {
        "floor": {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + r, "S2": p - r,
                  "R3": h + 2 * (p - l), "S3": l - 2 * (h - p)},
        "fibonacci": {"P": p, "R1": p + 0.382 * r, "S1": p - 0.382 * r, "R2": p + 0.618 * r,
                      "S2": p - 0.618 * r, "R3": p + r, "S3": p - r},
        "camarilla": {"R1": c + r * 1.1 / 12, "S1": c - r * 1.1 / 12, "R2": c + r * 1.1 / 6, "S2": c - r * 1.1 / 6,
                      "R3": c + r * 1.1 / 4, "S3": c - r * 1.1 / 4, "R4": c + r * 1.1 / 2, "S4": c - r * 1.1 / 2},
        "woodie": {"P": pw, "R1": 2 * pw - l, "S1": 2 * pw - h, "R2": pw + r, "S2": pw - r},
    }


def fib_levels(swing_low: float, swing_high: float, direction: str = "up") -> dict:
    """Retracements of the last move. direction='up' means the move was low->high."""
    r = swing_high - swing_low
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    if direction == "up":
        ret = {f"{x:.3f}": swing_high - x * r for x in ratios}
        ext = {f"{x:.3f}": swing_low + x * r for x in (1.272, 1.618)}
    else:
        ret = {f"{x:.3f}": swing_low + x * r for x in ratios}
        ext = {f"{x:.3f}": swing_high - x * r for x in (1.272, 1.618)}
    return {"retracement": ret, "extension": ext}


# ----------------------------------------------------------------- candles
def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["body"] = (c - o).abs()
    f["range"] = h - l
    f["upper_wick"] = h - np.maximum(o, c)
    f["lower_wick"] = np.minimum(o, c) - l
    f["close_location"] = (c - l) / rng  # 0 = closed at low, 1 = at high
    return f


def candle_patterns(df: pd.DataFrame) -> dict:
    """Patterns on the LAST completed bar. Descriptive only - not predictive by themselves."""
    f = candle_features(df)
    b, p = f.iloc[-1], f.iloc[-2]
    c, prev = df.iloc[-1], df.iloc[-2]
    bull = c.close > c.open
    pat = {
        "doji": bool(b.range > 0 and b.body <= 0.1 * b.range),
        "bullish_engulfing": bool(bull and prev.close < prev.open and c.close >= prev.open and c.open <= prev.close),
        "bearish_engulfing": bool(not bull and prev.close > prev.open and c.open >= prev.close and c.close <= prev.open),
        "bullish_pin": bool(b.range > 0 and b.lower_wick >= 2 * b.body and b.lower_wick >= 0.6 * b.range),
        "bearish_pin": bool(b.range > 0 and b.upper_wick >= 2 * b.body and b.upper_wick >= 0.6 * b.range),
        "inside_bar": bool(c.high <= prev.high and c.low >= prev.low),
        "outside_bar": bool(c.high > prev.high and c.low < prev.low),
    }
    return {k: v for k, v in pat.items() if v}


# ----------------------------------------------------------------- FX contract math
def pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def pip_value(symbol: str, price: float, lots: float = 1.0, contract: float = 100_000,
              account_ccy: str = "USD", quote_to_account: float | None = None) -> float:
    """Value of one pip in account currency.
    Pip value in QUOTE currency = contract * lots * pip_size. Convert quote -> account:
      quote == account (EURUSD, USD acct): factor 1
      base  == account (USDJPY, USD acct): divide by price
      otherwise: multiply by quote_to_account rate (must be supplied)."""
    sym = symbol.upper()
    base, quote = sym[:3], sym[3:6]
    v_quote = contract * lots * pip_size(sym)
    if quote == account_ccy:
        return v_quote
    if base == account_ccy:
        return v_quote / price
    if quote_to_account is None:
        raise ValueError(f"need {quote}->{account_ccy} rate for {sym}")
    return v_quote * quote_to_account


def position_size(risk_money: float, stop_pips: float, pip_value_per_lot: float, step: float = 0.01) -> float:
    lots = risk_money / (stop_pips * pip_value_per_lot)
    return np.floor(lots / step + 1e-9) * step
