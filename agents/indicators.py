"""Pure-pandas indicators. The math is done here in code, never by the AI."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def add_all(df: pd.DataFrame, q: dict) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema_fast"] = ema(c, q.get("ema_fast", 20))
    df["ema_slow"] = ema(c, q.get("ema_slow", 50))
    df["ema_trend"] = ema(c, q.get("ema_trend", 200))
    df["rsi"] = rsi(c, q.get("rsi_period", 14))
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
    df["atr"] = atr(df)
    df["bb_up"], df["bb_mid"], df["bb_low"] = bollinger(c)
    df["adx"] = adx(df)
    return df


def snapshot(df: pd.DataFrame) -> dict:
    """Last-bar values, rounded, for agents/AI to read."""
    last = df.iloc[-1]
    keys = ["close", "ema_fast", "ema_slow", "ema_trend", "rsi", "macd", "macd_signal",
            "macd_hist", "atr", "bb_up", "bb_mid", "bb_low", "adx"]
    out = {k: round(float(last[k]), 5) for k in keys}
    out["time"] = str(df.index[-1])
    out["change_24bars_pct"] = round(float((df["close"].iloc[-1] / df["close"].iloc[-25] - 1) * 100), 3)
    return out
