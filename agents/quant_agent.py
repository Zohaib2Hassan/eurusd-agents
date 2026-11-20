"""Quant agent: YOUR formulas. Pure code, no AI.

Each rule returns +1 (buy), -1 (sell) or 0 (no opinion).
Add your own rule: write a function rule_xxx(df, q) and add it to RULES.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind


def rule_ema_cross(df: pd.DataFrame, q) -> int:
    """EMA fast crossed EMA slow within the last 3 bars."""
    diff = df["ema_fast"] - df["ema_slow"]
    recent = diff.iloc[-4:]
    if (recent.iloc[0] < 0) and (recent.iloc[-1] > 0):
        return 1
    if (recent.iloc[0] > 0) and (recent.iloc[-1] < 0):
        return -1
    return 0


def rule_rsi_reversal(df, q) -> int:
    r = df["rsi"]
    if r.iloc[-2] < q.get("rsi_oversold", 30) <= r.iloc[-1]:
        return 1
    if r.iloc[-2] > q.get("rsi_overbought", 70) >= r.iloc[-1]:
        return -1
    return 0


def rule_trend_pullback(df, q) -> int:
    """Trend up + price touched EMA50 and closed back above it (and the reverse)."""
    last = df.iloc[-1]
    if last.close > last.ema_trend and last.low <= last.ema_slow < last.close:
        return 1
    if last.close < last.ema_trend and last.high >= last.ema_slow > last.close:
        return -1
    return 0


def rule_bollinger_breakout(df, q) -> int:
    last = df.iloc[-1]
    if last.close > last.bb_up and last.adx > 25:
        return 1
    if last.close < last.bb_low and last.adx > 25:
        return -1
    return 0


def rule_macd_cross(df, q) -> int:
    h = df["macd_hist"]
    if h.iloc[-2] < 0 < h.iloc[-1]:
        return 1
    if h.iloc[-2] > 0 > h.iloc[-1]:
        return -1
    return 0


RULES = [rule_ema_cross, rule_rsi_reversal, rule_trend_pullback, rule_bollinger_breakout, rule_macd_cross]


class QuantAgent:
    name = "quant"

    def __init__(self, cfg, market, llm=None):
        self.cfg, self.market = cfg, market

    def run(self) -> dict:
        q = self.cfg.get("quant", {})
        df = ind.add_all(self.market.rates(self.cfg["timeframes"]["entry"], self.cfg.get("bars", 500)), q)
        votes = {r.__name__.replace("rule_", ""): r(df, q) for r in RULES}
        active = [v for v in votes.values() if v != 0]
        bias = sum(active) / len(RULES)
        agree = abs(sum(active)) / len(active) if active else 0
        return {
            "agent": self.name, "bias": round(bias, 3), "confidence": round(0.3 + 0.5 * agree * (len(active) / len(RULES)) ** 0.5, 2),
            "summary": "Votes: " + ", ".join(f"{k}={v:+d}" for k, v in votes.items()),
            "details": {"votes": votes},
        }
