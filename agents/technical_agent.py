"""Technical agent: reads ONLY the EURUSD charts (trend + entry timeframe)."""
from __future__ import annotations

import json

from . import indicators as ind


def _score(s: dict) -> tuple[float, list[str]]:
    """Rule-based score in [-1, 1] from one timeframe snapshot."""
    pts, notes = 0.0, []
    if s["close"] > s["ema_trend"]:
        pts += 1; notes.append("price above EMA200")
    else:
        pts -= 1; notes.append("price below EMA200")
    if s["ema_fast"] > s["ema_slow"]:
        pts += 1; notes.append("EMA20 > EMA50")
    else:
        pts -= 1; notes.append("EMA20 < EMA50")
    if s["macd_hist"] > 0:
        pts += 0.5; notes.append("MACD histogram positive")
    else:
        pts -= 0.5; notes.append("MACD histogram negative")
    if s["rsi"] > 70:
        pts -= 0.5; notes.append(f"RSI overbought ({s['rsi']:.0f})")
    elif s["rsi"] < 30:
        pts += 0.5; notes.append(f"RSI oversold ({s['rsi']:.0f})")
    strength = min(s["adx"] / 40, 1.0)  # weak trend -> smaller conviction
    return max(-1, min(1, pts / 3)) * (0.5 + 0.5 * strength), notes


class TechnicalAgent:
    name = "technical"

    def __init__(self, cfg, market, llm):
        self.cfg, self.market, self.llm = cfg, market, llm

    def run(self) -> dict:
        tfs, bars, q = self.cfg["timeframes"], self.cfg.get("bars", 500), self.cfg.get("quant", {})
        snaps, frames = {}, {}
        for role, tf in tfs.items():
            df = ind.add_all(self.market.rates(tf, bars), q)
            frames[role], snaps[role] = df, ind.snapshot(df)
        t_score, t_notes = _score(snaps["trend"])
        e_score, e_notes = _score(snaps["entry"])
        bias = 0.6 * t_score + 0.4 * e_score
        if (t_score > 0) != (e_score > 0):
            bias *= 0.5  # timeframes disagree
        report = {
            "agent": self.name, "bias": round(bias, 3), "confidence": round(min(abs(bias) + 0.2, 1), 2),
            "summary": f"Trend({tfs['trend']}): {', '.join(t_notes)} | Entry({tfs['entry']}): {', '.join(e_notes)}",
            "details": {"snapshots": snaps, "atr_entry": snaps["entry"]["atr"]},
        }
        ai = self.llm.ask_json(
            "You are a forex technical analyst. You analyse ONLY EUR/USD. Use the computed indicator values "
            "given to you; do not invent numbers.",
            "Indicator snapshots (trend and entry timeframe):\n" + json.dumps(snaps, indent=1) +
            f"\nRule-based score: {report['bias']}.\nReturn JSON: "
            '{"bias": number -1..1, "confidence": number 0..1, "summary": "max 3 sentences", '
            '"key_levels": {"support": number, "resistance": number}}')
        if ai and "bias" in ai:
            report["ai"] = ai
            report["bias"] = round(0.5 * report["bias"] + 0.5 * _clip(ai["bias"]), 3)
            report["confidence"] = round(0.5 * report["confidence"] + 0.5 * _clip(ai.get("confidence", 0.5), 0, 1), 2)
            report["summary"] = ai.get("summary", report["summary"])
        return report


def _clip(x, lo=-1.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0
