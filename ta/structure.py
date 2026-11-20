"""Reproducible market structure (spec section 4): swings, trend state, S/R zones.

Swing definition (fractal, k bars each side): bar i is a swing high if high_i is strictly the
highest of bars i-k..i+k. It is only KNOWN at bar i+k ("confirmed_at") - no look-ahead.
Zones: confirmed swing prices clustered when within tol = zone_atr * ATR; a zone is [min, max]
of its cluster (padded to a minimum width), strength = number of touches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def swings(df: pd.DataFrame, k: int = 3) -> list[dict]:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    out = []
    for i in range(k, len(df) - k):
        win_h, win_l = h[i - k:i + k + 1], l[i - k:i + k + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            out.append({"type": "high", "price": float(h[i]), "time": df.index[i], "confirmed_at": df.index[i + k]})
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            out.append({"type": "low", "price": float(l[i]), "time": df.index[i], "confirmed_at": df.index[i + k]})
    return out


def trend_state(sw: list[dict]) -> dict:
    hs = [s for s in sw if s["type"] == "high"][-2:]
    ls = [s for s in sw if s["type"] == "low"][-2:]
    if len(hs) < 2 or len(ls) < 2:
        return {"state": "undetermined", "detail": "fewer than 2 confirmed swing highs/lows"}
    hh, hl = hs[1]["price"] > hs[0]["price"], ls[1]["price"] > ls[0]["price"]
    state = "uptrend" if hh and hl else "downtrend" if not hh and not hl else "range/transition"
    return {"state": state,
            "detail": f"{'HH' if hh else 'LH'} ({hs[0]['price']:.5f}->{hs[1]['price']:.5f}), "
                      f"{'HL' if hl else 'LL'} ({ls[0]['price']:.5f}->{ls[1]['price']:.5f})",
            "last_high": hs[1]["price"], "last_low": ls[1]["price"]}


def zones(sw: list[dict], atr_value: float, zone_atr: float = 0.5, min_width_atr: float = 0.15,
          max_zones: int = 12) -> list[dict]:
    if not sw or not atr_value or np.isnan(atr_value):
        return []
    pts = sorted(sw, key=lambda s: s["price"])
    tol = zone_atr * atr_value
    clusters, cur = [], [pts[0]]
    for s in pts[1:]:
        # join if close to previous point AND cluster stays narrower than max_width (no chaining)
        if s["price"] - cur[-1]["price"] <= tol and s["price"] - cur[0]["price"] <= 2 * tol:
            cur.append(s)
        else:
            clusters.append(cur); cur = [s]
    clusters.append(cur)
    out = []
    for c in clusters:
        lo, hi = min(s["price"] for s in c), max(s["price"] for s in c)
        pad = max(0.0, (min_width_atr * atr_value - (hi - lo)) / 2)
        out.append({"low": lo - pad, "high": hi + pad, "touches": len(c),
                    "last_touch": max(s["time"] for s in c),
                    "from": sorted({s["type"] for s in c})})
    out.sort(key=lambda z: (z["touches"], z["last_touch"]), reverse=True)
    return sorted(out[:max_zones], key=lambda z: z["low"])


def nearest(zs: list[dict], price: float) -> dict:
    below = [z for z in zs if z["high"] < price]
    above = [z for z in zs if z["low"] > price]
    inside = [z for z in zs if z["low"] <= price <= z["high"]]
    return {"support": below[-1] if below else None, "support2": below[-2] if len(below) > 1 else None,
            "resistance": above[0] if above else None, "resistance2": above[1] if len(above) > 1 else None,
            "inside": inside[0] if inside else None}
