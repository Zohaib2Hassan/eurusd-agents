"""Volatility engine: realised, EWMA, GARCH-family, volatility cone.

GARCH models are fitted to percentage log returns (r*100) with Student-t innovations (arch package).
GARCH(1,1):  s2_t = w + a*e2_{t-1} + b*s2_{t-1}
GJR-GARCH:   s2_t = w + (a + g*I[e_{t-1}<0])*e2_{t-1} + b*s2_{t-1}   (Glosten, Jagannathan & Runkle 1993)
Model choice by BIC. Persistence = a + b (+ g/2 for GJR); half-life = ln(0.5)/ln(persistence).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _f(x, d=4):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), d)
    except TypeError:
        return None


def realized(r: pd.Series, ppy: float, window: int = 20) -> pd.Series:
    return r.rolling(window).std(ddof=1) * np.sqrt(ppy)


def ewma(r: pd.Series, ppy: float, lam: float = 0.94, seed: int = 30) -> pd.Series:
    """RiskMetrics EWMA: s2_t = lam*s2_{t-1} + (1-lam)*r2_{t-1}. Value at t uses returns up to t-1."""
    v = r.to_numpy(float)
    s2 = np.full(len(v), np.nan)
    if len(v) <= seed:
        return pd.Series(s2, index=r.index)
    s2[seed] = np.var(v[:seed], ddof=1)
    for i in range(seed + 1, len(v)):
        s2[i] = lam * s2[i - 1] + (1 - lam) * v[i - 1] ** 2
    return pd.Series(np.sqrt(s2 * ppy), index=r.index)


def garch(r: pd.Series, ppy: float) -> dict:
    from arch import arch_model
    y = r * 100
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, o in (("GARCH(1,1)-t", 0), ("GJR-GARCH(1,1)-t", 1)):
            try:
                res = arch_model(y, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist="t").fit(disp="off")
                fits[name] = res
            except Exception as e:  # pragma: no cover
                fits[name] = e
    ok = {k: v for k, v in fits.items() if not isinstance(v, Exception)}
    if not ok:
        return {"error": "; ".join(str(v) for v in fits.values())}
    best_name = min(ok, key=lambda k: ok[k].bic)
    res = ok[best_name]
    p = res.params
    a, b, g = p.get("alpha[1]", np.nan), p.get("beta[1]", np.nan), p.get("gamma[1]", 0.0)
    pers = a + b + g / 2
    fc = res.forecast(horizon=10, reindex=False).variance.iloc[-1]
    ann = lambda v: _f(np.sqrt(v / 1e4 * ppy) * 100, 3)
    lr_var = p["omega"] / (1 - pers) if pers < 1 else np.nan
    return {
        "selected": best_name, "selection": "lowest BIC",
        "bic": {k: _f(v.bic, 2) for k, v in ok.items()},
        "params": {k: _f(v, 6) for k, v in p.items()},
        "pvalues": {k: _f(v, 4) for k, v in res.pvalues.items()},
        "persistence": _f(pers, 4),
        "half_life_bars": _f(np.log(0.5) / np.log(pers), 1) if 0 < pers < 1 else None,
        "long_run_vol_ann_pct": ann(lr_var) if np.isfinite(lr_var) else None,
        "current_cond_vol_ann_pct": ann(res.conditional_volatility.iloc[-1] ** 2),
        "forecast_vol_ann_pct": {"h1": ann(fc.iloc[0]), "h5": ann(fc.iloc[4]), "h10": ann(fc.iloc[9])},
        "converged": bool(res.convergence_flag == 0),
        "stationary": bool(pers < 1),
        "leverage_effect": None if best_name.startswith("GARCH") else
        ("negative shocks raise vol more" if g > 0 and res.pvalues.get("gamma[1]", 1) < 0.05 else "not significant"),
        "note": "in-sample fit; forecasts are model-conditional, not guarantees",
    }


def cone(r: pd.Series, ppy: float, windows=(10, 20, 60, 120)) -> dict:
    out = {}
    for w in windows:
        if len(r) < w * 3:
            continue
        s = realized(r, ppy, w).dropna() * 100
        cur = s.iloc[-1]
        out[f"w{w}"] = {"current": _f(cur, 2), "min": _f(s.min(), 2), "p25": _f(s.quantile(.25), 2),
                        "median": _f(s.median(), 2), "p75": _f(s.quantile(.75), 2), "max": _f(s.max(), 2),
                        "percentile_of_current": _f((s < cur).mean() * 100, 1)}
    return out
