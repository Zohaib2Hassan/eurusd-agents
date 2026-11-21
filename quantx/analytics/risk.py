"""Risk engine: VaR/CVaR (historical, parametric normal, Cornish-Fisher), VaR backtest (Kupiec),
drawdowns, and buy-and-hold performance statistics.

Sign convention: VaR and CVaR are reported as POSITIVE loss fractions of position value over one bar.
Performance stats describe HOLDING the instrument long (rf = 0) - they are not a strategy result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _f(x, d=6):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), d)
    except TypeError:
        return None


def var_cvar(r: pd.Series, level: float = 0.99) -> dict:
    a = 1 - level
    mu, sd = r.mean(), r.std(ddof=1)
    q = r.quantile(a)
    h_var, h_cvar = -q, -r[r <= q].mean()
    z = stats.norm.ppf(a)
    n_var = -(mu + z * sd)
    n_cvar = -(mu - sd * stats.norm.pdf(z) / a)
    s, k = stats.skew(r, bias=False), stats.kurtosis(r, fisher=True, bias=False)
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return {"level": level, "historical_var": _f(h_var), "historical_cvar": _f(h_cvar),
            "normal_var": _f(n_var), "normal_cvar": _f(n_cvar), "cornish_fisher_var": _f(-(mu + zcf * sd)),
            "unit": "fraction of position value, 1 bar"}


def kupiec_lr(n: int, x: int, p: float) -> float:
    """Kupiec POF likelihood ratio for x exceptions in n trials at exception probability p."""
    ph = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - ph) if ph < 1 else 0) + (x * np.log(ph) if x > 0 else 0)
    return float(-2 * (ll0 - ll1))


def kupiec(r: pd.Series, level: float = 0.99, window: int = 250) -> dict:
    """Out-of-sample backtest of rolling historical VaR: VaR_t from the previous `window` returns only.
    Kupiec (1995) proportion-of-failures LR test, chi2(1)."""
    v = r.to_numpy(float)
    if len(v) <= window + 50:
        return {"error": f"need > {window + 50} returns"}
    p = 1 - level
    hits = [v[t] < np.quantile(v[t - window:t], p) for t in range(window, len(v))]
    n, x = len(hits), int(sum(hits))
    ph = x / n
    lr = kupiec_lr(n, x, p)
    pv = 1 - stats.chi2.cdf(lr, 1)
    return {"level": level, "window": window, "test_bars": n, "exceptions": x, "expected": _f(n * p, 1),
            "exception_rate": _f(ph, 4), "lr_stat": _f(lr, 3), "p_value": _f(pv, 4),
            "conclusion": "VaR model rejected (exceptions inconsistent with level)" if pv < 0.05
            else "VaR model not rejected"}


def drawdown(close: pd.Series) -> dict:
    peak = close.cummax()
    dd = close / peak - 1
    trough = dd.idxmin()
    peak_t = close.loc[:trough].idxmax()
    under = dd < 0
    runs = (under != under.shift()).cumsum()
    longest = int(under.groupby(runs).sum().max()) if under.any() else 0
    return {"max_drawdown_pct": _f(dd.min() * 100, 3), "peak_utc": str(peak_t), "trough_utc": str(trough),
            "current_drawdown_pct": _f(dd.iloc[-1] * 100, 3), "longest_underwater_bars": longest}


def performance(close: pd.Series, ppy: float) -> dict:
    r = close.pct_change().dropna()
    n = len(r)
    years = n / ppy
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    sr_p = r.mean() / r.std(ddof=1)
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    mdd = (close / close.cummax() - 1).min()
    se = np.sqrt((1 + 0.5 * sr_p ** 2) / n)  # Lo (2002), iid returns
    return {"basis": "buy-and-hold long, rf=0, simple returns", "years": _f(years, 2),
            "cagr_pct": _f(cagr * 100, 3), "ann_vol_pct": _f(r.std(ddof=1) * np.sqrt(ppy) * 100, 3),
            "sharpe": _f(sr_p * np.sqrt(ppy), 3), "sharpe_95ci": [_f((sr_p - 1.96 * se) * np.sqrt(ppy), 3),
                                                                  _f((sr_p + 1.96 * se) * np.sqrt(ppy), 3)],
            "sortino": _f(r.mean() / dd * np.sqrt(ppy), 3) if dd > 0 else None,
            "calmar": _f(cagr / abs(mdd), 3) if mdd < 0 else None}


def beta_alpha(r: pd.Series, bench: pd.Series, ppy: float) -> dict:
    j = pd.concat([r, bench], axis=1, join="inner").dropna()
    if len(j) < 60:
        return {"error": "fewer than 60 overlapping returns"}
    y, x = j.iloc[:, 0], j.iloc[:, 1]
    res = stats.linregress(x, y)
    te = (y - x).std(ddof=1) * np.sqrt(ppy)
    return {"beta": _f(res.slope, 4), "beta_se": _f(res.stderr, 4), "alpha_ann_pct": _f(res.intercept * ppy * 100, 3),
            "r2": _f(res.rvalue ** 2, 4), "tracking_error_ann_pct": _f(te * 100, 3), "n": len(j)}
