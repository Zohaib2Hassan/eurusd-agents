"""Regime detection.

1) Markov-switching model (Hamilton 1989) on % log returns: 2 regimes with switching mean and variance
   (statsmodels MarkovRegression). The CURRENT regime uses FILTERED probabilities (information up to t),
   not smoothed ones (which use future data within the sample).
2) Volatility-percentile regime: current realised vol vs its own history.
3) Trend regime: sign/strength of 100-bar log-price regression slope relative to its standard error.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _f(x, d=4):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), d)
    except TypeError:
        return None


def markov_switching(r: pd.Series, ppy: float, k: int = 2) -> dict:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    y = (r * 100).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = MarkovRegression(y, k_regimes=k, trend="c", switching_variance=True).fit(disp=False, search_reps=5)
        except Exception as e:
            return {"error": f"fit failed: {e}"}
    prm = dict(zip(res.model.param_names, np.asarray(res.params)))
    sig = [prm[f"sigma2[{i}]"] for i in range(k)]
    mus = [prm[f"const[{i}]"] for i in range(k)]
    order = np.argsort(sig)  # 0 = low vol
    names = ["low-vol", "high-vol"] if k == 2 else [f"regime{i}" for i in range(k)]
    P = np.asarray(res.regime_transition)[:, :, 0]  # P[i, j] = Pr(S_t = i | S_{t-1} = j)
    filt = np.asarray(res.filtered_marginal_probabilities)[-1]
    smooth = np.asarray(res.smoothed_marginal_probabilities)
    out = {}
    for rank, i in enumerate(order):
        out[names[rank]] = {
            "mean_ann_pct": _f(mus[i] / 100 * ppy * 100, 3),
            "vol_ann_pct": _f(np.sqrt(sig[i] / 1e4 * ppy) * 100, 3),
            "stay_probability": _f(P[i, i], 4),
            "expected_duration_bars": _f(1 / (1 - P[i, i]), 1) if P[i, i] < 1 else None,
            "share_of_sample_pct": _f(smooth[:, i].mean() * 100, 1),
            "current_filtered_prob": _f(filt[i], 4)}
    cur = names[int(np.argmax([filt[i] for i in order]))]
    return {"model": f"Markov-switching mean & variance, k={k}", "regimes": out, "current_regime": cur,
            "log_likelihood": _f(res.llf, 2), "aic": _f(res.aic, 2), "bic": _f(res.bic, 2),
            "note": "current regime from filtered (real-time) probabilities"}


def vol_percentile(realized_vol: pd.Series, lookback: int = 500) -> dict:
    s = realized_vol.dropna().iloc[-lookback:]
    pct = (s < s.iloc[-1]).mean() * 100
    label = "low" if pct < 25 else "high" if pct > 75 else "normal"
    if pct > 95:
        label = "extreme"
    return {"current_ann_pct": _f(s.iloc[-1] * 100, 3), "percentile": _f(pct, 1), "label": label, "lookback": len(s)}


def trend(close: pd.Series, n: int = 100) -> dict:
    y = np.log(close.iloc[-n:].to_numpy(float))
    res = stats.linregress(np.arange(n), y)
    t = res.slope / res.stderr if res.stderr > 0 else 0
    label = "uptrend" if t > 3 and res.rvalue ** 2 > 0.3 else "downtrend" if t < -3 and res.rvalue ** 2 > 0.3 else "sideways"
    return {"window": n, "slope_pct_per_bar": _f(res.slope * 100, 5), "t_stat": _f(t, 2), "r2": _f(res.rvalue ** 2, 3),
            "label": label, "caveat": "OLS t-stats on trending prices are inflated by autocorrelated residuals"}
