"""Statistical engine: returns, distribution, dependence, stationarity, random-walk tests.

Conventions: log returns l_t = ln(P_t / P_{t-1}); sample statistics (ddof=1);
annualisation by periods_per_year(timeframe, calendar). All functions are deterministic.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module=__name__)

TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}


def periods_per_year(timeframe: str, calendar: str = "fx") -> float:
    days = 365.0 if calendar == "24x7" else 260.0
    if timeframe == "W1":
        return 52.0
    return days * 1440 / TF_MIN[timeframe] if timeframe != "D1" else days


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def _f(x, d=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def descriptive(r: pd.Series, ppy: float) -> dict:
    q = r.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {"n": int(len(r)), "start_utc": str(r.index[0]), "end_utc": str(r.index[-1]),
            "mean": _f(r.mean(), 8), "std": _f(r.std(ddof=1), 8),
            "ann_mean_log_pct": _f(r.mean() * ppy * 100, 3), "ann_vol_pct": _f(r.std(ddof=1) * np.sqrt(ppy) * 100, 3),
            "skew": _f(stats.skew(r, bias=False), 4), "excess_kurtosis": _f(stats.kurtosis(r, fisher=True, bias=False), 4),
            "min": _f(r.min(), 6), "max": _f(r.max(), 6), "min_time_utc": str(r.idxmin()), "max_time_utc": str(r.idxmax()),
            "q01": _f(q[0.01]), "q05": _f(q[0.05]), "median": _f(q[0.5]), "q95": _f(q[0.95]), "q99": _f(q[0.99]),
            "pct_positive": _f((r > 0).mean() * 100, 2), "periods_per_year": ppy}


def normality(r: pd.Series) -> dict:
    jb = stats.jarque_bera(r)
    return {"test": "Jarque-Bera", "stat": _f(jb.statistic, 3), "p_value": _f(jb.pvalue, 6),
            "conclusion": "normality rejected at 5%" if jb.pvalue < 0.05 else "normality not rejected at 5%"}


def dependence(r: pd.Series, lags=(10, 20)) -> dict:
    nl = min(20, len(r) // 4)
    a = acf(r, nlags=nl, fft=True)
    lb = acorr_ljungbox(r, lags=list(lags), return_df=True)
    lb2 = acorr_ljungbox(r ** 2, lags=list(lags), return_df=True)
    arch_lm = het_arch(r - r.mean(), nlags=10)
    band = 1.96 / np.sqrt(len(r))
    return {
        "acf_returns": {f"lag{k}": _f(a[k], 4) for k in (1, 2, 5, 10) if k <= nl},
        "acf_95pct_band": _f(band, 4),
        "ljung_box_returns": {f"lag{k}": {"stat": _f(lb.loc[k, "lb_stat"], 3), "p_value": _f(lb.loc[k, "lb_pvalue"], 6)} for k in lags},
        "ljung_box_squared": {f"lag{k}": {"stat": _f(lb2.loc[k, "lb_stat"], 3), "p_value": _f(lb2.loc[k, "lb_pvalue"], 6)} for k in lags},
        "arch_lm_10": {"stat": _f(arch_lm[0], 3), "p_value": _f(arch_lm[1], 6)},
        "interpretation": {
            "linear_dependence": "significant" if lb.loc[lags[0], "lb_pvalue"] < 0.05 else "not significant",
            "volatility_clustering": "present" if arch_lm[1] < 0.05 else "not detected"},
    }


def stationarity(close: pd.Series, r: pd.Series) -> dict:
    lp = np.log(close)
    adf_p = adfuller(lp, autolag="AIC")
    adf_r = adfuller(r, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS p-values are table-interpolated (bounded 0.01..0.10)
        kp = kpss(r, regression="c", nlags="auto")
    return {
        "adf_log_price": {"stat": _f(adf_p[0], 4), "p_value": _f(adf_p[1], 6), "lags": adf_p[2],
                          "conclusion": "unit root rejected (stationary)" if adf_p[1] < 0.05 else "unit root not rejected"},
        "adf_returns": {"stat": _f(adf_r[0], 4), "p_value": _f(adf_r[1], 6),
                        "conclusion": "stationary" if adf_r[1] < 0.05 else "unit root not rejected"},
        "kpss_returns": {"stat": _f(kp[0], 4), "p_value_bounded": _f(kp[1], 3),
                         "conclusion": "stationarity rejected" if kp[1] < 0.05 else "stationarity not rejected"},
    }


def variance_ratio(close: pd.Series, q: int) -> dict:
    """Lo & MacKinlay (1988) overlapping variance ratio with heteroskedasticity-robust z*.
    VR(q) ~ 1 under a random walk; >1 suggests positive autocorrelation (trending), <1 mean reversion."""
    x = np.log(close.to_numpy(float))
    nq = len(x) - 1
    r = np.diff(x)
    mu = (x[-1] - x[0]) / nq
    sa = np.sum((r - mu) ** 2) / (nq - 1)
    m = q * (nq - q + 1) * (1 - q / nq)
    sc = np.sum((x[q:] - x[:-q] - q * mu) ** 2) / m
    vr = sc / sa
    d = r - mu
    den = np.sum(d ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * nq * np.sum(d[j:] ** 2 * d[:-j] ** 2) / den for j in range(1, q))
    z = np.sqrt(nq) * (vr - 1) / np.sqrt(theta)  # z*(q) = sqrt(nq) * M_r(q) / sqrt(theta(q))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"q": q, "vr": _f(vr, 4), "z_robust": _f(z, 3), "p_value": _f(p, 6),
            "conclusion": ("random walk rejected: " + ("trending (VR>1)" if vr > 1 else "mean-reverting (VR<1)"))
            if p < 0.05 else "random walk not rejected"}


def hurst(close: pd.Series, max_lag: int = 100) -> dict:
    """Lagged-difference scaling estimator: std(x_{t+k}-x_t) ~ k^H. H~0.5 random walk.
    Biased in small samples and sensitive to lag range - treat as descriptive."""
    x = np.log(close.to_numpy(float))
    lags = np.unique(np.logspace(np.log10(2), np.log10(min(max_lag, len(x) // 4)), 20).astype(int))
    tau = np.array([np.std(x[k:] - x[:-k]) for k in lags])
    slope, intercept, rv, _, se = stats.linregress(np.log(lags), np.log(tau))
    return {"estimator": "lagged-difference scaling", "H": _f(slope, 4), "std_err": _f(se, 4),
            "lags": f"{lags[0]}-{lags[-1]}", "r2": _f(rv ** 2, 4)}
