"""QUANT-X Phase-1 tests: hand-computed values, simulation-based checks (known truth), safety.

Run:  python tests/test_quantx.py
Simulation tests use fixed seeds and generous tolerances; they check that estimators recover
parameters of processes whose truth we control.
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from quantx.analytics import regime, risk, stats, vol  # noqa: E402
from quantx.data.store import Store  # noqa: E402

idx = lambda n: pd.bdate_range("2015-01-01", periods=n, tz="UTC")


def price_from(r):
    return pd.Series(100 * np.exp(np.cumsum(np.r_[0, r])), index=idx(len(r) + 1))


# ---------------------------------------------------------------- hand-computed
def test_periods_per_year():
    assert stats.periods_per_year("D1", "fx") == 260 and stats.periods_per_year("D1", "24x7") == 365
    assert stats.periods_per_year("H1", "fx") == 260 * 24 and stats.periods_per_year("W1") == 52


def test_log_returns_golden():
    r = stats.log_returns(pd.Series([100.0, 110.0, 99.0]))
    assert abs(r.iloc[0] - math.log(1.1)) < 1e-12 and abs(r.iloc[1] - math.log(0.9)) < 1e-12


def test_historical_var_cvar_golden():
    # 100 returns -1%..-100%... use 1..100 bp losses: quantile(0.05) with linear interp
    r = pd.Series(np.arange(-50, 50) / 1000.0)  # -0.050 .. 0.049
    v = risk.var_cvar(r, 0.95)
    q = np.quantile(r, 0.05)
    assert abs(v["historical_var"] - round(-q, 6)) < 1e-9
    assert abs(v["historical_cvar"] - round(-r[r <= q].mean(), 6)) < 1e-9


def test_normal_var_golden():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 200000))
    v = risk.var_cvar(r, 0.99)
    assert abs(v["normal_var"] - 0.02326) < 2e-4           # z_0.99 * 1%
    assert abs(v["normal_cvar"] - 0.02665) < 2e-4          # sigma*phi(z)/0.01
    assert abs(v["cornish_fisher_var"] - v["normal_var"]) < 3e-4  # no skew/kurtosis -> equal


def test_drawdown_golden():
    c = pd.Series([100, 120, 90, 95, 130, 117.0], index=idx(6))
    d = risk.drawdown(c)
    assert d["max_drawdown_pct"] == -25.0 and abs(d["current_drawdown_pct"] - (-10.0)) < 1e-9


def test_kupiec_lr_golden():
    # n=1000, p=0.01, x=20 by hand:
    # ll0 = 980 ln .99 + 20 ln .01 = -9.8493 - 92.1034 = -101.9527
    # ll1 = 980 ln .98 + 20 ln .02 = -19.7987 - 78.2404 = -98.0391   -> LR = 7.8272
    assert abs(risk.kupiec_lr(1000, 20, 0.01) - 7.8272) < 1e-3
    assert abs(risk.kupiec_lr(1000, 10, 0.01)) < 1e-9      # exactly expected -> LR 0
    assert risk.kupiec_lr(1000, 0, 0.01) > 0               # zero exceptions handled


# ---------------------------------------------------------------- simulation: known truth
def test_variance_ratio_size_and_power():
    rng = np.random.default_rng(7)
    zs = [stats.variance_ratio(price_from(rng.normal(0, 0.01, 1500)), 5)["z_robust"] for _ in range(200)]
    assert 0.8 < np.std(zs) < 1.2 and abs(np.mean(zs)) < 0.2          # ~N(0,1) under random walk
    rej = np.mean([abs(z) > 1.96 for z in zs])
    assert rej < 0.10                                                   # size close to 5%
    e = rng.normal(0, 0.01, 1500); r = np.zeros(1500)
    for t in range(1, 1500):
        r[t] = 0.3 * r[t - 1] + e[t]                                    # positive autocorrelation
    v = stats.variance_ratio(price_from(r), 5)
    assert v["vr"] > 1.3 and v["p_value"] < 0.01


def test_hurst_random_walk_near_half():
    rng = np.random.default_rng(3)
    h = stats.hurst(price_from(rng.normal(0, 0.01, 4000)))["H"]
    assert 0.42 < h < 0.58


def test_stationarity_tests_distinguish():
    rng = np.random.default_rng(5)
    c = price_from(rng.normal(0, 0.01, 1500))
    s = stats.stationarity(c, stats.log_returns(c))
    assert s["adf_returns"]["p_value"] < 0.01 and s["adf_log_price"]["p_value"] > 0.05


def test_garch_recovers_parameters():
    rng = np.random.default_rng(11)
    n, w, a, b = 4000, 0.02, 0.08, 0.90   # in % units
    e = rng.standard_normal(n); s2 = np.empty(n); y = np.empty(n); s2[0] = w / (1 - a - b)
    for t in range(n):
        if t:
            s2[t] = w + a * y[t - 1] ** 2 + b * s2[t - 1]
        y[t] = math.sqrt(s2[t]) * e[t]
    g = vol.garch(pd.Series(y / 100, index=idx(n)), 260)
    assert g["converged"] and abs(g["persistence"] - 0.98) < 0.02
    assert abs(g["params"]["alpha[1]"] - a) < 0.04 and abs(g["params"]["beta[1]"] - b) < 0.05


def test_markov_switching_finds_two_vol_regimes():
    rng = np.random.default_rng(2)
    seg = [(0.004, 400), (0.015, 300), (0.004, 400), (0.015, 300)]
    r = np.concatenate([rng.normal(0, s, n) for s, n in seg])
    m = regime.markov_switching(pd.Series(r, index=idx(len(r))), 260)
    lo, hi = m["regimes"]["low-vol"]["vol_ann_pct"], m["regimes"]["high-vol"]["vol_ann_pct"]
    assert abs(lo - 0.4 * math.sqrt(260)) < 1.5 and abs(hi - 1.5 * math.sqrt(260)) < 4
    assert m["current_regime"] == "high-vol"   # sample ends in the high-vol segment


def test_ewma_uses_only_past():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 300), index=idx(300))
    a = vol.ewma(r.iloc[:200], 260)
    b = vol.ewma(r, 260).iloc[:200]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_kupiec_correct_model_not_rejected():
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0, 0.01, 3000), index=idx(3000))
    k = risk.kupiec(r, 0.99, 250)
    assert k["p_value"] > 0.05


def test_sharpe_ci_contains_point():
    rng = np.random.default_rng(6)
    c = price_from(rng.normal(0.0004, 0.01, 1000))
    p = risk.performance(c, 260)
    assert p["sharpe_95ci"][0] < p["sharpe"] < p["sharpe_95ci"][1]


# ---------------------------------------------------------------- store + safety
def test_store_roundtrip_and_provenance(tmp="/tmp/qx_test.db"):
    Path(tmp).unlink(missing_ok=True)
    s = Store(tmp)
    df = pd.DataFrame({"open": [1, 2.0], "high": [2, 3.0], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20.0]}, index=idx(2))
    meta = {"source": "test", "symbol": "X", "timeframe": "D1", "retrieved_at_utc": "t", "calendar": "fx"}
    run = s.save_bars(df, meta, 1.0, [])
    s.save_bars(df, meta, 1.0, [])  # upsert, no duplicates
    back = s.load_bars("test", "X", "D1")
    assert len(back) == 2 and back["close"].tolist() == [1.5, 2.5]
    assert s.db.execute("select count(*) from ingestion_runs").fetchone()[0] == 2 and run
    s.close()


def test_quantx_has_no_order_code():
    root = Path(__file__).resolve().parents[1] / "quantx"
    src = "".join(p.read_text() for p in root.rglob("*.py"))
    assert "order_send" not in src and "TRADE_ACTION" not in src


# ---------------------------------------------------------------- broker server time
def test_server_offset_weekend_and_live(tmpdir="/tmp/qx_srv"):
    from datetime import datetime, timezone
    import agents.server_time as st
    st.CACHE = Path(tmpdir) / "offset.json"
    st.CACHE.unlink(missing_ok=True)
    # Saturday 13:07 UTC, last tick Fri 23:59:58 server time on a UTC+2 broker (winter: NY close 22:00 UTC)
    now = datetime(2026, 11, 21, 13, 7, tzinfo=timezone.utc)
    last_tick_server = int(datetime(2026, 11, 20, 23, 59, 58, tzinfo=timezone.utc).timestamp())
    sec, how = st.infer_offset(last_tick_server, "TestSrv", None, now)
    assert sec == 2 * 3600 and "Friday" in how, (sec, how)
    # live: Wednesday, fresh tick 2h ahead -> detected and cached
    now2 = datetime(2026, 11, 18, 10, 0, 3, tzinfo=timezone.utc)
    sec2, how2 = st.infer_offset(int(now2.timestamp()) + 7200, "TestSrv", None, now2)
    assert sec2 == 7200 and how2 == "live tick"
    sec3, how3 = st.infer_offset(last_tick_server, "TestSrv", None, now)  # weekend again -> cache
    assert sec3 == 7200 and how3.startswith("cached")
    assert st.infer_offset(None, "X", "3")[0] == 3 * 3600  # override


def test_correlation_aligns_daily_bars_with_different_open_hours():
    from quantx.analytics import corr
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2024-01-01", periods=400, tz="UTC")
    common = rng.normal(0, 0.01, 400)
    fx = pd.Series(100 * np.exp(np.cumsum(common + rng.normal(0, 0.005, 400))), index=days - pd.Timedelta(hours=2))  # 22:00 prev day
    gold = pd.Series(100 * np.exp(np.cumsum(common + rng.normal(0, 0.005, 400))), index=days + pd.Timedelta(hours=1)) # 01:00
    c = corr.correlation({"FX": fx, "GOLD": gold}, daily=True)
    p = c["pairs"]["FX~GOLD"]
    assert p["n"] >= 390 and p["full"] > 0.7, p
    assert "error" in corr.correlation({"FX": fx, "GOLD": gold}, daily=False)  # exact stamps never match


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
