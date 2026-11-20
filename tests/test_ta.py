"""Golden tests, independent cross-checks and safety tests for the TA agent.

Run:  python -m pytest -q tests      (or)      python tests/test_ta.py
Golden values below were computed by hand (shown in comments) - not by the code under test.
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ta import calc  # noqa: E402
from ta.structure import swings, trend_state, zones  # noqa: E402
from ta.validate import validate  # noqa: E402

S = lambda a: pd.Series(a, dtype=float)
close = lambda a, b, tol=1e-9: abs(a - b) <= tol


def ohlc(n=400, seed=1, start="2026-01-05 00:00", freq="1h"):
    rng = np.random.default_rng(seed)
    c = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0007, n))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w, "close": c,
                         "tick_volume": rng.integers(100, 1000, n)}, index=idx)


# ------------------------------------------------------------------ golden values
def test_sma_golden():
    r = calc.sma(S([1, 2, 3, 4, 5]), 3)
    assert r.isna().iloc[:2].all() and list(r.iloc[2:]) == [2.0, 3.0, 4.0]


def test_ema_golden():
    # n=3 -> alpha=0.5, seed=SMA(1,2,3)=2 ; t3=0.5*4+0.5*2=3 ; t4=0.5*5+0.5*3=4
    r = calc.ema(S([1, 2, 3, 4, 5]), 3)
    assert list(r.iloc[2:]) == [2.0, 3.0, 4.0]


def test_wilder_golden():
    # n=3 -> seed 2 ; t3=(2*2+4)/3=2.666667 ; t4=(2.666667*2+5)/3=3.444444
    r = calc.rma(S([1, 2, 3, 4, 5]), 3)
    assert close(r.iloc[3], 8 / 3) and close(r.iloc[4], 31 / 9)


def test_wma_golden():
    # weights 1,2,3 : (1*1+2*2+3*3)/6 = 14/6
    assert close(calc.wma(S([1, 2, 3]), 3).iloc[-1], 14 / 6)


def test_rsi_golden():
    # closes 44,45,44,46,47,46 ; changes +1,-1,+2,+1,-1 ; n=3
    # t3: avgG=(1+0+2)/3=1, avgL=(0+1+0)/3=1/3 -> RS=3 -> RSI=75
    # t4: avgG=(1*2+1)/3=1, avgL=(1/3*2+0)/3=2/9 -> RS=4.5 -> RSI=81.8181...
    # t5: avgG=(1*2+0)/3=2/3, avgL=(2/9*2+1)/3=13/27 -> RS=18/13 -> RSI=58.0645...
    r = calc.rsi(S([44, 45, 44, 46, 47, 46]), 3)
    assert close(r.iloc[3], 75.0, 1e-9)
    assert close(r.iloc[4], 100 - 100 / 5.5, 1e-9)
    assert close(r.iloc[5], 100 - 100 / (1 + 18 / 13), 1e-9)


def test_true_range_and_atr_golden():
    df = pd.DataFrame({"open": [10, 11, 12, 11], "high": [11, 12.5, 12.2, 13],
                       "low": [9, 10.5, 10.8, 11.5], "close": [10.5, 12, 11, 12.8]}, dtype=float)
    tr = calc.true_range(df)
    # bar0 H-L=2 ; bar1 max(2, |12.5-10.5|=2, |10.5-10.5|=0)=2 ; bar2 max(1.4, 0.2, 1.2)=1.4 ; bar3 max(1.5, 2, 0.5)=2
    assert list(np.round(tr, 10)) == [2.0, 2.0, 1.4, 2.0]
    a = calc.atr(df, 3)  # seed (2+2+1.4)/3=1.8 ; next (1.8*2+2)/3=1.866667
    assert close(a.iloc[2], 1.8) and close(a.iloc[3], 5.6 / 3)


def test_bollinger_golden():
    # SMA(2,4,6)=4 ; population std = sqrt(((4)+(0)+(4))/3)=sqrt(8/3)
    b = calc.bollinger(S([2, 4, 6]), 3, 2)
    assert close(b["upper"].iloc[-1], 4 + 2 * math.sqrt(8 / 3)) and close(b["pct_b"].iloc[-1],
                                                                           (6 - b["lower"].iloc[-1]) / (b["upper"].iloc[-1] - b["lower"].iloc[-1]))


def test_pivots_golden():
    p = calc.pivots(1.10, 1.00, 1.05)
    f = p["floor"]
    assert close(f["P"], 1.05) and close(f["R1"], 1.10) and close(f["S1"], 1.00)
    assert close(f["R2"], 1.15) and close(f["S2"], 0.95)
    assert close(p["camarilla"]["R3"], 1.05 + 0.1 * 1.1 / 4)
    assert close(p["woodie"]["P"], (1.10 + 1.00 + 2 * 1.05) / 4)


def test_stochastic_williams_golden():
    df = pd.DataFrame({"high": [2, 3, 4], "low": [1, 1.5, 2], "close": [1.5, 2.5, 3.5], "open": [1.5, 1.5, 2.5]}, dtype=float)
    k, _ = calc.stochastic(df, 3, 1, 1)
    assert close(k.iloc[-1], 100 * (3.5 - 1) / (4 - 1))
    assert close(calc.williams_r(df, 3).iloc[-1], -100 * (4 - 3.5) / 3)


# ------------------------------------------------------------------ independent second implementation
def _ema_loop(x, n):
    a = 2 / (n + 1); out = [np.nan] * len(x); out[n - 1] = sum(x[:n]) / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return np.array(out)


def _rsi_loop(x, n):
    g = [max(x[i] - x[i - 1], 0) for i in range(1, len(x))]
    l = [max(x[i - 1] - x[i], 0) for i in range(1, len(x))]
    ag, al = sum(g[:n]) / n, sum(l[:n]) / n
    out = [np.nan] * len(x); out[n] = 100 - 100 / (1 + ag / al)
    for i in range(n, len(g)):
        ag = (ag * (n - 1) + g[i]) / n; al = (al * (n - 1) + l[i]) / n
        out[i + 1] = 100 - 100 / (1 + ag / al)
    return np.array(out)


def _atr_loop(h, l, c, n):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(h))]
    out = [np.nan] * len(h); out[n - 1] = sum(tr[:n]) / n
    for i in range(n, len(h)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return np.array(out)


def test_cross_check_second_implementation():
    df = ohlc(300, seed=7)
    c = df["close"].tolist()
    np.testing.assert_allclose(calc.ema(df["close"], 20).to_numpy(), _ema_loop(c, 20), rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(calc.rsi(df["close"], 14).to_numpy(), _rsi_loop(c, 14), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(calc.atr(df, 14).to_numpy(),
                               _atr_loop(df["high"].tolist(), df["low"].tolist(), c, 14), rtol=1e-12, equal_nan=True)


# ------------------------------------------------------------------ no look-ahead
def test_no_lookahead_all_indicators():
    full = ohlc(400, seed=3)
    part = full.iloc[:300]
    fns = {
        "ema": lambda d: calc.ema(d["close"], 20), "rsi": lambda d: calc.rsi(d["close"]),
        "macd": lambda d: calc.macd(d["close"])[2], "atr": lambda d: calc.atr(d),
        "adx": lambda d: calc.dmi(d)[2], "bb": lambda d: calc.bollinger(d["close"])["pct_b"],
        "cci": lambda d: calc.cci(d), "dpo": lambda d: calc.dpo(d["close"]), "psar": lambda d: calc.psar(d),
        "ichi_a": lambda d: calc.ichimoku(d)["span_a"], "yz": lambda d: calc.yang_zhang(d, 20, 6240),
        "aroon": lambda d: calc.aroon(d)[0], "keltner": lambda d: calc.keltner(d)["upper"],
    }
    for name, f in fns.items():
        a, b = f(part).to_numpy(), f(full).iloc[:300].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True, err_msg=f"look-ahead in {name}")


def test_swings_confirmed_only_after_k_bars():
    df = ohlc(200, seed=5)
    for s in swings(df, 3):
        i = df.index.get_loc(s["time"])
        assert df.index.get_loc(s["confirmed_at"]) == i + 3
    # a swing found on 150 bars must still exist identically with 200 bars
    a = {(s["type"], s["time"]) for s in swings(df.iloc[:150], 3)}
    b = {(s["type"], s["time"]) for s in swings(df, 3) if s["confirmed_at"] <= df.index[149]}
    assert a == b


def test_trend_state_and_zones():
    sw = [{"type": "low", "price": 1.0, "time": 1, "confirmed_at": 1}, {"type": "high", "price": 1.1, "time": 2, "confirmed_at": 2},
          {"type": "low", "price": 1.05, "time": 3, "confirmed_at": 3}, {"type": "high", "price": 1.2, "time": 4, "confirmed_at": 4}]
    assert trend_state(sw)["state"] == "uptrend"
    z = zones(sw, atr_value=0.1, zone_atr=0.5)
    assert all(zz["high"] - zz["low"] <= 0.1 + 1e-12 for zz in z)  # width capped (no chaining)


# ------------------------------------------------------------------ validation
def test_validation_drops_forming_bar_and_dupes():
    df = ohlc(50, start="2026-11-18 00:00")
    df = pd.concat([df, df.iloc[[-1]]])  # duplicate timestamp
    now = df.index[-1] + pd.Timedelta(minutes=30)  # last bar still forming
    out, warns, q, meta = validate(df, "H1", now.to_pydatetime())
    assert "dropped_forming_bar" in meta and out.index.is_unique and any("duplicate" in w for w in warns)


def test_validation_weekend_gap_not_flagged_but_weekday_gap_is():
    df = ohlc(120, start="2026-11-13 00:00")  # Fri
    keep = ~((df.index > "2026-11-13 21:00") & (df.index < "2026-11-15 22:00"))  # weekend removed
    _, w1, _, _ = validate(df[keep], "H1", datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert not any("unexpected gaps" in w for w in w1)
    keep2 = keep & ~((df.index > "2026-11-17 03:00") & (df.index < "2026-11-17 09:00"))  # Tue gap
    _, w2, _, _ = validate(df[keep2], "H1", datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert any("unexpected gaps" in w for w in w2)


def test_validation_bad_ohlc_removed():
    df = ohlc(30)
    df.iloc[5, df.columns.get_loc("high")] = df["low"].iloc[5] - 0.01
    out, warns, q, _ = validate(df, "H1", df.index[-1].to_pydatetime() + pd.Timedelta(hours=5))
    assert len(out) == 29 and q < 1


# ------------------------------------------------------------------ FX contract math (spec worked example 4)
def test_pip_value_and_position_size():
    assert close(calc.pip_value("EURUSD", 1.10), 10.0)                      # quote = USD
    assert close(calc.pip_value("USDJPY", 150.0), 100000 * 0.01 / 150)     # base = USD -> divide by price
    assert close(calc.pip_value("EURGBP", 0.85, quote_to_account=1.27), 10 * 1.27)
    # risk $50, stop 25 pips, $10/pip/lot -> 0.2 lots
    assert close(calc.position_size(50, 25, 10.0), 0.20)


# ------------------------------------------------------------------ safety
def test_agent_is_read_only():
    src = (Path(__file__).resolve().parents[1] / "ta" / "agent.py").read_text()
    assert "order_send" not in src and "Executor" not in src


def test_symbol_lock():
    from agents.market_data import Market, SymbolLockError
    m = Market({"symbol": "EURUSD", "mt5": {"use_mock_if_unavailable": True}})
    m.mock = True
    try:
        m.rates("H1", 10, symbol="GBPUSD")
        assert False, "symbol lock failed"
    except SymbolLockError:
        pass


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
