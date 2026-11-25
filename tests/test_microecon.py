"""Agent 3 tests: solver golden values (hand-checked), answer-bank integrity, grading rubric, microstructure recovery.

Run:  python tests/test_microecon.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microecon import econ, microstructure as ms, trading as tr  # noqa: E402
from microecon.bank import BANK  # noqa: E402
from microecon.grader import Session, extract_numbers, grade_step  # noqa: E402

close = lambda a, b, t=1e-6: abs(a - b) <= t * max(1, abs(b))


# ---------------------------------------------------------------- microeconomics (hand values)
def test_market_problem1():
    r, _ = econ.market(500, 10, -100, 15, floor=30)
    assert r["P_star"] == 24 and r["Q_star"] == 260 and r["CS"] == 3380
    assert close(r["PS"], 0.5 * (24 - 100 / 15) * 260)          # 2253.33 exact
    assert r["Q_traded"] == 200 and r["excess_supply"] == 150 and r["DWL"] == 300


def test_market_tax_incidence():
    # Qd = 100 - P, Qs = P (b = d = 1): tax 10 -> buyers and sellers each bear 5; Q 50 -> 45; revenue 450; DWL 25
    r, _ = econ.market(100, 1, 0, 1, tax=10)
    assert close(r["tax_Pbuyer"], 55) and close(r["tax_Pseller"], 45) and close(r["tax_Q"], 45)
    assert close(r["tax_revenue"], 450) and close(r["tax_DWL"], 25) and close(r["buyer_share"], 0.5)


def test_ceiling():
    r, _ = econ.market(100, 1, 0, 1, ceiling=40)          # P* = 50
    assert r["Q_traded"] == 40 and r["shortage"] == 20 and close(r["DWL"], 0.5 * (60 - 40) * 10)


def test_monopoly_problem2():
    r, _ = econ.monopoly(120, 2, 20, 100)
    assert (r["Qm"], r["Pm"], r["profit"], r["Qc"], r["Pc"], r["DWL"]) == (25, 70, 1150, 50, 20, 625)


def test_cournot_problem3_and_ordering():
    r, _ = econ.cournot(180, 1, 30, 30)
    assert (r["Q1"], r["Q2"], r["P"], r["profit1"]) == (50, 50, 80, 2500)
    assert (r["Q_cartel"], r["P_cartel"]) == (75, 105)
    assert r["Q_cartel"] < r["Q_total"] < r["stackelberg_Q_leader"] + r["stackelberg_Q_follower"] < r["bertrand_Q"]


def test_cournot_asymmetric():
    # P = 100 - Q, c1 = 10, c2 = 20: q1 = (100-20+20)/3 = 33.33, q2 = (100-40+10)/3 = 23.33
    r, _ = econ.cournot(100, 1, 10, 20)
    assert close(r["Q1"], 100 / 3) and close(r["Q2"], 70 / 3)


def test_nash():
    pd_game, _ = econ.nash_2player([[3, 0], [5, 1]], [[3, 5], [0, 1]], ["C", "D"], ["C", "D"])
    assert pd_game["pure_nash"] == [("D", "D")]
    mp, _ = econ.nash_2player([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    assert mp["pure_nash"] == [] and close(mp["mixed"]["p_row_first"], 0.5)
    bos, _ = econ.nash_2player([[2, 0], [0, 1]], [[1, 0], [0, 2]])      # battle of the sexes: 2 pure NE
    assert len(bos["pure_nash"]) == 2 and close(bos["mixed"]["p_row_first"], 2 / 3) and close(bos["mixed"]["q_col_first"], 1 / 3)


def test_slutsky_cobb_douglas():
    r, _ = econ.cobb_douglas_consumer(0.5, 100, 2, 1, 4)
    assert r["x"] == 25 and r["x_new"] == 12.5
    assert close(r["substitution_effect"] + r["income_effect"], r["total_effect"])
    assert close(r["substitution_effect"], -6.25)


def test_cost_min():
    r, _ = econ.cobb_douglas_cost(1, 0.5, 0.5, 4, 1, 100)
    assert close(r["L"], 50) and close(r["K"], 200) and close(r["cost"], 400)


def test_arc_elasticity():
    assert close(econ.elasticity(100, 80, 10, 12)["arc_elasticity"], (-20 / 90) / (2 / 11))


# ---------------------------------------------------------------- trading math
def test_fx_problem4_corrected():
    r, _ = tr.fx_position(50000, 1.5, 1.0850, 1.0810, "EURUSD", 1.0970, 0.45)
    assert r["dollar_risk"] == 750 and r["stop_pips"] == 40 and close(r["lots"], 1.875)
    assert close(r["profit_at_target"], 2250) and close(r["expectancy"], 600)   # NOT 937.50


def test_fx_jpy_pip_value():
    r, _ = tr.fx_position(10000, 1, 150.00, 149.50, "USDJPY")
    assert close(r["stop_pips"], 50) and close(r["pip_value_per_lot"], 1000 / 150)


def test_expectancy_and_breakeven():
    r, _ = tr.expectancy(0.35, 300, 100, 100)
    assert close(r["expectancy"], 40) and close(r["breakeven_win_rate"], 0.25) and close(r["expected_total"], 4000)


def test_dca_harmonic():
    r, _ = tr.dca(1000, [40000, 32000, 50000])
    assert close(r["total_units"], 0.07625) and close(r["cost_basis"], 3 / (1 / 40000 + 1 / 32000 + 1 / 50000))


def test_arbitrage_problem5():
    r, _ = tr.spatial_arbitrage(40000, 40800, 1, 0.0005, 0, 0.002, 10)
    assert close(r["profit"], 688.0408, 1e-6)


def test_funding_problem7():
    r, _ = tr.funding_carry(60000, 1, 0.0004, 90, 0.0005, 12000, 30)
    assert close(r["gross_funding"], 2160) and close(r["fees"], 120) and close(r["roc"], 2040 / 72000)


def test_liquidation_problem8_and_exact():
    a, _ = tr.liquidation(3000, 10, 0.005)
    assert close(a["liq_approx"], 2715) and close(a["liq_exact"], 3000 * 0.9 / 0.995)
    b, _ = tr.liquidation(3000, None, 0.005, margin=4500, notional=30000)
    assert close(b["liq_approx"], 2565)
    s, _ = tr.liquidation(3000, 10, 0.005, side="short")
    assert close(s["liq_approx"], 3000 * 1.095) and close(s["liq_exact"], 3000 * 1.1 / 1.005)


def test_kelly():
    r, _ = tr.kelly(0.55, 1)
    assert close(r["kelly"], 0.10) and close(r["fractional"], 0.05)
    assert tr.kelly(0.4, 1)[0]["kelly"] < 0      # negative edge -> don't bet


def test_cip():
    r, _ = tr.covered_interest_parity(1.10, 0.05, 0.03, 360, 360)
    assert close(r["forward"], 1.10 * 1.05 / 1.03)


def test_risk_of_ruin_monotone_in_risk():
    lo = tr.risk_of_ruin(0.35, 3, 0.01, 0.3, 300, 5000)[0]["p_ruin"]
    hi = tr.risk_of_ruin(0.35, 3, 0.05, 0.3, 300, 5000)[0]["p_ruin"]
    assert hi > lo


# ---------------------------------------------------------------- bank & grader
def test_bank_complete_and_keys_finite():
    assert sorted(BANK) == list(range(1, 9))
    for p in BANK.values():
        assert len(p.steps) == 4 and p.takeaway
        for s in p.steps:
            assert s.targets or s.concepts
            assert all(math.isfinite(t.value) for t in s.targets)
            assert s.hint and s.solution


def test_extract_numbers():
    assert extract_numbers("Q1* = 50, Q2* = 50, total $1,200.5 and 2.5k") == [50.0, 50.0, 1200.5, 2500.0]


def test_grading_bands():
    st = BANK[2].steps[3]                       # DWL = 625
    assert grade_step(st, "625")["fraction"] == 1.0
    assert grade_step(st, "630")["fraction"] == 0.85        # 0.8% off -> minor slip
    assert grade_step(st, "600")["fraction"] == 0.65        # 4% off
    assert grade_step(st, "1250")["fraction"] == 0.35       # conceptual error
    assert grade_step(st, "I don't know")["fraction"] == 0.0
    assert grade_step(st, "625", hint_used=True)["fraction"] == 0.8


def test_percent_target_accepts_decimal():
    st = BANK[8].steps[1]                       # 9.5%
    assert grade_step(st, "9.5%")["fraction"] == 1.0 and grade_step(st, "0.095")["fraction"] == 1.0


def test_text_step_concepts():
    st = BANK[7].steps[3]
    assert grade_step(st, "funding flip and liquidation in a rally")["passed"]
    assert not grade_step(st, "it might not work")["passed"]


def test_session_flow_show_solution_and_report():
    s = Session(BANK[4])
    assert s.submit("$750")["points"] == 25
    sol = s.show_solution()
    assert len(sol) == 3 and "937.50" in sol[-1] and s.done()
    assert abs(sum(s.scores) * s.weight - 25) < 1e-9 and "Needs Review" in s.report()


def test_wrong_original_key_is_not_full_credit():
    st = BANK[4].steps[3]
    assert not grade_step(st, "R = 3, expectancy $937.50")["passed"]
    assert grade_step(st, "R = 3, expectancy $600")["fraction"] == 1.0


# ---------------------------------------------------------------- microstructure
def test_roll_recovers_spread():
    rng = np.random.default_rng(1)
    n = 20000
    m = 1.10 + np.cumsum(rng.normal(0, 0.00005, n))
    p = m + rng.choice([-1, 1], n) * 0.0001          # spread 0.0002
    est = ms.roll(pd.Series(p))["estimate_price"]
    assert abs(est - 0.0002) < 0.00002


def test_corwin_schultz_order_of_magnitude():
    rng = np.random.default_rng(2)
    H, L, price, S = [], [], 100.0, 0.01
    for _ in range(600):
        path = price * np.exp(np.cumsum(rng.normal(0, 0.01 / np.sqrt(390), 390)))
        price = path[-1]
        H.append(path.max() * (1 + S / 2)); L.append(path.min() * (1 - S / 2))
    est = ms.corwin_schultz(pd.DataFrame({"high": H, "low": L}))["mean_bps"]
    assert 60 < est < 160          # true spread 100 bps; estimator is noisy and biased


def test_amihud_units():
    df = pd.DataFrame({"close": [100.0, 101.0, 100.0], "volume": [1e6, 1e6, 1e6]})
    r = ms.amihud(df)   # |ln(1.01)| / 1.01e8 and |ln(100/101)| / 1e8, mean, per $1B
    exp = np.mean([abs(np.log(1.01)) / 1.01e8, abs(np.log(100 / 101)) / 1e8]) * 1e9
    assert abs(r["illiq_per_billion_usd"] - round(exp, 6)) < 1e-6


def test_microecon_has_no_order_code():
    src = "".join(p.read_text() for p in (Path(__file__).resolve().parents[1] / "microecon").rglob("*.py"))
    assert "order_send" not in src


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
