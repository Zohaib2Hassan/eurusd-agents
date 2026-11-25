"""Seeded problem generator: unlimited practice with solver-computed answer keys.

Parameters are drawn so that answers stay readable (integer equilibria, round prices). Every key comes
from the same solver functions used in direct-assistance mode; each step lists common wrong answers so
the grader can explain the likely mistake.
"""
from __future__ import annotations

import random

from ta.calc import pip_size, pip_value
from . import econ
from . import trading as tr
from .bank import Problem, Step, Target

TOPICS: dict = {}


def topic(key, title, difficulty):
    def deco(fn):
        TOPICS[key] = (title, difficulty, fn)
        return fn
    return deco


def T(label, value, kind="number", tol=0.005):
    return Target(label, float(value), kind, tol)


def _r(x, d=2):
    return round(x, d)


# ----------------------------------------------------------------- microeconomics
@topic("tax", "Per-unit tax incidence", "Intermediate")
def g_tax(rng: random.Random):
    b, d = rng.choice([1, 2, 4, 5]), rng.choice([1, 2, 3, 5])
    P0, Q0 = rng.randrange(10, 60, 2), rng.randrange(100, 400, 10)
    a, c = Q0 + b * P0, Q0 - d * P0
    t = rng.choice([2, 4, 5, 6, 10])
    while a - b * ((a - c + d * t) / (b + d)) <= 0:
        t //= 2
    r, _ = econ.market(a, b, c, d, tax=t)
    return f"Qd = {a} - {b}P,  Qs = {c} + {d}P.  A per-unit tax of ${t} is levied on sellers.", [
        Step("Find the pre-tax equilibrium price and quantity.", [T("P*", r["P_star"]), T("Q*", r["Q_star"])],
             hint="Set Qd = Qs.", solution=f"P* = {r['P_star']:g}, Q* = {r['Q_star']:g}"),
        Step("Find the price buyers pay, the price sellers receive and the new quantity.",
             [T("buyer price", r["tax_Pbuyer"], tol=0.002), T("seller price", r["tax_Pseller"], tol=0.002), T("Q with tax", r["tax_Q"], tol=0.002)],
             hint="With the tax, Qd(Pb) = Qs(Pb - t). Solve for Pb.",
             solution=f"Pb = {r['tax_Pbuyer']:.4g}, Ps = {r['tax_Pseller']:.4g}, Q = {r['tax_Q']:.4g}",
             misconceptions=[(r["P_star"] + t, "Buyers do not pay the full tax: the price rises by t·d/(b+d), not by t.")]),
        Step("Compute tax revenue and deadweight loss.", [T("revenue", r["tax_revenue"], tol=0.002), T("DWL", r["tax_DWL"], tol=0.002)],
             hint="Revenue = t × Q_tax; DWL = 1/2 × t × (Q* - Q_tax).",
             solution=f"Revenue = {r['tax_revenue']:.4g}; DWL = {r['tax_DWL']:.4g}",
             misconceptions=[(t * r["Q_star"], "Revenue uses the quantity AFTER the tax, not Q*."),
                             (t * (r["Q_star"] - r["tax_Q"]), "That rectangle is twice the DWL triangle - include the 1/2.")]),
        Step("What percentage of the tax is borne by buyers?", [T("buyer share %", r["buyer_share"] * 100, "percent")],
             hint="Buyer share = (Pb - P*)/t = d/(b + d).", solution=f"{r['buyer_share']:.1%}",
             misconceptions=[(100.0, "Statutory incidence (who pays the government) is not economic incidence."),
                             (b / (b + d) * 100, "Reversed: buyers bear d/(b+d) - the steeper (less elastic) side bears more.")]),
    ], "Economic incidence depends only on relative elasticities, not on who legally remits the tax; the less elastic side bears more."


@topic("ceiling", "Price ceiling and shortage", "Foundational")
def g_ceiling(rng):
    b, d = rng.choice([1, 2, 5]), rng.choice([1, 2, 3, 4])
    P0, Q0 = rng.randrange(20, 80, 2), rng.randrange(100, 500, 10)
    a, c = Q0 + b * P0, Q0 - d * P0
    pc = P0 - rng.choice([2, 4, 6, 8])
    r, _ = econ.market(a, b, c, d, ceiling=pc)
    return f"Qd = {a} - {b}P,  Qs = {c} + {d}P.  A price ceiling of ${pc} is imposed.", [
        Step("Find the equilibrium price and quantity.", [T("P*", r["P_star"]), T("Q*", r["Q_star"])], hint="Qd = Qs.",
             solution=f"P* = {r['P_star']:g}, Q* = {r['Q_star']:g}"),
        Step("Find the quantity traded and the shortage under the ceiling.", [T("Q traded", r["Q_traded"]), T("shortage", r["shortage"])],
             hint="Under a binding ceiling, SELLERS determine how much is traded.",
             solution=f"Qs = {r['Q_traded']:g} traded; Qd = {r['Qd_at_ceiling']:g}; shortage {r['shortage']:g}",
             misconceptions=[(r["Qd_at_ceiling"], "That is quantity demanded; with a ceiling, trade is limited by quantity supplied.")]),
        Step("Find the demand price at the traded quantity (buyers' marginal willingness to pay).", [T("demand price", r["demand_price_at_Qt"], tol=0.002)],
             hint="Invert demand at Q = Q traded.", solution=f"P = (a - Q)/b = {r['demand_price_at_Qt']:.4g}"),
        Step("Compute the deadweight loss.", [T("DWL", r["DWL"], tol=0.002)],
             hint="Triangle between demand and supply from Q traded to Q*.", solution=f"DWL = {r['DWL']:.4g}",
             misconceptions=[(2 * r["DWL"], "Missing the 1/2 for a triangle.")]),
    ], "A binding ceiling creates a shortage and a deadweight loss; queues or black markets then ration the scarce units, often wasting even more surplus."


@topic("monopoly", "Monopoly pricing and the Lerner index", "Intermediate")
def g_monopoly(rng):
    B = rng.choice([1, 2, 4])
    mc = rng.randrange(10, 40, 2)
    Qm = rng.randrange(10, 50, 2)
    A = mc + 2 * B * Qm
    fc = rng.choice([0, 50, 100, 200])
    r, _ = econ.monopoly(A, B, mc, fc)
    e = -1 / r["lerner"]
    return f"Inverse demand P = {A} - {B}Q; total cost TC = {mc}Q + {fc}.", [
        Step("Give the marginal revenue function: intercept and slope.", [T("MR intercept", A), T("MR slope (abs)", 2 * B)],
             hint="TR = AQ - BQ²; MR = A - 2BQ.", solution=f"MR = {A} - {2 * B}Q",
             misconceptions=[(B, "MR falls twice as fast as demand for linear demand: slope 2B.")]),
        Step("Find the profit-maximising quantity, price and profit.", [T("Qm", r["Qm"]), T("Pm", r["Pm"]), T("profit", r["profit"])],
             hint="MR = MC, then price from demand, then subtract fixed cost.",
             solution=f"Qm = {r['Qm']:g}, Pm = {r['Pm']:g}, profit = {r['profit']:g}",
             misconceptions=[(r["profit"] + fc, "Subtract the fixed cost."), (r["Qc"], "That is the competitive quantity (P = MC).")]),
        Step("Compute the Lerner index and the price elasticity of demand at the optimum.",
             [T("Lerner", r["lerner"], tol=0.01), T("elasticity", e, tol=0.01)],
             hint="L = (P - MC)/P = -1/e at the monopoly optimum.", solution=f"L = {r['lerner']:.4f}, e = {e:.4f}",
             misconceptions=[(-e, "Elasticity of demand is negative.")]),
        Step("Compute the deadweight loss.", [T("DWL", r["DWL"])], hint="1/2 (Pm - MC)(Qc - Qm).", solution=f"{r['DWL']:g}",
             misconceptions=[(2 * r["DWL"], "Missing the 1/2.")]),
    ], "A monopolist always operates on the elastic part of demand (|e| > 1); the Lerner index links its markup directly to that elasticity."


@topic("cournot", "Asymmetric Cournot duopoly", "Advanced")
def g_cournot(rng):
    a = rng.randrange(120, 300, 30)
    c1 = rng.randrange(10, 40, 5)
    c2 = c1 + rng.choice([0, 5, 10, 15, 30])
    r, _ = econ.cournot(a, 1, c1, c2)
    return f"P = {a} - (Q1 + Q2); MC1 = {c1}, MC2 = {c2}.", [
        Step("Firm 1's reaction function Q1 = R1(Q2): intercept and slope.", [T("intercept", (a - c1) / 2), T("slope (abs)", 0.5)],
             hint="MR1 = a - 2Q1 - Q2 = MC1.", solution=r["R1"]),
        Step("Solve for the Cournot outputs.", [T("Q1", r["Q1"], tol=0.002), T("Q2", r["Q2"], tol=0.002)],
             hint="Substitute R2 into R1 (or use Qi = (a - 2ci + cj)/3).", solution=f"Q1 = {r['Q1']:.4g}, Q2 = {r['Q2']:.4g}",
             misconceptions=[((a - c1) / 2, "That is Firm 1's best response to Q2 = 0, not the equilibrium.")]),
        Step("Market price and each firm's profit.", [T("P", r["P"], tol=0.002), T("profit1", r["profit1"], tol=0.002), T("profit2", r["profit2"], tol=0.002)],
             hint="P from inverse demand; Πi = (P - ci)Qi.", solution=f"P = {r['P']:.4g}; Π1 = {r['profit1']:.4g}; Π2 = {r['profit2']:.4g}"),
        Step("Compute the Herfindahl-Hirschman Index (shares in %, squared and summed).",
             [T("HHI", (100 * r["Q1"] / r["Q_total"]) ** 2 + (100 * r["Q2"] / r["Q_total"]) ** 2, tol=0.005)],
             hint="HHI = s1² + s2² with shares in percent.", solution=f"HHI = {(100 * r['Q1'] / r['Q_total']) ** 2 + (100 * r['Q2'] / r['Q_total']) ** 2:.0f}",
             misconceptions=[((r["Q1"] / r["Q_total"]) ** 2 + (r["Q2"] / r["Q_total"]) ** 2, "Use shares in percent (0-100), not fractions.")]),
    ], "The lower-cost firm produces more and earns more; cost asymmetry raises concentration (HHI) even with only two firms."


@topic("stackelberg", "Stackelberg leadership", "Advanced")
def g_stackelberg(rng):
    a, c = rng.randrange(100, 260, 20), rng.randrange(10, 40, 10)
    r, _ = econ.cournot(a, 1, c, c)
    ql, qf, P = r["stackelberg_Q_leader"], r["stackelberg_Q_follower"], r["stackelberg_P"]
    return f"P = {a} - (Q1 + Q2); both firms have MC = {c}. Firm 1 moves first.", [
        Step("Follower's reaction function Q2 = R2(Q1): intercept.", [T("intercept", (a - c) / 2)], hint="Same as Cournot reaction.",
             solution=f"Q2 = {(a - c) / 2:g} - 0.5Q1"),
        Step("Leader's output (it maximises profit given R2).", [T("Q leader", ql)],
             hint="Substitute R2 into P, then set the leader's MR = MC.", solution=f"Q1 = (a - c)/2 = {ql:g}",
             misconceptions=[(r["Q1"], "That is the Cournot output; the leader internalises the follower's reaction and produces more.")]),
        Step("Follower output and market price.", [T("Q follower", qf), T("P", P)], hint="Plug Q1 into R2.",
             solution=f"Q2 = {qf:g}; P = {P:g}"),
        Step("Leader profit and its gain over Cournot profit.", [T("leader profit", (P - c) * ql), T("gain vs Cournot", (P - c) * ql - r["profit1"])],
             hint="Cournot profit per firm = (P_cournot - c) × Q_cournot.",
             solution=f"{(P - c) * ql:g}; Cournot {r['profit1']:g}; gain {(P - c) * ql - r['profit1']:g}"),
    ], "Commitment is valuable: moving first lets the leader capture a larger share, and total output rises above Cournot (price falls)."


@topic("nash", "2x2 game: pure and mixed Nash equilibria", "Intermediate")
def g_nash(rng):
    while True:
        A = [[rng.randint(0, 6) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(0, 6) for _ in range(2)] for _ in range(2)]
        r, _ = econ.nash_2player(A, B, ["Up", "Down"], ["Left", "Right"])
        if "mixed" in r and 0 < r["mixed"]["p_row_first"] < 1 and 0 < r["mixed"]["q_col_first"] < 1:
            break
    p, q = r["mixed"]["p_row_first"], r["mixed"]["q_col_first"]
    ev_row = q * A[0][0] + (1 - q) * A[0][1]
    return (f"Payoffs (Row, Column): Up/Left ({A[0][0]},{B[0][0]}), Up/Right ({A[0][1]},{B[0][1]}), "
            f"Down/Left ({A[1][0]},{B[1][0]}), Down/Right ({A[1][1]},{B[1][1]})."), [
        Step("How many pure-strategy Nash equilibria are there?", [T("count", len(r["pure_nash"]))],
             hint="Underline each player's best responses; a cell where both are best responses is a NE.",
             solution=f"{len(r['pure_nash'])}: {r['pure_nash']}"),
        Step("In the mixed equilibrium, with what probability does Column play Left (q)?", [T("q", q, tol=0.01)],
             hint="Column mixes to make ROW indifferent between Up and Down.", solution=f"q = {q:.4f}",
             misconceptions=[(p, "That is Row's probability; q is chosen to make Row indifferent, using ROW's payoffs.")]),
        Step("With what probability does Row play Up (p)?", [T("p", p, tol=0.01)],
             hint="Row mixes to make COLUMN indifferent between Left and Right.", solution=f"p = {p:.4f}"),
        Step("Row's expected payoff in the mixed equilibrium.", [T("E[Row]", ev_row, tol=0.01)],
             hint="Row is indifferent, so use either row: q·A(Up,Left) + (1-q)·A(Up,Right).", solution=f"{ev_row:.4f}"),
    ], "In a mixed equilibrium each player randomises to make the OTHER indifferent, so a player's own payoffs determine the opponent's mix."


@topic("slutsky", "Cobb-Douglas demand and Slutsky decomposition", "Advanced")
def g_slutsky(rng):
    alpha = rng.choice([0.25, 0.4, 0.5, 0.6, 0.75])
    M = rng.randrange(100, 1000, 50)
    px, py = rng.choice([2, 4, 5, 10]), rng.choice([1, 2, 5])
    pxn = px * rng.choice([2, 0.5, 1.25])
    r, _ = econ.cobb_douglas_consumer(alpha, M, px, py, pxn)
    return f"U = x^{alpha} y^{round(1 - alpha, 2)}, income M = {M}, Px = {px}, Py = {py}. Px changes to {pxn:g}.", [
        Step("Optimal bundle (x*, y*).", [T("x*", r["x"], tol=0.002), T("y*", r["y"], tol=0.002)],
             hint="Cobb-Douglas: x* = αM/Px, y* = (1-α)M/Py.", solution=f"x* = {r['x']:.4g}, y* = {r['y']:.4g}"),
        Step("New demand for x after the price change.", [T("x new", r["x_new"], tol=0.002)], hint="Same formula with the new Px.",
             solution=f"{r['x_new']:.4g}"),
        Step("Slutsky substitution effect on x.", [T("substitution", r["substitution_effect"], tol=0.01)],
             hint="Compensate income so the OLD bundle is just affordable at the new price: M' = M + ΔPx·x*.",
             solution=f"x_c = {r['x_compensated']:.4g}; substitution = {r['substitution_effect']:.4g}",
             misconceptions=[(r["total_effect"], "That is the TOTAL effect; split it into substitution and income parts.")]),
        Step("Income effect on x.", [T("income effect", r["income_effect"], tol=0.01)], hint="Income effect = total - substitution.",
             solution=f"{r['income_effect']:.4g}"),
    ], "For a normal good both effects move demand the same way; Cobb-Douglas spending shares are constant, so own-price elasticity is exactly -1."


@topic("costmin", "Cost minimisation with Cobb-Douglas technology", "Intermediate")
def g_costmin(rng):
    a = rng.choice([0.25, 0.5, 0.75]); b = rng.choice([0.25, 0.5, 0.75])
    w, r_ = rng.choice([2, 4, 8, 10]), rng.choice([1, 2, 4, 5])
    Q = rng.randrange(20, 200, 20)
    r, _ = econ.cobb_douglas_cost(1, a, b, w, r_, Q)
    return f"Q = K^{a} L^{b}; wage w = {w}, rental rate r = {r_}; target output Q = {Q}.", [
        Step("Cost-minimising capital/labour ratio K/L.", [T("K/L", r["K_over_L"], tol=0.002)], hint="MRTS = (b/a)(K/L) = w/r.",
             solution=f"K/L = (a/b)(w/r) = {r['K_over_L']:.4g}",
             misconceptions=[((b / a) * (w / r_), "Inverted: K/L = (a/b)·(w/r).")]),
        Step("Optimal L and K.", [T("L", r["L"], tol=0.005), T("K", r["K"], tol=0.005)], hint="Substitute K = ratio·L into the production function.",
             solution=f"L = {r['L']:.4g}, K = {r['K']:.4g}"),
        Step("Total and average cost.", [T("cost", r["cost"], tol=0.005), T("AC", r["AC"], tol=0.005)], hint="Cost = wL + rK.",
             solution=f"Cost = {r['cost']:.4g}, AC = {r['AC']:.4g}"),
        Step("Sum of exponents (a + b) - which determines returns to scale.", [T("a+b", r["returns_to_scale"])],
             hint=">1 increasing, =1 constant, <1 decreasing.", solution=f"{r['returns_to_scale']:g}"),
    ], "Firms substitute toward the relatively cheaper input; with a + b < 1 average cost rises with output, limiting firm size."


# ----------------------------------------------------------------- trading maths
@topic("fxsize", "Forex sizing across pair types", "Foundational")
def g_fxsize(rng):
    sym, price = rng.choice([("EURUSD", 1.0850), ("GBPUSD", 1.2650), ("AUDUSD", 0.6550), ("USDJPY", 151.20)])
    ps = pip_size(sym)
    eq = rng.choice([5000, 10000, 25000, 50000, 100000])
    risk = rng.choice([0.5, 1, 1.5, 2])
    stop_p, rr = rng.choice([15, 20, 25, 30, 40, 50]), rng.choice([1.5, 2, 3])
    side = rng.choice([1, -1])
    stop, target = round(price - side * stop_p * ps, 5), round(price + side * stop_p * rr * ps, 5)
    wr = rng.choice([0.35, 0.4, 0.45, 0.5])
    r, _ = tr.fx_position(eq, risk, price, stop, sym, target, wr)
    pv = pip_value(sym, price)
    return (f"Equity ${eq:,}, risk {risk}% per trade. {sym} {'long' if side > 0 else 'short'} at {price}, stop {stop}, target {target}. "
            f"Win rate {wr:.0%}. 1 standard lot = 100,000 units; account in USD."), [
        Step("Dollar risk and stop distance in pips.", [T("dollar risk", r["dollar_risk"]), T("pips", r["stop_pips"])],
             hint=f"One pip for {sym} is {ps}.", solution=f"${r['dollar_risk']:,.2f}; {r['stop_pips']:g} pips",
             misconceptions=[(abs(price - stop) / 0.0001 if ps == 0.01 else abs(price - stop) / 0.01, "Wrong pip size for this pair (JPY pairs: 0.01).")]),
        Step("Pip value per standard lot in USD.", [T("pip value", pv, tol=0.002)],
             hint="Pip value in quote currency = 100,000 × pip size; convert to USD (divide by price if USD is the BASE).",
             solution=f"${pv:.4f}",
             misconceptions=[(10.0, "$10 per pip only holds when USD is the QUOTE currency; for USDJPY divide 1,000 JPY by the price.")] if sym == "USDJPY" else []),
        Step("Position size in standard lots.", [T("lots", r["lots"], tol=0.005)], hint="Lots = dollar risk / (pips × pip value per lot).",
             solution=f"{r['lots']:.4f} lots",
             misconceptions=[(r["dollar_risk"] / (r["stop_pips"] * 10), "You assumed $10/pip; use this pair's pip value.")] if sym == "USDJPY" else []),
        Step("Expectancy per trade in dollars.", [T("expectancy", r["expectancy"], tol=0.005)],
             hint="Win = target pips × pip value × lots; loss = dollar risk.", solution=f"${r['expectancy']:,.2f}"),
    ], "Pip value depends on which side of the pair the account currency sits; sizing by dollar risk keeps every trade's loss equal regardless of pair."


@topic("liquidation", "Leverage and liquidation (long or short)", "Advanced")
def g_liq(rng):
    side = rng.choice(["long", "short"])
    entry = rng.choice([2000, 3000, 25000, 60000, 90000, 150])
    lev = rng.choice([3, 5, 10, 20, 25])
    mmr = rng.choice([0.004, 0.005, 0.01])
    notional = entry * rng.choice([1, 2, 5, 10])
    margin0 = notional / lev
    add = margin0 * rng.choice([0.5, 1.0])
    a, _ = tr.liquidation(entry, lev, mmr, side)
    b, _ = tr.liquidation(entry, None, mmr, side, margin=margin0 + add, notional=notional)
    return (f"Isolated-margin {side} at ${entry:,} with {lev}x leverage (notional ${notional:,}, margin ${margin0:,.2f}), MMR {mmr:.1%}. "
            f"Use P_liq ≈ P0(1 ∓ 1/L ± MMR)."), [
        Step("Approximate liquidation price.", [T("P_liq", a["liq_approx"], tol=0.001)],
             hint="Long: P0(1 - 1/L + MMR); short: P0(1 + 1/L - MMR).", solution=f"${a['liq_approx']:,.2f}",
             misconceptions=[(entry * (1 + 1 / lev - mmr) if side == "long" else entry * (1 - 1 / lev + mmr), "Sign error: a long is liquidated BELOW entry, a short ABOVE.")]),
        Step("Adverse move (%) that triggers liquidation.", [T("move %", a["adverse_move_pct"], "percent")],
             hint="|Entry - P_liq| / Entry.", solution=f"{a['adverse_move_pct']:.2f}%",
             misconceptions=[(100 / lev, "1/L ignores the maintenance margin.")]),
        Step(f"The trader adds ${add:,.2f} of margin. New approximate liquidation price?", [T("new P_liq", b["liq_approx"], tol=0.001)],
             hint="New IMR = total margin / notional.", solution=f"IMR {b['imr']:.2%} -> ${b['liq_approx']:,.2f}"),
        Step("Exact liquidation price for the ORIGINAL position (linear contract).", [T("exact P_liq", a["liq_exact"], tol=0.0005)],
             hint="Long: P0(1 - IMR)/(1 - MMR); short: P0(1 + IMR)/(1 + MMR).", solution=f"${a['liq_exact']:,.2f}"),
    ], "Liquidation distance ≈ 1/leverage - MMR; adding margin lowers effective leverage, and the approximation is close to the exact formula only when MMR is small."


@topic("kelly", "Expectancy, break-even and Kelly sizing", "Intermediate")
def g_kelly(rng):
    p = rng.choice([0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    R = rng.choice([1, 1.5, 2, 2.5, 3])
    if p * R - (1 - p) <= 0:
        p = round(1 / (1 + R) + 0.1, 2)
    e = p * R - (1 - p)
    k, _ = tr.kelly(p, R, 0.5)
    return f"A system wins {p:.0%} of trades with an average win of {R}R and an average loss of 1R.", [
        Step("Expectancy in R per trade.", [T("E (R)", e)], hint="E = p·R - (1-p)·1.", solution=f"{e:.4f}R",
             misconceptions=[(p * R, "Subtract the expected loss (1-p)·1R.")]),
        Step("Break-even win rate (%).", [T("break-even %", 100 / (1 + R), "percent")], hint="p* = 1/(1 + R).", solution=f"{100 / (1 + R):.2f}%"),
        Step("Full Kelly fraction f*.", [T("f*", k["kelly"], tol=0.005)], hint="f* = (bp - q)/b with b = R.", solution=f"{k['kelly']:.4f}",
             misconceptions=[(e, "Kelly is (bp - q)/b: divide the R-expectancy by b.")]),
        Step("Half-Kelly fraction (as % of equity).", [T("half Kelly %", k["fractional"] * 100, "percent", tol=0.005)],
             hint="0.5 × f*.", solution=f"{k['fractional']:.2%}"),
    ], "Kelly maximises long-run growth only if p and R are known exactly; estimation error makes fractional Kelly (or smaller fixed risk) the practical choice."


@topic("cip", "Covered interest parity and carry", "Intermediate")
def g_cip(rng):
    spot = rng.choice([1.0850, 1.2650, 0.6550, 1.3500])
    rq, rb = rng.choice([0.0525, 0.045, 0.04, 0.035]), rng.choice([0.04, 0.0375, 0.0425, 0.01, 0.005])
    days = rng.choice([30, 90, 180, 360])
    r, _ = tr.covered_interest_parity(spot, rq, rb, days)
    notional = 1_000_000
    carry = notional * (rb - rq) * days / 360  # long base, short quote (in quote currency, approx)
    return (f"Spot = {spot} (quote per base). Quote-currency rate {rq:.2%}, base-currency rate {rb:.2%} (simple, ACT/360). "
            f"Horizon {days} days."), [
        Step("CIP forward rate.", [T("F", r["forward"], tol=0.00002)], hint="F = S(1 + r_q·d/360)/(1 + r_b·d/360).",
             solution=f"{r['forward']:.5f}",
             misconceptions=[(spot * (1 + rb * days / 360) / (1 + rq * days / 360), "Rates inverted: the QUOTE rate goes in the numerator.")]),
        Step("Forward points (pips).", [T("points", r["forward_points"], tol=0.01)], hint="(F - S) × 10,000.", solution=f"{r['forward_points']:.1f}"),
        Step(f"Interest-differential carry on a 1,000,000 base-currency LONG held {days} days if spot is unchanged (quote currency, approx).",
             [T("carry", carry * spot, tol=0.01)], hint="≈ notional × spot × (r_base - r_quote) × d/360.", solution=f"{carry * spot:,.0f}"),
        Step("Spot move (%) that wipes out the carry (break-even).", [T("break-even %", abs((r["forward"] - spot) / spot) * 100, "percent", tol=0.01)],
             hint="Under CIP the forward premium/discount equals the carry: |F - S|/S.", solution=f"{abs((r['forward'] - spot) / spot) * 100:.3f}%"),
    ], "Covered parity is an arbitrage relation; the carry trade bets that UNcovered parity fails - that spot will not move to offset the rate differential."


@topic("funding", "Perpetual funding carry", "Intermediate")
def g_funding(rng):
    spot = rng.choice([30000, 45000, 60000, 90000])
    fr = rng.choice([0.0001, 0.0002, 0.0003, 0.0005])
    days = rng.choice([7, 14, 30, 60])
    periods = days * 3
    taker = rng.choice([0.0004, 0.0005, 0.001])
    lev = rng.choice([2, 3, 5])
    r, _ = tr.funding_carry(spot, 1, fr, periods, taker, spot / lev, days)
    return (f"Long 1 BTC spot at ${spot:,} and short 1 BTC perpetual with {lev}x (margin ${spot / lev:,.0f}). "
            f"Funding {fr:.3%} per 8 h received by shorts for {days} days ({periods} payments). Taker {taker:.2%} on each of 4 legs."), [
        Step("Gross funding received.", [T("gross", r["gross_funding"])], hint="notional × rate × periods.", solution=f"${r['gross_funding']:,.2f}",
             misconceptions=[(spot * fr, "That is a single payment."), (spot * fr * days, "Funding is paid every 8 hours: 3 times a day.")]),
        Step("Total fees.", [T("fees", r["fees"])], hint="4 × notional × taker.", solution=f"${r['fees']:,.2f}"),
        Step("Net profit and return on total capital (%).", [T("net", r["net"]), T("return %", r["roc"] * 100, "percent", tol=0.003)],
             hint="Capital = spot cost + perp margin.", solution=f"${r['net']:,.2f}; {r['roc']:.3%}",
             misconceptions=[(r["net"] / spot * 100, "Include the perp margin in capital.")]),
        Step("Simple annualised return (%).", [T("annualised %", r["annualised_simple"] * 100, "percent", tol=0.005)],
             hint="× 365/days.", solution=f"{r['annualised_simple']:.2%}"),
    ], "Funding carry earns the premium leveraged longs pay; the main risks are funding turning negative and the short leg being liquidated in a rally."


# ----------------------------------------------------------------- public API
def generate(key: str, seed: int | None = None) -> Problem:
    if key not in TOPICS:
        raise KeyError(f"unknown topic {key}; choose from {sorted(TOPICS)}")
    seed = random.randrange(10 ** 6) if seed is None else seed
    title, diff, fn = TOPICS[key]
    given, steps, takeaway = fn(random.Random(f"{key}-{seed}"))
    return Problem(1000 + seed % 1000, title, diff, given, steps, takeaway, topic_key=key, seed=seed)
