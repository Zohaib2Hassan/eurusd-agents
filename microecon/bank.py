"""Embedded practice problem bank. Answer keys are COMPUTED by the solvers at import time - never typed in.
Where the original key in the agent specification was wrong, the step carries a `correction` note.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import econ
from . import trading as tr


@dataclass
class Target:
    label: str
    value: float
    kind: str = "number"      # number | percent (accepts 9.5 or 0.095) | lots
    tol: float = 0.005        # relative tolerance for full credit


@dataclass
class Step:
    prompt: str
    targets: list = field(default_factory=list)      # numeric answers
    concepts: list = field(default_factory=list)     # text answers: list of synonym-groups
    min_concepts: int = 0
    hint: str = ""
    solution: str = ""
    correction: str = ""


@dataclass
class Problem:
    pid: int
    topic: str
    difficulty: str
    given: str
    steps: list
    takeaway: str


def _p1():
    r, _ = econ.market(500, 10, -100, 15, floor=30)
    return Problem(1, "Supply, demand & price floor", "Foundational",
                   "Qd = 500 - 10P,  Qs = -100 + 15P", [
        Step("Find the market equilibrium price P* and quantity Q*.",
             [Target("P*", r["P_star"]), Target("Q*", r["Q_star"])],
             hint="Set Qd = Qs and solve the linear equation for P, then substitute back.",
             solution=f"500 - 10P = -100 + 15P => 25P = 600 => P* = {r['P_star']:g}; Q* = {r['Q_star']:g}"),
        Step("Calculate consumer surplus (CS) and producer surplus (PS) at equilibrium.",
             [Target("CS", r["CS"]), Target("PS", r["PS"], tol=0.002)],
             hint="Find the demand choke price (Qd = 0) and the supply intercept (Qs = 0); each surplus is a triangle with base Q*.",
             solution=f"Choke 50, supply intercept 6.667; CS = 1/2(50-24)(260) = {r['CS']:,.0f}; PS = 1/2(24-6.667)(260) = {r['PS']:,.2f}",
             correction="Original key gave PS = $2,252.90 by rounding the intercept to 6.67; exact value is $2,253.33 (both accepted)."),
        Step("A binding price floor Pf = $30 is imposed. Find the quantity traded and the excess supply.",
             [Target("Q traded", r["Q_traded"]), Target("excess supply", r["excess_supply"])],
             hint="Evaluate Qd and Qs at P = 30. With a floor, buyers decide how much is actually traded.",
             solution=f"Qd(30) = 200, Qs(30) = 350 -> traded {r['Q_traded']:g}, surplus {r['excess_supply']:g}"),
        Step("Calculate the deadweight loss (DWL) caused by the floor.",
             [Target("DWL", r["DWL"])],
             hint="DWL is the triangle between demand and supply from the traded quantity up to Q*. You need the supply price at Q = 200.",
             solution=f"Supply price at Q=200 is 20; DWL = 1/2(30-20)(260-200) = {r['DWL']:g}"),
    ], "Price floors transfer surplus to sellers who still sell but destroy the gains from the trades that no longer happen; "
       "the unsold surplus is a symptom of the price no longer clearing the market.")


def _p2():
    r, _ = econ.monopoly(120, 2, 20, 100)
    return Problem(2, "Monopoly vs perfect competition", "Intermediate", "P = 120 - 2Q,  TC = 20Q + 100 (MC = 20)", [
        Step("Write total revenue TR(Q) and marginal revenue MR(Q). Give the MR intercept and slope.",
             [Target("MR intercept", 120), Target("MR slope (absolute value)", 4)],
             hint="TR = P(Q)·Q; differentiate with respect to Q.",
             solution="TR = 120Q - 2Q²; MR = 120 - 4Q"),
        Step("Find the profit-maximising quantity, price and profit.",
             [Target("Qm", r["Qm"]), Target("Pm", r["Pm"]), Target("profit", r["profit"])],
             hint="Set MR = MC, then read the price off the DEMAND curve; remember the fixed cost in profit.",
             solution=f"120 - 4Q = 20 => Q = {r['Qm']:g}; P = {r['Pm']:g}; profit = 1,750 - 600 = {r['profit']:,.0f}"),
        Step("Find price and quantity under perfect competition.",
             [Target("Pc", r["Pc"]), Target("Qc", r["Qc"])],
             hint="Competitive firms produce where P = MC.",
             solution=f"120 - 2Q = 20 => Qc = {r['Qc']:g}, Pc = {r['Pc']:g}"),
        Step("Calculate the deadweight loss of monopoly.",
             [Target("DWL", r["DWL"])],
             hint="Triangle between demand and MC from Qm to Qc.",
             solution=f"DWL = 1/2(70-20)(50-25) = {r['DWL']:g}"),
    ], "A monopolist restricts output to where MR = MC, so price exceeds marginal cost; the Lerner index (P-MC)/P = "
       f"{r['lerner']:.2f} measures that markup and the DWL is the value of trades society loses.")


def _p3():
    r, _ = econ.cournot(180, 1, 30, 30)
    return Problem(3, "Cournot duopoly", "Advanced", "P = 180 - (Q1 + Q2),  MC1 = MC2 = 30", [
        Step("Derive Firm 1's reaction function Q1 = R1(Q2). Give its intercept and slope.",
             [Target("intercept", 75), Target("slope (absolute)", 0.5)],
             hint="Write Firm 1's profit (P - 30)Q1, differentiate with respect to Q1 holding Q2 fixed.",
             solution="MR1 = 180 - 2Q1 - Q2 = 30 => Q1 = 75 - 0.5Q2"),
        Step("Solve for the Cournot-Nash outputs Q1*, Q2* and total output.",
             [Target("Q1*", r["Q1"]), Target("Q2*", r["Q2"]), Target("Q total", r["Q_total"])],
             hint="By symmetry Q1 = Q2; substitute into the reaction function.",
             solution=f"Q1* = Q2* = {r['Q1']:g}, Q = {r['Q_total']:g}"),
        Step("Find the market price and each firm's profit.",
             [Target("P*", r["P"]), Target("profit per firm", r["profit1"])],
             hint="Use the inverse demand at total Q; profit = (P - MC)·Qi.",
             solution=f"P = 180 - 100 = {r['P']:g}; Πi = (80-30)×50 = {r['profit1']:,.0f}"),
        Step("Find cartel (joint-monopoly) output and price, and compare with Cournot.",
             [Target("Q cartel", r["Q_cartel"]), Target("P cartel", r["P_cartel"])],
             hint="A cartel acts as one monopolist: total MR = 180 - 2Q = MC.",
             solution=f"Q = {r['Q_cartel']:g}, P = {r['P_cartel']:g}; Cournot output is higher and price lower"),
    ], "Each Cournot firm ignores the price cut its output imposes on its rival, so output exceeds the cartel level; "
       "the cartel is unstable because each member gains by producing more - a prisoner's dilemma.")


def _p4():
    r, _ = tr.fx_position(50000, 1.5, 1.0850, 1.0810, "EURUSD", 1.0970, 0.45)
    return Problem(4, "Forex position sizing & expectancy", "Foundational",
                   "Equity $50,000; risk 1.5%; EUR/USD (pip $10/lot); entry 1.0850; stop 1.0810; target 1.0970; win rate 45%", [
        Step("Calculate the maximum dollar risk.", [Target("dollar risk", r["dollar_risk"])],
             hint="Risk % of equity.", solution="50,000 × 0.015 = $750"),
        Step("Calculate the stop distance in pips.", [Target("pips", r["stop_pips"])],
             hint="For EUR/USD one pip is 0.0001.", solution="(1.0850 - 1.0810)/0.0001 = 40 pips"),
        Step("Calculate the position size in standard lots.", [Target("lots", r["lots"], tol=0.002)],
             hint="Lots = dollar risk / (stop pips × pip value per lot).", solution="750 / (40 × 10) = 1.875 lots"),
        Step("Find the reward:risk ratio and the expectancy per trade at a 45% win rate.",
             [Target("reward:risk (R)", r["reward_risk"]), Target("expectancy $", r["expectancy"])],
             hint="Target is 120 pips. Profit at target = target pips × pip value × LOTS - use your position size from Step 3.",
             solution=f"R = 3; profit at target = 120 × 10 × 1.875 = $2,250; E = 0.45×2,250 - 0.55×750 = ${r['expectancy']:,.2f}",
             correction="Original key used a $3,000 win and E = +$937.50. With 1.875 lots a 120-pip win is $2,250 (= 3 × $750), "
                        "so the correct expectancy is +$600.00."),
    ], "Position size comes from the stop distance and the risk budget, not from conviction; with a 1:3 payoff, "
       "any win rate above 25% gives positive expectancy before costs.")


def _p5():
    r, _ = tr.dca(1000, [40000, 32000, 50000])
    a, _ = tr.spatial_arbitrage(40000, 40800, 1, 0.0005, 0, 0.002, 10)
    return Problem(5, "Crypto DCA & spatial arbitrage", "Intermediate",
                   "$1,000/week into BTC at $40,000, $32,000, $50,000. Arbitrage: Ex A $40,000, Ex B $40,800; "
                   "withdrawal fee 0.0005 BTC; Ex B taker 0.2%; network fee $10", [
        Step("Total BTC acquired and the weighted average cost basis.",
             [Target("BTC", r["total_units"], tol=0.001), Target("cost basis", r["cost_basis"], tol=0.001)],
             hint="Units each week = 1,000 / price; basis = total spent / total units.",
             solution=f"0.025 + 0.03125 + 0.02 = {r['total_units']} BTC; basis = 3,000/0.07625 = ${r['cost_basis']:,.2f}"),
        Step("Compare the DCA cost basis with the simple arithmetic mean price.",
             [Target("arithmetic mean", r["arithmetic_mean"], tol=0.001)],
             hint="Average the three prices; then explain which is lower and why.",
             solution=f"Mean = ${r['arithmetic_mean']:,.2f} > basis ${r['cost_basis']:,.2f}: more units bought at low prices (harmonic ≤ arithmetic mean)"),
        Step("Net arbitrage profit on 1 BTC (buy on A, sell on B).",
             [Target("net profit", a["profit"], tol=0.002)],
             hint="Subtract the withdrawal fee in BTC first, then apply the 0.2% taker fee to the proceeds and the $10 fixed fee.",
             solution=f"0.9995 × 40,800 = 40,779.60; - 81.56 taker - 10 network = 40,688.04; profit = ${a['profit']:,.2f}",
             correction="Assumes no taker fee on the buy at Exchange A (as in the original key)."),
        Step("Name the microstructure limits to continuous arbitrage.",
             concepts=[["settlement", "transfer time", "delay", "confirmation", "latency"],
                       ["counterparty", "exchange risk", "custody", "insolvency"],
                       ["depth", "slippage", "order book", "liquidity", "market impact"],
                       ["fee", "fees", "withdrawal", "gas", "network fee"],
                       ["price risk", "price move", "volatility", "converge", "competition"]], min_concepts=3,
             hint="Think about time, who holds your coins, how much size the book can absorb, and costs.",
             solution="Settlement/transfer delays (price risk while in transit), counterparty/custody risk, order-book depth and slippage, "
                      "changing fees, and competition closing the gap."),
    ], "DCA lowers the average cost relative to the average price by construction, not by prediction. Quoted price gaps are "
       "not free money: fees, transfer time and depth determine what is realisable.")


def _p6():
    f, _ = tr.fx_position(10000, 1, 1.1000, 1.0975, "EURUSD", 1.1075)
    c, _ = tr.crypto_units(10000, 1, 90000, 88200, 95400)
    e, _ = tr.expectancy(0.35, 300, 100, 100)
    return Problem(6, "Forex vs crypto sizing & expectancy", "Foundational",
                   "Equity $10,000, risk 1% ($100). EUR/USD 1.1000, stop 1.0975, target 1.1075. BTC $90,000, stop $88,200, target $95,400.", [
        Step("Forex position size in standard lots.", [Target("lots", f["lots"], tol=0.002)],
             hint="25-pip stop, $10 per pip per lot.", solution="100 / (25 × 10) = 0.40 lots (40,000 units)"),
        Step("Crypto position size in BTC and its USD notional.",
             [Target("BTC", c["units"], tol=0.01), Target("notional", c["notional"], tol=0.01)],
             hint="Units = dollar risk / stop distance in dollars.", solution="100 / 1,800 = 0.0556 BTC; notional $5,000 (0.5x equity)"),
        Step("Dollar profit at the 1:3 target for both trades.",
             [Target("forex profit", f["profit_at_target"], tol=0.01), Target("crypto profit", c["profit_at_target"], tol=0.01)],
             hint="Target distance × position size.", solution="75 pips × $4/pip = $300; 0.0556 × 5,400 ≈ $300"),
        Step("Expectancy per trade at a 35% win rate, and expected total over 100 trades.",
             [Target("expectancy", e["expectancy"]), Target("total over 100", e["expected_total"])],
             hint="E = p·W - (1-p)·L with W = 300, L = 100.",
             solution=f"0.35×300 - 0.65×100 = $40; × 100 = $4,000 (1 s.d. ≈ ${e['sd_total']:,.0f}, so outcomes vary widely)"),
    ], "Risk-based sizing equalises dollar risk across instruments with very different volatility, so the same payoff "
       "structure produces the same expectancy. Expected totals hide large dispersion: one standard deviation over 100 trades is about $1,900.")


def _p7():
    r, _ = tr.funding_carry(60000, 1, 0.0004, 90, 0.0005, 12000, 30)
    return Problem(7, "Perpetual funding carry", "Intermediate",
                   "Long 1 BTC spot $60,000 + short 1 BTC perp ($12,000 margin, 5x). Funding +0.04% per 8h × 90 periods; taker 0.05% per leg (4 legs).", [
        Step("Gross funding earned over 30 days.", [Target("gross funding", r["gross_funding"])],
             hint="Total rate = periods × rate per period, applied to the notional.", solution="90 × 0.04% = 3.6%; 60,000 × 0.036 = $2,160"),
        Step("Total taker fees for entering and unwinding both legs.", [Target("fees", r["fees"])],
             hint="Four executions, each on $60,000 notional.", solution="4 × 60,000 × 0.0005 = $120"),
        Step("Net profit and 30-day return on total capital ($72,000).",
             [Target("net", r["net"]), Target("return %", r["roc"] * 100, kind="percent", tol=0.003)],
             hint="Net = gross - fees; capital = spot + margin.", solution=f"$2,040; 2,040/72,000 = {r['roc']:.3%} (~34% simple annualised)"),
        Step("Main risks to this delta-neutral strategy.",
             concepts=[["negative funding", "funding flip", "funding turns", "funding rate change", "funding risk"],
                       ["liquidation", "margin call", "rally", "short squeeze"],
                       ["slippage", "execution", "basis"],
                       ["exchange", "counterparty", "custody", "smart contract", "platform"]], min_concepts=2,
             hint="What if funding changes sign? What happens to a 5x short if BTC rallies 20%?",
             solution="Funding turning negative, liquidation of the short perp in a sharp rally, execution/basis slippage, exchange/custody risk."),
    ], "Funding carry is a yield for providing leverage to longs; it is market-neutral in price but not risk-free - "
       "liquidation and counterparty risks are the hidden short options in the trade.")


def _p8():
    a, _ = tr.liquidation(3000, 10, 0.005)
    b, _ = tr.liquidation(3000, None, 0.005, margin=4500, notional=30000)
    c, _ = tr.liquidation(3000, 50, 0.005)
    return Problem(8, "Isolated-margin liquidation", "Advanced",
                   "Long 10 ETH at $3,000 (notional $30,000), 10x (margin $3,000), MMR 0.5%. Use P_liq ≈ P0(1 - 1/L + MMR).", [
        Step("Approximate liquidation price.", [Target("P_liq", a["liq_approx"], tol=0.001)],
             hint="1/L = 0.10 for 10x.", solution=f"3,000 × (1 - 0.10 + 0.005) = ${a['liq_approx']:,.0f} (exact formula gives ${a['liq_exact']:,.2f})"),
        Step("Percentage drop from entry that triggers liquidation.", [Target("drop %", a["adverse_move_pct"], kind="percent")],
             hint="(Entry - P_liq)/Entry.", solution="285 / 3,000 = 9.5%"),
        Step("The trader adds $1,500 margin (total $4,500). New liquidation price?", [Target("new P_liq", b["liq_approx"], tol=0.001)],
             hint="New initial margin fraction = 4,500 / 30,000.", solution=f"IMR 15% -> 3,000 × 0.855 = ${b['liq_approx']:,.0f}"),
        Step("Liquidation price and trigger move for a 50x long at $3,000.",
             [Target("P_liq 50x", c["liq_approx"], tol=0.001), Target("drop %", c["adverse_move_pct"], kind="percent")],
             hint="1/50 = 0.02.", solution=f"3,000 × 0.985 = ${c['liq_approx']:,.0f}; a 1.5% move liquidates"),
    ], "Liquidation distance is roughly 1/leverage minus the maintenance margin; at 50x a 1.5% move - smaller than "
       "ETH's typical daily range - wipes the position, so leverage choice dominates outcome risk.")


BANK = {p.pid: p for p in (_p1(), _p2(), _p3(), _p4(), _p5(), _p6(), _p7(), _p8())}
