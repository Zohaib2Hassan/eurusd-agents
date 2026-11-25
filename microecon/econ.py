"""Microeconomics solvers (deterministic). Linear demand/supply conventions:
   Qd = a - b*P,  Qs = c + d*P   (b, d > 0)     inverse demand P = A - B*Q
All areas are triangles under linear curves; every function returns numbers AND a worked Solution.
"""
from __future__ import annotations

import itertools

import numpy as np

from .solution import Solution, fmt


# ----------------------------------------------------------------- competitive market
def market(a: float, b: float, c: float, d: float, floor: float | None = None, ceiling: float | None = None,
           tax: float | None = None) -> tuple[dict, Solution]:
    s = Solution("Competitive market: equilibrium, surplus, interventions",
                 given={"Demand": f"Qd = {fmt(a)} - {fmt(b)}P", "Supply": f"Qs = {fmt(c)} + {fmt(d)}P"})
    s.formulas = ["Equilibrium: Qd = Qs", "CS = 1/2 (P_choke - P*) Q*,  PS = 1/2 (P* - P_min) Q*",
                  "DWL = 1/2 x (price wedge) x (lost quantity)"]
    P = (a - c) / (b + d)
    Q = a - b * P
    p_choke, p_min = a / b, -c / d
    cs, ps = 0.5 * (p_choke - P) * Q, 0.5 * (P - p_min) * Q
    r = {"P_star": P, "Q_star": Q, "P_choke": p_choke, "P_min_supply": p_min, "CS": cs, "PS": ps, "TS": cs + ps}
    s.step(f"{fmt(a)} - {fmt(b)}P = {fmt(c)} + {fmt(d)}P  =>  {fmt(b + d)}P = {fmt(a - c)}  =>  P* = {fmt(P, 4)}")
    s.step(f"Q* = {fmt(a)} - {fmt(b)}({fmt(P, 4)}) = {fmt(Q, 4)}")
    s.step(f"Choke price (Qd=0): P = {fmt(a)}/{fmt(b)} = {fmt(p_choke, 4)};  supply intercept (Qs=0): P = {fmt(-c)}/{fmt(d)} = {fmt(p_min, 4)}")
    s.step(f"CS = 1/2({fmt(p_choke, 4)} - {fmt(P, 4)})({fmt(Q, 4)}) = {fmt(cs)};  PS = 1/2({fmt(P, 4)} - {fmt(p_min, 4)})({fmt(Q, 4)}) = {fmt(ps)}")
    e_d = -b * P / Q
    r["elasticity_demand_at_eq"] = e_d
    s.step(f"Point price elasticity of demand at equilibrium: e = -b·P/Q = -{fmt(b)}·{fmt(P, 4)}/{fmt(Q, 4)} = {fmt(e_d, 4)}")
    s.interpretation.append(f"Market clears at P* = {fmt(P)} with Q* = {fmt(Q)}; total surplus {fmt(cs + ps)}. "
                            f"Demand is {'elastic' if abs(e_d) > 1 else 'inelastic' if abs(e_d) < 1 else 'unit-elastic'} at equilibrium.")
    if floor is not None:
        s.given["Price floor"] = fmt(floor)
        if floor <= P:
            s.step(f"Floor {fmt(floor)} <= P* {fmt(P)}: not binding, no effect.")
            r.update(floor_binding=False)
        else:
            qd, qs = a - b * floor, c + d * floor
            p_supply = (qd - c) / d
            dwl = 0.5 * (floor - p_supply) * (Q - qd)
            r.update(floor_binding=True, Q_traded=qd, Qs_at_floor=qs, excess_supply=qs - qd, supply_price_at_Qt=p_supply, DWL=dwl)
            s.step(f"Binding floor: Qd({fmt(floor)}) = {fmt(qd)}, Qs({fmt(floor)}) = {fmt(qs)} -> traded Q = min = {fmt(qd)}, surplus = {fmt(qs - qd)}")
            s.step(f"Supply price at Q = {fmt(qd)}: P = (Q - c)/d = {fmt(p_supply, 4)};  DWL = 1/2({fmt(floor)} - {fmt(p_supply, 4)})({fmt(Q, 4)} - {fmt(qd)}) = {fmt(dwl)}")
            s.interpretation.append(f"The floor cuts trade by {fmt(Q - qd)} units, leaves {fmt(qs - qd)} units unsold and destroys {fmt(dwl)} of surplus.")
    if ceiling is not None:
        s.given["Price ceiling"] = fmt(ceiling)
        if ceiling >= P:
            s.step(f"Ceiling {fmt(ceiling)} >= P*: not binding.")
            r.update(ceiling_binding=False)
        else:
            qd, qs = a - b * ceiling, c + d * ceiling
            p_demand = (a - qs) / b
            dwl = 0.5 * (p_demand - ceiling) * (Q - qs)
            r.update(ceiling_binding=True, Q_traded=qs, Qd_at_ceiling=qd, shortage=qd - qs, demand_price_at_Qt=p_demand, DWL=dwl)
            s.step(f"Binding ceiling: Qs = {fmt(qs)} traded, Qd = {fmt(qd)}, shortage = {fmt(qd - qs)}; DWL = 1/2({fmt(p_demand, 4)} - {fmt(ceiling)})({fmt(Q - qs, 4)}) = {fmt(dwl)}")
    if tax is not None:
        s.given["Per-unit tax on sellers"] = fmt(tax)
        Pb = (a - c + d * tax) / (b + d)
        Ps = Pb - tax
        Qt = a - b * Pb
        rev, dwl = tax * Qt, 0.5 * tax * (Q - Qt)
        r.update(tax_Pbuyer=Pb, tax_Pseller=Ps, tax_Q=Qt, tax_revenue=rev, tax_DWL=dwl,
                 buyer_share=(Pb - P) / tax, seller_share=(P - Ps) / tax)
        s.step(f"Tax t={fmt(tax)}: Qd(Pb) = Qs(Pb - t) => Pb = {fmt(Pb, 4)}, Ps = {fmt(Ps, 4)}, Q = {fmt(Qt, 4)}")
        s.step(f"Revenue = t·Q = {fmt(rev)}; DWL = 1/2·t·(Q* - Qt) = {fmt(dwl)}; buyers bear {fmt((Pb - P) / tax * 100, 1)}% (= d/(b+d))")
        s.interpretation.append("Tax incidence depends only on relative slopes, not on who legally pays.")
    return r, s


def elasticity(q1, q2, p1, p2) -> dict:
    """Arc (midpoint) elasticity."""
    e = ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2))
    return {"arc_elasticity": e, "class": "elastic" if abs(e) > 1 else "inelastic" if abs(e) < 1 else "unit"}


# ----------------------------------------------------------------- monopoly
def monopoly(A: float, B: float, mc: float, fc: float = 0.0) -> tuple[dict, Solution]:
    s = Solution("Single-price monopoly vs perfect competition",
                 given={"Inverse demand": f"P = {fmt(A)} - {fmt(B)}Q", "Cost": f"TC = {fmt(mc)}Q + {fmt(fc)}  (MC = {fmt(mc)})"})
    s.formulas = ["TR = P·Q, MR = dTR/dQ = A - 2BQ", "Monopoly: MR = MC;  competition: P = MC",
                  "Lerner index L = (P - MC)/P = -1/e_d at the optimum", "DWL = 1/2 (Pm - MC)(Qc - Qm)"]
    Qm = (A - mc) / (2 * B); Pm = A - B * Qm
    profit = Pm * Qm - (mc * Qm + fc)
    Qc = (A - mc) / B; Pc = mc
    dwl = 0.5 * (Pm - mc) * (Qc - Qm)
    lerner = (Pm - mc) / Pm
    s.step(f"TR = {fmt(A)}Q - {fmt(B)}Q²;  MR = {fmt(A)} - {fmt(2 * B)}Q")
    s.step(f"MR = MC: {fmt(A)} - {fmt(2 * B)}Q = {fmt(mc)} => Qm = {fmt(Qm, 4)}, Pm = {fmt(Pm, 4)}")
    s.step(f"Profit = {fmt(Pm, 4)}×{fmt(Qm, 4)} - ({fmt(mc)}×{fmt(Qm, 4)} + {fmt(fc)}) = {fmt(profit)}")
    s.step(f"Competition P = MC: {fmt(A)} - {fmt(B)}Q = {fmt(mc)} => Qc = {fmt(Qc, 4)}, Pc = {fmt(Pc)}")
    s.step(f"DWL = 1/2({fmt(Pm, 4)} - {fmt(mc)})({fmt(Qc, 4)} - {fmt(Qm, 4)}) = {fmt(dwl)};  Lerner = {fmt(lerner, 4)}")
    s.interpretation.append(f"Market power raises price by {fmt(Pm - Pc)} and halves output (linear demand), costing {fmt(dwl)} in lost surplus.")
    return {"TR": f"{A}Q - {B}Q^2", "MR": f"{A} - {2 * B}Q", "Qm": Qm, "Pm": Pm, "profit": profit, "Qc": Qc, "Pc": Pc,
            "DWL": dwl, "lerner": lerner}, s


# ----------------------------------------------------------------- oligopoly
def cournot(a: float, b: float, c1: float, c2: float) -> tuple[dict, Solution]:
    """Inverse demand P = a - b(Q1+Q2), constant marginal costs c1, c2."""
    s = Solution("Cournot duopoly vs cartel, Stackelberg and Bertrand",
                 given={"Inverse demand": f"P = {fmt(a)} - {fmt(b)}(Q1+Q2)", "MC1": fmt(c1), "MC2": fmt(c2)})
    s.formulas = ["Firm i: max (P - ci)Qi  =>  MRi = a - 2bQi - bQj = ci", "Reaction: Qi = (a - ci)/(2b) - Qj/2",
                  "Cartel: MR(Q) = a - 2bQ = c (if equal costs)", "Stackelberg leader internalises follower reaction",
                  "Bertrand (homogeneous good, equal MC): P = MC"]
    q1 = (a - 2 * c1 + c2) / (3 * b); q2 = (a - 2 * c2 + c1) / (3 * b)
    P = a - b * (q1 + q2)
    p1, p2 = (P - c1) * q1, (P - c2) * q2
    s.step(f"Firm 1: {fmt(a)} - {fmt(2 * b)}Q1 - {fmt(b)}Q2 = {fmt(c1)} => R1: Q1 = {fmt((a - c1) / (2 * b), 4)} - 0.5·Q2")
    s.step(f"Firm 2: R2: Q2 = {fmt((a - c2) / (2 * b), 4)} - 0.5·Q1")
    s.step(f"Solve jointly: Q1* = {fmt(q1, 4)}, Q2* = {fmt(q2, 4)}, Q = {fmt(q1 + q2, 4)}, P* = {fmt(P, 4)}")
    s.step(f"Profits: Π1 = ({fmt(P, 4)} - {fmt(c1)})·{fmt(q1, 4)} = {fmt(p1)},  Π2 = {fmt(p2)}")
    r = {"R1": f"Q1 = {(a - c1) / (2 * b)} - 0.5*Q2", "R2": f"Q2 = {(a - c2) / (2 * b)} - 0.5*Q1",
         "Q1": q1, "Q2": q2, "Q_total": q1 + q2, "P": P, "profit1": p1, "profit2": p2}
    if c1 == c2:
        c = c1
        Qk = (a - c) / (2 * b); Pk = a - b * Qk
        Qs1 = (a - c) / (2 * b); Qs2 = (a - c) / (4 * b); Ps = a - b * (Qs1 + Qs2)
        r.update(Q_cartel=Qk, P_cartel=Pk, profit_cartel_total=(Pk - c) * Qk,
                 stackelberg_Q_leader=Qs1, stackelberg_Q_follower=Qs2, stackelberg_P=Ps, bertrand_P=c, bertrand_Q=(a - c) / b)
        s.step(f"Cartel: {fmt(a)} - {fmt(2 * b)}Q = {fmt(c)} => Q = {fmt(Qk, 4)}, P = {fmt(Pk, 4)} (joint profit {fmt((Pk - c) * Qk)})")
        s.step(f"Stackelberg: leader {fmt(Qs1, 4)}, follower {fmt(Qs2, 4)}, P = {fmt(Ps, 4)};  Bertrand: P = MC = {fmt(c)}, Q = {fmt((a - c) / b, 4)}")
        s.interpretation.append(f"Ordering of output: cartel {fmt(Qk)} < Cournot {fmt(q1 + q2)} < Stackelberg {fmt(Qs1 + Qs2)} < Bertrand {fmt((a - c) / b)}; "
                                "prices move the opposite way. Competition in quantities leaves some market power; price competition with identical goods removes it.")
    return r, s


def nash_2player(payoff_row: list[list[float]], payoff_col: list[list[float]],
                 row_names=None, col_names=None) -> tuple[dict, Solution]:
    """Pure-strategy Nash equilibria of a bimatrix game, plus the mixed equilibrium for 2x2 games."""
    A, B = np.asarray(payoff_row, float), np.asarray(payoff_col, float)
    m, n = A.shape
    rn = row_names or [f"R{i + 1}" for i in range(m)]
    cn = col_names or [f"C{j + 1}" for j in range(n)]
    s = Solution("Nash equilibrium of a 2-player game", given={"Row payoffs": A.tolist(), "Column payoffs": B.tolist()})
    s.formulas = ["Pure NE: each strategy is a best response to the other",
                  "2x2 mixed NE: each player mixes to make the OTHER indifferent"]
    pure = [(rn[i], cn[j]) for i, j in itertools.product(range(m), range(n))
            if A[i, j] >= A[:, j].max() - 1e-12 and B[i, j] >= B[i, :].max() - 1e-12]
    for j in range(n):
        s.step(f"Row best response to {cn[j]}: {rn[int(A[:, j].argmax())]}")
    for i in range(m):
        s.step(f"Column best response to {rn[i]}: {cn[int(B[i, :].argmax())]}")
    s.step(f"Pure-strategy NE: {pure or 'none'}")
    r = {"pure_nash": pure}
    if (m, n) == (2, 2):
        den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
        den_p = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
        if den_q != 0 and den_p != 0:
            q = (A[1, 1] - A[0, 1]) / den_q   # prob column plays C1
            p = (B[1, 1] - B[1, 0]) / den_p   # prob row plays R1
            if 0 <= p <= 1 and 0 <= q <= 1:
                r["mixed"] = {"p_row_first": p, "q_col_first": q}
                s.step(f"Mixed NE: row plays {rn[0]} with p = {fmt(p, 4)}, column plays {cn[0]} with q = {fmt(q, 4)}")
    s.interpretation.append("A dominant-strategy equilibrium can still be Pareto-inferior (prisoner's dilemma) - "
                            "the same logic explains liquidity withdrawal by competing market makers in stress.")
    return r, s


# ----------------------------------------------------------------- consumer & producer
def cobb_douglas_consumer(alpha: float, income: float, px: float, py: float, px_new: float | None = None) -> tuple[dict, Solution]:
    """U = x^alpha y^(1-alpha). Marshallian demand, MRS, and Slutsky decomposition for a change in px."""
    s = Solution("Utility maximisation (Cobb-Douglas)", given={"U": f"x^{alpha} y^{round(1 - alpha, 4)}", "M": income, "Px": px, "Py": py})
    s.formulas = ["Tangency: MRS = MUx/MUy = (α/(1-α))(y/x) = Px/Py", "Budget: Px·x + Py·y = M",
                  "Demand: x* = αM/Px, y* = (1-α)M/Py", "Slutsky: total = substitution (compensated) + income effect"]
    x, y = alpha * income / px, (1 - alpha) * income / py
    U = x ** alpha * y ** (1 - alpha)
    s.step(f"x* = {alpha}×{fmt(income)}/{fmt(px)} = {fmt(x, 4)};  y* = {fmt(1 - alpha, 4)}×{fmt(income)}/{fmt(py)} = {fmt(y, 4)};  U = {fmt(U, 4)}")
    r = {"x": x, "y": y, "U": U, "MRS_at_opt": px / py}
    if px_new is not None:
        x_new = alpha * income / px_new
        M_comp = income + (px_new - px) * x            # Slutsky compensation: afford the old bundle
        x_comp = alpha * M_comp / px_new
        sub, inc = x_comp - x, x_new - x_comp
        r.update(x_new=x_new, x_compensated=x_comp, substitution_effect=sub, income_effect=inc, total_effect=x_new - x)
        s.step(f"New price {fmt(px_new)}: x_new = {fmt(x_new, 4)}")
        s.step(f"Slutsky compensated income M' = M + ΔPx·x = {fmt(M_comp, 4)} -> x_c = {fmt(x_comp, 4)}")
        s.step(f"Substitution effect = {fmt(sub, 4)}, income effect = {fmt(inc, 4)}, total = {fmt(x_new - x, 4)}")
    s.interpretation.append("With Cobb-Douglas preferences the consumer spends a fixed share α of income on x, so own-price elasticity is exactly -1.")
    return r, s


def cobb_douglas_cost(A: float, a: float, b: float, w: float, r_: float, Q: float) -> tuple[dict, Solution]:
    """Q = A K^a L^b; minimise wL + rK."""
    s = Solution("Cost minimisation (Cobb-Douglas production)", given={"Q": f"{A} K^{a} L^{b}", "w": w, "r": r_, "Target Q": Q})
    s.formulas = ["Tangency: MRTS = MPL/MPK = (b/a)(K/L) = w/r", "Returns to scale: a + b (>1 increasing, =1 constant, <1 decreasing)"]
    ratio = (a / b) * (w / r_)                     # K/L
    L = (Q / (A * ratio ** a)) ** (1 / (a + b)); K = ratio * L
    cost = w * L + r_ * K
    s.step(f"K/L = (a/b)(w/r) = {fmt(ratio, 4)}")
    s.step(f"Q = A(ratio·L)^a L^b => L* = {fmt(L, 4)}, K* = {fmt(K, 4)}, cost = {fmt(cost)}")
    rts = a + b
    s.step(f"a + b = {rts} -> {'increasing' if rts > 1 else 'constant' if rts == 1 else 'decreasing'} returns to scale")
    s.interpretation.append("Factor mix depends on relative factor prices; scale economies depend on a + b.")
    return {"K_over_L": ratio, "L": L, "K": K, "cost": cost, "AC": cost / Q, "returns_to_scale": rts}, s
