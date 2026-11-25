"""Trading mathematics: expectancy, sizing, FX/crypto units, DCA, arbitrage, funding carry,
liquidation, Kelly, risk of ruin, covered interest parity. Deterministic; each returns (results, Solution)."""
from __future__ import annotations

import math

import numpy as np

from ta.calc import pip_size, pip_value
from .solution import Solution, fmt


def expectancy(win_rate: float, avg_win: float, avg_loss: float, n_trades: int | None = None) -> tuple[dict, Solution]:
    s = Solution("Trade expectancy", given={"Win rate": f"{win_rate:.2%}", "Average win": fmt(avg_win), "Average loss": fmt(avg_loss)})
    s.formulas = ["E = p·W - (1-p)·L", "Break-even win rate p* = L/(W+L) = 1/(1+R)", "R = W/L"]
    e = win_rate * avg_win - (1 - win_rate) * avg_loss
    R = avg_win / avg_loss
    p_be = 1 / (1 + R)
    s.step(f"E = {win_rate}×{fmt(avg_win)} - {round(1 - win_rate, 6)}×{fmt(avg_loss)} = {fmt(win_rate * avg_win)} - {fmt((1 - win_rate) * avg_loss)} = {fmt(e)}")
    s.step(f"R = {fmt(R, 4)} -> break-even win rate = 1/(1+R) = {p_be:.2%}")
    r = {"expectancy": e, "R": R, "breakeven_win_rate": p_be, "expectancy_R": e / avg_loss}
    if n_trades:
        sd = math.sqrt(win_rate * avg_win ** 2 + (1 - win_rate) * avg_loss ** 2 - e ** 2)
        r.update(n_trades=n_trades, expected_total=e * n_trades, sd_total=sd * math.sqrt(n_trades))
        s.step(f"Over {n_trades} trades: expected {fmt(e * n_trades)} ± {fmt(sd * math.sqrt(n_trades))} (1 s.d., independent trades)")
        s.interpretation.append(f"Even with positive expectancy, the 1-s.d. range over {n_trades} trades is wide; "
                                f"P(total < 0) ≈ {_phi(-e * n_trades / (sd * math.sqrt(n_trades))):.1%} under a normal approximation.")
    s.interpretation.insert(0, f"{'Positive' if e > 0 else 'Negative'} expectancy: {fmt(e / avg_loss, 3)}R per trade. "
                               f"The system needs a win rate above {p_be:.1%} at this payoff ratio.")
    s.assumptions.append("Fixed win/loss sizes, independent trades, costs already inside W and L.")
    return r, s


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def fx_position(equity: float, risk_pct: float, entry: float, stop: float, symbol: str = "EURUSD",
                target: float | None = None, win_rate: float | None = None, quote_to_account: float | None = None,
                account_ccy: str = "USD") -> tuple[dict, Solution]:
    ps = pip_size(symbol)
    pv = pip_value(symbol, entry, 1.0, account_ccy=account_ccy, quote_to_account=quote_to_account)
    s = Solution(f"Forex position sizing ({symbol})",
                 given={"Equity": fmt(equity), "Risk": f"{risk_pct}%", "Entry": entry, "Stop": stop, "Target": target,
                        "Pip size": ps, "Pip value / standard lot": f"{fmt(pv, 4)} {account_ccy}"})
    s.formulas = ["Dollar risk = Equity × risk%", "Stop (pips) = |Entry - Stop| / pip size",
                  "Lots = Dollar risk / (Stop pips × pip value per lot)", "1 standard lot = 100,000 units (mini 10k, micro 1k)"]
    risk = equity * risk_pct / 100
    pips = round(abs(entry - stop) / ps, 6)
    lots = risk / (pips * pv)
    s.step(f"Dollar risk = {fmt(equity)} × {risk_pct / 100} = {fmt(risk)}")
    s.step(f"Stop = |{entry} - {stop}| / {ps} = {fmt(pips, 1)} pips")
    s.step(f"Lots = {fmt(risk)} / ({fmt(pips, 1)} × {fmt(pv, 4)}) = {fmt(lots, 4)} standard lots = {fmt(lots * 100000, 0)} units")
    r = {"dollar_risk": risk, "stop_pips": pips, "lots": lots, "units": lots * 100000, "pip_value_per_lot": pv}
    if target is not None:
        tp = round(abs(target - entry) / ps, 6)
        gain = tp * pv * lots
        rr = tp / pips
        r.update(target_pips=tp, profit_at_target=gain, reward_risk=rr)
        s.step(f"Target = {fmt(tp, 1)} pips -> R:R = 1:{fmt(rr, 3)}; profit at target = {fmt(tp, 1)} × {fmt(pv, 4)} × {fmt(lots, 4)} = {fmt(gain)}")
        if win_rate is not None:
            e = win_rate * gain - (1 - win_rate) * risk
            r["expectancy"] = e
            s.step(f"Expectancy = {win_rate}×{fmt(gain)} - {round(1 - win_rate, 6)}×{fmt(risk)} = {fmt(e)} per trade")
    s.interpretation.append(f"Size is set by the stop, not by conviction: a wider stop means fewer lots for the same {fmt(risk)} risk.")
    s.assumptions.append("No spread, commission, swap or slippage; stop fills exactly (gaps can make losses larger).")
    return r, s


def crypto_units(equity: float, risk_pct: float, entry: float, stop: float, target: float | None = None) -> tuple[dict, Solution]:
    s = Solution("Crypto position sizing", given={"Equity": fmt(equity), "Risk": f"{risk_pct}%", "Entry": fmt(entry), "Stop": fmt(stop)})
    s.formulas = ["Units = Dollar risk / |Entry - Stop|", "Notional = Units × Entry;  leverage needed = Notional / Equity"]
    risk = equity * risk_pct / 100
    units = risk / abs(entry - stop)
    notional = units * entry
    s.step(f"Units = {fmt(risk)} / {fmt(abs(entry - stop))} = {fmt(units, 6)}")
    s.step(f"Notional = {fmt(units, 6)} × {fmt(entry)} = {fmt(notional)} ({notional / equity:.1%} of equity)")
    r = {"dollar_risk": risk, "units": units, "notional": notional, "leverage_needed": notional / equity}
    if target is not None:
        r["profit_at_target"] = units * abs(target - entry)
        s.step(f"Profit at target = {fmt(units, 6)} × {fmt(abs(target - entry))} = {fmt(r['profit_at_target'])}")
    s.interpretation.append("Leverage needed" + (" exceeds 1x - requires margin." if notional > equity else " is ≤ 1x - spot is enough."))
    return r, s


def dca(amount_per_buy: float, prices: list[float]) -> tuple[dict, Solution]:
    s = Solution("Dollar-cost averaging", given={"Amount per buy": fmt(amount_per_buy), "Prices": [fmt(p) for p in prices]})
    s.formulas = ["Units_i = Amount / P_i", "Cost basis = total spent / total units = harmonic mean of prices",
                  "Harmonic mean ≤ arithmetic mean (equality only if all prices equal)"]
    units = [amount_per_buy / p for p in prices]
    tot_u, spent = sum(units), amount_per_buy * len(prices)
    basis = spent / tot_u
    mean = sum(prices) / len(prices)
    s.step("Units: " + " + ".join(fmt(u, 6) for u in units) + f" = {fmt(tot_u, 6)}")
    s.step(f"Cost basis = {fmt(spent)} / {fmt(tot_u, 6)} = {fmt(basis)};  arithmetic mean = {fmt(mean)}")
    s.interpretation.append(f"DCA basis is {fmt(mean - basis)} below the simple mean because fixed dollars buy more units when prices are low.")
    s.interpretation.append("This is an identity about averages; it does not make DCA outperform a lump sum in expectation.")
    return {"units": units, "total_units": tot_u, "cost_basis": basis, "arithmetic_mean": mean}, s


def spatial_arbitrage(p_buy: float, p_sell: float, qty: float = 1.0, withdraw_fee_units: float = 0.0,
                      taker_buy: float = 0.0, taker_sell: float = 0.0, fixed_fee: float = 0.0) -> tuple[dict, Solution]:
    s = Solution("Cross-exchange arbitrage", given={"Buy price (A)": fmt(p_buy), "Sell price (B)": fmt(p_sell), "Qty": qty,
                                                    "Withdrawal fee (units)": withdraw_fee_units, "Taker A": taker_buy, "Taker B": taker_sell,
                                                    "Fixed fee": fixed_fee})
    s.formulas = ["Net = (qty - withdrawal fee) × P_B × (1 - taker_B) - fixed fee - qty × P_A × (1 + taker_A)"]
    cost = qty * p_buy * (1 + taker_buy)
    recv_units = qty - withdraw_fee_units
    gross = recv_units * p_sell
    fee_b = gross * taker_sell
    net_proceeds = gross - fee_b - fixed_fee
    profit = net_proceeds - cost
    s.step(f"Cost on A = {qty} × {fmt(p_buy)} × (1 + {taker_buy}) = {fmt(cost)}")
    s.step(f"Units arriving on B = {qty} - {withdraw_fee_units} = {recv_units};  gross = {fmt(gross)}")
    s.step(f"Taker fee B = {fmt(fee_b)}; fixed fee = {fmt(fixed_fee)} -> net proceeds = {fmt(net_proceeds)}")
    s.step(f"Profit = {fmt(net_proceeds)} - {fmt(cost)} = {fmt(profit)} ({profit / cost:.3%})")
    s.interpretation.append("The quoted gap is not the realisable profit: transfer time exposes the position to price moves, "
                            "and order-book depth limits size before slippage erases the edge.")
    return {"cost": cost, "gross": gross, "fee_sell": fee_b, "net_proceeds": net_proceeds, "profit": profit}, s


def funding_carry(spot: float, qty: float, funding_rate_per_period: float, periods: int, taker: float,
                  margin: float, days: float) -> tuple[dict, Solution]:
    s = Solution("Delta-neutral funding carry (long spot + short perpetual)",
                 given={"Spot": fmt(spot), "Qty": qty, "Funding/period": f"{funding_rate_per_period:.4%}", "Periods": periods,
                        "Taker fee": f"{taker:.3%}", "Perp margin": fmt(margin), "Days": days})
    s.formulas = ["Funding payment = notional × funding rate (paid longs -> shorts when positive)",
                  "Fees = 4 legs × notional × taker", "Return on capital = net / (spot cost + margin)"]
    notional = spot * qty
    gross = notional * funding_rate_per_period * periods
    fees = 4 * notional * taker
    net = gross - fees
    cap = notional + margin
    roc = net / cap
    s.step(f"Total funding rate = {periods} × {funding_rate_per_period:.4%} = {funding_rate_per_period * periods:.3%}; gross = {fmt(gross)}")
    s.step(f"Fees = 4 × {fmt(notional)} × {taker} = {fmt(fees)}")
    s.step(f"Net = {fmt(net)}; capital = {fmt(cap)}; return = {roc:.3%} over {days} days "
           f"(simple annualised {roc * 365 / days:.1%}, compounded {(1 + roc) ** (365 / days) - 1:.1%})")
    s.interpretation.append("Risks: funding turning negative, short-perp liquidation on a sharp rally (collateral only covers a "
                            f"{margin / notional:.0%} move), exchange/custody risk, and basis moves at unwind.")
    return {"gross_funding": gross, "fees": fees, "net": net, "capital": cap, "roc": roc,
            "annualised_simple": roc * 365 / days}, s


def liquidation(entry: float, leverage: float | None = None, mmr: float = 0.005, side: str = "long",
                margin: float | None = None, notional: float | None = None) -> tuple[dict, Solution]:
    if margin is not None and notional is not None:
        imr = margin / notional
    else:
        imr = 1 / leverage
    s = Solution(f"Isolated-margin liquidation ({side})", given={"Entry": fmt(entry), "Initial margin fraction": f"{imr:.4%}",
                                                                 "Effective leverage": f"{1 / imr:.2f}x", "MMR": f"{mmr:.3%}"})
    s.formulas = ["Approximation: long P_liq ≈ P0(1 - IMR + MMR);  short P_liq ≈ P0(1 + IMR - MMR)",
                  "Exact (linear contract, equity = MMR × notional at P_liq): long P_liq = P0(1 - IMR)/(1 - MMR), short P_liq = P0(1 + IMR)/(1 + MMR)"]
    if side == "long":
        approx = entry * (1 - imr + mmr); exact = entry * (1 - imr) / (1 - mmr)
    else:
        approx = entry * (1 + imr - mmr); exact = entry * (1 + imr) / (1 + mmr)
    move = abs(approx - entry) / entry
    s.step(f"Approx P_liq = {fmt(entry)} × {fmt((approx / entry), 4)} = {fmt(approx)} -> {move:.2%} adverse move")
    s.step(f"Exact P_liq = {fmt(exact)} (difference {fmt(abs(exact - approx))}); exchanges also add fees/ insurance-fund buffers")
    s.interpretation.append(f"At {1 / imr:.1f}x effective leverage a {move:.1%} move wipes the margin; "
                            "compare with the instrument's daily volatility to judge how likely that is.")
    return {"imr": imr, "liq_approx": approx, "liq_exact": exact, "adverse_move_pct": move * 100}, s


def kelly(p: float, b: float, fraction: float = 0.5) -> tuple[dict, Solution]:
    s = Solution("Kelly criterion", given={"Win probability p": p, "Net odds b (win/loss)": b, "Kelly fraction used": fraction})
    s.formulas = ["f* = (b·p - q)/b,  q = 1 - p", "Growth rate g(f) = p·ln(1 + b f) + q·ln(1 - f)"]
    q = 1 - p
    f = (b * p - q) / b
    g = lambda x: p * math.log(1 + b * x) + q * math.log(1 - x) if 0 <= x < 1 else float("nan")
    s.step(f"f* = ({b}×{p} - {round(q, 6)})/{b} = {f:.4f}")
    r = {"kelly": f, "fractional": max(0.0, f * fraction)}
    if f > 0:
        s.step(f"Growth per bet: full Kelly {g(f):.5f}, {fraction}×Kelly {g(f * fraction):.5f} ({g(f * fraction) / g(f):.0%} of max growth)")
        r.update(growth_full=g(f), growth_fraction=g(f * fraction))
    s.interpretation.append("Kelly assumes p and b are KNOWN. With estimated edges, overbetting is costly (growth turns negative "
                            "above 2×f*), so fractional Kelly or fixed-fractional risk is the practical choice.")
    return r, s


def risk_of_ruin(win_rate: float, rr: float, risk_frac: float, ruin_drawdown: float = 0.5, n_trades: int = 500,
                 sims: int = 20000, seed: int = 7) -> tuple[dict, Solution]:
    """Monte Carlo: probability equity falls by `ruin_drawdown` within n_trades, fixed-fractional sizing."""
    s = Solution("Risk of ruin (Monte Carlo)", given={"Win rate": win_rate, "R:R": rr, "Risk per trade": f"{risk_frac:.2%}",
                                                       "Ruin level": f"-{ruin_drawdown:.0%}", "Horizon": n_trades, "Simulations": sims})
    rng = np.random.default_rng(seed)
    wins = rng.random((sims, n_trades)) < win_rate
    mult = np.where(wins, 1 + risk_frac * rr, 1 - risk_frac)
    eq = np.cumprod(mult, axis=1)
    ruined = (eq.min(axis=1) <= 1 - ruin_drawdown).mean()
    dd = 1 - (eq / np.maximum.accumulate(eq, axis=1)).min(axis=1)
    s.formulas = ["Equity_t = Equity_{t-1} × (1 + f·R) on a win, × (1 - f) on a loss (fixed fractional)"]
    s.step(f"P(drawdown ≥ {ruin_drawdown:.0%} within {n_trades} trades) = {ruined:.2%}")
    s.step(f"Max drawdown: median {np.median(dd):.1%}, 95th percentile {np.quantile(dd, .95):.1%}")
    s.step(f"Terminal equity multiple: median {np.median(eq[:, -1]):.2f}x, 5th percentile {np.quantile(eq[:, -1], .05):.2f}x")
    s.interpretation.append("Drawdowns are path-dependent; a positive-expectancy system can still suffer deep drawdowns at high risk per trade.")
    s.assumptions.append(f"Independent trades, constant win rate and payoff; seed {seed} for reproducibility.")
    return {"p_ruin": float(ruined), "dd_median": float(np.median(dd)), "dd_p95": float(np.quantile(dd, .95)),
            "terminal_median": float(np.median(eq[:, -1]))}, s


def covered_interest_parity(spot: float, r_quote: float, r_base: float, days: int, basis: int = 360) -> tuple[dict, Solution]:
    s = Solution("Covered interest parity forward", given={"Spot (quote per base)": spot, "r_quote": r_quote, "r_base": r_base, "Days": days})
    s.formulas = ["F = S × (1 + r_quote·d/basis) / (1 + r_base·d/basis)", "Forward points = (F - S) × 10,000 (4-decimal pairs)"]
    F = spot * (1 + r_quote * days / basis) / (1 + r_base * days / basis)
    s.step(f"F = {spot} × (1 + {r_quote}×{days}/{basis}) / (1 + {r_base}×{days}/{basis}) = {F:.5f}; points = {(F - spot) * 1e4:.1f}")
    s.interpretation.append("The higher-yield currency trades at a forward discount; carry trades earn the differential only "
                            "if the spot rate does NOT move to offset it (uncovered parity failing).")
    return {"forward": F, "forward_points": (F - spot) * 1e4}, s
