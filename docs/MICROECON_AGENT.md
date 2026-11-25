# Agent 3: MicroEcon & Quant tutor (design)

Code: `microecon/` · CLI: `micro_cli.py` · Tests: `tests/test_microecon.py` (37)

Agent 3 tutors and checks microeconomics, market microstructure and trading-risk maths. It has two
modes from the specification (direct assistance and practice with automated grading) plus a bridge to
real data (liquidity estimators on MT5/Binance bars). Agents 1 (Technical Analysis) and 2 (QUANT-X)
are unchanged.

## Principles
1. **Answers are computed, never typed.** Every answer key in the problem bank is produced by the solver
   functions when the program loads. A wrong key can't silently grade students wrong.
2. **Same 4-part format everywhere:** Given → Conditions/formulas → Step-by-step → Interpretation, with
   assumptions stated.
3. **Deterministic grading.** The rubric is code, so the same answer always gets the same score.
4. **Analytical framing only.** There is no order code, and outputs end with "not financial advice".

## Components
| Module | Contents |
|---|---|
| `econ.py` | Linear markets (equilibrium, CS/PS, point elasticity, floors, ceilings, per-unit tax incidence and DWL), arc elasticity, monopoly (MR = MC, Lerner, DWL), Cournot (symmetric and asymmetric), cartel, Stackelberg, Bertrand, bimatrix Nash (pure and 2×2 mixed), Cobb-Douglas utility maximisation with Slutsky decomposition, Cobb-Douglas cost minimisation and returns to scale |
| `trading.py` | Expectancy, break-even win rate and dispersion over N trades; FX lot sizing with correct pip value for quote-USD, base-USD (e.g. USDJPY) and cross pairs; crypto unit sizing; DCA (harmonic-mean basis); cross-exchange arbitrage; perpetual funding carry; isolated-margin liquidation (spec approximation **and** exact formula); Kelly and fractional Kelly with growth rates; Monte Carlo risk of ruin and drawdowns; covered interest parity |
| `bank.py` | The 8 specification problems, each 4 steps × 25 pts, with hints, worked solutions, takeaways and answer-key correction notes |
| `grader.py` | Number extraction, rubric, sessions (hint cap 80%, "show solution" = 0 for remaining steps, 3 attempts then reveal), score table and mastery tier |
| `generator.py` | 13 seeded topics with unlimited new problems: tax incidence, price ceiling, monopoly & Lerner index, asymmetric Cournot + HHI, Stackelberg, 2×2 Nash (pure & mixed), Slutsky decomposition, Cobb-Douglas cost minimisation, forex sizing across pair types (incl. USDJPY pip value), long/short liquidation, Kelly & break-even, covered interest parity & carry, funding carry. Keys come from the solvers; the same seed always gives the same problem |
| `progress.py` | Local practice log (`data/microecon_progress.jsonl`, git-ignored): attempts, recent average and tier per topic, suggested next topic |
| `microstructure.py` | MT5 quoted spread, Roll (1984) effective spread, Corwin-Schultz (2012) high-low spread, Amihud (2002) illiquidity (exchange volume only) |

## Grading rubric (implemented)
| Result per numeric target | Credit |
|---|---|
| within tolerance (default 0.5%) | 100% |
| ≤ 2% off | 85% (minor slip) |
| ≤ 10% off | 65% (minor slip) |
| a number, further off | 35% (conceptual/algebraic flaw) |
| no number | 0% |

**Mistake diagnosis:** each step can list common wrong answers (forgetting the fixed cost, using quantity
supplied under a floor, $10/pip on USDJPY, a missing ½ in a triangle, statutory vs economic incidence, ...).
If a student's number matches one, the agent explains the likely mistake instead of giving a generic hint,
and the step counts as a conceptual error (at most 59%), never a "minor slip".

The step score is the mean over its targets. Text steps (risks, limits) score by coverage of concept
groups. The pass mark is 75% per step; below that the student gets a targeted hint and a retry.
Mastery tiers: ≥90 Mastery, ≥75 Proficient, ≥50 Developing, <50 Needs Review.

## Answer-key review of the specification
| Problem | Status |
|---|---|
| 1 | PS: the key gives $2,252.90 by rounding the intercept to 6.67; the exact value is **$2,253.33**. Both are accepted within tolerance |
| 2, 3, 6, 7, 8 | correct |
| **4 (Step 4)** | **Wrong in the specification.** With 1.875 lots a 120-pip win is $2,250 (= 3 × $750), not $3,000, so expectancy = 0.45×2,250 − 0.55×750 = **+$600**, not +$937.50 |
| 5 | correct under the key's assumption of no taker fee on the buy leg (now stated) |

The liquidation formula in the specification is an approximation. The agent also shows the exact
linear-contract value: long P_liq = P0(1 − IMR)/(1 − MMR), for example $2,713.57 vs $2,715 at 10×.

## About the attached paper
Wang et al. (2022), *Mathematical processing of trading strategy based on long short-term memory neural
network model*, Front. Comput. Neurosci. 16:1052140. It is an LSTM gold/Bitcoin trading strategy
(MCM-style), so it belongs with the ML agent (QUANT-X Phase 3), not here. Read it with care: decision
thresholds are fixed by hand, there is no transaction-cost or walk-forward evidence in the excerpt, and
much of the reference list is unrelated to finance (nanoparticle optics, oncology, fibre optics). It is
a candidate for a replication exercise once Phase 3 has a walk-forward framework.

## Next extensions
- Bertrand with capacity constraints, price discrimination, externalities/Pigouvian tax, adverse selection (lemons) topics.
- Adaptive difficulty: pick the next topic from the progress log automatically.
- Optional LLM layer that explains a student's mistake in words while the score stays from the rubric.
- Order-book microstructure (Kyle's lambda, order-flow imbalance) when tick/trade data are available.
