# QUANT-X: Quantitative Research & Intelligence Platform: system design

Version 0.1 (Phase 1 implemented) · Owner: Zohaib Hassan · Lives in: `quantx/`

QUANT-X is a **research platform**, not a trading bot. Its job is to produce numbers that are
correct, reproducible and honestly qualified, then let a human decide. Every number traces back to
a data hash, a code version and a stored run.

---

## 1. Principles (non-negotiable)

| # | Principle | How it is enforced |
|---|---|---|
| P1 | Numbers come from code, never from a language model | All statistics in `quantx/analytics/*`; LLMs may only explain stored results |
| P2 | Point-in-time correctness | Forming bars dropped; filtered (not smoothed) regime probabilities; rolling VaR uses past data only; macro data stored by *release time* and vintage |
| P3 | Reproducibility | `ingestion_runs` (data hash, retrieval time, timezone conversion) + `analysis_runs` (request, data ids, code version, result) |
| P4 | Out-of-sample over in-sample | In-sample fits labelled as such; anything called "performance" needs walk-forward or hold-out evidence |
| P5 | Count every trial | `experiment` table with `parent_id`; selection reports number of variants tried (multiple testing) |
| P6 | Research ≠ advice; paper first | No order code in `quantx/`; Phase 8 paper trading needs human approval per order |
| P7 | Legal data only | Each source has terms reviewed (`data_source.terms_url`); public/official endpoints preferred |

---

## 2. Architecture

```
                         ┌──────────────────────────────┐
  CLI / dashboard / chat │  14. SUPERVISOR & ORCHESTRATOR │  plans, routes, checks evidence, blocks unsupported claims
                         └──────────────┬───────────────┘
        ┌───────────────┬───────────────┼────────────────┬─────────────────┐
   DATA LAYER       ANALYTICS LAYER   MODEL LAYER     DECISION-SUPPORT    OUTPUT LAYER
   1 Collection     3 Market analysis 4 Math models   8 Strategy dev      13 Reporting
   2 Validation     5 Statistics      6 ML            9 Backtesting       Dashboard (Ph 7)
                    10 Risk analytics 7 Deep learning 11 Portfolio opt
                                       12 Literature (RAG)
        └───────────────┴───────────────┴────────────────┴─────────────────┘
                 RESEARCH STORE: SQLite (Ph 1) -> PostgreSQL + TimescaleDB (Ph 2+)
                 instruments · ohlcv · ticks · macro (vintaged) · features · experiments · runs
```

Data flow for one request: Supervisor → Collection → Validation (quality gate) → analytics agents
(independent, deterministic) → Supervisor checks (sample size, convergence, consistency, multiple
testing) → Reporting → `analysis_runs`.

---

## 3. Agents

Status: ✅ built (Phase 1) · 🔜 next phase · ⏳ later

| # | Agent | Role | Tools / methods | Inputs → outputs | Validation | Status |
|---|---|---|---|---|---|---|
| 1 | Data Collection | Fetch market data read-only with provenance | MT5 (any broker symbol), Binance public klines, CSV; later FRED, SEC EDGAR, Polygon/FMP (API keys) | symbol, tf, bars → bars + metadata | schema check, non-empty, data hash stored | ✅ |
| 2 | Data Validation | Quality gate before any statistics | duplicates, OHLC consistency, forming bar, calendar-aware gaps (FX 24×5 vs crypto 24×7), robust outliers, staleness | raw bars → clean bars, warnings, quality 0–1 | blocks if quality < 0.5 or < 100 bars | ✅ |
| 3 | Market Analysis | Describe one market | moments, vol cone, EWMA, GARCH/GJR-GARCH, regimes (Markov-switching, vol percentile, trend) | clean bars → structured stats | min sample per method (GARCH/Markov ≥ 500) | ✅ |
| 4 | Mathematical Modelling | Stochastic models & calibration | GBM, OU, Merton jump-diffusion, Heston, Black-Scholes/Garman-Kohlhagen, Greeks, Kalman/state space | series / option quotes → calibrated params, diagnostics | parameter identifiability, Feller condition, residual tests, simulation-recovery tests | 🔜 Ph 2 |
| 5 | Statistical Research | Hypothesis tests & econometrics | JB, Ljung-Box, ARCH-LM, ADF/KPSS, variance ratio, Hurst ✅; ARIMA, VAR/VECM, cointegration (Engle-Granger/Johansen), EVT, bootstrap 🔜 | data → test results with p-values | multiple-testing log, robust (HAC) variants | ✅ partial |
| 6 | Machine Learning | Tabular ML forecasting | linear/logit, RF, XGBoost, LightGBM, CatBoost, SVM, clustering, PCA | point-in-time features → predictions + uncertainty | purged/embargoed walk-forward CV, calibration, no-skill baselines | ⏳ Ph 3 |
| 7 | Deep Learning | Sequence models | LSTM, GRU, TCN, Transformer, TFT | sequences → quantile forecasts | same as 6 + must beat 6 and baselines after costs | ⏳ Ph 3 |
| 8 | Strategy Development | Turn a hypothesis into explicit rules | rule DSL, signal timing spec, sizing rules | hypothesis → machine-testable strategy | pre-registration in `experiment.hypothesis` | ⏳ Ph 4 |
| 9 | Backtesting | Realistic simulation | event-driven + vectorised engines, fill model, costs, walk-forward, Monte Carlo, Deflated Sharpe / PBO | strategy + data → trade log, metrics, CIs | look-ahead tests, cost sensitivity, trial count | ⏳ Ph 4 |
| 10 | Risk Analytics | Measure market risk | VaR/CVaR (hist, normal, Cornish-Fisher), Kupiec backtest, drawdowns, Sharpe CI, beta ✅; EVT tail, stress scenarios, liquidity 🔜 | returns → risk report | VaR backtested out-of-sample | ✅ partial |
| 11 | Portfolio Optimisation | Allocation research | min-variance, max-Sharpe (shrinkage covariance), risk parity, HRP, Black-Litterman, fractional Kelly (as framework only) | returns/views → weights + risk budget | out-of-sample comparison vs 1/N | ⏳ Ph 5 |
| 12 | Research Literature | Ground claims in papers | RAG over owned papers, citation records | question → cited summary | no citation without source record; prior-art check before "novel" | ⏳ Ph 6 |
| 13 | Reporting | Standard 15-section report | Markdown/JSON now; PDF/dashboard later | results → report | every finding cites its statistic | ✅ |
| 14 | Supervisor | Orchestrate + guard | plan, quality gates, sample-size rules, convergence/consistency checks, multiple-testing note | request → validated result | refuses forecasts/signals in Phase 1 | ✅ |

The existing **Technical Analysis agent** (`ta/`) and the EUR/USD trading pipeline (`agents/`, dry-run)
stay separate. QUANT-X feeds them research; it never sends orders.

---

## 4. Data sources

| Source | Data | Access | Notes |
|---|---|---|---|
| MetaTrader 5 (your broker) | FX, metals, indices, CFDs: OHLC, spread, tick volume | local terminal, read-only | BID prices; tick volume is broker-specific; server time converted to UTC ✅ |
| Binance public API | crypto spot klines | no key | exchange-specific volume; geo-restrictions possible (fallback endpoint built in) ✅ |
| FRED (St. Louis Fed) | rates, CPI, employment | free API key | store **release time and vintage** to avoid look-ahead 🔜 |
| SEC EDGAR | filings, XBRL fundamentals | free; requires a descriptive User-Agent and rate limits | 🔜 |
| Polygon / FMP / Alpha Vantage | equities, options, fundamentals | paid/free keys | check licence before storing/redistributing ⏳ |
| Yahoo Finance | equities | unofficial endpoints | avoid for production; terms unclear |

---

## 5. Database

Target schema: `docs/quantx_schema.sql` (PostgreSQL + TimescaleDB hypertables).
Phase 1 uses the same core tables on SQLite (`data/quantx.db`) so nothing needs a server yet.

Key design choices: bar-open UTC timestamps; `CHECK` on OHLC consistency; delisted instruments kept
(survivorship); macro values keyed by **vintage + release time**; point-in-time feature store with
`calc_version`; `experiment`/`experiment_result` record every trial with its split.

---

## 6. Method catalogue by phase

**Phase 1 ✅** log returns, moments, Jarque-Bera, ACF, Ljung-Box (levels and squares), ARCH-LM, ADF,
KPSS, Lo-MacKinlay variance ratio (robust z*), Hurst scaling, realised/EWMA vol, GARCH(1,1)-t and
GJR-GARCH-t (BIC selection, persistence, half-life, forecasts), volatility cone, 2-state
Markov-switching (filtered probabilities), VaR/CVaR ×3 methods, Kupiec backtest, drawdowns,
buy-and-hold Sharpe with Lo (2002) confidence interval, beta/alpha/tracking error, correlation
matrices with rolling stability.

**Phase 2 🔜** ARIMA/SARIMA (AIC/BIC, residual diagnostics), EGARCH, VAR/VECM, Engle-Granger and
Johansen cointegration, Kalman filter (dynamic hedge ratios), HMM with >2 states, change-point
detection (PELT/Bayesian), extreme-value theory (POT/GPD tail VaR), OU half-life, GBM/Merton/Heston
simulation **with parameter-recovery tests**, Black-Scholes / Garman-Kohlhagen pricing and Greeks.

**Phase 3 ⏳** ML/DL per agents 6–7. Protocol: target and horizon fixed in advance; features computed
point-in-time; scaling fitted on train only; purged + embargoed walk-forward CV; hyperparameters chosen
inside the training folds (nested); report against naive, AR and buy-and-hold baselines; calibration
plots for probabilities; drift monitoring. Accuracy alone never counts as edge.

**Phase 4 ⏳** Backtesting. Signal at bar close t, order at t+1 open; buys at ask / sells at bid;
commission, swap, slippage model; when stop and target are hit in the same bar, assume the stop
first; gaps fill at the open. Walk-forward, Monte Carlo resampling (block bootstrap), cost
sensitivity, Deflated Sharpe Ratio and Probability of Backtest Overfitting using the trial count
from `experiment`.

**Phase 5 ⏳** Portfolio construction (Ledoit-Wolf shrinkage), HRP, risk parity, Black-Litterman,
risk budgeting, stress tests, correlation-breakdown scenarios.

---

## 7. Supervisor rules
1. Quality gate: no analysis on data with quality < 0.5 or < 100 bars.
2. Minimum evidence: GARCH/Markov need ≥ 500 returns; VR ≥ 200; correlation ≥ 70 overlapping returns.
3. Model checks: optimiser convergence, GARCH persistence < 1, VaR backtest result reported.
4. Multiple testing: the report states how many tests ran and the expected false positives.
5. Claims: every finding names the statistic and p-value it rests on; no direction forecasts or trade
   signals before Phase 4 evidence exists; in-sample results always labelled.
6. Failures: a failed source is reported and excluded, never silently replaced.

---

## 8. Technology stack

| Layer | Now (Phase 1) | Later, when it is actually needed |
|---|---|---|
| Language | Python 3.11, pandas, NumPy, SciPy, statsmodels, arch | + scikit-learn, LightGBM/XGBoost, PyTorch (Ph 3) |
| Storage | SQLite (`data/quantx.db`) | PostgreSQL + TimescaleDB when data exceeds ~10M rows or several users |
| Backtesting | (Ph 4) own event-driven engine | vectorbt for fast parameter sweeps, always cross-checked with the event engine |
| Portfolio | (Ph 5) PyPortfolioOpt / Riskfolio-Lib, cross-checked by own code | |
| Orchestration | plain Python Supervisor | LangGraph only if the workflow needs branching tool use |
| Dashboard | Markdown/JSON reports | Streamlit + Plotly (Ph 7) |
| Ops | git + tests | Docker, CI running the test suite, scheduled data refresh |

Kafka/Spark/Redis are **not** planned. At this scale (single researcher, bar data) they add
operational cost without benefit. Revisit only for tick-level, multi-venue streaming.

---

## 9. Roadmap and acceptance criteria

| Phase | Deliverable | Done when |
|---|---|---|
| 1 ✅ | Data layer + market analytics + risk + Supervisor + reports | 16 tests pass (golden + simulation recovery); runs on live MT5 and Binance data |
| 2 | Stochastic & econometric models; Postgres/Timescale migration | each model has a recovery test on simulated data, residual diagnostics, and a documented limitation |
| 3 | ML/DL research environment | walk-forward framework with purging; every model compared with baselines after costs |
| 4 | Strategy + backtesting engine | look-ahead test suite; cost sensitivity; DSR/PBO reported with trial counts |
| 5 | Risk & portfolio engine | out-of-sample comparison against 1/N and min-variance |
| 6 | Multi-agent research workflow + literature RAG | Supervisor blocks unsupported claims in adversarial tests |
| 7 | Dashboard | every chart reproducible from a stored run id |
| 8 | Paper trading + continuous evaluation | live-vs-backtest tracking error monitored; human approval per order |

---

## 10. Review of the earlier draft code (Word documents)
The drafts are useful for scope but should not be copied in as they are:
- **HRP**: the recursive-bisection loop pairs clusters incorrectly and would fail or mis-weight. Use a
  tested library implementation and cross-check it.
- **Vectorised backtester**: P&L uses price differences per unit, then turns equity into % returns.
  Costs are charged on price × turnover in mixed units, and annualisation uses √252 regardless of
  timeframe.
- **Black-Scholes theta (put)**: the second term uses the call sign convention.
- **Feature extractor**: "RSI" uses simple means of log returns (not Wilder), and volatility is
  annualised with √365 on hourly data.
- **Agent nodes**: signals and weights are hard-coded (`BTC_LONG: 0.82`, `BTC 0.70 / ETH 0.30`), and the
  script prints "Out-Of-Sample Validation Complete" without running any validation.
- **Merton simulation**: the jump term draws a normal scaled by the jump count. For N > 1 jumps in a
  step, the sum of N normals has variance N·σ_J², not N²·σ_J².
QUANT-X re-implements these in later phases with recovery tests.

---

## 11. Safety
Read-only data adapters; no order code anywhere in `quantx/` (enforced by a test). API keys only in
`.env` (git-ignored). Paper trading is the only execution mode planned (Phase 8), with human approval.
Reports separate findings from advice and state limitations.
