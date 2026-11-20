# Technical Analysis Agent — design (v1.0)

First agent of the system: a **read-only** agent that researches, calculates, explains and evaluates
technical analysis for one instrument (EUR/USD). Built against section 17 of the TA specification.

## 1. Capabilities and boundaries (17.A)

| | |
|---|---|
| Users | the owner of this project (education + analysis for own trading research) |
| Instrument | EURUSD only (symbol lock in `agents/market_data.py`) |
| Timeframes | D1 → H4 → H1 (configurable in `config.yaml → ta.timeframes`) |
| Does | validate data, compute indicators, read structure, derive zones/pivots, scenarios + invalidation, report uncertainty, cite method sources |
| Does NOT | place/modify/cancel orders, move funds, change broker settings, give personalised advice, state probabilities or price targets |
| Output | structured JSON (`ta.analysis_response/1`) + brief or deep markdown report |

Execution is a **separate** agent (`agents/execution.py`) behind the risk agent, dry-run by default and
human approval per order. The TA agent has no order code at all (enforced by a test).

## 2. Reference architecture (17.H)

```
User / main.py / ta_cli.py
        │  request (symbol, timeframes, timezone, depth)
        ▼
 Policy & intent ──────── read-only by construction; consequential actions live elsewhere
        ▼
 Orchestrator (TAAgent.analyze) — bounded plan: for tf in [D1,H4,H1] → validate → calc → structure
        ├── Market-data adapter ........ agents/market_data.py   (MT5, symbol-locked, BID bars)
        ├── Calendar adapter ........... agents/news_agent.py    (scheduled events, strict mode)
        ├── Validation layer ........... ta/validate.py          (dupes, OHLC, forming bar, gaps, outliers, stale)
        ├── Deterministic calc service . ta/calc.py              (CALC_VERSION, all formulas)
        ├── Structure service .......... ta/structure.py         (swings, trend state, zones)
        ├── LLM (optional) ............. narrative only, fed computed JSON, labelled MODEL INTERPRETATION
        ├── Report service ............. ta/report.py            (brief / deep / data-quality)
        └── Audit store ................ logs/ta/*.json          (inputs hash, versions, full response)
        ▼
 Human-review boundary — orchestrator + risk agent + executor (dry_run / demo / approval)
```

### Deterministic vs language model

| Component | Deterministic code | Language model |
|---|---|---|
| Indicators, pips, ATR, volatility, pivots | ✅ always | ❌ never |
| Data validation, timestamps, staleness | ✅ | ❌ |
| Swings, trend state, zones | ✅ fixed rules | ❌ |
| Rule labels, composite score, scenarios | ✅ fixed rules | ❌ |
| Plain-language explanation | — | ✅ optional, cannot add numbers, clearly labelled |

## 3. Analysis workflow (17.D) — what `analyze()` does

1. Instrument + convention (`price = USD per 1 EUR`, pip 0.0001, BID bars), source + server.
2. Timestamps: generated UTC + display timezone, last completed bar per timeframe, data age.
3. Validation per timeframe → warnings + quality score (0–1). Forming bar is always dropped.
4. Indicators (≈40 values per timeframe) from `ta/calc.py`.
5. Higher timeframe first: D1, then H4, then H1.
6. Facts / rule interpretation / model interpretation kept in separate fields.
7. Zones: confirmed fractal swings clustered at 0.5×ATR, width capped at 1×ATR; touches + last touch.
8. Trend (structure, EMAs, ADX, LinReg R²), momentum (RSI, MACD, Stoch, CCI, ROC), volatility regime (ATR percentile).
9. Events only from the calendar feed, shown in display timezone; if the feed fails the report says *not checked*.
10. Scenarios: bullish / bearish / range, each with trigger, next reference zone and invalidation.
11. Uncertainty list (timeframe disagreement, low ADX, events, BID/tick-volume limits).
12. Method references (Wilder 1978, Bollinger 2001, Parkinson 1980, Garman–Klass 1980, Rogers–Satchell 1991, Yang–Zhang 2000).

**Composite score** = Σ tf_weight × score_tf (D1 .40, H4 .35, H1 .25). Each score_tf is built from fixed
rules (structure ±1, EMA200 ±0.5, EMA50/200 ±0.5, RSI ±0.5, MACD ±0.25, cloud ±0.25)/3 × ADX conviction.
It is a **descriptive summary**, not a validated forecast — no backtest has been run on it yet.

## 4. State, memory and failure handling (17.I)

| Kind | Where | Retention |
|---|---|---|
| User preferences | `config.yaml` | until edited |
| Task parameters | request args | per call |
| Retrieved facts (bars, events) | memory during call; hash in audit | hash only |
| Calculated results | `logs/ta/*.json` | user deletes; no credentials inside |
| Model interpretation | `narrative` field, labelled | with the audit file |
| Unverified hypotheses | not stored | — |

Failures: a failed timeframe is reported in `failures` and excluded (status `partial`); all failed →
status `failed`, no score. Calendar failure → "events NOT checked". Stale data lowers quality by 0.3 and is
shown in the report. LLM failure → narrative omitted, numbers unaffected. API keys live only in `.env`.

## 5. Security (17.I)
- TA agent ingests only numeric bars + calendar fields; headlines never reach it. The optional LLM is told
  that any text inside data is data, not instructions.
- Symbol lock blocks other instruments; no order functions in `ta/`.
- No secrets in logs (audit files contain market data + results only).

## 6. Tests (`tests/test_ta.py`, 19 tests)
Hand-computed golden values (SMA, EMA, Wilder, WMA, RSI, TR/ATR, Bollinger, pivots, Stochastic, Williams %R);
independent loop implementations cross-checking EMA/RSI/ATR; no-look-ahead test over 13 indicators;
swing confirmation timing; zone width cap; validation (forming bar, duplicates, weekend vs weekday gaps,
bad OHLC); pip value for quote-USD, base-USD and cross pairs; position sizing; read-only and symbol-lock checks.

## 7. Not built yet (next steps)
- Backtesting agent (17.E) with the fill model in `docs/ta_schemas.json → backtest_config`.
- Daylight-saving / broker server-time tests (MT5 server time vs UTC needs confirming per broker).
- Research/RAG connector for books and papers (17.B) and citation capture with URLs/pages.
- Monitoring dashboard (freshness, failures, drift) and an evaluation benchmark set.
- Volume-profile / VWAP tools (low value on OTC FX tick volume; deferred).
