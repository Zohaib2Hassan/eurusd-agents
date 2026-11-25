# EUR/USD Multi-Agent Trader (MT5)

One lead agent (orchestrator) controls a small team of agents that look at **EUR/USD only**.

```
                 ┌──────────────── Orchestrator (lead) ────────────────┐
                 │  combines reports -> BUY / SELL / HOLD               │
   ┌─────────────┼──────────────┬──────────────┐                       │
Technical     Quant (your     News agent    →  Risk agent (hard rules, can veto)
agent         formulas)       calendar+RSS     → Execution agent (only one touching MT5 orders)
H4 + H1       5 rule votes    blackout
```

| Agent | File | What it does |
|---|---|---|
| Technical | `agents/technical_agent.py` | Reads EURUSD H4 (trend) + H1 (entry): EMA 20/50/200, RSI, MACD, ATR, Bollinger, ADX |
| Quant | `agents/quant_agent.py` | Your formulas as code rules that vote +1 / -1 / 0. **Add your own rules here.** |
| News | `agents/news_agent.py` | EUR+USD economic calendar, high-impact **blackout** window, forex headlines |
| Orchestrator | `agents/orchestrator.py` | Weighted decision; AI can only confirm or downgrade to HOLD, never flip direction |
| Risk | `agents/risk_agent.py` | Lot size from ATR stop, max daily loss, max positions, spread limit, news veto |
| Execution | `agents/execution.py` | Sends orders - only in `demo`/`live` mode, with your approval |

**Symbol lock:** `agents/market_data.py` refuses any symbol except the one in `config.yaml`.

## Safety (default settings)
- `mode: dry_run` - analyses and logs, never sends orders
- `mode: demo` - sends orders only if the MT5 account is a **DEMO** account
- `mode: live` - needs `allow_live: true` too. Don't, until it has proven itself on demo for weeks.
- `require_approval: true` - you must type `y` for every order
- Indicators are calculated in Python, not by the AI.

## Setup (Windows)
1. Install MetaTrader 5 from your broker, open a **demo account**, log in.
2. In MT5: **Tools → Options → Expert Advisors → Allow algorithmic trading**, and click the **Algo Trading** button (green).
3. In this folder:
   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   ```
4. Test: `python main.py --check` (shows account, EURUSD tick)
5. Run once: `python main.py`  · Run continuously: `python main.py --loop`

## Turning on the AI (optional)
In `config.yaml` → `llm.provider`:
- `anthropic` + `ANTHROPIC_API_KEY` in `.env`
- `openai` + `OPENAI_API_KEY` in `.env`
- `ollama` or `lemonade` - free local models (base_url defaults to localhost)
- `none` - rule-based only (default, free, offline)

## Logs
- `logs/decisions.jsonl` - every cycle: reports, decision, risk result, order result
- `logs/run.log` - runtime log

> Research/education tool. Not financial advice. Forex trading can lose more than you expect;
> test on demo and backtest before risking money.

## Technical Analysis agent (`ta/`)
Read-only, multi-timeframe (D1 → H4 → H1) analysis with ~40 deterministic indicators, data validation,
swing-based support/resistance zones, pivots, scenarios with invalidation, and an audit log.
It is also the "technical" analyst inside `main.py`.

```
.venv\Scripts\python.exe ta_cli.py           # deep report (also saved to logs\ta\latest_report.md)
.venv\Scripts\python.exe ta_cli.py --brief   # short read
.venv\Scripts\python.exe ta_cli.py --json    # structured JSON
.venv\Scripts\python.exe tests\test_ta.py    # 19 golden / cross-check / safety tests
```
Design: `docs/TA_AGENT_DESIGN.md` · Schemas: `docs/ta_schemas.json`

## QUANT-X quantitative research (`quantx/`) - Phase 1
Read-only research platform: data collection with provenance (MT5 any symbol, Binance public crypto),
calendar-aware validation, statistics (normality, autocorrelation, stationarity, variance ratio, Hurst),
volatility (EWMA, GARCH/GJR-GARCH), regimes (Markov-switching), risk (VaR/CVaR x3 + Kupiec backtest,
drawdowns), cross-asset correlation, Supervisor checks, 15-section reports, SQLite research store.

```
.venv\Scripts\python.exe quantx_cli.py analyze EURUSD
.venv\Scripts\python.exe quantx_cli.py analyze XAUUSD --tf H4 --bars 3000
.venv\Scripts\python.exe quantx_cli.py compare EURUSD GBPUSD USDJPY XAUUSD
.venv\Scripts\python.exe quantx_cli.py analyze BTCUSDT --source binance
.venv\Scripts\python.exe tests\test_quantx.py
```
Design & roadmap: `docs/QUANTX_DESIGN.md` · DB schema: `docs/quantx_schema.sql`

## Agent 3 - MicroEcon & Quant tutor (`microecon/`)
Microeconomics + trading-risk maths with worked solutions and automated grading. Answer keys are computed
by the solvers (the spec's Problem 4 key was wrong: expectancy is +$600, not +$937.50).

```
python micro_cli.py list                                   # problem bank
python micro_cli.py practice --problem 3                   # graded practice (hint / show solution / quit)
python micro_cli.py solve market 500 10 -100 15 --floor 30 # direct assistance, step by step
python micro_cli.py solve fx 50000 1.5 1.0850 1.0810 --target 1.0970 --winrate 0.45
python micro_cli.py solve liq 3000 --lev 10
python micro_cli.py liquidity EURUSD                       # spread / Roll / Corwin-Schultz on real data
python micro_cli.py liquidity BTCUSDT --source binance     # + Amihud illiquidity
python tests\test_microecon.py
```
Design: `docs/MICROECON_AGENT.md`
