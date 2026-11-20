"""EUR/USD multi-agent trader.

  python main.py              one analysis cycle
  python main.py --loop       run every loop_minutes (Ctrl+C to stop)
  python main.py --check      only test the MT5 connection
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agents.execution import Executor
from agents.llm import LLM
from agents.market_data import Market
from agents.news_agent import NewsAgent
from agents.orchestrator import Orchestrator
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskAgent
from ta.agent import TAAgent

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-10s %(levelname)-7s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / "run.log", encoding="utf-8")])
log = logging.getLogger("main")


def load_cfg(path: str) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cycle(cfg, market, llm, interactive=True) -> dict:
    news = NewsAgent(cfg, market, llm)
    tech = TAAgent(cfg, market, llm, events_provider=news.calendar_strict)   # multi-timeframe TA agent
    quant = QuantAgent(cfg, market)
    with ThreadPoolExecutor(max_workers=1) as pool:   # news (network + AI) runs in parallel
        news_future = pool.submit(news.run)
        reports = {"technical": tech.run(), "quant": quant.run()}  # MT5 calls stay on one thread
        reports["news"] = news_future.result()

    decision = Orchestrator(cfg, llm).decide(reports)
    risk = RiskAgent(cfg, market).review(decision, reports)
    result = Executor(cfg, market).execute(risk["order"], interactive) if risk["approved"] else {"status": "no_trade"}

    print_report(market, reports, decision, risk, result)
    entry = {"time": datetime.now(timezone.utc).isoformat(), "mode": cfg.get("mode"), "mock": market.mock,
             "reports": {k: {kk: v[kk] for kk in ("bias", "confidence", "summary")} for k, v in reports.items()},
             "decision": decision, "risk": risk, "result": result}
    with open(LOGS / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def print_report(market, reports, decision, risk, result):
    line = "=" * 72
    print(f"\n{line}\n {market.symbol} | {datetime.now():%Y-%m-%d %H:%M} | "
          f"{'MOCK DATA' if market.mock else market.account_type()}\n{line}")
    for k, r in reports.items():
        print(f" [{k.upper():9}] bias {r['bias']:+.2f}  conf {r['confidence']:.2f}  | {r['summary'][:160]}")
    print(f"\n DECISION : {decision['action'].upper()}  (score {decision['score']:+.2f}, "
          f"conf {decision['confidence']:.2f}, {decision['source']})\n reason   : {decision['reason']}")
    if risk.get("order"):
        o = risk["order"]
        print(f" order    : {o['side']} {o['volume']} lots @ {o['price']}  SL {o['sl']} ({o['sl_pips']} pips)  "
              f"TP {o['tp']} ({o['tp_pips']} pips)  risk {o['risk_money']}")
    print(f" risk     : {'APPROVED' if risk['approved'] else 'VETO -> ' + '; '.join(risk['vetoes'])}")
    print(f" result   : {result}\n{line}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--yes-no-prompt", action="store_true", help="non-interactive (orders needing approval are blocked)")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_cfg(a.config)
    market = Market(cfg)
    if not market.connect():
        sys.exit("Could not connect to MetaTrader 5. Open MT5, log in, enable Algo Trading, then retry.")
    try:
        if a.check:
            print(json.dumps({"account": market.account(), "tick": market.tick(), "symbol": market.symbol_info(),
                              "mock": market.mock}, indent=2, default=str))
            return
        llm = LLM(cfg)
        log.info("Mode=%s | AI provider=%s %s | approval=%s", cfg.get("mode"), llm.provider, llm.model,
                 cfg.get("require_approval"))
        while True:
            try:
                cycle(cfg, market, llm, interactive=not a.yes_no_prompt)
            except Exception as e:
                log.exception("Cycle failed: %s", e)
            if not a.loop:
                break
            time.sleep(60 * cfg.get("loop_minutes", 15))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        market.shutdown()


if __name__ == "__main__":
    main()
