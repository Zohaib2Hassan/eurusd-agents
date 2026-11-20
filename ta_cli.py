"""Run the Technical Analysis agent on its own (read-only).

  python ta_cli.py                 deep report (markdown) + JSON saved in logs/ta/
  python ta_cli.py --brief         short chart read
  python ta_cli.py --json          print the structured JSON response
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agents.llm import LLM
from agents.market_data import Market
from agents.news_agent import NewsAgent
from ta import report
from ta.agent import TAAgent

ROOT = Path(__file__).parent
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(open(ROOT / a.config, encoding="utf-8"))
    m = Market(cfg)
    if not m.connect():
        sys.exit("MT5 not available")
    try:
        news = NewsAgent(cfg)
        agent = TAAgent(cfg, m, LLM(cfg), events_provider=news.calendar_strict)
        res = agent.analyze(depth="brief" if a.brief else "deep")
        if a.json:
            print(json.dumps(res, indent=1, default=str))
        else:
            text = report.brief(res) if a.brief else report.deep(res)
            print(text)
            if not a.brief:
                out = ROOT / "logs" / "ta" / "latest_report.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
    finally:
        m.shutdown()


if __name__ == "__main__":
    main()
