"""QUANT-X Phase-1 command line (read-only research).

  python quantx_cli.py analyze EURUSD                         # MT5, D1, 2000 bars
  python quantx_cli.py analyze XAUUSD --tf H4 --bars 3000
  python quantx_cli.py analyze BTCUSDT --source binance
  python quantx_cli.py compare EURUSD GBPUSD USDJPY XAUUSD
  python quantx_cli.py compare BTCUSDT ETHUSDT SOLUSDT --source binance
Reports are saved to reports/quantx/, data and run records to data/quantx.db.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from quantx.agents import reporting
from quantx.agents.core import Supervisor
from quantx.data.sources import make_source
from quantx.data.store import Store

ROOT = Path(__file__).parent
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main():
    ap = argparse.ArgumentParser(description="QUANT-X quantitative research (Phase 1)")
    ap.add_argument("command", choices=["analyze", "compare"])
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--source", default="mt5", choices=["mt5", "binance"])
    ap.add_argument("--tf", default="D1", choices=["M15", "M30", "H1", "H4", "D1", "W1"])
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--json", action="store_true", help="print raw JSON result")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    src = make_source(a.source, cfg)
    if not src.connect():
        sys.exit(f"Could not connect to {a.source}")
    store = Store(ROOT / "data" / "quantx.db")
    try:
        sup = Supervisor(src, store)
        if a.command == "analyze":
            res = sup.analyze(a.symbols[0], a.tf, a.bars)
            text = reporting.single(res)
        else:
            res = sup.compare(a.symbols, a.tf, a.bars)
            text = reporting.compare(res)
        print(json.dumps(res, indent=1, default=str) if a.json else text)
        out = ROOT / "reports" / "quantx"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        name = f"{stamp}_{a.command}_{'-'.join(a.symbols)}_{a.tf}"
        (out / f"{name}.md").write_text(text, encoding="utf-8")
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"\nSaved: reports/quantx/{name}.md", file=sys.stderr)
    finally:
        store.close()
        src.close()


if __name__ == "__main__":
    main()
