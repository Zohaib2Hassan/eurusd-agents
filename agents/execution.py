"""Execution agent: the ONLY part that can send orders to MT5.

Safety gates, in order:
 1. mock data      -> never trades
 2. mode dry_run   -> logs only (default)
 3. mode demo      -> only if the MT5 account is DEMO
 4. mode live      -> only if allow_live: true AND account is REAL
 5. require_approval -> you type 'y' in the console
"""
from __future__ import annotations

import logging

from .market_data import mt5

log = logging.getLogger("execution")


class Executor:
    name = "execution"

    def __init__(self, cfg, market):
        self.cfg, self.market = cfg, market

    def execute(self, order: dict, interactive: bool = True) -> dict:
        mode = self.cfg.get("mode", "dry_run")
        if self.market.mock:
            return {"status": "skipped", "why": "mock data - orders disabled"}
        if mode == "dry_run":
            return {"status": "dry_run", "why": "mode is dry_run (would have sent order)", "order": order}
        acct = self.market.account_type()
        if mode == "demo" and acct != "DEMO":
            return {"status": "blocked", "why": f"mode demo but account is {acct}"}
        if mode == "live" and not (self.cfg.get("allow_live") and acct == "REAL"):
            return {"status": "blocked", "why": "live mode needs allow_live: true and a REAL account"}
        if mode not in ("demo", "live"):
            return {"status": "blocked", "why": f"unknown mode {mode}"}

        if self.cfg.get("require_approval", True):
            if not interactive:
                return {"status": "blocked", "why": "approval required but running non-interactively"}
            ans = input(f"\n>>> APPROVE {order['side'].upper()} {order['volume']} {self.market.symbol} "
                        f"SL {order['sl']} TP {order['tp']} (risk {order['risk_money']})? [y/N]: ").strip().lower()
            if ans != "y":
                return {"status": "rejected_by_user"}
        return self._send(order)

    def _send(self, o: dict) -> dict:
        sym = self.market.symbol
        tick = mt5.symbol_info_tick(sym)
        is_buy = o["side"] == "buy"
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(o["volume"]),
               "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
               "price": tick.ask if is_buy else tick.bid, "sl": float(o["sl"]), "tp": float(o["tp"]),
               "deviation": 10, "magic": int(self.cfg["mt5"].get("magic", 260411)),
               "comment": "eurusd-agents", "type_time": mt5.ORDER_TIME_GTC}
        last = None
        for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
            req["type_filling"] = filling
            res = mt5.order_send(req)
            last = res
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info("ORDER FILLED ticket=%s price=%s", res.order, res.price)
                return {"status": "filled", "ticket": res.order, "price": res.price, "volume": res.volume}
            if res is not None and res.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                break
        return {"status": "failed", "retcode": getattr(last, "retcode", None),
                "comment": getattr(last, "comment", str(mt5.last_error()))}
