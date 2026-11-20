"""Risk agent: hard rules in code. It can VETO any trade; the AI cannot override it."""
from __future__ import annotations

import math


class RiskAgent:
    name = "risk"

    def __init__(self, cfg, market):
        self.cfg, self.market = cfg, market
        self.r = cfg.get("risk", {})

    def review(self, decision: dict, reports: dict) -> dict:
        action = decision.get("action", "hold")
        vetoes = []
        if action == "hold":
            return {"approved": False, "vetoes": ["orchestrator chose HOLD"], "order": None}

        news = reports.get("news", {}).get("details", {})
        if news.get("blackout"):
            vetoes.append("news blackout: " + "; ".join(news.get("blocking_events", [])))
        if decision.get("confidence", 0) < self.r.get("min_confidence", 0.6):
            vetoes.append(f"confidence {decision.get('confidence', 0):.2f} < {self.r.get('min_confidence', 0.6)}")

        tick = self.market.tick()
        if tick["spread_points"] > self.r.get("max_spread_points", 20):
            vetoes.append(f"spread {tick['spread_points']} pts too wide")
        if len(self.market.positions()) >= self.r.get("max_open_positions", 1):
            vetoes.append("max open positions reached")

        acc = self.market.account()
        day_pnl = self.market.today_closed_pnl()
        if day_pnl <= -acc["balance"] * self.r.get("max_daily_loss_pct", 2.0) / 100:
            vetoes.append(f"daily loss limit hit ({day_pnl:.2f})")

        order = self._size(action, tick, acc, reports)
        if order is None:
            vetoes.append("could not size position (ATR missing or lot too small)")
        return {"approved": not vetoes, "vetoes": vetoes, "order": order,
                "context": {"spread_points": tick["spread_points"], "balance": acc["balance"], "today_pnl": day_pnl}}

    def _size(self, action, tick, acc, reports):
        atr = reports.get("technical", {}).get("details", {}).get("atr_entry")
        if not atr:
            return None
        info = self.market.symbol_info()
        sl_dist = max(atr * self.r.get("atr_sl_multiplier", 1.5), (info["stops_level"] + 5) * info["point"])
        tp_dist = sl_dist * self.r.get("reward_risk", 2.0)
        price = tick["ask"] if action == "buy" else tick["bid"]
        sl = price - sl_dist if action == "buy" else price + sl_dist
        tp = price + tp_dist if action == "buy" else price - tp_dist

        risk_money = acc["balance"] * self.r.get("risk_per_trade_pct", 0.5) / 100
        loss_per_lot = sl_dist / info["tick_size"] * info["tick_value"]
        lots = risk_money / loss_per_lot if loss_per_lot > 0 else 0
        step = info["volume_step"]
        lots = math.floor(lots / step) * step
        lots = min(lots, self.r.get("max_lot", 1.0), info["volume_max"])
        min_lot = max(self.r.get("min_lot", 0.01), info["volume_min"])
        if lots < min_lot:
            # small account: allow the minimum lot only if its risk stays under the cap
            min_lot_risk = min_lot * loss_per_lot
            if min_lot_risk <= acc["balance"] * self.r.get("max_risk_pct_min_lot", 2.0) / 100:
                lots, risk_money = min_lot, min_lot_risk
            else:
                return None
        d = info["digits"]
        return {"side": action, "volume": round(lots, 2), "price": round(price, d), "sl": round(sl, d),
                "tp": round(tp, d), "risk_money": round(risk_money, 2),
                "sl_pips": round(sl_dist / (info["point"] * 10), 1), "tp_pips": round(tp_dist / (info["point"] * 10), 1)}
