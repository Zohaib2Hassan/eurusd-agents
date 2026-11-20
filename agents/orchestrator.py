"""Orchestrator: the 'lead' agent. Combines the analysts' reports into BUY / SELL / HOLD.

The AI (if enabled) may only confirm the rule-based direction or downgrade to HOLD.
It can never flip the direction against the combined evidence.
"""
from __future__ import annotations

import json

WEIGHTS = {"technical": 0.45, "quant": 0.25, "news": 0.30}
THRESHOLD = 0.2


class Orchestrator:
    name = "orchestrator"

    def __init__(self, cfg, llm):
        self.cfg, self.llm = cfg, llm

    def decide(self, reports: dict) -> dict:
        score = sum(WEIGHTS[k] * reports[k]["bias"] * max(reports[k]["confidence"], 0.1) for k in WEIGHTS if k in reports)
        norm = sum(WEIGHTS[k] * max(reports[k]["confidence"], 0.1) for k in WEIGHTS if k in reports) or 1
        score /= norm
        signs = [1 if reports[k]["bias"] > 0.05 else -1 if reports[k]["bias"] < -0.05 else 0 for k in WEIGHTS if k in reports]
        agreement = abs(sum(signs)) / len(signs) if signs else 0
        action = "buy" if score > THRESHOLD else "sell" if score < -THRESHOLD else "hold"
        conf = round(min(1.0, abs(score) * 0.7 + agreement * 0.4), 2)
        decision = {"action": action, "score": round(score, 3), "confidence": conf,
                    "reason": f"weighted score {score:+.2f}, agreement {agreement:.2f}", "source": "rules"}

        brief = {k: {"bias": v["bias"], "confidence": v["confidence"], "summary": v["summary"]} for k, v in reports.items()}
        ai = self.llm.ask_json(
            "You are the head trader of a desk that trades ONLY EUR/USD. You receive reports from a technical "
            "analyst, a quant (rule votes) and a news analyst. Decide buy, sell or hold. Prefer HOLD when evidence "
            "is mixed or a major event is near. Capital preservation first.",
            "Analyst reports:\n" + json.dumps(brief, indent=1) +
            f"\nRule-based combined score: {score:+.3f} (threshold +/-{THRESHOLD}).\n"
            'Return JSON: {"action": "buy|sell|hold", "confidence": number 0..1, "reason": "max 3 sentences"}')
        if ai and ai.get("action") in ("buy", "sell", "hold"):
            decision["ai"] = ai
            if ai["action"] == "hold" or ai["action"] == action:
                decision["action"] = ai["action"]
                try:
                    decision["confidence"] = round(min(conf, float(ai.get("confidence", conf))) if ai["action"] != "hold" else conf, 2)
                except Exception:
                    pass
                decision["reason"] = ai.get("reason", decision["reason"])
                decision["source"] = "ai+rules"
            else:
                decision["action"] = "hold"
                decision["reason"] = f"AI wanted {ai['action']} but evidence score says {action}: conflict -> HOLD"
                decision["source"] = "conflict"
        return decision
