"""News agent: economic calendar (EUR + USD) + headlines.

- Finds high-impact events (ECB, Fed, NFP, CPI...) and sets a BLACKOUT around them.
- Reads RSS headlines that mention EUR/USD drivers; AI (if enabled) judges sentiment.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("news")
UA = {"User-Agent": "Mozilla/5.0 (eurusd-agents)"}
KEYWORDS = re.compile(r"\b(EUR|USD|euro|dollar|ECB|Fed|FOMC|Lagarde|Powell|inflation|CPI|NFP|payrolls|"
                      r"PMI|GDP|rate|yield|tariff|treasur)", re.I)
BULL_EUR = re.compile(r"(euro (rises|gains|jumps|climbs)|ECB.*(hawkish|hike)|dollar (falls|slides|weakens|drops)|"
                      r"Fed.*(dovish|cut)|weak US|EUR/USD (rises|gains|rallies))", re.I)
BEAR_EUR = re.compile(r"(euro (falls|slides|drops|weakens)|ECB.*(dovish|cut)|dollar (rises|gains|jumps|strengthens)|"
                      r"Fed.*(hawkish|hike)|strong US|EUR/USD (falls|drops|slides))", re.I)


class NewsAgent:
    name = "news"

    def __init__(self, cfg, market=None, llm=None):
        self.cfg, self.llm = cfg, llm
        self.ncfg = cfg.get("news", {})

    def calendar_strict(self) -> list[dict]:
        """Like calendar() but raises on failure, so callers can say 'events NOT checked'."""
        r = requests.get(self.ncfg["calendar_url"], headers=UA, timeout=20)
        r.raise_for_status()
        cur = set(self.ncfg.get("currencies", ["EUR", "USD"]))
        return [{"time": datetime.fromisoformat(e["date"]).astimezone(timezone.utc), "currency": e["country"],
                 "title": e["title"], "impact": e["impact"], "forecast": e.get("forecast", ""),
                 "previous": e.get("previous", "")}
                for e in r.json() if e.get("country") in cur and e.get("impact") in ("High", "Medium")]

    def calendar(self) -> list[dict]:
        try:
            r = requests.get(self.ncfg["calendar_url"], headers=UA, timeout=20)
            r.raise_for_status()
            cur = set(self.ncfg.get("currencies", ["EUR", "USD"]))
            out = []
            for e in r.json():
                if e.get("country") in cur and e.get("impact") in ("High", "Medium"):
                    out.append({"time": datetime.fromisoformat(e["date"]).astimezone(timezone.utc),
                                "currency": e["country"], "title": e["title"], "impact": e["impact"],
                                "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
            return out
        except Exception as ex:
            log.warning("Calendar unavailable: %s", ex)
            return []

    def headlines(self) -> list[str]:
        items = []
        for url in self.ncfg.get("rss_feeds", []):
            try:
                r = requests.get(url, headers=UA, timeout=15)
                root = ET.fromstring(r.content)
                for it in root.iter():
                    if it.tag.endswith("item") or it.tag.endswith("entry"):
                        t = next((c.text for c in it if c.tag.endswith("title")), None)
                        if t and KEYWORDS.search(t):
                            items.append(t.strip())
            except Exception as ex:
                log.debug("RSS failed %s: %s", url, ex)
        seen, uniq = set(), []
        for h in items:
            if h not in seen:
                seen.add(h); uniq.append(h)
        return uniq[: self.ncfg.get("max_headlines", 25)]

    def run(self) -> dict:
        if not self.ncfg.get("enabled", True):
            return {"agent": self.name, "bias": 0, "confidence": 0, "summary": "news disabled",
                    "details": {"blackout": False}}
        now = datetime.now(timezone.utc)
        before = timedelta(minutes=self.ncfg.get("blackout_minutes_before", 30))
        after = timedelta(minutes=self.ncfg.get("blackout_minutes_after", 30))
        events = self.calendar()
        high = [e for e in events if e["impact"] == "High"]
        blocking = [e for e in high if e["time"] - before <= now <= e["time"] + after]
        upcoming = sorted([e for e in events if now <= e["time"] <= now + timedelta(hours=24)], key=lambda e: e["time"])
        heads = self.headlines()

        bull = sum(bool(BULL_EUR.search(h)) for h in heads)
        bear = sum(bool(BEAR_EUR.search(h)) for h in heads)
        bias = (bull - bear) / max(bull + bear, 1) * min((bull + bear) / 5, 1)
        conf = 0.2 + 0.3 * min((bull + bear) / 5, 1)
        summary = f"{len(heads)} relevant headlines (bull EUR {bull} / bear EUR {bear})."

        ev_txt = [f"{e['time']:%a %H:%M} UTC {e['currency']} {e['title']} ({e['impact']}) f:{e['forecast']} p:{e['previous']}"
                  for e in upcoming[:12]]
        ai = self.llm.ask_json(
            "You are a forex fundamental/news analyst for EUR/USD only. Positive bias = EUR/USD likely UP "
            "(euro stronger or dollar weaker). Be cautious; if news is mixed or old, keep bias near 0.",
            f"UTC now: {now:%Y-%m-%d %H:%M}\nUpcoming EUR/USD events (24h):\n" + "\n".join(ev_txt or ["none"]) +
            "\n\nHeadlines:\n" + "\n".join(f"- {h}" for h in heads or ["none"]) +
            '\n\nReturn JSON: {"bias": number -1..1, "confidence": number 0..1, "summary": "max 3 sentences", '
            '"key_risks": ["..."]}') if self.llm else None
        report = {"agent": self.name, "bias": round(bias, 3), "confidence": round(conf, 2), "summary": summary,
                  "details": {"blackout": bool(blocking),
                              "blocking_events": [f"{e['currency']} {e['title']} @ {e['time']:%H:%M} UTC" for e in blocking],
                              "upcoming": ev_txt, "headlines": heads[:10]}}
        if ai and "bias" in ai:
            try:
                report["bias"] = round(max(-1, min(1, float(ai["bias"]))), 3)
                report["confidence"] = round(max(0, min(1, float(ai.get("confidence", 0.4)))), 2)
            except Exception:
                pass
            report["summary"] = ai.get("summary", summary)
            report["ai"] = ai
        if blocking:
            report["summary"] = "BLACKOUT: " + "; ".join(report["details"]["blocking_events"]) + ". " + report["summary"]
        return report
