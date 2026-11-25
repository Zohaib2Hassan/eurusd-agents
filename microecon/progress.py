"""Local progress log (data/microecon_progress.jsonl - stays on your machine, git-ignored)."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .grader import tier

LOG = Path(__file__).resolve().parents[1] / "data" / "microecon_progress.jsonl"


def record(topic_key: str, title: str, seed, score: float, hints: int, revealed: bool, path: Path | None = None):
    path = path or LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "topic": topic_key,
                            "title": title, "seed": seed, "score": round(score, 1), "hints": hints, "revealed": revealed}) + "\n")


def summary(path: Path | None = None, last_n: int = 5) -> dict:
    path = path or LOG
    if not path.exists():
        return {}
    by = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["topic"]].append(r)
    out = {}
    for k, rows in by.items():
        recent = [r["score"] for r in rows[-last_n:]]
        avg = sum(recent) / len(recent)
        out[k] = {"title": rows[-1]["title"], "attempts": len(rows), "recent_avg": round(avg, 1), "tier": tier(avg),
                  "best": max(r["score"] for r in rows), "last": rows[-1]["time_utc"][:16]}
    return out


def render(path: Path | None = None) -> str:
    s = summary(path)
    if not s:
        return "No practice recorded yet. Try: python micro_cli.py practice --topic tax"
    rows = sorted(s.items(), key=lambda kv: kv[1]["recent_avg"])
    L = ["| Topic | Attempts | Recent avg | Best | Tier | Last |", "|---|---|---|---|---|---|"]
    L += [f"| {v['title']} | {v['attempts']} | {v['recent_avg']} | {v['best']} | {v['tier']} | {v['last']} |" for _, v in rows]
    weakest = rows[0]
    L += ["", f"Suggested next: **{weakest[1]['title']}** (recent average {weakest[1]['recent_avg']}%)"
          + (f" -> python micro_cli.py practice --topic {weakest[0]}" if not weakest[0].startswith("bank") else "")]
    return "\n".join(L)
