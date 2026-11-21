"""Broker server-time -> UTC offset for MetaTrader 5 (shared by trading agents and QUANT-X).

MT5 stamps ticks and bars in the broker's SERVER time. Comparing the last tick with 'now' only works
while the market is trading - on weekends the last tick is from Friday's close and gives nonsense.

Order of methods:
 1. config override (mt5.server_utc_offset_hours)
 2. market open and tick fresh -> live detection; result cached to data/server_offset.json
 3. market closed -> cached value from an earlier live detection
 4. no cache -> infer from Friday's FX close (17:00 America/New_York) vs the last tick time
 5. otherwise 0 with a warning
Any result outside [-12h, +14h] is rejected.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("server_time")
CACHE = Path(__file__).resolve().parents[1] / "data" / "server_offset.json"
NY = ZoneInfo("America/New_York")


def _plausible(sec: float) -> bool:
    return -12 * 3600 <= sec <= 14 * 3600


def fx_market_open(now: datetime) -> bool:
    wd, h = now.weekday(), now.hour
    if wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22):
        return False
    return True


def _read_cache(server: str):
    try:
        return json.loads(CACHE.read_text()).get(server)
    except Exception:
        return None


def _write_cache(server: str, sec: float):
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        d = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        d[server] = {"offset_seconds": sec, "detected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        CACHE.write_text(json.dumps(d, indent=1))
    except Exception as e:
        log.debug("cache write failed: %s", e)


def infer_offset(tick_time_server: int | None, server: str, override=None, now: datetime | None = None) -> tuple[float, str]:
    """Returns (offset_seconds, method)."""
    now = now or datetime.now(timezone.utc)
    if override not in (None, ""):
        return float(override) * 3600, "config override"
    if tick_time_server:
        diff = tick_time_server - now.timestamp()
        off = round(diff / 1800) * 1800
        if fx_market_open(now) and abs(diff - off) <= 300 and _plausible(off):
            _write_cache(server, off)
            return off, "live tick"
    cached = _read_cache(server)
    if cached and _plausible(cached["offset_seconds"]):
        return cached["offset_seconds"], f"cached (detected {cached['detected_at_utc']})"
    if tick_time_server:
        # last tick (server clock) should be just before Friday 17:00 New York (FX weekly close)
        t_server = datetime.fromtimestamp(tick_time_server, tz=timezone.utc)  # naive server clock read as UTC
        fri = t_server.date() - timedelta(days=(t_server.weekday() - 4) % 7)
        close_utc = datetime(fri.year, fri.month, fri.day, 17, tzinfo=NY).astimezone(timezone.utc)
        off = round((t_server - close_utc).total_seconds() / 3600) * 3600
        if _plausible(off) and t_server.weekday() in (4, 5):
            return off, "inferred from Friday FX close (17:00 New York)"
    log.warning("Broker server offset unknown; assuming UTC. Set mt5.server_utc_offset_hours in config.yaml")
    return 0.0, "unknown (assumed 0)"
