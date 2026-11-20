"""MetaTrader 5 connection, hard-locked to ONE symbol (EURUSD).

Every function refuses any other symbol, so no agent can read or trade other charts.
If MT5 isn't available and use_mock_if_unavailable is true, synthetic data is used
(for testing the pipeline only - mock mode can never send orders).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("market")

try:
    import MetaTrader5 as mt5  # Windows only
except Exception:  # pragma: no cover
    mt5 = None

TF_NAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


class SymbolLockError(Exception):
    pass


class Market:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.symbol = cfg["symbol"].upper()
        self.mcfg = cfg.get("mt5", {})
        self.mock = False
        self.connected = False
        self.server_offset = pd.Timedelta(0)

    # ---------- guard ----------
    def _check(self, symbol: str | None):
        if symbol is not None and symbol.upper() != self.symbol:
            raise SymbolLockError(f"Blocked: only {self.symbol} is allowed, got {symbol}")

    # ---------- connection ----------
    def connect(self) -> bool:
        if mt5 is None:
            return self._fallback("MetaTrader5 package not installed / not Windows")
        kw = {}
        if self.mcfg.get("terminal_path"):
            kw["path"] = self.mcfg["terminal_path"]
        if self.mcfg.get("login"):
            kw.update(login=int(self.mcfg["login"]), password=str(self.mcfg.get("password", "")),
                      server=str(self.mcfg.get("server", "")))
        if not mt5.initialize(**kw):
            return self._fallback(f"mt5.initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            return self._fallback(f"Symbol {self.symbol} not available at this broker")
        self.connected = True
        self.server_offset = self._detect_server_offset()
        acc = mt5.account_info()
        log.info("Connected to MT5: %s | %s | balance %.2f %s", acc.server, self.account_type(), acc.balance, acc.currency)
        return True

    def _detect_server_offset(self) -> pd.Timedelta:
        """MT5 timestamps are broker SERVER time. Offset = server - UTC, rounded to 30 min,
        estimated from the latest tick (only reliable while the market is trading)."""
        override = self.mcfg.get("server_utc_offset_hours")
        if override not in (None, ""):
            return pd.Timedelta(hours=float(override))
        t = mt5.symbol_info_tick(self.symbol)
        if t is None or not t.time:
            return pd.Timedelta(0)
        diff = t.time - datetime.now(timezone.utc).timestamp()
        off = round(diff / 1800) * 1800
        if abs(diff - off) > 600:  # tick is old (weekend) -> can't infer reliably
            log.warning("Could not infer broker server offset (old tick); set mt5.server_utc_offset_hours")
            return pd.Timedelta(0)
        log.info("Broker server time offset: UTC%+.1f h", off / 3600)
        return pd.Timedelta(seconds=off)

    def _fallback(self, why: str) -> bool:
        if self.mcfg.get("use_mock_if_unavailable", True):
            log.warning("%s -> using MOCK data (orders disabled)", why)
            self.mock = True
            return True
        log.error(why)
        return False

    def shutdown(self):
        if self.connected and mt5:
            mt5.shutdown()

    # ---------- data ----------
    def rates(self, timeframe: str, bars: int, symbol: str | None = None) -> pd.DataFrame:
        self._check(symbol)
        if self.mock:
            return _mock_rates(timeframe, bars)
        tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
        r = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)
        if r is None or len(r) == 0:
            raise RuntimeError(f"No rates: {mt5.last_error()}")
        df = pd.DataFrame(r)
        # MT5 'time' is broker server time -> convert to true UTC
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - self.server_offset
        return df.set_index("time")[["open", "high", "low", "close", "tick_volume", "spread"]]

    def tick(self, symbol: str | None = None) -> dict:
        self._check(symbol)
        if self.mock:
            return {"bid": 1.08500, "ask": 1.08508, "spread_points": 8}
        t = mt5.symbol_info_tick(self.symbol)
        info = mt5.symbol_info(self.symbol)
        return {"bid": t.bid, "ask": t.ask, "spread_points": int(round((t.ask - t.bid) / info.point))}

    def symbol_info(self) -> dict:
        if self.mock:
            return {"point": 0.00001, "digits": 5, "tick_value": 1.0, "tick_size": 0.00001,
                    "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01, "stops_level": 0}
        i = mt5.symbol_info(self.symbol)
        return {"point": i.point, "digits": i.digits, "tick_value": i.trade_tick_value,
                "tick_size": i.trade_tick_size, "volume_min": i.volume_min, "volume_max": i.volume_max,
                "volume_step": i.volume_step, "stops_level": i.trade_stops_level}

    def account(self) -> dict:
        if self.mock:
            return {"balance": 10000.0, "equity": 10000.0, "currency": "USD", "type": "MOCK", "server": "mock"}
        a = mt5.account_info()
        return {"balance": a.balance, "equity": a.equity, "currency": a.currency,
                "type": self.account_type(), "server": a.server, "login": a.login}

    def account_type(self) -> str:
        if self.mock:
            return "MOCK"
        m = mt5.account_info().trade_mode
        return {mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO", mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
                mt5.ACCOUNT_TRADE_MODE_REAL: "REAL"}.get(m, "UNKNOWN")

    def positions(self) -> list[dict]:
        if self.mock:
            return []
        ps = mt5.positions_get(symbol=self.symbol) or []
        return [{"ticket": p.ticket, "type": "buy" if p.type == 0 else "sell", "volume": p.volume,
                 "price": p.price_open, "sl": p.sl, "tp": p.tp, "profit": p.profit, "magic": p.magic} for p in ps]

    def today_closed_pnl(self) -> float:
        if self.mock:
            return 0.0
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(start, now) or []
        return float(sum(d.profit + d.commission + d.swap for d in deals if d.symbol == self.symbol))


def _mock_rates(timeframe: str, bars: int) -> pd.DataFrame:
    minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[timeframe]
    rng = np.random.default_rng(42 + minutes)
    rets = rng.normal(0.00002, 0.0009 * np.sqrt(minutes / 60), bars)
    close = 1.085 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0006 * np.sqrt(minutes / 60), bars))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("min"), periods=bars, freq=f"{minutes}min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(100, 3000, bars), "spread": 8}, index=idx)
