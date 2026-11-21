"""Data Collection layer: READ-ONLY market-data adapters with provenance.

Every fetch returns (DataFrame, meta). DataFrame: UTC DatetimeIndex (bar open), columns
open/high/low/close/volume[/spread]. meta records source, price type, volume meaning,
retrieval time and how timestamps were converted - so every later number is traceable.

Adapters
- MT5Source     : any symbol your broker offers (forex, metals, indices, CFDs). Bars are BID,
                  volume = broker tick count. Broker server time is converted to UTC.
- BinanceSource : public spot klines (no API key). Volume = base-asset traded volume on Binance only.
- CSVSource     : your own files (columns time,open,high,low,close[,volume]).
None of these adapters can place orders.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("quantx.data")

MT5_TF = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
          "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1"}
BINANCE_TF = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w"}


def _now():
    return datetime.now(timezone.utc)


def _with_timeout(fn, seconds: float, msg: str):
    """MT5 blocks while it downloads missing history (can take minutes the first time)."""
    import sys
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
    t0 = datetime.now()
    th.start()
    th.join(5)
    if th.is_alive():
        print(f"  ...waiting for MT5 history download ({msg.split(':')[0]})", file=sys.stderr, flush=True)
        th.join(max(0, seconds - 5))
    if th.is_alive():
        raise TimeoutError(f"{msg} (waited {int((datetime.now() - t0).total_seconds())}s)")
    return box.get("v")


class MT5Source:
    name = "mt5"
    calendar = "fx"

    def __init__(self, terminal_path: str = "", server_utc_offset_hours=None, timeout_s: float = 240):
        self.path = terminal_path
        self.timeout_s = timeout_s
        self.override = server_utc_offset_hours
        self.mt5 = None
        self.offset = pd.Timedelta(0)
        self.offset_method = "not detected"
        self._offset_done = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except Exception:
            log.error("MetaTrader5 package unavailable (Windows only)")
            return False
        ok = mt5.initialize(path=self.path) if self.path else mt5.initialize()
        if not ok:
            log.error("mt5.initialize failed: %s", mt5.last_error())
            return False
        self.mt5 = mt5
        self.server = mt5.account_info().server
        return True

    def _offset_for(self, symbol: str) -> pd.Timedelta:
        """Offset is a property of the broker SERVER, not the symbol: detect once per session,
        preferably from EURUSD (FX weekly close is well defined), so all symbols share one clock."""
        if self._offset_done:
            return self.offset
        from agents.server_time import infer_offset
        ref = "EURUSD" if self.mt5.symbol_info("EURUSD") is not None else symbol
        self.mt5.symbol_select(ref, True)
        t = self.mt5.symbol_info_tick(ref)
        sec, self.offset_method = infer_offset(t.time if t else None, self.server, self.override)
        self.offset_method += f" [ref {ref}]"
        self.offset, self._offset_done = pd.Timedelta(seconds=sec), True
        return self.offset

    def asset_class(self, symbol: str) -> str:
        i = self.mt5.symbol_info(symbol)
        path = (i.path if i else "").lower()
        for key, cls in (("forex", "forex"), ("crypto", "crypto"), ("metal", "metals"), ("index", "index"),
                         ("indices", "index"), ("stock", "equity"), ("energ", "commodity"), ("commod", "commodity")):
            if key in path:
                return cls
        return "forex" if len(symbol) == 6 and symbol.isalpha() else "other"

    def fetch(self, symbol: str, timeframe: str, bars: int):
        mt5 = self.mt5
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"{symbol} not offered by broker {self.server}")
        r = _with_timeout(lambda: mt5.copy_rates_from_pos(symbol, getattr(mt5, MT5_TF[timeframe]), 0, bars),
                          self.timeout_s, f"{symbol} {timeframe}: MT5 is still downloading history. Open its chart in MT5 "
                                          f"once (or retry in a few minutes)")
        if r is None or len(r) == 0:
            raise RuntimeError(f"no rates for {symbol}: {mt5.last_error()}")
        off = self._offset_for(symbol)
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) - off
        df = df.set_index("time").rename(columns={"tick_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume", "spread"]].astype(float)
        meta = {"source": "MetaTrader 5", "server": self.server, "symbol": symbol, "timeframe": timeframe,
                "price_type": "bid", "volume_type": "broker tick count (not market volume)",
                "server_utc_offset_hours": off.total_seconds() / 3600, "offset_method": self.offset_method,
                "calendar": self.calendar,
                "asset_class": self.asset_class(symbol), "retrieved_at_utc": _now().isoformat(timespec="seconds")}
        return df, meta

    def close(self):
        if self.mt5:
            self.mt5.shutdown()


class BinanceSource:
    name = "binance"
    calendar = "24x7"
    BASES = ["https://api.binance.com", "https://data-api.binance.vision"]  # public market-data endpoints

    def connect(self) -> bool:
        return True

    def fetch(self, symbol: str, timeframe: str, bars: int):
        symbol = symbol.upper().replace("/", "")
        rows, end, err = [], None, None
        for base in self.BASES:
            try:
                rows, end = [], None
                while len(rows) < bars:
                    params = {"symbol": symbol, "interval": BINANCE_TF[timeframe], "limit": min(1000, bars - len(rows))}
                    if end:
                        params["endTime"] = end
                    r = requests.get(f"{base}/api/v3/klines", params=params, timeout=20)
                    r.raise_for_status()
                    chunk = r.json()
                    if not chunk:
                        break
                    rows = chunk + rows
                    end = chunk[0][0] - 1
                    if len(chunk) < params["limit"]:
                        break
                err = None
                break
            except Exception as e:
                err = e
        if err or not rows:
            raise RuntimeError(f"Binance fetch failed for {symbol}: {err}")
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "i"])
        df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        meta = {"source": "Binance public spot klines", "server": base, "symbol": symbol, "timeframe": timeframe,
                "price_type": "last trade (OHLC of trades)", "volume_type": "base-asset volume on Binance only",
                "server_utc_offset_hours": 0.0, "calendar": self.calendar, "asset_class": "crypto",
                "retrieved_at_utc": _now().isoformat(timespec="seconds")}
        return df, meta

    def close(self):
        pass


class CSVSource:
    name = "csv"

    def __init__(self, path: str, calendar: str = "fx", asset_class: str = "other"):
        self.path, self.calendar, self.cls = path, calendar, asset_class

    def connect(self) -> bool:
        return True

    def fetch(self, symbol: str, timeframe: str, bars: int):
        df = pd.read_csv(self.path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        if "volume" not in df:
            df["volume"] = np.nan
        df = df[["open", "high", "low", "close", "volume"]].astype(float).iloc[-bars:]
        return df, {"source": f"CSV {self.path}", "server": None, "symbol": symbol, "timeframe": timeframe,
                    "price_type": "as provided", "volume_type": "as provided", "server_utc_offset_hours": 0.0,
                    "calendar": self.calendar, "asset_class": self.cls, "retrieved_at_utc": _now().isoformat(timespec="seconds")}

    def close(self):
        pass


def make_source(name: str, cfg: dict):
    m = cfg.get("mt5", {})
    if name == "mt5":
        return MT5Source(m.get("terminal_path", ""), m.get("server_utc_offset_hours"))
    if name == "binance":
        return BinanceSource()
    raise ValueError(f"unknown source {name}")
