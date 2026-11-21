"""QUANT-X Phase-1 agents + Supervisor.

Each agent declares ROLE, INPUTS, OUTPUTS and VALIDATION, and returns an AgentResult.
The Supervisor plans the pipeline, enforces minimum-evidence rules and blocks unsupported conclusions
(no price forecasts, no trade signals, every claim tied to a computed statistic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ta.validate import validate
from ..analytics import corr, regime, risk, stats, vol
from ..data.store import Store, data_hash

log = logging.getLogger("quantx")
CODE_VERSION = "quantx/0.1.0-phase1"


@dataclass
class AgentResult:
    agent: str
    ok: bool
    payload: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class DataCollectionAgent:
    ROLE = "Fetch OHLCV from a read-only source and persist it with provenance"
    INPUTS = "source, symbol, timeframe, bars"
    OUTPUTS = "bars DataFrame (UTC), source metadata, ingestion run id"
    VALIDATION = "non-empty response; schema columns present; stored with data hash"

    def __init__(self, source, store: Store):
        self.source, self.store = source, store

    def run(self, symbol, timeframe, bars) -> AgentResult:
        import sys
        print(f"[data] {symbol} {timeframe} x{bars} from {self.source.name}", file=sys.stderr, flush=True)
        try:
            df, meta = self.source.fetch(symbol, timeframe, bars)
        except Exception as e:
            return AgentResult("data_collection", False, errors=[f"{symbol}: {e}"])
        if df.empty or not {"open", "high", "low", "close"} <= set(df.columns):
            return AgentResult("data_collection", False, errors=[f"{symbol}: empty or malformed data"])
        return AgentResult("data_collection", True, {"df": df, "meta": meta})


class DataValidationAgent:
    ROLE = "Check bars before any statistics (calendar-aware)"
    INPUTS = "raw bars + metadata"
    OUTPUTS = "clean completed bars, warnings, quality score 0-1"
    VALIDATION = "drops forming bar/duplicates/bad OHLC; flags gaps, outliers, staleness; blocks if quality < 0.5"
    MIN_QUALITY = 0.5

    def __init__(self, store: Store):
        self.store = store

    def run(self, df, meta) -> AgentResult:
        clean, warns, q, vmeta = validate(df, meta["timeframe"] if meta["timeframe"] != "W1" else "D1",
                                          calendar=meta.get("calendar", "fx"))
        run_id = self.store.save_bars(clean, meta, q, warns) if len(clean) else None
        ok = q >= self.MIN_QUALITY and len(clean) >= 100
        errs = [] if ok else [f"quality {q} / {len(clean)} bars below minimum (0.5 / 100)"]
        return AgentResult("data_validation", ok, {"df": clean, "quality": q, "meta": vmeta, "ingestion_run": run_id,
                                                   "data_sha256": data_hash(clean) if len(clean) else None},
                           warnings=warns, errors=errs)


class MarketAnalysisAgent:
    ROLE = "Statistical, volatility and regime analysis of one instrument"
    INPUTS = "clean bars, timeframe, calendar"
    OUTPUTS = "descriptive stats, normality, dependence, stationarity, random-walk tests, GARCH, regimes"
    VALIDATION = "minimum sample sizes per method; GARCH convergence & stationarity checked"
    MIN_N = {"garch": 500, "markov": 500, "vr": 200}

    def run(self, df, timeframe, calendar) -> AgentResult:
        ppy = stats.periods_per_year(timeframe, calendar)
        c = df["close"]
        r = stats.log_returns(c)
        out, warns = {"periods_per_year": ppy}, []
        out["descriptive"] = stats.descriptive(r, ppy)
        out["normality"] = stats.normality(r)
        out["dependence"] = stats.dependence(r)
        out["stationarity"] = stats.stationarity(c, r)
        out["hurst"] = stats.hurst(c)
        if len(r) >= self.MIN_N["vr"]:
            out["variance_ratio"] = {f"q{q}": stats.variance_ratio(c, q) for q in (2, 5, 10)}
        rv = vol.realized(r, ppy, 20)
        out["vol_cone"] = vol.cone(r, ppy)
        out["ewma_vol_ann_pct"] = round(float(vol.ewma(r, ppy).iloc[-1] * 100), 3)
        if len(r) >= self.MIN_N["garch"]:
            out["garch"] = vol.garch(r, ppy)
            if "error" not in out["garch"] and not out["garch"]["converged"]:
                warns.append("GARCH optimiser did not converge - parameters unreliable")
            if "error" not in out["garch"] and not out["garch"]["stationary"]:
                warns.append("GARCH persistence >= 1 (integrated volatility) - long-run vol undefined")
        else:
            out["garch"] = {"skipped": f"needs >= {self.MIN_N['garch']} returns, have {len(r)}"}
        out["regime"] = {"volatility": regime.vol_percentile(rv), "trend": regime.trend(c)}
        if len(r) >= self.MIN_N["markov"]:
            out["regime"]["markov"] = regime.markov_switching(r, ppy)
        else:
            out["regime"]["markov"] = {"skipped": f"needs >= {self.MIN_N['markov']} returns"}
        return AgentResult("market_analysis", True, out, warnings=warns)


class RiskAnalyticsAgent:
    ROLE = "Market-risk measurement of holding the instrument (not trade approval)"
    INPUTS = "clean bars, timeframe, calendar, optional benchmark"
    OUTPUTS = "VaR/CVaR (3 methods, 95/99%), Kupiec backtest, drawdowns, buy&hold stats, beta"
    VALIDATION = "VaR backtested out-of-sample; methods compared; Sharpe reported with confidence interval"

    def run(self, df, timeframe, calendar, bench: pd.Series | None = None) -> AgentResult:
        ppy = stats.periods_per_year(timeframe, calendar)
        c = df["close"]
        r = stats.log_returns(c)
        out = {"var_95": risk.var_cvar(r, 0.95), "var_99": risk.var_cvar(r, 0.99),
               "var_backtest_99": risk.kupiec(r, 0.99), "drawdown": risk.drawdown(c), "buy_and_hold": risk.performance(c, ppy)}
        warns = []
        v = out["var_99"]
        if v["historical_var"] and v["normal_var"] and v["historical_var"] > 1.2 * v["normal_var"]:
            warns.append("historical 99% VaR exceeds normal VaR by >20%: fat left tail, normal VaR understates risk")
        if bench is not None:
            out["beta"] = risk.beta_alpha(r, stats.log_returns(bench), ppy)
        return AgentResult("risk_analytics", True, out, warnings=warns)


class Supervisor:
    ROLE = "Plan, run and check the pipeline; block unsupported conclusions; record every run"

    def __init__(self, source, store: Store):
        self.source, self.store = source, store
        self.collect = DataCollectionAgent(source, store)
        self.validate = DataValidationAgent(store)
        self.market = MarketAnalysisAgent()
        self.risk = RiskAnalyticsAgent()

    def _load(self, symbol, tf, bars):
        trace = []
        c = self.collect.run(symbol, tf, bars)
        trace.append(c)
        if not c.ok:
            return None, None, trace
        v = self.validate.run(c.payload["df"], c.payload["meta"])
        trace.append(v)
        return (v.payload["df"], {**c.payload["meta"], **v.payload["meta"], "quality": v.payload["quality"],
                                  "data_sha256": v.payload["data_sha256"], "ingestion_run": v.payload["ingestion_run"],
                                  "validation_warnings": v.warnings}) if v.ok else (None, None), trace

    def analyze(self, symbol: str, timeframe: str = "D1", bars: int = 2000) -> dict:
        req = {"kind": "single", "symbol": symbol, "timeframe": timeframe, "bars": bars, "source": self.source.name}
        (df, meta), trace = self._load(symbol, timeframe, bars)
        if df is None:
            return {"status": "failed", "request": req, "errors": sum((t.errors for t in trace), [])}
        cal = meta.get("calendar", "fx")
        m = self.market.run(df, timeframe, cal)
        rk = self.risk.run(df, timeframe, cal)
        checks = self._checks(m.payload, rk.payload, len(df))
        res = {"status": "ok", "request": req, "code_version": CODE_VERSION,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "data": {k: v for k, v in meta.items()}, "last_close": float(df["close"].iloc[-1]),
               "last_bar_open_utc": str(df.index[-1]), "market": m.payload, "risk": rk.payload,
               "warnings": meta["validation_warnings"] + m.warnings + rk.warnings, "supervisor_checks": checks,
               "findings": self._findings(m.payload, rk.payload)}
        res["analysis_run"] = self.store.save_analysis("single", req, {symbol: meta["data_sha256"]}, CODE_VERSION, res)
        return res

    def compare(self, symbols: list[str], timeframe: str = "D1", bars: int = 2000) -> dict:
        req = {"kind": "compare", "symbols": symbols, "timeframe": timeframe, "bars": bars, "source": self.source.name}
        closes, rows, errors, ids = {}, {}, [], {}
        for s in symbols:
            (df, meta), trace = self._load(s, timeframe, bars)
            if df is None:
                errors += sum((t.errors for t in trace), [])
                continue
            cal = meta["calendar"]
            ppy = stats.periods_per_year(timeframe, cal)
            r = stats.log_returns(df["close"])
            d = stats.descriptive(r, ppy)
            v99 = risk.var_cvar(r, 0.99)
            rows[s] = {"n": d["n"], "ann_vol_pct": d["ann_vol_pct"], "skew": d["skew"], "excess_kurtosis": d["excess_kurtosis"],
                       "hist_var99_pct": round(v99["historical_var"] * 100, 3), "hist_cvar99_pct": round(v99["historical_cvar"] * 100, 3),
                       "max_dd_pct": risk.drawdown(df["close"])["max_drawdown_pct"],
                       "vol_regime": regime.vol_percentile(vol.realized(r, ppy)).get("label"),
                       "hurst": stats.hurst(df["close"])["H"], "quality": meta["quality"]}
            closes[s] = df["close"]
            ids[s] = meta["data_sha256"]
        res = {"status": "ok" if closes else "failed", "request": req, "code_version": CODE_VERSION,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "summary": rows, "errors": errors,
               "correlation": corr.correlation(closes, daily=timeframe in ("D1", "W1")) if len(closes) > 1 else {"error": "need >= 2 symbols"}}
        res["analysis_run"] = self.store.save_analysis("compare", req, ids, CODE_VERSION, res)
        return res

    # ---------------------------------------------------------- guards
    def _checks(self, m, rk, n) -> list[str]:
        c = [f"sample size: {n} bars", "multiple testing: ~14 hypothesis tests run; at 5% each, ~0.7 false rejections expected by chance"]
        g = m.get("garch", {})
        if "persistence" in g:
            c.append(f"GARCH: converged={g['converged']}, persistence={g['persistence']}")
        vb = rk.get("var_backtest_99", {})
        if "conclusion" in vb:
            c.append(f"VaR backtest: {vb['exceptions']} exceptions vs {vb['expected']} expected -> {vb['conclusion']}")
        c.append("no price forecasts or trade signals are produced by Phase-1 agents")
        return c

    def _findings(self, m, rk) -> list[str]:
        """Every finding cites the statistic it rests on."""
        f = []
        d, nm = m["descriptive"], m["normality"]
        f.append(f"Annualised volatility {d['ann_vol_pct']}% (sample std of log returns x sqrt({d['periods_per_year']:.0f})).")
        f.append(f"Return distribution: skew {d['skew']}, excess kurtosis {d['excess_kurtosis']}; "
                 f"Jarque-Bera p={nm['p_value']} -> {nm['conclusion']}.")
        dep = m["dependence"]
        f.append(f"Linear autocorrelation {dep['interpretation']['linear_dependence']} "
                 f"(Ljung-Box lag10 p={dep['ljung_box_returns']['lag10']['p_value']}); volatility clustering "
                 f"{dep['interpretation']['volatility_clustering']} (ARCH-LM p={dep['arch_lm_10']['p_value']}).")
        vr = m.get("variance_ratio", {}).get("q10")
        if vr:
            f.append(f"Variance ratio VR(10)={vr['vr']} (robust z={vr['z_robust']}, p={vr['p_value']}) -> {vr['conclusion']}.")
        g = m.get("garch", {})
        if "selected" in g:
            f.append(f"{g['selected']} (BIC-selected): persistence {g['persistence']}, half-life {g['half_life_bars']} bars, "
                     f"current conditional vol {g['current_cond_vol_ann_pct']}% vs long-run {g['long_run_vol_ann_pct']}%.")
        mk = m["regime"].get("markov", {})
        if "current_regime" in mk:
            f.append(f"Markov-switching model: current regime {mk['current_regime']} "
                     f"(filtered probability {mk['regimes'][mk['current_regime']]['current_filtered_prob']}).")
        v = rk["var_99"]
        f.append(f"1-bar 99% VaR: historical {v['historical_var']*100:.3f}%, normal {v['normal_var']*100:.3f}%, "
                 f"Cornish-Fisher {v['cornish_fisher_var']*100:.3f}%; CVaR (hist) {v['historical_cvar']*100:.3f}%.")
        dd = rk["drawdown"]
        f.append(f"Max drawdown {dd['max_drawdown_pct']}% ({dd['peak_utc'][:10]} -> {dd['trough_utc'][:10]}); current {dd['current_drawdown_pct']}%.")
        return f
