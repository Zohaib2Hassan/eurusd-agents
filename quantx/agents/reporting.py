"""Reporting Agent: renders the QUANT-X standard 15-section report (spec section 15)."""
from __future__ import annotations

REFERENCES = [
    "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3).",
    "Cornish, E. A. & Fisher, R. A. (1938). Moments and cumulants in the specification of distributions. Revue de l'Institut International de Statistique, 5.",
    "Dickey, D. A. & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. JASA, 74.",
    "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. Econometrica, 50(4).",
    "Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance, 48(5).",
    "Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2).",
    "Jarque, C. M. & Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence. Economics Letters, 6(3).",
    "J.P. Morgan / Reuters (1996). RiskMetrics - Technical Document, 4th ed.",
    "Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives, 3(2).",
    "Kwiatkowski, D., Phillips, P., Schmidt, P. & Shin, Y. (1992). Testing the null hypothesis of stationarity. Journal of Econometrics, 54.",
    "Ljung, G. M. & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2).",
    "Lo, A. W. & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. Review of Financial Studies, 1(1).",
    "Lo, A. W. (2002). The statistics of Sharpe ratios. Financial Analysts Journal, 58(4).",
]


def _kv(d: dict, keys=None) -> str:
    keys = keys or d.keys()
    return ", ".join(f"{k} {d.get(k)}" for k in keys if k in d)


def single(res: dict) -> str:
    if res.get("status") != "ok":
        return "# QUANT-X analysis FAILED\n\n" + "\n".join(f"- {e}" for e in res.get("errors", []))
    rq, d, m, rk = res["request"], res["data"], res["market"], res["risk"]
    ds, g, rg = m["descriptive"], m.get("garch", {}), m["regime"]
    L = [f"# QUANT-X quantitative report: {rq['symbol']} ({rq['timeframe']})",
         f"_Run {res['analysis_run']} - {res['generated_at_utc']} UTC - {res['code_version']}_", "",
         "## 1. Executive summary", *[f"- {x}" for x in res["findings"][:4]],
         f"- Current volatility regime: **{rg['volatility']['label']}** ({rg['volatility']['percentile']}th percentile); "
         f"trend filter: **{rg['trend']['label']}**.", "- Descriptive research only: no forecast, signal or advice.", "",
         "## 2. Research objective",
         f"Characterise the return distribution, dependence structure, volatility dynamics, regimes and market risk of {rq['symbol']}.", "",
         "## 3. Dataset and data sources",
         f"- Source: {d['source']} ({d.get('server')}); asset class {d.get('asset_class')}; calendar {d.get('calendar')}",
         f"- Price type: {d.get('price_type')}; volume: {d.get('volume_type')}",
         f"- {ds['n']} log returns from {ds['start_utc']} to {ds['end_utc']}; last close {round(res['last_close'], 6)} (bar opened {res['last_bar_open_utc']} UTC)",
         f"- Timestamps converted to UTC (server offset {d.get('server_utc_offset_hours'):+.1f} h, method: {d.get('offset_method', 'n/a')}); retrieved {d['retrieved_at_utc']}",
         f"- Data hash {d['data_sha256']}, ingestion run {d['ingestion_run']}", "",
         "## 4. Data quality assessment", f"- Quality score {d['quality']} / 1.0",
         *([f"- {w}" for w in res["warnings"]] or ["- no issues detected"]), "",
         "## 5. Mathematical methodology",
         "- Log returns l_t = ln(P_t/P_{t-1}); sample moments (ddof=1); annualisation x periods/year "
         f"({ds['periods_per_year']:.0f}).",
         "- Tests: Jarque-Bera (normality), Ljung-Box on l_t and l_t^2, Engle ARCH-LM, ADF and KPSS (stationarity), "
         "Lo-MacKinlay variance ratio with heteroskedasticity-robust z*, Hurst scaling (descriptive).",
         "- Volatility: rolling realised, RiskMetrics EWMA (lambda 0.94), GARCH(1,1) and GJR-GARCH(1,1) with Student-t errors, BIC selection.",
         "- Regimes: 2-state Markov-switching mean/variance (filtered probabilities), volatility percentile, regression trend filter.",
         "- Risk: VaR/CVaR historical, normal, Cornish-Fisher; rolling out-of-sample VaR backtest (Kupiec POF); drawdowns.", "",
         "## 6. Statistical analysis",
         f"- Moments: {_kv(ds, ['ann_mean_log_pct', 'ann_vol_pct', 'skew', 'excess_kurtosis', 'pct_positive'])}",
         f"- Extremes: worst {ds['min']} ({ds['min_time_utc'][:16]}), best {ds['max']} ({ds['max_time_utc'][:16]})",
         f"- Normality: {m['normality']['test']} stat {m['normality']['stat']}, p {m['normality']['p_value']} -> {m['normality']['conclusion']}",
         f"- ACF: {m['dependence']['acf_returns']} (95% band +/-{m['dependence']['acf_95pct_band']})",
         f"- Ljung-Box returns: {m['dependence']['ljung_box_returns']}; squared: {m['dependence']['ljung_box_squared']}",
         f"- Stationarity: ADF log-price p {m['stationarity']['adf_log_price']['p_value']} ({m['stationarity']['adf_log_price']['conclusion']}); "
         f"ADF returns p {m['stationarity']['adf_returns']['p_value']}; KPSS returns p {m['stationarity']['kpss_returns']['p_value_bounded']} "
         f"({m['stationarity']['kpss_returns']['conclusion']})"]
    for k, v in m.get("variance_ratio", {}).items():
        L.append(f"- Variance ratio {k}: VR {v['vr']}, z* {v['z_robust']}, p {v['p_value']} -> {v['conclusion']}")
    L.append(f"- Hurst ({m['hurst']['estimator']}): H {m['hurst']['H']} +/- {m['hurst']['std_err']} (lags {m['hurst']['lags']})")
    L += ["", "## 7. Model architecture"]
    if "selected" in g:
        L += [f"- Volatility model: **{g['selected']}** ({g['selection']}; BIC {g['bic']})",
              f"- Parameters: {g['params']}", f"- p-values: {g['pvalues']}",
              f"- Persistence {g['persistence']}, half-life {g['half_life_bars']} bars, leverage effect: {g.get('leverage_effect')}",
              f"- Conditional vol now {g['current_cond_vol_ann_pct']}%, long-run {g['long_run_vol_ann_pct']}%, "
              f"forecast {g['forecast_vol_ann_pct']} (annualised %, model-conditional)"]
    else:
        L.append(f"- GARCH: {g}")
    mk = rg.get("markov", {})
    if "regimes" in mk:
        L.append(f"- {mk['model']} (BIC {mk['bic']}); current regime **{mk['current_regime']}**")
        for name, v in mk["regimes"].items():
            L.append(f"  - {name}: vol {v['vol_ann_pct']}%, mean {v['mean_ann_pct']}%/yr, stay prob {v['stay_probability']}, "
                     f"expected duration {v['expected_duration_bars']} bars, {v['share_of_sample_pct']}% of sample, P(now) {v['current_filtered_prob']}")
    else:
        L.append(f"- Markov-switching: {mk}")
    L += ["", "## 8. Experimental design",
          "- Descriptive statistics, tests and model fits are IN-SAMPLE on the full window.",
          f"- VaR is evaluated OUT-OF-SAMPLE: each day's 99% VaR uses only the previous {rk['var_backtest_99'].get('window')} returns.",
          "- Current regime uses filtered (real-time) probabilities; regime shares use smoothed probabilities (in-sample).", "",
          "## 9. Performance metrics (buy-and-hold long, not a strategy)",
          f"- {_kv(rk['buy_and_hold'], ['years', 'cagr_pct', 'ann_vol_pct', 'sharpe', 'sharpe_95ci', 'sortino', 'calmar'])}", "",
          "## 10. Risk assessment"]
    for lvl in ("var_95", "var_99"):
        v = rk[lvl]
        L.append(f"- {int(v['level']*100)}% 1-bar: VaR hist {v['historical_var']*100:.3f}% | normal {v['normal_var']*100:.3f}% | "
                 f"Cornish-Fisher {v['cornish_fisher_var']*100:.3f}% ; CVaR hist {v['historical_cvar']*100:.3f}% | normal {v['normal_cvar']*100:.3f}%")
    vb = rk["var_backtest_99"]
    if "conclusion" in vb:
        L.append(f"- VaR backtest (99%, Kupiec): {vb['exceptions']} exceptions in {vb['test_bars']} bars vs {vb['expected']} expected, "
                 f"LR {vb['lr_stat']}, p {vb['p_value']} -> {vb['conclusion']}")
    dd = rk["drawdown"]
    L += [f"- Drawdown: max {dd['max_drawdown_pct']}% ({dd['peak_utc'][:10]} -> {dd['trough_utc'][:10]}), current {dd['current_drawdown_pct']}%, "
          f"longest underwater {dd['longest_underwater_bars']} bars",
          f"- Volatility cone (annualised %): " + "; ".join(f"{k}: now {v['current']} (pct {v['percentile_of_current']}, median {v['median']})"
                                                         for k, v in m["vol_cone"].items()), "",
          "## 11. Visualisation", "- Interactive charts arrive with the Phase-7 dashboard; all series are stored in data/quantx.db.", "",
          "## 12. Limitations",
          "- Single instrument, single source; broker BID bars and tick volume are not consolidated market data.",
          "- In-sample model fits overstate fit quality; GARCH/Markov estimates are sensitive to window and starting values.",
          "- ~14 hypothesis tests: some 'significant' results are expected by chance (no multiple-testing correction applied).",
          "- Ljung-Box assumes homoskedasticity; with volatility clustering its p-values are too small (use VR z* instead).",
          "- Hurst scaling is biased in finite samples; OLS trend t-stats are inflated by autocorrelation.",
          "- Structural breaks, holidays and news spikes are not modelled separately.", "",
          "## 13. Conclusions", *[f"- {x}" for x in res["findings"]], "",
          "## 14. Recommendations (research, not trading)",
          "- Re-run on a second data source to confirm results are not broker-specific.",
          "- If dependence/VR tests reject the random walk, test a pre-registered strategy in Phase 4 with costs and walk-forward splits.",
          "- Use the GARCH/EWMA volatility for position sizing research rather than for direction.", "",
          "## 15. References", *[f"- {r}" for r in REFERENCES], "",
          "## Supervisor checks", *[f"- {c}" for c in res["supervisor_checks"]], "",
          "_Research output. Not financial advice; past statistics do not guarantee future behaviour._"]
    return "\n".join(L)


def compare(res: dict) -> str:
    rq = res["request"]
    L = [f"# QUANT-X comparison ({rq['timeframe']}, source {rq['source']})", f"_Run {res['analysis_run']} - {res['generated_at_utc']} UTC_", ""]
    if res.get("errors"):
        L += ["## Data errors", *[f"- {e}" for e in res["errors"]], ""]
    cols = ["n", "ann_vol_pct", "skew", "excess_kurtosis", "hist_var99_pct", "hist_cvar99_pct", "max_dd_pct", "vol_regime", "hurst", "quality"]
    L += ["## Per-instrument statistics", "| symbol | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for s, r in res["summary"].items():
        L.append(f"| {s} | " + " | ".join(str(r.get(c)) for c in cols) + " |")
    c = res["correlation"]
    L += ["", "## Correlation of log returns"]
    if "error" in c:
        L.append(f"- {c['error']}")
    else:
        syms = list(c["pearson"].keys())
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        L += [f"Aligned on {c['alignment']}; overlap {c['n_overlap']}+ returns per pair (Pearson, pairwise-complete)", "",
              "| | " + " | ".join(syms) + " |", "|" + "---|" * (len(syms) + 1)]
        for a in syms:
            L.append(f"| {a} | " + " | ".join(fmt(c['pearson'][a][b]) for b in syms) + " |")
        L += ["", f"### Stability ({c['rolling_window']}-bar rolling correlation)"]
        for p, v in c["pairs"].items():
            L.append(f"- {p}: " + (v["error"] if "error" in v else
                     f"n {v['n']}, full {v['full']} (Spearman {v['spearman']}), now {v['rolling_now']}, "
                     f"range {v['rolling_min']} to {v['rolling_max']}"))
        L.append(f"- {c['note']}")
    L += ["", "_Research output. Not financial advice._"]
    return "\n".join(L)
