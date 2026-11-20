"""Report templates (spec 17.F): brief chart read, deep multi-timeframe report, data-quality warning."""
from __future__ import annotations


def _z(z):
    return f"{z['low']}-{z['high']} ({z['touches']}x)" if z else "none found"


def data_quality_block(a: dict) -> list[str]:
    lines = []
    for tf, r in a["timeframes"].items():
        if r.get("error"):
            lines.append(f"- {tf}: FAILED - {r['error']}")
        elif r.get("warnings"):
            lines.append(f"- {tf}: quality {r['quality']} - " + "; ".join(r["warnings"]))
    for f in a.get("failures", []):
        lines.append(f"- {f}")
    return lines


def brief(a: dict) -> str:
    if a.get("status") == "failed":
        return "TA FAILED: " + "; ".join(a["failures"])
    s, i, q = a["summary"], a["instrument"], a["quote_now"]
    ex = a["timeframes"][s["execution_timeframe"]]
    L = [f"**{i['symbol']} brief read** - {a['generated_at_local']} ({a['display_timezone']}) - {a['source']['provider']}",
         f"Bid {q['bid']} / Ask {q['ask']} (spread {q['spread_points']} pts) - {i['convention']}",
         f"Lean: **{s['lean']}** (composite {s['composite_score']:+.2f}, agreement {s['agreement']}, confidence {s['confidence']})"]
    for tf, r in a["timeframes"].items():
        if "score" in r:
            L.append(f"- {tf}: {r['structure']['state']}, score {r['score']:+.2f}, RSI {r['indicators']['rsi14']}, "
                     f"ADX {r['indicators']['adx14']}, vol {r['volatility_regime']}")
    L.append(f"{s['execution_timeframe']} support {_z(ex['nearest']['support'])} | resistance {_z(ex['nearest']['resistance'])}")
    if a["events"].get("high_impact_next_24h"):
        L.append("Events 24h: " + "; ".join(a["events"]["high_impact_next_24h"]))
    dq = data_quality_block(a)
    if dq:
        L.append("Data warnings:\n" + "\n".join(dq))
    L.append("_Descriptive analysis, not a forecast or advice._")
    return "\n".join(L)


def deep(a: dict) -> str:
    if a.get("status") == "failed":
        return "TA FAILED: " + "; ".join(a["failures"])
    s, i = a["summary"], a["instrument"]
    L = [f"# {i['symbol']} - multi-timeframe technical report",
         f"- Generated: {a['generated_at_utc']} UTC / {a['generated_at_local']} ({a['display_timezone']})",
         f"- Source: {a['source']['provider']} ({a['source']['server']}, {a['source']['account_type']}) - {i['price_type']}",
         f"- Convention: {i['convention']}, pip = {i['pip_size']}",
         f"- Versions: {a['agent_version']}, {a['calc_version']}", ""]
    dq = data_quality_block(a)
    L += ["## Data quality", *(dq or ["- no issues detected"]), ""]
    L += ["## Summary (rule-based)",
          f"- Lean **{s['lean']}**, composite {s['composite_score']:+.3f}, timeframe agreement {s['agreement']}, "
          f"data quality {s['data_quality']}, confidence {s['confidence']}", ""]
    for tf, r in a["timeframes"].items():
        if "score" not in r:
            continue
        ind = r["indicators"]
        L += [f"## {tf} (last completed bar opened {r['as_of_bar_open_utc']} UTC)",
              f"- Structure: **{r['structure']['state']}** - {r['structure'].get('detail', '')}",
              f"- Rule labels: " + "; ".join(r["rule_labels"]),
              f"- Trend: EMA20 {ind['ema20']} / EMA50 {ind['ema50']} / EMA200 {ind['ema200']}; "
              f"ADX {ind['adx14']} (+DI {ind['plus_di']}, -DI {ind['minus_di']}); LinReg50 R² {ind['linreg50_r2']}",
              f"- Momentum: RSI {ind['rsi14']}, MACD hist {ind['macd_hist']}, Stoch {ind['stoch_k']}/{ind['stoch_d']}, "
              f"CCI {ind['cci20']}, ROC10 {ind['roc10']}%",
              f"- Volatility: ATR14 {ind['atr14_pips']} pips (pct-rank {ind['atr_pct_rank_250']}) -> {r['volatility_regime']}; "
              f"BB %B {ind['bb_pct_b']}; ann. vol C2C {ind['vol_close_to_close_20_ann_pct']}% / YZ {ind['vol_yang_zhang_20_ann_pct']}%",
              f"- Zones (from confirmed swings, 0.5×ATR clustering): support {_z(r['nearest']['support'])}, "
              f"resistance {_z(r['nearest']['resistance'])}",
              f"- Last candle: {', '.join(r['candles_last_bar']) or 'no named pattern'}", ""]
    pv = s.get("pivots_daily", {})
    if "floor" in pv:
        f = pv["floor"]
        L += ["## Daily floor pivots (previous completed day " + pv["from_day_utc"] + ")",
              f"S3 {f['S3']} | S2 {f['S2']} | S1 {f['S1']} | **P {f['P']}** | R1 {f['R1']} | R2 {f['R2']} | R3 {f['R3']}", ""]
    L += ["## Scenarios"]
    for sc in a["scenarios"]:
        L.append(f"- **{sc['name']}**{' (aligned with composite)' if sc['aligned_with_composite'] else ''}: "
                 f"{sc['trigger']}" + (f"; next reference {sc['next_reference']}" if sc.get("next_reference") else "") +
                 (f"; invalidation {sc['invalidation']}" if sc.get("invalidation") else ""))
    L += ["", "## Events", *(a["events"].get("high_impact_next_24h") or [a["events"].get("note", "no high-impact events in 24h")]), ""]
    if a.get("narrative"):
        n = a["narrative"]
        L += [f"## {n['label']}", f"- Observed: {n.get('observed', '')}", f"- Interpretation: {n.get('interpretation', '')}",
              f"- What would change the view: {n.get('what_would_change_the_view', '')}", ""]
    L += ["## Uncertainty", *[f"- {u}" for u in a["uncertainty"]], "",
          "## Method references", *[f"- {r['text']} - {r['supports']}" for r in a["methods"]["references"]], "",
          "_Read-only descriptive analysis. Not a forecast, signal guarantee or financial advice._"]
    return "\n".join(L)
