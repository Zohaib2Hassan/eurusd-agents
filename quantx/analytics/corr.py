"""Cross-asset dependence: correlation matrices and rolling correlation stability.

Alignment: returns are computed per instrument first, then aligned.
- D1/W1: aligned on SESSION DATE (date of bar_open + 12h), because brokers/exchanges open daily bars at
  different hours (FX 22:00 UTC, metals 23:00 UTC, crypto 00:00 UTC) - exact timestamps would not match.
- Intraday: aligned on exact bar-open timestamp.
Correlations use pairwise-complete observations; each pair reports its own overlap count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rekey(r: pd.Series, daily: bool) -> pd.Series:
    if not daily:
        return r
    k = (r.index + pd.Timedelta(hours=12)).normalize()
    return r.groupby(k).last()


def aligned_returns(closes: dict[str, pd.Series], daily: bool = True) -> pd.DataFrame:
    return pd.concat({k: _rekey(np.log(v).diff().dropna(), daily) for k, v in closes.items()}, axis=1, join="outer")


def correlation(closes: dict[str, pd.Series], rolling: int = 60, daily: bool = True) -> dict:
    R = aligned_returns(closes, daily)
    cols = list(R.columns)
    pairs, n_overlap = {}, {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pr = R[[cols[i], cols[j]]].dropna()
            key = f"{cols[i]}~{cols[j]}"
            n_overlap[key] = len(pr)
            if len(pr) < rolling + 10:
                pairs[key] = {"error": f"only {len(pr)} overlapping returns"}
                continue
            rc = pr[cols[i]].rolling(rolling).corr(pr[cols[j]]).dropna()
            pairs[key] = {"n": len(pr), "full": round(float(pr.corr().iloc[0, 1]), 3),
                          "spearman": round(float(pr.corr(method="spearman").iloc[0, 1]), 3),
                          "rolling_now": round(float(rc.iloc[-1]), 3), "rolling_min": round(float(rc.min()), 3),
                          "rolling_max": round(float(rc.max()), 3), "rolling_std": round(float(rc.std()), 3)}
    if not any("full" in p for p in pairs.values()):
        return {"error": "no pair has enough overlapping returns", "pairs": pairs}
    pear = R.corr(method="pearson", min_periods=rolling + 10).round(3)
    spear = R.corr(method="spearman", min_periods=rolling + 10).round(3)
    both = R.dropna()
    return {"alignment": "session date" if daily else "exact timestamp",
            "n_overlap": int(min(v for v in n_overlap.values())), "n_overlap_by_pair": n_overlap,
            "start_utc": str(both.index[0]) if len(both) else str(R.index[0]),
            "end_utc": str(both.index[-1]) if len(both) else str(R.index[-1]),
            "pearson": pear.where(pd.notna(pear), None).to_dict(), "spearman": spear.where(pd.notna(spear), None).to_dict(),
            "pairs": pairs, "rolling_window": rolling,
            "note": "correlation is not causation and is unstable; see rolling range"}
