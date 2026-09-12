"""Pairwise association (Pearson r, R^2, n, optional p-value). Association, not causation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.deps import available
from powermet.schema import FEATURE_COLUMNS, FE_ESTIMATE_COLUMNS, label
from powermet.textfmt import fmt_r, table

ANALYSIS_FEATURES = tuple(FEATURE_COLUMNS) + ("wire_cap_fraction",)


@dataclass
class Association:
    x: str
    y: str
    r: float
    n: int
    p: float = float("nan")

    @property
    def r2(self) -> float:
        return self.r * self.r if np.isfinite(self.r) else float("nan")


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, int, float]:
    xv = pd.to_numeric(x, errors="coerce").astype(float)
    yv = pd.to_numeric(y, errors="coerce").astype(float)
    ok = xv.notna() & yv.notna()
    n = int(ok.sum())
    if n < 3:
        return float("nan"), n, float("nan")
    xa, ya = xv[ok].to_numpy(), yv[ok].to_numpy()
    if xa.std() == 0 or ya.std() == 0:
        return float("nan"), n, float("nan")
    if available("scipy"):
        from scipy import stats

        res = stats.pearsonr(xa, ya)
        return float(res[0]), n, float(res[1])
    r = float(np.corrcoef(xa, ya)[0, 1])
    return r, n, float("nan")


def associations(df: pd.DataFrame, target: str, candidates: list[str] | tuple[str, ...]) -> list[Association]:
    out = []
    for c in candidates:
        if c == target or c not in df.columns:
            continue
        r, n, p = pearson(df[c], df[target])
        out.append(Association(c, target, r, n, p))
    out.sort(key=lambda a: (-(abs(a.r) if np.isfinite(a.r) else -1)))
    return out


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    num = df[present].apply(pd.to_numeric, errors="coerce")
    return num.corr(method="pearson")


def be_correlations(df: pd.DataFrame) -> list[Association]:
    cands = list(FE_ESTIMATE_COLUMNS) + list(ANALYSIS_FEATURES)
    return associations(df, "be_mw", cands)


def render_associations(assocs: list[Association], title: str, show_p: bool = True) -> str:
    has_p = show_p and any(np.isfinite(a.p) for a in assocs)
    headers = ["Variable", "r", "R^2", "n"] + (["p"] if has_p else [])
    rows = []
    for a in assocs:
        row = [label(a.x), fmt_r(a.r), fmt_r(a.r2), f"{a.n:,}"]
        if has_p:
            row.append("<0.001" if a.p < 0.001 else f"{a.p:.3f}" if np.isfinite(a.p) else "n/a")
        rows.append(row)
    return f"{title}\n\n" + table(headers, rows)


def strength_word(r: float) -> str:
    a = abs(r) if np.isfinite(r) else 0.0
    if a >= 0.9:
        return "very strongly"
    if a >= 0.7:
        return "strongly"
    if a >= 0.5:
        return "moderately"
    if a >= 0.3:
        return "weakly"
    return "not meaningfully"
