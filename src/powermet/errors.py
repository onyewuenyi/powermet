"""FE -> BE error analysis: ranked FUBs, error-feature associations, error by build."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.correlation import ANALYSIS_FEATURES, Association, associations
from powermet.metrics import prediction_metrics
from powermet.schema import label
from powermet.textfmt import fmt_mw, fmt_pct, fmt_r, table

STAGES = {
    "logical": ("fe_logical_mw", "logical_error_mw", "logical_error_pct", "FE Logical"),
    "physical": ("fe_physical_mw", "physical_error_mw", "physical_error_pct", "FE Physical"),
}


@dataclass
class ErrorAnalysis:
    stage: str
    fe_col: str
    metrics: dict[str, float]
    top_abs: pd.DataFrame
    top_pct: pd.DataFrame
    assoc_mw: list[Association]
    assoc_pct: list[Association]
    by_build: pd.DataFrame
    by_design: pd.DataFrame


def stage_metrics(df: pd.DataFrame, stage: str) -> dict[str, float]:
    fe_col = STAGES[stage][0]
    return prediction_metrics(df["be_mw"], df[fe_col])


def analyze_errors(df: pd.DataFrame, stage: str = "physical", top_n: int = 10) -> ErrorAnalysis:
    fe_col, err_mw, err_pct, _ = STAGES[stage]
    d = df.copy()
    d["_abs_mw"] = d[err_mw].abs()
    d["_abs_pct"] = d[err_pct].abs()
    id_cols = [c for c in ("design", "build", "fub") if c in d.columns]
    show = id_cols + [fe_col, "be_mw", err_mw, err_pct]
    top_abs = d.sort_values("_abs_mw", ascending=False).head(top_n)[show].reset_index(drop=True)
    top_pct = d.sort_values("_abs_pct", ascending=False).head(top_n)[show].reset_index(drop=True)

    assoc_mw = associations(d, err_mw, ANALYSIS_FEATURES)
    assoc_pct = associations(d, err_pct, ANALYSIS_FEATURES)

    def _group(col: str) -> pd.DataFrame:
        if col not in d.columns:
            return pd.DataFrame()
        g = d.groupby(col, sort=True)
        return pd.DataFrame({
            "n": g.size(),
            "mean_error_pct": g[err_pct].mean(),
            "mape": g["_abs_pct"].mean(),
            "p95_ape": g["_abs_pct"].quantile(0.95),
        }).reset_index()

    return ErrorAnalysis(
        stage=stage,
        fe_col=fe_col,
        metrics=stage_metrics(d, stage),
        top_abs=top_abs,
        top_pct=top_pct,
        assoc_mw=assoc_mw,
        assoc_pct=assoc_pct,
        by_build=_group("build"),
        by_design=_group("design"),
    )


# ----------------------------------------------------------------------------- rendering

def render_top(df: pd.DataFrame, fe_col: str, title: str) -> str:
    err_mw = [c for c in df.columns if c.endswith("_error_mw")][0]
    err_pct = [c for c in df.columns if c.endswith("_error_pct")][0]
    id_cols = [c for c in ("design", "build", "fub") if c in df.columns]
    headers = [c.capitalize() for c in id_cols] + ["FE (mW)", "BE (mW)", "Error (mW)", "Error"]
    rows = []
    for _, r in df.iterrows():
        rows.append([r[c] for c in id_cols] + [f"{r[fe_col]:.1f}", f"{r['be_mw']:.1f}",
                                                f"{r[err_mw]:+.1f}", fmt_pct(r[err_pct], signed=True)])
    align = ["l"] * len(id_cols) + ["r"] * 4
    return f"{title}\n\n" + table(headers, rows, align)


def render_assoc(assocs: list[Association], title: str) -> str:
    rows = [[label(a.x), fmt_r(a.r), f"{a.n:,}"] for a in assocs]
    return f"{title}\n\n" + table(["Feature", "r", "n"], rows)


def render_by_group(g: pd.DataFrame, key: str, title: str) -> str:
    if g.empty:
        return ""
    rows = [[r[key], f"{int(r['n']):,}", fmt_pct(r["mean_error_pct"], signed=True),
             fmt_pct(r["mape"]), fmt_pct(r["p95_ape"])] for _, r in g.iterrows()]
    return f"{title}\n\n" + table([key.capitalize(), "n", "Mean Err", "MAPE", "P95"], rows)


def build_trend(by_build: pd.DataFrame) -> tuple[float, str]:
    """Slope of mean % error across build order and a one-line reading."""
    if by_build.empty or len(by_build) < 3:
        return float("nan"), "Too few builds to assess whether error drifts across builds."
    x = np.arange(len(by_build))
    y = by_build["mean_error_pct"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    first, last = y[0], y[-1]
    if abs(last - first) < 2.0:
        word = "is roughly stable"
    elif last > first:
        word = "grows"
    else:
        word = "shrinks"
    return slope, (f"Mean error {word} across builds "
                   f"({fmt_pct(first, True)} in {by_build['build'].iloc[0]} -> "
                   f"{fmt_pct(last, True)} in {by_build['build'].iloc[-1]}, "
                   f"{slope:+.1f} pts/build).")


def interpret(ea: ErrorAnalysis) -> str:
    m = ea.metrics
    name = STAGES[ea.stage][3]
    # prediction_metrics reports (FE - BE)/BE; flip to the BE - FE convention used everywhere else.
    bias = -m.get("mean_error_pct", float("nan"))
    direction = "underestimates" if bias > 0 else "overestimates"
    lines = [
        f"{name} power {direction} BE power by {fmt_pct(abs(bias))} on average "
        f"(MAPE {fmt_pct(m.get('mape'))}, P95 {fmt_pct(m.get('p95_ape'))}, R^2 {fmt_r(m.get('r2'))}).",
    ]
    if ea.assoc_pct:
        top = ea.assoc_pct[0]
        lines.append(
            f"Percentage error is most strongly associated with {label(top.x)} (r={fmt_r(top.r)}). "
            "This is an association in this dataset, not evidence of cause."
        )
    if ea.assoc_mw:
        top = ea.assoc_mw[0]
        lines.append(
            f"Absolute (mW) error is most associated with {label(top.x)} (r={fmt_r(top.r)}); "
            "mW error correlations are partly a block-size effect, so prefer the percentage view."
        )
    lines.append(build_trend(ea.by_build)[1])
    return "\n".join(lines)
