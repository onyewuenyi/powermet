"""Estimator / engine qualification: compare two estimates of the same quantity across FUBs.

Use it to qualify a second signoff engine (`be_mw` from PrimePower vs `be_voltus_mw` from Voltus),
an engine version bump, vectorless vs vector-based activity, or any FE stage against BE. The verdict
is against an explicit tolerance so a tool assessment is a reproducible number, not an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.correlation import pearson
from powermet.metrics import prediction_metrics
from powermet.schema import identity_key, label
from powermet.selection import DatasetSlice
from powermet.textfmt import fmt_mw, fmt_pct, fmt_r, table


@dataclass
class Qualification:
    a: str
    b: str
    n: int
    metrics: dict
    r: float
    bias_pct: float
    by_partition: pd.DataFrame
    worst: pd.DataFrame
    tolerance_pct: float
    verdict: str
    scope: str


def qualify(df: pd.DataFrame, a: str, b: str, tolerance_pct: float = 5.0, design: str | None = None,
            build: str | None = None, workload: str | None = None, operating_point: str | None = None, top: int = 8) -> Qualification:
    """`b` is judged against `a` (the reference). Verdict: PASS if MAPE <= tolerance and P95 <= 2x tolerance."""
    if a not in df.columns or b not in df.columns:
        raise ValueError(f"both metrics must be columns of the dataset: {a}, {b}")
    sel = DatasetSlice(design=design, build=build, workload=workload, operating_point=operating_point,
                       latest_only=build is None, default_workload=False, default_operating_point=False).apply(df)
    sel = sel.dropna(subset=[a, b])
    if not len(sel):
        raise ValueError(f"no rows where both {a} and {b} are present")
    m = prediction_metrics(sel[a], sel[b])
    r, n, _ = pearson(sel[a], sel[b])
    bias = float((sel[b].sum() - sel[a].sum()) / sel[a].sum() * 100) if sel[a].sum() else float("nan")
    key = identity_key(sel)
    sel = sel.assign(err_pct=(sel[b] - sel[a]) / sel[a].where(sel[a] != 0) * 100)
    parts = pd.DataFrame()
    if "partition" in sel.columns:
        g = sel.groupby("partition")
        parts = pd.DataFrame({"n": g.size(), "a_mw": g[a].sum(), "b_mw": g[b].sum(),
                              "mape": g["err_pct"].apply(lambda s: float(s.abs().mean())),
                              "bias_pct": (g[b].sum() - g[a].sum()) / g[a].sum().where(lambda s: s != 0) * 100}).reset_index()
    worst = sel.reindex(sel["err_pct"].abs().sort_values(ascending=False).index).head(top)[[c for c in (key, "build", "workload", "operating_point", a, b, "err_pct") if c in sel.columns]]
    verdict = "PASS" if (m["mape"] <= tolerance_pct and m["p95_ape"] <= 2 * tolerance_pct) else "FAIL"
    scope = DatasetSlice(design=design, build=build, workload=workload, operating_point=operating_point).describe(sel)
    return Qualification(a, b, n, m, r, bias, parts, worst, tolerance_pct, verdict, scope)


def render_qualification(q: Qualification) -> str:
    m = q.metrics
    out = [f"QUALIFICATION  {label(q.b)} vs reference {label(q.a)}   [{q.verdict}]   {q.scope}", ""]
    out.append(table(["n", "MAPE", "P50", "P95", "bias (sum)", "r", "R^2", "tolerance"],
                     [[f"{q.n:,}", fmt_pct(m["mape"]), fmt_pct(m["p50_ape"]), fmt_pct(m["p95_ape"]), fmt_pct(q.bias_pct, True),
                       fmt_r(q.r), fmt_r(m["r2"]), f"MAPE <= {q.tolerance_pct:g}%, P95 <= {2 * q.tolerance_pct:g}%"]]))
    if len(q.by_partition):
        out += ["", "Per partition", ""]
        out.append(table(["Partition", "n", label(q.a), label(q.b), "MAPE", "bias"],
                         [[r["partition"], int(r["n"]), fmt_mw(r["a_mw"]), fmt_mw(r["b_mw"]), fmt_pct(r["mape"]), fmt_pct(r["bias_pct"], True)]
                          for _, r in q.by_partition.iterrows()]))
    if len(q.worst):
        key = q.worst.columns[0]
        out += ["", f"Largest disagreements (top {len(q.worst)})", ""]
        out.append(table([key, "Build", "WL/OP", label(q.a), label(q.b), "diff"],
                         [[r[key], r.get("build", ""), f"{r.get('workload', '')}/{r.get('operating_point', '')}", fmt_mw(r[q.a]), fmt_mw(r[q.b]),
                           fmt_pct(r["err_pct"], True)] for _, r in q.worst.iterrows()]))
    out.append("")
    out.append("A qualification is evidence about agreement on this dataset; a systematic bias with low scatter is a calibration, a wide scatter is a methodology gap.")
    return "\n".join(out)
