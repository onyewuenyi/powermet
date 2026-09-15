"""Power budgets tracked across design milestones.

Budgets are checked against the RAW dataset (every measured FUB), not the sanitized one: a FUB excluded
from correlation for a data-quality reason still burns power.

Budgets live in a TOML file (templates/budgets.template.toml). Each entry names a scope
(design, partition or FUB), a workload and operating point, a metric (be_mw by default; cdyn_pf for
Cdyn, be_leakage_mw for leakage power, be_dynamic_mw), a target value, and a tolerance per
milestone: early estimates are allowed more headroom than signoff. Each build carries its
milestone in metadata.json, so `budget check` places every build against the tolerance that
applies at that stage and classifies it ON TRACK / AT RISK / OVER, with the trend across builds
and the model's cross-validated error band so a number inside tolerance but inside the noise is
called out as such.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from powermet.metrics import add_convergence_metrics
from powermet.schema import CONVERGENCE_METRICS, METRIC_UNITS
from powermet.selection import DatasetSlice, build_order
from powermet.textfmt import fmt_mw, fmt_pct, table

MILESTONES = ("rtl", "synthesis", "placement", "route", "signoff")
DEFAULT_TOLERANCE = {"rtl": 25.0, "synthesis": 15.0, "placement": 10.0, "route": 5.0, "signoff": 0.0}
AT_RISK_BAND = 0.5      # within this fraction of the tolerance -> AT RISK


def fmt_metric(value: float, metric: str) -> str:
    """Value in the metric's unit: mW through fmt_mw, pF with one decimal, else plain."""
    unit = METRIC_UNITS.get(metric, "")
    if unit == "mW":
        return fmt_mw(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.1f} {unit}".strip()


@dataclass
class Budget:
    design: str
    scope: str                     # "design" | "partition:<name>" | "fub:<name>" | "model_root:<id>"
    target: float
    workload: str | None = None
    operating_point: str | None = None
    tolerance_pct: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOLERANCE))
    owner: str | None = None
    note: str | None = None
    metric: str = "be_mw"          # be_mw | cdyn_pf | be_leakage_mw | be_dynamic_mw (extensive: sums over the scope)

    @property
    def be_mw(self) -> float:      # compatibility alias for the original mW-only budgets
        return self.target

    @property
    def short_metric(self) -> str:
        return CONVERGENCE_METRICS.get(self.metric, (self.metric, "", ""))[0]

    @property
    def unit(self) -> str:
        return METRIC_UNITS.get(self.metric, "")

    @property
    def label(self) -> str:
        return f"{self.design} {self.scope} {self.short_metric} @ {self.workload or 'default'}/{self.operating_point or 'default'}"


@dataclass
class BudgetStatus:
    budget: Budget
    build: str
    milestone: str
    actual: float
    tolerance_pct: float
    margin_pct: float              # (budget*(1+tol) - actual) / budget * 100 ; negative = over
    status: str
    trend_pct_per_build: float
    interval_pct: tuple[float, float] | None
    history: pd.DataFrame          # build, milestone, actual, tolerance_pct, margin_pct, status
    fubs_measured: int = 0
    fubs_expected: int = 0

    @property
    def actual_mw(self) -> float:  # compatibility alias
        return self.actual

    @property
    def coverage_pct(self) -> float:
        return self.fubs_measured / self.fubs_expected * 100 if self.fubs_expected else float("nan")

    @property
    def complete(self) -> bool:
        return not self.fubs_expected or self.fubs_measured >= self.fubs_expected


def load_budgets(path: str | Path) -> list[Budget]:
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    out = []
    defaults = doc.get("defaults", {})
    for b in doc.get("budget", []):
        d = {**defaults, **b}
        tol = dict(DEFAULT_TOLERANCE)
        tol.update({k: float(v) for k, v in (d.get("tolerance_pct") or {}).items()})
        metric = str(d.get("metric") or next((m for m in CONVERGENCE_METRICS if m in d), "be_mw"))
        if metric not in CONVERGENCE_METRICS:
            raise ValueError(f"budget metric '{metric}' is not a convergence metric ({', '.join(CONVERGENCE_METRICS)})")
        value = d.get("target", d.get(metric))
        if value is None:
            raise ValueError(f"budget for {d.get('design')} {d.get('scope', 'design')}: no target (use `target = ...` or `{metric} = ...`)")
        out.append(Budget(str(d["design"]), str(d.get("scope", "design")), float(value), d.get("workload"),
                          d.get("operating_point"), tol, d.get("owner"), d.get("note"), metric))
    return out


def _scope_rows(df: pd.DataFrame, b: Budget) -> pd.DataFrame:
    kind, _, name = b.scope.partition(":")
    kw = {"design": b.design, "workload": b.workload, "operating_point": b.operating_point, "latest_only": False}
    if kind == "partition":
        kw["partition"] = name
    elif kind == "fub":
        kw["fub"] = name
    elif kind == "model_root":
        kw["model_root"] = name
    elif kind != "design":
        raise ValueError(f"unknown budget scope '{b.scope}'")
    return DatasetSlice(**kw).apply(df)


def classify_status(actual: float, budget: float, tol_pct: float) -> tuple[str, float]:
    limit = budget * (1 + tol_pct / 100)
    margin = (limit - actual) / budget * 100
    if actual > limit:
        return "OVER", margin
    band = max(tol_pct, 2.0) * AT_RISK_BAND
    if margin < band:
        return "AT RISK", margin
    return "ON TRACK", margin


def _expected_fubs(lineage: pd.DataFrame | None, b: Budget, build: str) -> int:
    """How many FUBs the model root says belong to this scope (0 when no lineage table)."""
    if lineage is None or not len(lineage):
        return 0
    sel = lineage[(lineage["design"].astype(str) == b.design) & (lineage["build"].astype(str) == build)]
    kind, _, name = b.scope.partition(":")
    if kind == "partition" and "partition" in sel.columns:
        sel = sel[sel["partition"].astype(str) == name]
    elif kind == "fub":
        sel = sel[sel["fub"].astype(str) == name]
    elif kind == "model_root" and "model_root" in sel.columns:
        sel = sel[sel["model_root"].astype(str) == name]
    return int(sel["fub"].nunique())


def check_budgets(df: pd.DataFrame, budgets: list[Budget], milestones: dict[tuple[str, str], str] | None = None,
                  interval: tuple[float, float] | None = None, lineage: pd.DataFrame | None = None) -> list[BudgetStatus]:
    """milestones: (design, build) -> milestone name; falls back to the dataset's `milestone` column, then 'signoff'.
    lineage: the lineage table, used to detect scopes whose FUBs are not all measured (undercounted totals)."""
    out = []
    if any(b.metric not in df.columns for b in budgets):
        df = add_convergence_metrics(df.copy())
    for b in budgets:
        try:
            rows = _scope_rows(df, b)
        except ValueError:
            continue
        if b.metric not in rows.columns or rows[b.metric].isna().all():
            continue
        per_build = rows.groupby("build")[b.metric].sum(min_count=1).dropna()
        if not len(per_build):
            continue
        order = build_order(per_build.index)
        hist = []
        for bl in order:
            ms = (milestones or {}).get((b.design, bl))
            if ms is None and "milestone" in rows.columns:
                v = rows[rows["build"] == bl]["milestone"].dropna()
                ms = str(v.iloc[0]) if len(v) else None
            ms = ms or "signoff"
            tol = b.tolerance_pct.get(ms, 0.0)
            status, margin = classify_status(float(per_build[bl]), b.target, tol)
            hist.append({"build": bl, "milestone": ms, "actual": float(per_build[bl]), "tolerance_pct": tol,
                         "margin_pct": margin, "status": status})
        h = pd.DataFrame(hist)
        last = h.iloc[-1]
        slope = float(np.polyfit(np.arange(len(h)), h["actual"].to_numpy(), 1)[0] / b.target * 100) if len(h) >= 3 else float("nan")
        measured = int(rows[rows["build"].astype(str) == last["build"]]["fub"].nunique())
        expected = _expected_fubs(lineage, b, last["build"])
        out.append(BudgetStatus(b, last["build"], last["milestone"], last["actual"], last["tolerance_pct"], last["margin_pct"],
                                last["status"], slope, interval, h, measured, expected))
    return out


def render_budgets(statuses: list[BudgetStatus]) -> str:
    if not statuses:
        return "No budgets matched the dataset."
    rows = []
    for s in statuses:
        b = s.budget
        cov = f"{s.fubs_measured}/{s.fubs_expected}" if s.fubs_expected else str(s.fubs_measured)
        status = s.status if s.complete else f"{s.status} (INCOMPLETE)"
        rows.append([b.design, b.scope, b.short_metric, f"{b.workload or '-'}/{b.operating_point or '-'}", s.build, s.milestone,
                     fmt_metric(b.target, b.metric), fmt_metric(s.actual, b.metric), cov, fmt_pct(s.tolerance_pct), fmt_pct(s.margin_pct, True),
                     fmt_pct(s.trend_pct_per_build, True) + "/build" if np.isfinite(s.trend_pct_per_build) else "n/a", status])
    out = ["POWER BUDGET STATUS", "",
           table(["Design", "Scope", "Metric", "WL/OP", "Build", "Milestone", "Budget", "Actual", "FUBs", "Tol", "Margin", "Trend", "Status"], rows,
                 ["l", "l", "l", "l", "l", "l", "r", "r", "r", "r", "r", "r", "l"])]
    n_over = sum(s.status == "OVER" for s in statuses)
    n_risk = sum(s.status == "AT RISK" for s in statuses)
    n_inc = sum(not s.complete for s in statuses)
    out.append("")
    out.append(f"{len(statuses)} budgets: {n_over} OVER, {n_risk} AT RISK, {len(statuses) - n_over - n_risk} ON TRACK; "
               f"{n_inc} with FUBs missing from the measurement (INCOMPLETE: the actual is an undercount). "
               "Margin = headroom to budget x (1 + milestone tolerance), relative to budget.")
    if statuses and statuses[0].interval_pct:
        lo, hi = statuses[0].interval_pct
        out.append(f"Model error band (leave-one-build-out, 90%): {lo:+.1f}% .. {hi:+.1f}%; a margin inside that band is not evidence of closure.")
    return "\n".join(out)


def render_history(s: BudgetStatus) -> str:
    m = s.budget.metric
    rows = [[r["build"], r["milestone"], fmt_metric(r["actual"], m), fmt_pct(r["tolerance_pct"]), fmt_pct(r["margin_pct"], True), r["status"]]
            for _, r in s.history.iterrows()]
    return f"{s.budget.label}  target {fmt_metric(s.budget.target, m)}\n\n" + table(["Build", "Milestone", "Actual", "Tol", "Margin", "Status"], rows)
