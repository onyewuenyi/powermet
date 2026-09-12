"""Power convergence: is the design going to hit its power targets, and what closes the gap?

The targets a program signs up to are usually not one total-power number but a pair of design-owned
quantities per milestone:

* **CdynTot** (`cdyn_pf`): effective switched capacitance, dynamic power with V^2 f divided out.
  It is independent of the corner the report was run at, so it compares FE to BE, build to build and
  corner to corner, and it is what RTL and physical changes actually move. mW / (V^2 GHz) is pF exactly.
* **LkgPwr** (`be_leakage_mw`): leakage at the signoff corner. Tracked separately because its levers
  (Vt mix, power gating, area, memory sleep) and its sensitivity (V^3, temperature, process) differ
  from the dynamic ones, and because a total-power target lets one component hide the other.

`converge()` places every target against the latest build with the milestone tolerance (budgets.py does
the classification), then adds what closure needs: the gap in the metric's unit, the reduction required,
the trend across builds and a projection of how many builds it takes at that rate. `plan()` ranks the
technique assessments by how much of the gap each covers in the target's own component, so a CdynTot gap
is answered with dynamic levers (clock gating, wire cap, ...) and a LkgPwr gap with leakage levers
(Vt swap, power gating, ...). The plan is a prioritisation under the techniques' stated assumptions,
not a commitment: each line is an order-of-magnitude estimate to be confirmed with a what-if and the
next build.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.budgets import Budget, BudgetStatus, _scope_rows, check_budgets, fmt_metric
from powermet.metrics import add_convergence_metrics
from powermet.schema import CONVERGENCE_METRICS
from powermet.techniques import BY_KEY, TechniqueResult, assess_all
from powermet.textfmt import fmt_pct, table

VERDICTS = ("CONVERGED", "CONVERGING", "FLAT", "DIVERGING")


@dataclass
class Convergence:
    status: BudgetStatus
    gap: float                       # actual - target in the metric's unit; > 0 means over target
    gap_pct: float                   # gap / target * 100
    required_cut_pct: float          # reduction of the actual needed to reach target (0 when under)
    trend_per_build: float           # change of the actual per build in metric units (NaN with < 3 builds)
    builds_to_target: float          # gap / -trend when converging; inf when flat/diverging; 0 when under
    verdict: str

    @property
    def budget(self) -> Budget:
        return self.status.budget

    @property
    def component(self) -> str:
        return CONVERGENCE_METRICS.get(self.budget.metric, ("", "", "total"))[2]


@dataclass
class PlanLine:
    technique: str
    name: str
    saving: float                    # in the target's unit
    cumulative: float
    covers_pct: float                # saving / gap * 100
    assumptions: list[str]


@dataclass
class Plan:
    convergence: Convergence
    lines: list[PlanLine]
    remainder: float                 # gap left after every applicable technique (>= 0)
    voltage_v: float
    frequency_ghz: float
    not_applicable: list[str] = field(default_factory=list)

    @property
    def covered_pct(self) -> float:
        g = self.convergence.gap
        return (g - self.remainder) / g * 100 if g > 0 else 100.0


def _verdict(gap: float, trend: float, tol_margin_pct: float) -> str:
    if gap <= 0:
        return "CONVERGED"
    if not np.isfinite(trend) or abs(trend) < 1e-12:
        return "FLAT"
    return "CONVERGING" if trend < 0 else "DIVERGING"


def converge(df: pd.DataFrame, budgets: list[Budget], milestones: dict[tuple[str, str], str] | None = None,
             lineage: pd.DataFrame | None = None) -> list[Convergence]:
    """Convergence view of every budget: classification plus gap, required cut, trend and projection."""
    if any(b.metric not in df.columns for b in budgets):
        df = add_convergence_metrics(df.copy())
    out = []
    for s in check_budgets(df, budgets, milestones=milestones, lineage=lineage):
        b = s.budget
        gap = s.actual - b.target
        gap_pct = gap / b.target * 100 if b.target else float("nan")
        cut = max(gap, 0.0) / s.actual * 100 if s.actual > 0 else 0.0
        h = s.history["actual"].to_numpy(dtype=float)
        trend = float(np.polyfit(np.arange(len(h)), h, 1)[0]) if len(h) >= 3 else float("nan")
        verdict = _verdict(gap, trend, s.margin_pct)
        if verdict == "CONVERGING":
            builds = gap / -trend
        elif verdict == "CONVERGED":
            builds = 0.0
        else:
            builds = float("inf")
        out.append(Convergence(s, gap, gap_pct, cut, trend, builds, verdict))
    return out


def _corner(df: pd.DataFrame, b: Budget) -> tuple[float, float]:
    rows = _scope_rows(df, b)
    v = pd.to_numeric(rows.get("voltage_v"), errors="coerce").dropna() if "voltage_v" in rows.columns else pd.Series(dtype=float)
    f = pd.to_numeric(rows.get("frequency_ghz"), errors="coerce").dropna() if "frequency_ghz" in rows.columns else pd.Series(dtype=float)
    return (float(v.iloc[0]) if len(v) else float("nan"), float(f.iloc[0]) if len(f) else float("nan"))


def saving_in_metric(saving_mw: float, metric: str, voltage_v: float, frequency_ghz: float) -> float:
    """A dynamic-power saving expressed in the target metric: mW stays mW; Cdyn divides out V^2 f."""
    if metric in ("cdyn_pf", "fe_cdyn_pf"):
        den = voltage_v ** 2 * frequency_ghz
        return saving_mw / den if np.isfinite(den) and den > 0 else float("nan")
    return saving_mw


def plan(df: pd.DataFrame, c: Convergence, results: list[TechniqueResult] | None = None, ctx: dict | None = None) -> Plan:
    """Rank technique assessments by the share of this target's gap each covers, in the target's component."""
    b = c.budget
    ctx = dict(ctx or {})
    ctx.setdefault("design", b.design)
    ctx.setdefault("workload", b.workload)
    ctx.setdefault("operating_point", b.operating_point)
    if results is None:
        results = assess_all(df, ctx)
    v, f = _corner(df, b)
    comp = c.component
    lines, skipped = [], []
    for r in results:
        if not r.assessable:
            continue
        t = BY_KEY[r.technique]
        if t.reduces == "corner" and (comp != "total" or b.operating_point):
            skipped.append(f"{t.name}: changes the corner; the target is {b.short_metric}" + (f" at {b.operating_point}" if b.operating_point else ""))
            continue
        s_mw = r.saving_for(comp)
        if not s_mw > 0:
            skipped.append(f"{t.name}: no {comp} saving")
            continue
        sav = saving_in_metric(s_mw, b.metric, v, f)
        if not np.isfinite(sav):
            skipped.append(f"{t.name}: no voltage / frequency for the scope to convert mW to pF")
            continue
        lines.append(PlanLine(t.key, t.name, sav, 0.0, sav / c.gap * 100 if c.gap > 0 else float("nan"), r.assumptions))
    lines.sort(key=lambda l: l.saving, reverse=True)
    cum = 0.0
    for l in lines:
        cum += l.saving
        l.cumulative = cum
    remainder = max(c.gap - cum, 0.0) if c.gap > 0 else 0.0
    return Plan(c, lines, remainder, v, f, skipped)


def render_convergence(items: list[Convergence]) -> str:
    if not items:
        return "No convergence targets matched the dataset."
    rows = []
    for c in items:
        b, s = c.budget, c.status
        proj = "-" if c.verdict == "CONVERGED" else (f"{c.builds_to_target:.1f} builds" if np.isfinite(c.builds_to_target) else "never at this trend")
        trend = fmt_metric(c.trend_per_build, b.metric) + "/build" if np.isfinite(c.trend_per_build) else "n/a"
        rows.append([b.design, b.scope, b.short_metric, s.build, s.milestone, fmt_metric(b.target, b.metric), fmt_metric(s.actual, b.metric),
                     fmt_metric(c.gap, b.metric) if c.gap > 0 else "-", fmt_pct(c.required_cut_pct) if c.gap > 0 else "-",
                     trend, proj, s.status + ("" if s.complete else " (INCOMPLETE)"), c.verdict])
    out = ["POWER CONVERGENCE", "",
           table(["Design", "Scope", "Metric", "Build", "Milestone", "Target", "Actual", "Gap", "Cut needed", "Trend", "Projection", "Status", "Verdict"],
                 rows, ["l", "l", "l", "l", "l", "r", "r", "r", "r", "r", "r", "l", "l"]), ""]
    n = {v: sum(c.verdict == v for c in items) for v in VERDICTS}
    out.append(f"{len(items)} targets: " + ", ".join(f"{k} {n[k]}" for k in VERDICTS if n[k]) + ".")
    out.append("CdynTot = (BE power - leakage) / (V^2 f) summed over the scope, in pF; LkgPwr = signoff leakage summed over the scope. "
               "Status applies the milestone tolerance; the verdict looks at the trend across builds toward the target itself.")
    return "\n".join(out)


def render_plan(p: Plan, top: int = 6) -> str:
    c, b = p.convergence, p.convergence.budget
    head = f"Closure plan  {b.label}   gap {fmt_metric(c.gap, b.metric)} ({fmt_pct(c.gap_pct, True)} over target), component: {c.component}"
    if c.gap <= 0:
        return head + "\n  target met; no plan needed."
    if not p.lines:
        return head + "\n  no assessable technique moves this component" + ("; " + "; ".join(p.not_applicable) if p.not_applicable else "")
    rows = [[l.name, fmt_metric(l.saving, b.metric), fmt_pct(l.covers_pct), fmt_metric(l.cumulative, b.metric)] for l in p.lines[:top]]
    out = [head, "", table(["Technique (ranked)", "Est. saving", "Covers", "Cumulative"], rows, ["l", "r", "r", "r"]), ""]
    if p.remainder > 0:
        out.append(f"  remainder after every applicable technique: {fmt_metric(p.remainder, b.metric)} ({fmt_pct(100 - p.covered_pct)} of the gap) "
                   "-> needs an architectural change, a target renegotiation or data the pipeline lacks.")
    else:
        need = next((i + 1 for i, l in enumerate(p.lines) if l.cumulative >= c.gap), len(p.lines))
        out.append(f"  the top {need} technique{'s cover' if need > 1 else ' covers'} the gap on paper; confirm each with a what-if and the next build.")
    if b.metric == "cdyn_pf" and np.isfinite(p.voltage_v):
        out.append(f"  mW -> pF at {p.voltage_v:.3f} V, {p.frequency_ghz:.2f} GHz (V^2 f = {p.voltage_v ** 2 * p.frequency_ghz:.3f})")
    if p.not_applicable:
        out.append("  not counted: " + "; ".join(p.not_applicable))
    out.append("  assumptions per technique are listed by `powermet techniques assess`; savings are estimates, not measurements.")
    return "\n".join(out)
