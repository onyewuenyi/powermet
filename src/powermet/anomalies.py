"""Comparative power analysis: spot the anomalies that warrant scrutiny, name the likely power bug, and
say who owns it.

A hotspot is where power is large; an anomaly is where power is *wrong for what the block is doing*.
Every rule compares a FUB against something that should agree with it: the same FUB at an idle workload,
its peers at the same activity and capacitance, its own previous builds, its replica instances, or the
category split of its own power. Findings carry the evidence, the threshold, the technique that usually
fixes the class of bug, and the owner from the FUB map, so the output is a list to drive rather than a
chart to admire.

Rules are a registry (`RULES`): adding one is a `Rule` entry plus a function returning findings; the CLI,
the catalog record and the report follow. Thresholds live in Config (`anomaly_*`) so a company tunes them
once against real magnitudes. Associations only: a rule says "this looks like a gating bug", the designer
confirms it in the RTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from powermet.config import Config
from powermet.schema import identity_key
from powermet.selection import DatasetSlice, build_order, default_value
from powermet.textfmt import fmt_pct, table


@dataclass
class Finding:
    design: str
    build: str
    model_root: str
    fub: str
    partition: str | None
    owner: str | None
    rule: str
    severity: str            # high | medium
    value: float
    threshold: float
    unit: str
    evidence: str
    technique: str | None
    status: str = "new"      # new | persisting (present in the previous build too)
    workload: str | None = None
    operating_point: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.model_root, self.rule)


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    question: str            # what it compares
    bug: str                 # the class of power bug it points at
    severity: str
    technique: str | None    # techniques.py key that usually addresses it
    needs: tuple[str, ...]   # columns that must be present, else the rule is skipped (and says so)
    fn: Callable[["Context"], list[Finding]]


@dataclass
class Context:
    df: pd.DataFrame          # sanitized wide dataset (all builds of one design)
    design: str
    build: str
    prev_build: str | None
    workload: str
    operating_point: str
    cfg: Config
    key: str = "model_root"
    skipped: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers

def _rows(ctx: Context, build: str | None = None, workload: str | None = None, op: str | None = None) -> pd.DataFrame:
    d = ctx.df
    sel = (d["build"].astype(str) == (build or ctx.build)) & (d["workload"].astype(str) == (workload or ctx.workload)) \
        & (d["operating_point"].astype(str) == (op or ctx.operating_point))
    return d[sel].copy()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _finding(ctx: Context, r: pd.Series, rule: Rule, value: float, threshold: float, unit: str, evidence: str,
             workload: str | None = None, op: str | None = None, model_root: str | None = None) -> Finding:
    """`r` is a dataset row (Series); when the frame was indexed by the identity key, pass `model_root` explicitly."""
    root = model_root if model_root is not None else str(r[ctx.key])
    owner = r.get("owner")
    part = r.get("partition")
    return Finding(ctx.design, ctx.build, root, str(r["fub"]), str(part) if pd.notna(part) else None,
                   str(owner) if pd.notna(owner) else None, rule.key, rule.severity, float(value), float(threshold), unit, evidence,
                   rule.technique, workload=workload or ctx.workload, operating_point=op or ctx.operating_point)


def _prev_rows(ctx: Context) -> pd.DataFrame:
    """For each FUB, its row from the most recent earlier build that has one (a FUB excluded by sanitize in the
    design's previous build is compared with its own last good build); indexed by the identity key, with `_build`."""
    d = ctx.df
    order = build_order(d["build"])
    earlier = order[:order.index(ctx.build)]
    sel = d["build"].astype(str).isin(earlier) & (d["workload"].astype(str) == ctx.workload) & (d["operating_point"].astype(str) == ctx.operating_point)
    rows = d[sel].copy()
    if not len(rows):
        return rows.set_index(ctx.key) if ctx.key in rows.columns else rows
    rows["_order"] = rows["build"].astype(str).map({b: i for i, b in enumerate(order)})
    rows = rows.sort_values("_order").drop_duplicates(subset=[ctx.key], keep="last").rename(columns={"build": "_build"})
    return rows.set_index(ctx.key)


def _base_fub(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split("@", n=1).str[0]


# ----------------------------------------------------------------------------- rules

def rule_idle_dynamic(ctx: Context) -> list[Finding]:
    """Dynamic power at the idle workload as a fraction of dynamic power at the reference workload."""
    idle = ctx.cfg.idle_workload
    if idle not in set(ctx.df["workload"].astype(str)) or idle == ctx.workload:
        ctx.skipped["idle_dynamic"] = f"no '{idle}' workload in the dataset"
        return []
    ref = _rows(ctx).set_index(ctx.key)
    idl = _rows(ctx, workload=idle).set_index(ctx.key)
    common = ref.index.intersection(idl.index)
    out = []
    for k in common:
        d_ref, d_idle = float(_num(ref, "be_dynamic_mw")[k]), float(_num(idl, "be_dynamic_mw")[k])
        if not (d_ref > 0):
            continue
        ratio = d_idle / d_ref
        if ratio > ctx.cfg.anomaly_idle_ratio:
            a_ref, a_idle = _num(ref, "activity")[k], _num(idl, "activity")[k]
            act = f"; activity ratio {a_idle / a_ref:.2f}" if pd.notna(a_ref) and a_ref > 0 and pd.notna(a_idle) else ""
            out.append(_finding(ctx, ref.loc[k], RULES["idle_dynamic"], ratio, ctx.cfg.anomaly_idle_ratio, "ratio",
                                f"idle dynamic {d_idle:,.1f} mW is {ratio:.0%} of {ctx.workload} ({d_ref:,.1f} mW){act}",
                                workload=idle, model_root=str(k)))
    return out


def rule_activity_power_mismatch(ctx: Context) -> list[Finding]:
    """Dynamic power per unit of activity x capacitance x V^2 f, against the design median (robust)."""
    cur = _rows(ctx)
    act, cap = _num(cur, "activity"), _num(cur, "wire_cap_pf") + _num(cur, "cell_cap_pf")
    v, f = _num(cur, "voltage_v"), _num(cur, "frequency_ghz")
    drive = act * cap * v ** 2 * f
    eff = _num(cur, "be_dynamic_mw") / drive.where(drive > 0)
    ok = eff.notna() & (eff > 0)
    if ok.sum() < 8:
        ctx.skipped["activity_power_mismatch"] = "fewer than 8 FUBs with activity, capacitance and corner"
        return []
    le = np.log(eff[ok])
    med = float(np.median(le))
    mad = float(np.median(np.abs(le - med))) * 1.4826 or 1e-9
    out = []
    for i in cur.index[ok]:
        ratio = float(np.exp(le[i] - med))
        z = (le[i] - med) / mad
        if ratio >= ctx.cfg.anomaly_eff_ratio and z >= 3:
            out.append(_finding(ctx, cur.loc[i], RULES["activity_power_mismatch"], ratio, ctx.cfg.anomaly_eff_ratio, "x median",
                                f"dynamic power per activity*C*V^2f is {ratio:.1f}x the design median (robust z {z:.1f}); "
                                f"activity {act[i]:.2f}, dynamic {_num(cur, 'be_dynamic_mw')[i]:,.1f} mW"))
    return out


def rule_unexplained_regression(ctx: Context) -> list[Finding]:
    """Dynamic power up vs the previous build by more than the change in activity x capacitance explains."""
    if not ctx.prev_build:
        ctx.skipped["unexplained_regression"] = "no previous build"
        return []
    cur, prev = _rows(ctx).set_index(ctx.key), _prev_rows(ctx)
    out = []
    for k in cur.index.intersection(prev.index):
        d0, d1 = float(_num(prev, "be_dynamic_mw")[k]), float(_num(cur, "be_dynamic_mw")[k])
        if not (d0 > 0) or not np.isfinite(d1):
            continue
        growth = (d1 - d0) / d0 * 100
        if growth <= ctx.cfg.anomaly_regression_pct:
            continue
        # what the physical and activity changes explain: dynamic ~ activity x (wire + cell) capacitance at a fixed corner
        changes = {}
        for col, lab in (("activity", "activity"), ("wire_cap_pf", "wire cap"), ("cell_cap_pf", "cell cap"), ("area", "area")):
            a, b = _num(prev, col)[k], _num(cur, col)[k]
            if pd.notna(a) and a > 0 and pd.notna(b):
                changes[lab] = (b - a) / a * 100
        cap0 = _num(prev, "wire_cap_pf")[k] + _num(prev, "cell_cap_pf")[k]
        cap1 = _num(cur, "wire_cap_pf")[k] + _num(cur, "cell_cap_pf")[k]
        a0, a1 = _num(prev, "activity")[k], _num(cur, "activity")[k]
        if pd.notna(cap0) and cap0 > 0 and pd.notna(cap1):
            drive0, drive1 = cap0 * (a0 if pd.notna(a0) and a0 > 0 else 1.0), cap1 * (a1 if pd.notna(a1) and a1 > 0 else 1.0)
            explained = (drive1 - drive0) / drive0 * 100
        elif "area" in changes:
            explained = changes["area"]
        else:
            explained = 0.0
        unexplained = growth - explained
        if unexplained <= ctx.cfg.anomaly_regression_pct:
            continue
        ev = ", ".join(f"{lab} {c:+.1f}%" for lab, c in changes.items()) or "no physical data"
        out.append(_finding(ctx, cur.loc[k], RULES["unexplained_regression"], unexplained, ctx.cfg.anomaly_regression_pct, "%",
                            f"dynamic {d0:,.1f} -> {d1:,.1f} mW ({growth:+.1f}%) vs {prev.loc[k, '_build']}; activity x capacitance explains "
                            f"{explained:+.1f}% ({ev})", model_root=str(k)))
    return out


def rule_creeping_growth(ctx: Context) -> list[Finding]:
    """Dynamic power rising in each of the last N builds with a cumulative growth above the threshold."""
    order = build_order(ctx.df["build"])
    i = order.index(ctx.build)
    n = ctx.cfg.anomaly_creep_builds
    if i < n:
        ctx.skipped["creeping_growth"] = f"fewer than {n + 1} builds"
        return []
    builds = order[i - n:i + 1]
    series = {b: _rows(ctx, build=b).set_index(ctx.key)["be_dynamic_mw"] for b in builds}
    cur = _rows(ctx).set_index(ctx.key)
    out = []
    for k in cur.index:
        vals = [float(series[b].get(k, np.nan)) for b in builds]
        if any(not np.isfinite(x) or x <= 0 for x in vals):
            continue
        if all(b > a for a, b in zip(vals, vals[1:])):
            growth = (vals[-1] - vals[0]) / vals[0] * 100
            if growth > ctx.cfg.anomaly_creep_pct:
                out.append(_finding(ctx, cur.loc[k], RULES["creeping_growth"], growth, ctx.cfg.anomaly_creep_pct, "%",
                                    f"dynamic rose in {n} consecutive builds {builds[0]} -> {builds[-1]}: " + " -> ".join(f"{x:,.0f}" for x in vals) + " mW",
                                    model_root=str(k)))
    return out


def rule_clock_dominant(ctx: Context) -> list[Finding]:
    """Clock-network share of dynamic power (needs the power-groups report)."""
    cur = _rows(ctx)
    frac = _num(cur, "clock_fraction")
    if frac.notna().sum() == 0:
        ctx.skipped["clock_dominant"] = "no power-groups report (be_clock_mw)"
        return []
    out = []
    for i in cur.index[frac.notna()]:
        if frac[i] > ctx.cfg.anomaly_clock_fraction and _num(cur, "be_dynamic_mw")[i] > 0:
            cg = _num(cur, "cg_efficiency")[i]
            ev = f"clock network is {frac[i]:.0%} of dynamic power ({_num(cur, 'be_clock_mw')[i]:,.1f} of {_num(cur, 'be_dynamic_mw')[i]:,.1f} mW)"
            if pd.notna(cg):
                ev += f"; clock-gating efficiency {cg:.2f}"
            out.append(_finding(ctx, cur.loc[i], RULES["clock_dominant"], frac[i], ctx.cfg.anomaly_clock_fraction, "fraction", ev))
    return out


def rule_replica_divergence(ctx: Context) -> list[Finding]:
    """Replica instances (FUB@i rows) of one module whose dynamic power differs by more than the ratio."""
    cur = _rows(ctx)
    inst = cur["fub"].astype(str).str.contains("@")
    if not inst.any():
        ctx.skipped["replica_divergence"] = "no per-instance replica rows (replica_policy = per_instance)"
        return []
    out = []
    g = cur[inst].assign(_base=_base_fub(cur.loc[inst, "fub"])).groupby("_base")
    for base, grp in g:
        dyn = _num(grp, "be_dynamic_mw")
        if dyn.notna().sum() < 2 or dyn.min() <= 0:
            continue
        ratio = float(dyn.max() / dyn.min())
        if ratio > ctx.cfg.anomaly_replica_ratio:
            hi, lo = grp.loc[dyn.idxmax()], grp.loc[dyn.idxmin()]
            out.append(_finding(ctx, hi, RULES["replica_divergence"], ratio, ctx.cfg.anomaly_replica_ratio, "max/min",
                                f"{base}: {hi['fub']} {dyn.max():,.1f} mW vs {lo['fub']} {dyn.min():,.1f} mW across {len(grp)} instances"))
    return out


def rule_leakage_share(ctx: Context) -> list[Finding]:
    """Leakage fraction far above the design median at the reference corner."""
    cur = _rows(ctx)
    lf = _num(cur, "leakage_fraction")
    ok = lf.notna() & (_num(cur, "be_mw") > 0)
    if ok.sum() < 8:
        ctx.skipped["leakage_share"] = "fewer than 8 FUBs with a leakage split"
        return []
    med = float(lf[ok].median())
    out = []
    for i in cur.index[ok]:
        if lf[i] > max(2 * med, 0.25):
            out.append(_finding(ctx, cur.loc[i], RULES["leakage_share"], lf[i], max(2 * med, 0.25), "fraction",
                                f"leakage is {lf[i]:.0%} of total vs design median {med:.0%} ({_num(cur, 'be_leakage_mw')[i]:,.1f} mW)"))
    return out


RULES: dict[str, Rule] = {r.key: r for r in (
    Rule("idle_dynamic", "Dynamic power at idle", "idle vs reference workload, same FUB",
         "clocks or datapaths not gated when the block is idle", "high", "clock_gating", ("be_dynamic_mw",), rule_idle_dynamic),
    Rule("activity_power_mismatch", "Power high for its activity", "FUB vs design peers at the same activity, capacitance and corner",
         "glitching, ungated clock tree, or activity not reaching the power run", "high", "glitch_reduction",
         ("be_dynamic_mw", "activity", "wire_cap_pf", "cell_cap_pf"), rule_activity_power_mismatch),
    Rule("unexplained_regression", "Unexplained build regression", "this build vs the previous one, net of activity x capacitance change",
         "gating lost, a new always-on path, or a tool setting changed", "high", "clock_gating",
         ("be_dynamic_mw",), rule_unexplained_regression),
    Rule("creeping_growth", "Creeping growth across builds", "the last N builds of the same FUB",
         "incremental ECOs adding power nobody signed up for", "medium", None, ("be_dynamic_mw",), rule_creeping_growth),
    Rule("clock_dominant", "Clock network dominant", "clock-network share of the FUB's own dynamic power",
         "clock tree feeding idle registers; gating enable coverage", "medium", "clock_gating", ("be_clock_mw",), rule_clock_dominant),
    Rule("replica_divergence", "Replica instances disagree", "instances of one module against each other",
         "placement, activity mapping or a per-instance constraint difference", "medium", "wire_cap_reduction",
         ("be_dynamic_mw",), rule_replica_divergence),
    Rule("leakage_share", "Leakage share far above peers", "leakage fraction vs the design median",
         "low-Vt mix or missing power gating on a mostly-idle block", "medium", "vt_swap", ("be_leakage_mw",), rule_leakage_share),
)}


# ----------------------------------------------------------------------------- driver

@dataclass
class AnomalyReport:
    design: str
    build: str
    prev_build: str | None
    workload: str
    operating_point: str
    findings: list[Finding]
    cleared: list[Finding]                 # present in the previous build, absent now
    skipped: dict[str, str]

    def by_owner(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["owner", "findings", "high", "fubs"])
        t = pd.DataFrame([{"owner": f.owner or "(no owner)", "high": f.severity == "high", "fub": f.model_root} for f in self.findings])
        g = t.groupby("owner")
        return pd.DataFrame({"findings": g.size(), "high": g["high"].sum().astype(int), "fubs": g["fub"].nunique()}).reset_index() \
            .sort_values(["high", "findings"], ascending=False).reset_index(drop=True)


def _run_rules(ctx: Context, keys: list[str] | None) -> list[Finding]:
    out: list[Finding] = []
    for key, rule in RULES.items():
        if keys and key not in keys:
            continue
        missing = [c for c in rule.needs if c not in ctx.df.columns or ctx.df[c].isna().all()]
        if missing:
            ctx.skipped[key] = f"needs {', '.join(missing)}"
            continue
        out.extend(rule.fn(ctx))
    return out


def anomalies(df: pd.DataFrame, cfg: Config, design: str, build: str | None = None, workload: str | None = None,
              operating_point: str | None = None, rules: list[str] | None = None) -> AnomalyReport:
    """Run every rule for one design at one build (default latest) and mark each finding new or persisting."""
    hist = DatasetSlice(design=design, latest_only=False, default_workload=False, default_operating_point=False).apply(df)
    if "be_dynamic_mw" not in hist.columns:
        from powermet.metrics import add_convergence_metrics
        add_convergence_metrics(hist)
    order = build_order(hist["build"])
    cur_b = build or order[-1]
    if cur_b not in order:
        raise ValueError(f"build {cur_b} not in dataset for {design}")
    prev_b = order[order.index(cur_b) - 1] if order.index(cur_b) > 0 else None
    wl = workload or default_value(hist, "workload", "typical")
    op = operating_point or default_value(hist, "operating_point", "nom")
    key = identity_key(hist)
    ctx = Context(hist, design, cur_b, prev_b, wl, op, cfg, key)
    findings = _run_rules(ctx, rules)
    cleared: list[Finding] = []
    if prev_b:
        pctx = Context(hist, design, prev_b, order[order.index(prev_b) - 1] if order.index(prev_b) > 0 else None, wl, op, cfg, key)
        prev = {f.key: f for f in _run_rules(pctx, rules)}
        now = {f.key for f in findings}
        for f in findings:
            f.status = "persisting" if f.key in prev else "new"
        cleared = [f for k, f in prev.items() if k not in now]
    sev = {"high": 0, "medium": 1}
    findings.sort(key=lambda f: (sev.get(f.severity, 2), -f.value if f.unit in ("%",) else 0, f.model_root))
    return AnomalyReport(design, cur_b, prev_b, wl, op, findings, cleared, ctx.skipped)


def render_anomalies(rep: AnomalyReport, top: int = 20, owner: str | None = None) -> str:
    out = [f"Power anomalies  design={rep.design}  build={rep.build}" + (f"  (vs {rep.prev_build})" if rep.prev_build else "")
           + f"  reference {rep.workload}/{rep.operating_point}", ""]
    fs = [f for f in rep.findings if owner is None or (f.owner or "") == owner]
    if not fs:
        out.append("no anomalies" + (f" for owner {owner}" if owner else ""))
    else:
        rows = []
        for f in fs[:top]:
            val = fmt_pct(f.value) if f.unit == "%" else (f"{f.value:.2f}" if f.unit in ("ratio", "fraction") else f"{f.value:.2f} {f.unit}")
            rows.append([f.model_root, f.owner or "-", f.rule, f.severity, val, f.status, f.evidence])
        out.append(table(["Model root", "Owner", "Rule", "Sev", "Value", "Status", "Evidence"], rows, ["l"] * 7))
        if len(fs) > top:
            out.append(f"... {len(fs)} findings in total (showing {top})")
        bo = rep.by_owner()
        if len(bo) and owner is None:
            out += ["", "By owner (who to drive)", "", table(["Owner", "Findings", "High", "FUBs"],
                                                             [[r.owner, int(r.findings), int(r.high), int(r.fubs)] for r in bo.itertuples()])]
    if rep.cleared:
        out += ["", f"Cleared since {rep.prev_build}: " + ", ".join(f"{f.model_root} [{f.rule}]" for f in rep.cleared[:10])
                + (f" (+{len(rep.cleared) - 10} more)" if len(rep.cleared) > 10 else "")]
    if rep.skipped:
        out += ["", "Rules skipped: " + "; ".join(f"{k}: {v}" for k, v in rep.skipped.items())]
    out += ["", "Rules compare a FUB with what should agree with it (idle vs busy, peers, previous builds, replicas, its own groups). "
                 "Evidence is an association; confirm the bug in RTL or the flow before fixing."]
    return "\n".join(out)


def render_rules() -> str:
    return table(["Rule", "Compares", "Points at", "Sev", "Technique", "Needs"],
                 [[r.key, r.question, r.bug, r.severity, r.technique or "-", ", ".join(r.needs)] for r in RULES.values()], ["l"] * 6)
