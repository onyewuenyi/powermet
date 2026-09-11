"""Workload and performance summary (V3).

Joins FUB power with design-level performance per (design, build, workload, operating point)
to produce energy per operation, workload sensitivity and activity association. Curve fits live in curves.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.correlation import pearson
from powermet.curves import DvfsCurve, PerfModel, TimingModel  # noqa: F401  (re-exported)
from powermet.selection import latest_build_per_design
from powermet.textfmt import fmt_mw, fmt_pct, fmt_r, table

KEYS = ["design", "build", "workload", "operating_point"]


def energy_pj_per_op(power_mw, throughput_gops):
    """mW / (Gops/s) = 1e-3 W / 1e9 ops/s = 1e-12 J/op = pJ/op."""
    p = pd.to_numeric(pd.Series(power_mw), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(throughput_gops), errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p) & np.isfinite(t) & (t > 0)
    out[ok] = p[ok] / t[ok]
    return out


def attach_op_params(perf: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the performance table carries frequency/voltage (merge from measurements if absent)."""
    if perf is None or not len(perf):
        return perf
    out = perf.copy()
    need = [c for c in ("frequency_ghz", "voltage_v") if c not in out.columns or out[c].isna().all()]
    if need and all(c in df.columns for c in ("design", "build", "operating_point")):
        keys = ["design", "build", "operating_point"]
        src = df[keys + [c for c in need if c in df.columns]].drop_duplicates(subset=keys)
        out = out.drop(columns=[c for c in need if c in out.columns]).merge(src, on=keys, how="left")
    return out


def design_power(df: pd.DataFrame) -> pd.DataFrame:
    """Sum FUB power per (design, build, workload, operating_point)."""
    keys = [k for k in KEYS if k in df.columns]
    g = df.groupby(keys, sort=True)
    out = pd.DataFrame({
        "n_fubs": g.size(),
        "be_mw": g["be_mw"].sum(),
        "fe_physical_mw": g["fe_physical_mw"].sum(),
        "activity_mean": g["activity"].mean() if "activity" in df.columns else np.nan,
        "frequency_ghz": g["frequency_ghz"].first() if "frequency_ghz" in df.columns else np.nan,
        "voltage_v": g["voltage_v"].first() if "voltage_v" in df.columns else np.nan,
    }).reset_index()
    return out


def join_performance(power: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    if perf is None or not len(perf):
        out = power.copy()
        out["throughput_gops"] = np.nan
        out["energy_pj_per_op"] = np.nan
        return out
    keys = [k for k in KEYS if k in power.columns and k in perf.columns]
    cols = keys + [c for c in ("ipc", "throughput_gops") if c in perf.columns]
    out = power.merge(perf[cols], on=keys, how="left")
    out["energy_pj_per_op"] = energy_pj_per_op(out["be_mw"], out["throughput_gops"])
    return out


@dataclass
class WorkloadSummary:
    table: pd.DataFrame                 # design x build x workload x op with power, throughput, energy/op
    latest: pd.DataFrame                # same, latest build per design
    sensitivity: pd.DataFrame           # per design/FUB: power range across workloads
    activity_assoc: pd.DataFrame        # per design: corr(activity, be_mw)
    perf_model: PerfModel
    dvfs: DvfsCurve
    timing: TimingModel = field(default_factory=TimingModel)


def summarize_workloads(df: pd.DataFrame, perf: pd.DataFrame | None) -> WorkloadSummary:
    perf = attach_op_params(perf, df)
    power = design_power(df)
    joined = join_performance(power, perf)
    lb = latest_build_per_design(joined) if len(joined) else {}
    latest = joined[[lb.get(str(d)) == str(b) for d, b in zip(joined["design"], joined["build"])]].reset_index(drop=True)

    sens = pd.DataFrame()
    if "workload" in df.columns and df["workload"].nunique() > 1:
        lat = df[df["build"].isin(latest["build"].unique())]
        piv = lat.pivot_table(index=["design", "fub"], columns="workload", values="be_mw", aggfunc="mean")
        sens = pd.DataFrame({"min_mw": piv.min(axis=1), "max_mw": piv.max(axis=1)})
        sens["dynamic_range"] = sens["max_mw"] / sens["min_mw"].where(sens["min_mw"] > 0)
        sens = sens.reset_index().sort_values("dynamic_range", ascending=False)

    assoc_rows = []
    if "activity" in df.columns:
        for d, g in df.groupby("design"):
            r, n, _ = pearson(g["activity"], g["be_mw"])
            r_norm, _, _ = pearson(g["activity"], g["be_mw"] / g["fe_physical_mw"].where(g["fe_physical_mw"] > 0))
            assoc_rows.append({"design": d, "r_activity_be": r, "r_activity_ratio": r_norm, "n": n})
    return WorkloadSummary(joined, latest, sens, pd.DataFrame(assoc_rows),
                           PerfModel().fit(perf) if perf is not None and len(perf) else PerfModel(),
                           DvfsCurve().fit(df), TimingModel().fit(df))


def render_workload_summary(ws: WorkloadSummary, top: int = 8) -> str:
    out = ["Power, performance and energy per workload x operating point (latest build per design)", ""]
    rows = []
    for _, r in ws.latest.sort_values(["design", "workload", "operating_point"]).iterrows():
        rows.append([r["design"], r["build"], r["workload"], r["operating_point"],
                     f"{r['frequency_ghz']:.2f}" if pd.notna(r["frequency_ghz"]) else "", 
                     f"{r['voltage_v']:.2f}" if pd.notna(r["voltage_v"]) else "",
                     fmt_mw(r["be_mw"]),
                     f"{r['throughput_gops']:.1f}" if pd.notna(r["throughput_gops"]) else "n/a",
                     f"{r['energy_pj_per_op']:.2f}" if pd.notna(r["energy_pj_per_op"]) else "n/a"])
    out.append(table(["Design", "Build", "Workload", "OP", "f GHz", "V", "BE power", "Gops/s", "pJ/op"], rows,
                     ["l", "l", "l", "l", "r", "r", "r", "r", "r"]))
    if len(ws.latest) and ws.latest["energy_pj_per_op"].notna().any():
        out.append("")
        out.append("Most energy-efficient operating point per design/workload:")
        best = ws.latest.loc[ws.latest.groupby(["design", "workload"])["energy_pj_per_op"].idxmin().dropna()]
        out.append(table(["Design", "Workload", "OP", "pJ/op"],
                         [[r["design"], r["workload"], r["operating_point"], f"{r['energy_pj_per_op']:.2f}"] for _, r in best.iterrows()]))
    if len(ws.sensitivity):
        out += ["", f"FUBs with the widest power range across workloads (top {top}):", ""]
        rows = [[r["design"], r["fub"], fmt_mw(r["min_mw"]), fmt_mw(r["max_mw"]), f"{r['dynamic_range']:.1f}x"]
                for _, r in ws.sensitivity.head(top).iterrows()]
        out.append(table(["Design", "FUB", "min", "max", "range"], rows))
    if len(ws.activity_assoc):
        out += ["", "Association of activity with BE power (Pearson r; within design):", ""]
        rows = [[r["design"], fmt_r(r["r_activity_be"]), fmt_r(r["r_activity_ratio"]), f"{int(r['n']):,}"] for _, r in ws.activity_assoc.iterrows()]
        out.append(table(["Design", "r(activity, BE)", "r(activity, BE/FE)", "n"], rows))
    out += ["", ws.perf_model.render(), "", ws.dvfs.render()]
    if ws.timing.params:
        out += ["", ws.timing.render()]
    return "\n".join(out)
