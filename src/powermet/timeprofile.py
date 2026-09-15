"""Time-based power profile analysis: what the averaged number hides inside a workload.

From `power_profile.parquet` (design power per time window, per build / workload / operating point):
    average, peak window and peak-to-average ratio      -> the vector for thermal / IR signoff vs the energy number
    largest window-to-window step (a di/dt proxy)       -> where a power-delivery review should look
    energy over the run                                 -> what the workload costs, independent of its length
    reconciliation with the averaged hierarchical report -> the two runs used the same netlist and activity, or not
Workloads are then ranked by peak (which one sets TDP) and by energy per op when throughput is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.selection import build_order
from powermet.textfmt import fmt_mw, fmt_pct, table


@dataclass
class ProfileSummary:
    design: str
    build: str
    table: pd.DataFrame        # per workload x operating point
    tolerance_pct: float

    @property
    def peak_workload(self) -> str | None:
        return str(self.table.sort_values("peak_mw", ascending=False)["workload"].iloc[0]) if len(self.table) else None


def summarize_profile(profile: pd.DataFrame, design: str, build: str | None = None, wide: pd.DataFrame | None = None,
                      perf: pd.DataFrame | None = None, tolerance_pct: float = 5.0) -> ProfileSummary:
    p = profile[profile["design"].astype(str) == design]
    if not len(p):
        raise ValueError(f"no power profile for {design} (optional source power_profile)")
    order = build_order(p["build"])
    b = build or order[-1]
    p = p[p["build"].astype(str) == b].copy()
    if not len(p):
        raise ValueError(f"no power profile for {design} build {b}")
    rows = []
    for (wl, op), g in p.groupby(["workload", "operating_point"], sort=True):
        g = g.sort_values("t_start_ns")
        dt = (g["t_end_ns"] - g["t_start_ns"]).astype(float)
        pw = pd.to_numeric(g["profile_total_mw"], errors="coerce").astype(float)
        ok = dt > 0
        if not ok.any():
            continue
        dur = float(dt[ok].sum())
        avg = float((pw[ok] * dt[ok]).sum() / dur)
        i_peak = int(pw.idxmax())
        steps = pw.diff().abs() / dt.shift(0)
        i_step = int(steps.idxmax()) if steps.notna().any() else i_peak
        energy_uj = avg * dur * 1e-6                     # mW * ns = 1e-12 J = 1e-6 uJ
        row = {"workload": wl, "operating_point": op, "n_windows": int(len(g)), "duration_ns": dur,
               "avg_mw": avg, "peak_mw": float(pw.max()), "min_mw": float(pw.min()),
               "peak_t_start_ns": float(g.loc[i_peak, "t_start_ns"]), "peak_t_end_ns": float(g.loc[i_peak, "t_end_ns"]),
               "peak_to_avg": float(pw.max() / avg) if avg > 0 else np.nan,
               "max_step_mw_per_ns": float(steps.max()) if steps.notna().any() else np.nan,
               "max_step_t_ns": float(g.loc[i_step, "t_start_ns"]),
               "energy_uj": energy_uj, "reported_avg_mw": np.nan, "avg_gap_pct": np.nan, "energy_pj_per_op": np.nan}
        if wide is not None and len(wide):
            w = wide[(wide["design"].astype(str) == design) & (wide["build"].astype(str) == b)
                     & (wide["workload"].astype(str) == str(wl)) & (wide["operating_point"].astype(str) == str(op))]
            if len(w):
                rep = float(pd.to_numeric(w["be_mw"], errors="coerce").sum())
                row["reported_avg_mw"] = rep
                row["avg_gap_pct"] = (avg - rep) / rep * 100 if rep > 0 else np.nan
        if perf is not None and len(perf) and "throughput_gops" in perf.columns:
            q = perf[(perf["design"].astype(str) == design) & (perf["build"].astype(str) == b)
                     & (perf["workload"].astype(str) == str(wl)) & (perf["operating_point"].astype(str) == str(op))]
            if len(q) and pd.notna(q["throughput_gops"].iloc[0]) and float(q["throughput_gops"].iloc[0]) > 0:
                row["energy_pj_per_op"] = avg / float(q["throughput_gops"].iloc[0])
        rows.append(row)
    t = pd.DataFrame(rows)
    t["profile_ok"] = ~(t["avg_gap_pct"].abs() > tolerance_pct)
    return ProfileSummary(design, b, t.sort_values("peak_mw", ascending=False).reset_index(drop=True), tolerance_pct)


def render_profile(s: ProfileSummary) -> str:
    out = [f"Workload power profile  design={s.design}  build={s.build}", ""]
    rows = []
    for r in s.table.itertuples():
        gap = "n/a" if not np.isfinite(r.avg_gap_pct) else (fmt_pct(r.avg_gap_pct, True) + ("" if r.profile_ok else " MISMATCH"))
        rows.append([r.workload, r.operating_point, fmt_mw(r.avg_mw), fmt_mw(r.peak_mw), f"{r.peak_to_avg:.2f}",
                     f"{r.peak_t_start_ns:,.0f}-{r.peak_t_end_ns:,.0f} ns", f"{r.max_step_mw_per_ns:,.1f}" if np.isfinite(r.max_step_mw_per_ns) else "n/a",
                     f"{r.energy_uj:,.1f} uJ", f"{r.energy_pj_per_op:.1f}" if np.isfinite(r.energy_pj_per_op) else "n/a", gap])
    out.append(table(["Workload", "Op", "Average", "Peak", "Peak/avg", "Peak window", "Max step mW/ns", "Energy", "pJ/op", "vs report"], rows,
                     ["l", "l", "r", "r", "r", "l", "r", "r", "r", "r"]))
    if len(s.table):
        pk = s.table.iloc[0]
        out += ["", f"Peak-power vector: {pk.workload}/{pk.operating_point} at {pk.peak_t_start_ns:,.0f}-{pk.peak_t_end_ns:,.0f} ns "
                    f"({fmt_mw(pk.peak_mw)}, {pk.peak_to_avg:.2f}x its average); use that window for IR / thermal signoff, the averages for energy."]
        bad = s.table[~s.table["profile_ok"]]
        if len(bad):
            out.append(f"MISMATCH: profile average differs from the averaged report by > {s.tolerance_pct:g}% for "
                       + ", ".join(f"{r.workload}/{r.operating_point} ({fmt_pct(r.avg_gap_pct, True)})" for r in bad.itertuples())
                       + "; the two runs used different netlists, parasitics or activity windows.")
    return "\n".join(out)
