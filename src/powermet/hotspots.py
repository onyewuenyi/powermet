"""Power hotspots and inefficiencies: where power concentrates, where it grew, and where low-power
techniques are under-used.

Rankings for the latest build (default workload / operating point):
    hotspot        high share of design power and high power density (mW per um^2)
    regressed      largest increase vs the previous build
    inefficient    high power with low clock-gating efficiency (when PPRTL reports it)
    movement-bound data-movement share of predicted power is high (when the datamove model exists)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.schema import identity_key, label
from powermet.selection import DatasetSlice, build_order
from powermet.textfmt import fmt_mw, fmt_pct, table

DENSITY_TOP_PCT = 80      # percentile of power density above which a FUB counts as dense
SHARE_MIN_PCT = 2.0       # minimum share of design power to be called a hotspot
CG_LOW = 0.6              # clock-gating efficiency below this is "low"


@dataclass
class HotspotReport:
    design: str
    build: str
    prev_build: str | None
    table: pd.DataFrame        # per FUB
    partitions: pd.DataFrame   # per partition rollup


def hotspots(df: pd.DataFrame, design: str, workload: str | None = None, operating_point: str | None = None,
             build: str | None = None) -> HotspotReport:
    hist = DatasetSlice(design=design, workload=workload, operating_point=operating_point, latest_only=False).apply(df)
    order = build_order(hist["build"])
    cur_b = build or order[-1]
    prev_b = order[order.index(cur_b) - 1] if cur_b in order and order.index(cur_b) > 0 else None
    key = identity_key(hist)
    cur = hist[hist["build"].astype(str) == cur_b].copy()
    total = float(cur["be_mw"].sum())
    t = cur[[c for c in (key, "fub", "partition", "be_mw", "area", "cg_efficiency", "wire_cap_fraction") if c in cur.columns]].copy()
    t["share_pct"] = t["be_mw"] / total * 100 if total else np.nan
    t["power_density"] = t["be_mw"] / pd.to_numeric(t["area"], errors="coerce").where(lambda a: a > 0) if "area" in t.columns else np.nan
    if prev_b:
        prev = hist[hist["build"].astype(str) == prev_b].set_index(key)["be_mw"]
        t["prev_mw"] = t[key].map(prev)
        t["delta_pct"] = (t["be_mw"] - t["prev_mw"]) / t["prev_mw"].where(t["prev_mw"] > 0) * 100
    else:
        t["prev_mw"], t["delta_pct"] = np.nan, np.nan
    dens_cut = float(np.nanpercentile(t["power_density"], DENSITY_TOP_PCT)) if t["power_density"].notna().any() else np.inf
    flags = []
    for _, r in t.iterrows():
        f = []
        if r["share_pct"] >= SHARE_MIN_PCT and r["power_density"] >= dens_cut:
            f.append("hotspot")
        if pd.notna(r.get("delta_pct")) and r["delta_pct"] > 5:
            f.append("regressed")
        if "cg_efficiency" in t.columns and pd.notna(r.get("cg_efficiency")) and r["cg_efficiency"] < CG_LOW and r["share_pct"] >= SHARE_MIN_PCT / 2:
            f.append("low-cg")
        flags.append(";".join(f))
    t["flags"] = flags
    parts = pd.DataFrame()
    if "partition" in t.columns:
        g = t.groupby("partition")
        parts = pd.DataFrame({"be_mw": g["be_mw"].sum(), "share_pct": g["share_pct"].sum(),
                              "area": g["area"].sum() if "area" in t.columns else np.nan,
                              "n_hotspots": g["flags"].apply(lambda s: int(s.str.contains("hotspot").sum())),
                              "delta_pct": (g["be_mw"].sum() - g["prev_mw"].sum()) / g["prev_mw"].sum().where(lambda s: s > 0) * 100 if prev_b else np.nan})
        parts["power_density"] = parts["be_mw"] / parts["area"].where(parts["area"] > 0)
        parts = parts.sort_values("be_mw", ascending=False).reset_index()
    return HotspotReport(design, cur_b, prev_b, t.sort_values("be_mw", ascending=False).reset_index(drop=True), parts)


def render_hotspots(rep: HotspotReport, top: int = 10) -> str:
    key = "model_root" if "model_root" in rep.table.columns else "fub"
    out = [f"Power hotspots  design={rep.design}  build={rep.build}" + (f"  (vs {rep.prev_build})" if rep.prev_build else ""), ""]
    has_cg = "cg_efficiency" in rep.table.columns and rep.table["cg_efficiency"].notna().any()
    hdr = ["Model root" if key == "model_root" else "FUB", "Power", "Share", "mW/um2", "dPrev"] + (["CG eff"] if has_cg else []) + ["Flags"]
    rows = []
    for _, r in rep.table.head(top).iterrows():
        row = [r[key], fmt_mw(r["be_mw"]), fmt_pct(r["share_pct"]), f"{r['power_density']:.4f}" if pd.notna(r["power_density"]) else "n/a",
               fmt_pct(r["delta_pct"], True) if pd.notna(r["delta_pct"]) else "n/a"]
        if has_cg:
            row.append(f"{r['cg_efficiency']:.2f}" if pd.notna(r["cg_efficiency"]) else "n/a")
        row.append(r["flags"])
        rows.append(row)
    out.append(table(hdr, rows, ["l", "r", "r", "r", "r"] + (["r"] if has_cg else []) + ["l"]))
    flagged = rep.table[rep.table["flags"] != ""]
    if len(flagged) > top:
        out.append(f"... {len(flagged)} FUBs carry flags in total (showing the top {top} by power)")
    if len(rep.partitions):
        out += ["", "Per partition", ""]
        out.append(table(["Partition", "Power", "Share", "mW/um2", "dPrev", "Hotspots"],
                         [[r["partition"], fmt_mw(r["be_mw"]), fmt_pct(r["share_pct"]),
                           f"{r['power_density']:.4f}" if pd.notna(r["power_density"]) else "n/a",
                           fmt_pct(r["delta_pct"], True) if pd.notna(r["delta_pct"]) else "n/a", int(r["n_hotspots"])]
                          for _, r in rep.partitions.iterrows()]))
    out.append("")
    out.append(f"hotspot = share >= {SHARE_MIN_PCT:g}% and density in the top {100 - DENSITY_TOP_PCT}%; regressed = > +5% vs previous build; "
               f"low-cg = clock-gating efficiency < {CG_LOW:g} with material power.")
    return "\n".join(out)
