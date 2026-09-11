"""Power x timing frontier across builds (Pareto marking, ASCII scatter)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.deltas import _slice, classify
from powermet.selection import build_order
from powermet.textfmt import fmt_mw, table



@dataclass
class FrontierPoint:
    design: str
    build: str
    power_mw: float
    fmax_ghz: float
    wns_ps: float
    throughput_gops: float
    pareto: bool = False
    status: str = ""


def frontier(df: pd.DataFrame, design: str, workload: str | None = None, operating_point: str | None = None,
             perf: pd.DataFrame | None = None) -> list[FrontierPoint]:
    d = _slice(df, design, workload, operating_point)
    pts = []
    for b in build_order(d["build"]):
        g = d[d["build"].astype(str) == b]
        power = float(g["be_mw"].sum())
        fmax = float(g["fmax_ghz"].min()) if "fmax_ghz" in g.columns and g["fmax_ghz"].notna().any() else np.nan
        wns = float(g["wns_ps"].min()) if "wns_ps" in g.columns and g["wns_ps"].notna().any() else np.nan
        thr = np.nan
        if perf is not None and len(perf):
            wl = g["workload"].iloc[0] if "workload" in g.columns else None
            o = g["operating_point"].iloc[0] if "operating_point" in g.columns else None
            sel = perf[(perf["design"] == design) & (perf["build"] == b)]
            if wl is not None and "workload" in sel.columns:
                sel = sel[sel["workload"] == wl]
            if o is not None and "operating_point" in sel.columns:
                sel = sel[sel["operating_point"] == o]
            if len(sel):
                thr = float(sel["throughput_gops"].iloc[0])
        pts.append(FrontierPoint(design, b, power, fmax, wns, thr))
    for p in pts:
        p.pareto = np.isfinite(p.fmax_ghz) and not any(
            o is not p and np.isfinite(o.fmax_ghz) and o.power_mw <= p.power_mw and o.fmax_ghz >= p.fmax_ghz
            and (o.power_mw < p.power_mw or o.fmax_ghz > p.fmax_ghz) for o in pts)
    for prev, cur in zip(pts[:-1], pts[1:]):
        dp = (cur.power_mw - prev.power_mw) / prev.power_mw * 100 if prev.power_mw else 0.0
        cur.status = classify(dp, cur.wns_ps - prev.wns_ps) if np.isfinite(cur.wns_ps) and np.isfinite(prev.wns_ps) else ""
    return pts


def render_frontier(pts: list[FrontierPoint], workload: str | None, operating_point: str | None) -> str:
    if not pts:
        return "No builds."
    out = [f"Power x Timing frontier  design={pts[0].design}  workload={workload or 'default'}  operating point={operating_point or 'default'}", ""]
    rows = [[p.build, fmt_mw(p.power_mw), f"{p.fmax_ghz:.3f}" if np.isfinite(p.fmax_ghz) else "n/a",
             f"{p.wns_ps:+.1f}" if np.isfinite(p.wns_ps) else "n/a",
             f"{p.throughput_gops:.1f}" if np.isfinite(p.throughput_gops) else "n/a",
             f"{p.power_mw / p.fmax_ghz:.1f}" if np.isfinite(p.fmax_ghz) and p.fmax_ghz > 0 else "n/a",
             "*" if p.pareto else "", p.status] for p in pts]
    out.append(table(["Build", "Power", "Fmax GHz", "WNS ps", "Gops/s", "mW per GHz", "Pareto", "vs previous"], rows,
                     ["l", "r", "r", "r", "r", "r", "l", "l"]))
    out.append("")
    out.append(ascii_scatter(pts))
    out.append("")
    par = [p.build for p in pts if p.pareto]
    out.append("Pareto-optimal builds (no other build has both lower power and higher Fmax): " + (", ".join(par) or "none"))
    out.append("Performance axis = worst-partition Fmax from PrimeTime; throughput shown for reference only.")
    return "\n".join(out)


def ascii_scatter(pts: list[FrontierPoint], width: int = 50, height: int = 12) -> str:
    ok = [p for p in pts if np.isfinite(p.fmax_ghz)]
    if len(ok) < 2:
        return ""
    xs, ys = np.array([p.fmax_ghz for p in ok]), np.array([p.power_mw for p in ok])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    grid = [[" "] * (width + 1) for _ in range(height + 1)]
    for p in ok:
        i = int(round((p.fmax_ghz - x0) / (x1 - x0) * width))
        j = height - int(round((p.power_mw - y0) / (y1 - y0) * height))
        grid[j][i] = "*" if p.pareto else "o"
    lines = [f"  Power (mW)  {y1:,.0f} ^"]
    for j, row in enumerate(grid):
        lines.append("              " + ("|" if j < height else "+") + "".join(row))
    lines.append(f"              {x0:.2f} GHz" + " " * (width - 16) + f"{x1:.2f} GHz  -> Fmax")
    lines.append("              * = Pareto-optimal build, o = dominated")
    return "\n".join(lines)
