"""Build-to-build deltas: power, timing (partition WNS / Fmax), physical metrics, and trade classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.schema import identity_key, label
from powermet.selection import DatasetSlice, build_order
from powermet.textfmt import fmt_mw, fmt_pct, table



DELTA_METRICS = ("be_mw", "fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "wire_length_um")
TIMING_METRICS = ("wns_ps", "tns_ps", "fmax_ghz")


@dataclass
class BuildDelta:
    design: str
    build_from: str
    build_to: str
    totals: dict[str, tuple[float, float]]      # metric -> (from, to)
    fub_table: pd.DataFrame                     # per FUB deltas
    classification: str
    partition_table: pd.DataFrame

    def pct(self, m: str) -> float:
        a, b = self.totals.get(m, (np.nan, np.nan))
        return (b - a) / a * 100 if a else np.nan


POWER_TOL_PCT = 0.5
TIMING_TOL_PS = 1.0


def classify(d_power: float, d_timing: float, tol: float = POWER_TOL_PCT, timing_tol: float = TIMING_TOL_PS) -> str:
    """Classify a build move. d_power in percent (up = worse), d_timing = change in WNS in ps (up = better).

    Changes within +/-tol (power) or +/-timing_tol (timing) count as flat.
    """
    if not np.isfinite(d_timing):
        return "timing unknown"
    if not np.isfinite(d_power):
        return "power unknown"
    p_up, p_down = d_power > tol, d_power < -tol
    t_up, t_down = d_timing > timing_tol, d_timing < -timing_tol
    if not (p_up or p_down) and not (t_up or t_down):
        return "neutral"
    if (p_down or not p_up) and (t_up or not t_down):
        return "pareto improvement"
    if (p_up or not p_down) and (t_down or not t_up):
        return "regression"
    if p_up and t_up:
        return "power-for-performance trade"
    return "performance-for-power trade"


def _slice(df: pd.DataFrame, design: str, workload: str | None, operating_point: str | None) -> pd.DataFrame:
    """All builds of one design at one workload x operating point (defaults: typical / nom)."""
    return DatasetSlice(design=design, workload=workload, operating_point=operating_point, latest_only=False).apply(df)


def build_deltas(df: pd.DataFrame, design: str, workload: str | None = None, operating_point: str | None = None,
                 pairs: list[tuple[str, str]] | None = None) -> list[BuildDelta]:
    d = _slice(df, design, workload, operating_point)
    builds = build_order(d["build"])
    if pairs is None:
        pairs = list(zip(builds[:-1], builds[1:]))
    key = identity_key(d)
    out = []
    for a, b in pairs:
        da, db = d[d["build"].astype(str) == a], d[d["build"].astype(str) == b]
        common = sorted(set(da[key]) & set(db[key]))
        da, db = da.set_index(key).loc[common], db.set_index(key).loc[common]
        totals = {}
        for m in DELTA_METRICS:
            if m in d.columns:
                totals[m] = (float(da[m].sum()), float(db[m].sum()))
        parts = pd.DataFrame()
        if "partition" in d.columns and "wns_ps" in d.columns:
            pa = da.groupby("partition")[["wns_ps", "tns_ps", "fmax_ghz"]].first()
            pb = db.groupby("partition")[["wns_ps", "tns_ps", "fmax_ghz"]].first()
            parts = pa.join(pb, lsuffix="_from", rsuffix="_to", how="inner")
            pw_a = da.groupby("partition")["be_mw"].sum()
            pw_b = db.groupby("partition")["be_mw"].sum()
            parts["be_mw_from"], parts["be_mw_to"] = pw_a, pw_b
            parts = parts.reset_index()
            totals["wns_ps"] = (float(pa["wns_ps"].min()), float(pb["wns_ps"].min()))
            totals["fmax_ghz"] = (float(pa["fmax_ghz"].min()), float(pb["fmax_ghz"].min()))
        ft = pd.DataFrame({key: common})
        for m in DELTA_METRICS + TIMING_METRICS:
            if m in d.columns:
                ft[m + "_from"] = da[m].to_numpy()
                ft[m + "_to"] = db[m].to_numpy()
                ft[m + "_pct"] = (db[m].to_numpy() - da[m].to_numpy()) / np.where(da[m].to_numpy() != 0, da[m].to_numpy(), np.nan) * 100
        if "partition" in d.columns:
            ft["partition"] = da["partition"].to_numpy()
        d_power = (totals["be_mw"][1] - totals["be_mw"][0]) / totals["be_mw"][0] * 100 if totals.get("be_mw", (0, 0))[0] else np.nan
        d_timing = totals["wns_ps"][1] - totals["wns_ps"][0] if "wns_ps" in totals else np.nan
        out.append(BuildDelta(design, a, b, totals, ft, classify(d_power, d_timing), parts))
    return out


def render_deltas(deltas: list[BuildDelta], top: int = 5) -> str:
    out = []
    for bd in deltas:
        out.append(f"{bd.design}: {bd.build_from} -> {bd.build_to}    [{bd.classification.upper()}]")
        out.append("")
        rows = []
        for m in ("be_mw", "fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "wire_length_um"):
            if m in bd.totals:
                a, b = bd.totals[m]
                rows.append([label(m), f"{a:,.1f}", f"{b:,.1f}", fmt_pct(bd.pct(m), signed=True)])
        if "wns_ps" in bd.totals:
            a, b = bd.totals["wns_ps"]
            rows.append(["WNS (worst partition, ps)", f"{a:+.1f}", f"{b:+.1f}", f"{b - a:+.1f} ps"])
            a, b = bd.totals["fmax_ghz"]
            rows.append(["Fmax (worst partition)", f"{a:.3f}", f"{b:.3f}", fmt_pct((b - a) / a * 100, signed=True)])
        out.append(table(["Metric", bd.build_from, bd.build_to, "Change"], rows))
        if len(bd.partition_table):
            out.append("")
            out.append("Per partition (timing is a partition attribute):")
            out.append("")
            rows = [[r["partition"], fmt_mw(r["be_mw_from"]), fmt_mw(r["be_mw_to"]),
                     fmt_pct((r["be_mw_to"] - r["be_mw_from"]) / r["be_mw_from"] * 100, True) if r["be_mw_from"] else "n/a",
                     f"{r['wns_ps_from']:+.1f}", f"{r['wns_ps_to']:+.1f}", f"{r['wns_ps_to'] - r['wns_ps_from']:+.1f}",
                     classify((r["be_mw_to"] - r["be_mw_from"]) / r["be_mw_from"] * 100 if r["be_mw_from"] else 0, r["wns_ps_to"] - r["wns_ps_from"])]
                    for _, r in bd.partition_table.iterrows()]
            out.append(table(["Partition", "Power from", "to", "dP", "WNS from", "to", "dWNS ps", "Class"], rows))
        ft = bd.fub_table
        if len(ft) and "be_mw_pct" in ft.columns:
            out.append("")
            out.append(f"Largest power movers (top {top}):")
            out.append("")
            key = ft.columns[0]
            mv = ft.reindex(ft["be_mw_pct"].abs().sort_values(ascending=False).index).head(top)
            cols = [c for c in ("wire_cap_pf_pct", "cell_cap_pf_pct", "area_pct") if c in ft.columns]
            rows = [[r[key], fmt_mw(r["be_mw_from"]), fmt_mw(r["be_mw_to"]), fmt_pct(r["be_mw_pct"], True)] + [fmt_pct(r[c], True) for c in cols]
                    for _, r in mv.iterrows()]
            out.append(table(["Model root" if key == "model_root" else "FUB", "Power from", "to", "dP"] + [label(c[:-4]) + " d" for c in cols], rows))
        out.append("")
    out.append("Classification: pareto improvement = power down & timing up; regression = power up & timing down;")
    out.append("power-for-performance trade = both up; performance-for-power trade = both down. Timing = worst-partition WNS.")
    return "\n".join(out)
