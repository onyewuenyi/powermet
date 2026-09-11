"""Performance-tool integration (V3): export a compact power model and evaluate workload traces.

The compact model is a JSON file a performance simulator can load without powermet:
per design, per FUB: the physical features of the latest build and the fitted
coefficients of the data-movement model (or physics model), plus the DVFS curve,
per-workload activity/traffic profiles, per-workload throughput scaling and the
timing feasibility model. `CompactPowerModel` evaluates it; `run_trace` walks a
phase trace (workload, operating point, activity scale, duration) and produces a
power / throughput / energy timeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from powermet import __version__
from powermet.curves import DvfsCurve, PerfModel, TimingModel
from powermet.extract import metrics_with_scope
from powermet.features import add_engineered_features
from powermet.modeling import LinearModel, pick_model_key
from powermet.schema import FEATURE_COLUMNS, identity_key
from powermet.selection import latest_build
from powermet.textfmt import fmt_mw, fmt_pct, table
from powermet.workload import attach_op_params, energy_pj_per_op

# per-FUB physical features (build-level) and workload-level features, derived from the metric scopes
PHYS_COLS = tuple(m for m in metrics_with_scope() if m in FEATURE_COLUMNS)
WL_COLS = tuple(m for m in metrics_with_scope("workload") if m in FEATURE_COLUMNS)


def export_compact(df: pd.DataFrame, perf: pd.DataFrame | None, payload: dict, meta: dict, model_key: str = "datamove") -> dict:
    models = payload["models"]
    model_key = pick_model_key(models, model_key if isinstance(models.get(model_key), LinearModel) else None)
    m: LinearModel = models[model_key]
    dv = DvfsCurve().fit(df)
    tm = TimingModel().fit(df)
    perf = attach_op_params(perf, df) if perf is not None else None
    pm = PerfModel().fit(perf) if perf is not None and len(perf) else PerfModel()
    iv = (meta.get("cv") or {}).get("intervals", {}).get(model_key)
    out = {
        "format": "powermet-compact-power-model", "version": 1, "powermet": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_model": meta.get("model_file"), "model_kind": model_key,
        "terms": {"features": m.features, "coefficients": m.coefficients(), "intercept": m.intercept_, "fill": m.fill_},
        "error_interval_pct": iv, "designs": {},
    }
    key = identity_key(df)
    for design, g in df.groupby("design"):
        latest = latest_build(g)
        gl = g[g["build"].astype(str) == latest]
        fubs = {}
        for root, gf in gl.groupby(key):
            base = gf.iloc[0]
            entry = {"fub": str(base.get("fub", root)), "partition": None if pd.isna(base.get("partition")) else str(base.get("partition")),
                     "physical": {c: float(base[c]) for c in PHYS_COLS if c in gf.columns and pd.notna(base[c])},
                     "workloads": {}}
            if "workload" in gf.columns:
                for wl, gw in gf.groupby("workload"):
                    entry["workloads"][str(wl)] = {c: float(gw[c].mean()) for c in WL_COLS if c in gw.columns and gw[c].notna().any()}
            fubs[str(root)] = entry
        ops = {}
        if "operating_point" in gl.columns:
            for op, go in gl.groupby("operating_point"):
                ops[str(op)] = {"voltage_v": float(go["voltage_v"].iloc[0]), "frequency_ghz": float(go["frequency_ghz"].iloc[0])}
        out["designs"][str(design)] = {
            "latest_build": latest, "operating_points": ops, "fubs": fubs,
            "dvfs": {"c0": dv.params[design][0], "c1": dv.params[design][1]} if design in dv.params else None,
            "perf": {w: {"a": a, "b": b, "r2": r2} for (d, w), (a, b, r2, n) in pm.params.items() if d == design},
            "timing": {p: {"a": a, "k": k} for (d, p), (a, k, n) in tm.params.items() if d == design},
        }
    return out


class CompactPowerModel:
    """Evaluate an exported compact model. Standalone: only needs numpy/pandas."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.features = doc["terms"]["features"]
        self.coef = doc["terms"]["coefficients"]
        self.intercept = doc["terms"]["intercept"]
        self.fill = doc["terms"].get("fill", {})

    @classmethod
    def load(cls, path: str | Path) -> "CompactPowerModel":
        return cls(json.loads(Path(path).read_text()))

    def design(self, name: str) -> dict:
        if name not in self.doc["designs"]:
            raise KeyError(f"design '{name}' not in compact model")
        return self.doc["designs"][name]

    def rows(self, design: str, workload: str, frequency_ghz: float, voltage_v: float, activity_scale: float = 1.0) -> pd.DataFrame:
        d = self.design(design)
        recs = []
        for root, e in d["fubs"].items():
            wl = e["workloads"].get(workload) or (next(iter(e["workloads"].values())) if e["workloads"] else {})
            r = {"model_root": root, "fub": e["fub"], "partition": e["partition"], "frequency_ghz": frequency_ghz, "voltage_v": voltage_v}
            r.update(e["physical"])
            r["activity"] = wl.get("activity", np.nan) * activity_scale
            r["bits_per_cycle"] = wl.get("bits_per_cycle", np.nan) * activity_scale
            recs.append(r)
        return add_engineered_features(pd.DataFrame(recs))

    def power_by_fub(self, design: str, workload: str, frequency_ghz: float, voltage_v: float, activity_scale: float = 1.0) -> pd.DataFrame:
        X = self.rows(design, workload, frequency_ghz, voltage_v, activity_scale)
        p = np.full(len(X), self.intercept)
        for f in self.features:
            v = pd.to_numeric(X[f], errors="coerce").fillna(self.fill.get(f, 0.0)).to_numpy(float) if f in X.columns else np.full(len(X), self.fill.get(f, 0.0))
            p = p + self.coef[f] * v
        X["power_mw"] = p          # no clipping: must reproduce LinearModel.predict exactly
        return X[["model_root", "fub", "partition", "power_mw"]]

    def power(self, design: str, workload: str, frequency_ghz: float, voltage_v: float, activity_scale: float = 1.0) -> float:
        return float(self.power_by_fub(design, workload, frequency_ghz, voltage_v, activity_scale)["power_mw"].sum())

    def voltage_for(self, design: str, frequency_ghz: float) -> float:
        dv = self.design(design).get("dvfs")
        return float("nan") if not dv else dv["c0"] + dv["c1"] * frequency_ghz

    def throughput(self, design: str, workload: str, frequency_ghz: float) -> float:
        p = self.design(design).get("perf", {}).get(workload)
        return float("nan") if not p or frequency_ghz <= 0 else float(np.exp(p["a"] + p["b"] * np.log(frequency_ghz)))

    def fmax(self, design: str, voltage_v: float) -> tuple[float, str]:
        best, crit = float("nan"), ""
        if not np.isfinite(voltage_v) or voltage_v <= 0:
            return best, crit
        for part, t in self.design(design).get("timing", {}).items():
            f = 1000.0 / float(np.exp(t["a"] + t["k"] * np.log(voltage_v)))
            if not np.isfinite(best) or f < best:
                best, crit = f, part
        return best, crit

    def operating_point(self, design: str, name: str) -> tuple[float, float]:
        op = self.design(design)["operating_points"].get(name)
        if op is None:
            raise KeyError(f"operating point '{name}' not in compact model for {design}")
        return op["frequency_ghz"], op["voltage_v"]


@dataclass
class TraceResult:
    timeline: pd.DataFrame
    totals: dict


def run_trace(model: CompactPowerModel, trace: pd.DataFrame) -> TraceResult:
    """trace columns: design, interval, duration_s, workload, and either operating_point or (frequency_ghz, voltage_v);
    optional activity_scale."""
    rows = []
    for _, t in trace.iterrows():
        design, wl = str(t["design"]), str(t["workload"])
        if "operating_point" in t and pd.notna(t.get("operating_point")):
            f, v = model.operating_point(design, str(t["operating_point"]))
        else:
            f = float(t["frequency_ghz"])
            v = float(t["voltage_v"]) if "voltage_v" in t and pd.notna(t.get("voltage_v")) else model.voltage_for(design, f)
        scale = float(t.get("activity_scale", 1.0)) if pd.notna(t.get("activity_scale", 1.0)) else 1.0
        dur = float(t["duration_s"])
        p = model.power(design, wl, f, v, scale)
        thr = model.throughput(design, wl, f)
        fmax, crit = model.fmax(design, v)
        e_mj = p * dur                           # mW * s = mJ
        ops = thr * 1e9 * dur if np.isfinite(thr) else np.nan
        rows.append({"interval": int(t.get("interval", len(rows))), "design": design, "workload": wl,
                     "operating_point": t.get("operating_point"), "frequency_ghz": f, "voltage_v": v, "activity_scale": scale,
                     "duration_s": dur, "power_mw": p, "throughput_gops": thr, "energy_mj": e_mj, "ops": ops,
                     "energy_pj_per_op": float(energy_pj_per_op([p], [thr])[0]), "fmax_ghz": fmax,
                     "timing_ok": (not np.isfinite(fmax)) or f <= fmax * 1.005, "critical_partition": crit})
    tl = pd.DataFrame(rows)
    tot_t = float(tl["duration_s"].sum())
    tot_e = float(tl["energy_mj"].sum())
    tot_ops = float(tl["ops"].sum(skipna=True))
    totals = {"duration_s": tot_t, "energy_mj": tot_e, "avg_power_mw": tot_e / tot_t if tot_t else np.nan,
              "peak_power_mw": float(tl["power_mw"].max()), "ops": tot_ops,
              "energy_pj_per_op": tot_e * 1e9 / tot_ops if tot_ops else np.nan,
              "timing_violations": int((~tl["timing_ok"]).sum())}
    by_wl = tl.groupby("workload").agg(time_s=("duration_s", "sum"), energy_mj=("energy_mj", "sum"), ops=("ops", "sum")).reset_index()
    by_wl["share_time"] = by_wl["time_s"] / tot_t
    by_wl["share_energy"] = by_wl["energy_mj"] / tot_e
    totals["by_workload"] = by_wl
    return TraceResult(tl, totals)


def render_trace(res: TraceResult, model_kind: str) -> str:
    tl = res.timeline
    out = ["Workload trace evaluated with the compact power model  [" + model_kind + "]", ""]
    rows = [[int(r["interval"]), r["workload"], r["operating_point"] or "", f"{r['frequency_ghz']:.2f}", f"{r['voltage_v']:.3f}",
             f"{r['activity_scale']:.2f}", f"{r['duration_s'] * 1e6:.0f}", fmt_mw(r["power_mw"]),
             f"{r['throughput_gops']:.1f}" if np.isfinite(r["throughput_gops"]) else "n/a",
             f"{r['energy_mj'] * 1e3:.3f}", f"{r['energy_pj_per_op']:.2f}" if np.isfinite(r["energy_pj_per_op"]) else "n/a",
             "ok" if r["timing_ok"] else f"VIOLATES ({r['critical_partition']})"] for _, r in tl.iterrows()]
    out.append(table(["#", "Workload", "OP", "f GHz", "V", "act x", "dur us", "Power", "Gops/s", "Energy uJ", "pJ/op", "Timing"], rows,
                     ["r", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "l"]))
    t = res.totals
    out.append("")
    out.append("Totals")
    out.append(f"  duration        {t['duration_s'] * 1e6:,.0f} us")
    out.append(f"  energy          {t['energy_mj'] * 1e3:,.2f} uJ")
    out.append(f"  average power   {fmt_mw(t['avg_power_mw'])}    peak {fmt_mw(t['peak_power_mw'])}")
    if np.isfinite(t["energy_pj_per_op"]):
        out.append(f"  energy per op   {t['energy_pj_per_op']:.2f} pJ/op over {t['ops']:.3g} ops")
    out.append(f"  timing          {t['timing_violations']} interval(s) exceed the partition Fmax model")
    out.append("")
    out.append("Energy by workload:")
    out.append("")
    bw = t["by_workload"]
    out.append(table(["Workload", "Time share", "Energy share", "Energy uJ"],
                     [[r["workload"], fmt_pct(r["share_time"] * 100), fmt_pct(r["share_energy"] * 100), f"{r['energy_mj'] * 1e3:.2f}"] for _, r in bw.iterrows()]))
    return "\n".join(out)
