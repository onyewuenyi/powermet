"""Design-space exploration (V3): sweep a parameter or compare scenarios; predict power,
throughput and energy per op using the saved power model plus the perf/DVFS models.

  sweep    : one parameter, several values (frequency follows the DVFS curve unless fixed voltage)
  opmap    : existing operating points + candidate (V, f) points
  scenario : named what-if scenarios from a TOML file

Every number is a model prediction; the tables carry the model's CV error interval.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from powermet.features import add_engineered_features, rescale_fe_physical
from powermet.modeling import MODEL_NAMES, pick_model_key
from powermet.selection import DatasetSlice
from powermet.textfmt import fmt_mw, fmt_pct, table
from powermet.whatif import Override, parse_override
from powermet.curves import DvfsCurve, PerfModel, TimingModel
from powermet.workload import attach_op_params, energy_pj_per_op


@dataclass
class Scenario:
    name: str
    design: str
    workload: str
    operating_point: str | None = None
    overrides: list[Override] = field(default_factory=list)
    frequency_ghz: float | None = None      # convenience: sets frequency and (unless voltage given) DVFS voltage
    voltage_v: float | None = None


@dataclass
class ExploreRow:
    scenario: str
    changes: str
    frequency_ghz: float
    voltage_v: float
    power_mw: float
    power_lo: float
    power_hi: float
    throughput_gops: float
    energy_pj: float
    delta_power_pct: float
    delta_perf_pct: float
    delta_energy_pct: float
    pareto: bool = False
    fmax_ghz: float = float("nan")
    feasible: bool = True
    critical_partition: str = ""


class Explorer:
    def __init__(self, df: pd.DataFrame, perf: pd.DataFrame | None, payload: dict, meta: dict, model_key: str = "physics"):
        self.df = df
        self.perf = attach_op_params(perf, df) if perf is not None else pd.DataFrame()
        self.models = payload["models"]
        self.meta = meta
        self.model_key = pick_model_key(self.models, model_key, allow_tree=True)
        self.interval = (meta.get("cv") or {}).get("intervals", {}).get(self.model_key)
        self.perf_model = PerfModel().fit(self.perf) if len(self.perf) else PerfModel()
        self.dvfs = DvfsCurve().fit(df)
        self.timing = TimingModel().fit(df)

    # ---- helpers
    def base_rows(self, design: str, workload: str, operating_point: str | None) -> pd.DataFrame:
        """Latest build of `design` for the workload; operating point defaults to nom when not given."""
        return DatasetSlice(design=design, workload=workload, operating_point=operating_point,
                            default_workload=False).apply(self.df)

    def predict_power(self, rows: pd.DataFrame) -> float:
        return float(np.nansum(self.models[self.model_key].predict(add_engineered_features(rows))))

    def evaluate(self, sc: Scenario, baseline: ExploreRow | None = None) -> ExploreRow:
        rows = self.base_rows(sc.design, sc.workload, sc.operating_point)
        cur = add_engineered_features(rows)
        prop = cur.copy()
        changes = []
        f0 = float(pd.to_numeric(cur["frequency_ghz"], errors="coerce").iloc[0]) if "frequency_ghz" in cur else float("nan")
        v0 = float(pd.to_numeric(cur["voltage_v"], errors="coerce").iloc[0]) if "voltage_v" in cur else float("nan")
        f, v = f0, v0
        if sc.frequency_ghz is not None:
            f = sc.frequency_ghz
            prop["frequency_ghz"] = f
            changes.append(f"f={f:g}GHz")
            if sc.voltage_v is None and np.isfinite(self.dvfs.voltage(sc.design, f)):
                v = self.dvfs.voltage(sc.design, f)
                prop["voltage_v"] = v
                changes.append(f"V={v:.3f} (DVFS)")
        if sc.voltage_v is not None:
            v = sc.voltage_v
            prop["voltage_v"] = v
            changes.append(f"V={v:g}")
        for o in sc.overrides:
            prop = o.apply(prop)
            changes.append(o.describe())
            if o.feature == "frequency_ghz":
                f = float(prop["frequency_ghz"].iloc[0])
            if o.feature == "voltage_v":
                v = float(prop["voltage_v"].iloc[0])
        prop = add_engineered_features(prop)
        changed = {o.feature for o in sc.overrides} | ({"frequency_ghz"} if sc.frequency_ghz is not None else set()) \
            | ({"voltage_v"} if sc.voltage_v is not None else set())
        prop, _ = rescale_fe_physical(cur, prop, changed)
        power = self.predict_power(prop)
        lo = power * (1 + self.interval["p05"] / 100) if self.interval else float("nan")
        hi = power * (1 + self.interval["p95"] / 100) if self.interval else float("nan")
        thr = self.perf_model.predict(sc.design, sc.workload, f) if np.isfinite(f) else float("nan")
        energy = float(energy_pj_per_op([power], [thr])[0])
        row = ExploreRow(sc.name, "; ".join(changes) or "baseline", f, v, power, lo, hi, thr, energy, 0.0, 0.0, 0.0)
        if self.timing.params and np.isfinite(v) and np.isfinite(f):
            fmax, crit = self.timing.fmax(sc.design, v)
            row.fmax_ghz, row.critical_partition = fmax, crit
            row.feasible = (not np.isfinite(fmax)) or f <= fmax * 1.005
        if baseline is not None:
            row.delta_power_pct = (power - baseline.power_mw) / baseline.power_mw * 100 if baseline.power_mw else float("nan")
            row.delta_perf_pct = (thr - baseline.throughput_gops) / baseline.throughput_gops * 100 if baseline.throughput_gops else float("nan")
            row.delta_energy_pct = (energy - baseline.energy_pj) / baseline.energy_pj * 100 if baseline.energy_pj else float("nan")
        return row

    # ---- entry points
    def sweep(self, design: str, workload: str, param: str, values: list[float], mode: str = "set",
              operating_point: str | None = None, fixed_voltage: bool = False) -> list[ExploreRow]:
        base = self.evaluate(Scenario("baseline", design, workload, operating_point))
        out = [base]
        for val in values:
            if param == "frequency_ghz" and mode == "set" and not fixed_voltage:
                sc = Scenario(f"f={val:g}", design, workload, operating_point, frequency_ghz=val)
            else:
                sc = Scenario(f"{param}{'=' if mode == 'set' else 'x'}{val:g}", design, workload, operating_point,
                              overrides=[Override(param, mode, val)])
            out.append(self.evaluate(sc, base))
        mark_pareto(out)
        return out

    def opmap(self, design: str, workload: str, candidates: list[tuple[float, float]] | None = None) -> list[ExploreRow]:
        sel = self.df[self.df["design"].astype(str) == design]
        ops = sorted(sel["operating_point"].astype(str).unique()) if "operating_point" in sel.columns else [None]
        rows = []
        base = None
        for op in ops:
            sc = Scenario(f"{op} (measured OP)", design, workload, op)
            r = self.evaluate(sc, base)
            if base is None:
                base = r
                r = self.evaluate(sc, base)
            rows.append(r)
        for v, f in candidates or []:
            sc = Scenario(f"candidate V={v:g} f={f:g}", design, workload, None, frequency_ghz=f, voltage_v=v)
            rows.append(self.evaluate(sc, base))
        mark_pareto(rows)
        return rows

    def scenarios(self, scs: list[Scenario]) -> list[ExploreRow]:
        out = []
        base_by_key: dict[tuple, ExploreRow] = {}
        for sc in scs:
            key = (sc.design, sc.workload, sc.operating_point)
            if key not in base_by_key:
                base_by_key[key] = self.evaluate(Scenario(f"baseline {sc.design}/{sc.workload}", sc.design, sc.workload, sc.operating_point))
                out.append(base_by_key[key])
            out.append(self.evaluate(sc, base_by_key[key]))
        mark_pareto(out)
        return out


def mark_pareto(rows: list[ExploreRow]) -> None:
    """Pareto-optimal = feasible and no other feasible row has both lower power and >= throughput."""
    for r in rows:
        r.pareto = r.feasible and np.isfinite(r.throughput_gops) and not any(
            (o is not r) and o.feasible and np.isfinite(o.power_mw) and np.isfinite(o.throughput_gops)
            and o.power_mw <= r.power_mw and o.throughput_gops >= r.throughput_gops
            and (o.power_mw < r.power_mw or o.throughput_gops > r.throughput_gops)
            for o in rows
        )


def load_scenarios(path: str | Path) -> list[Scenario]:
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    out = []
    defaults = doc.get("defaults", {})
    for sc in doc.get("scenario", []):
        d = {**defaults, **sc}
        overrides = []
        for k, v in (d.get("set") or {}).items():
            overrides.append(Override(k, "set", float(v)))
        for k, v in (d.get("scale") or {}).items():
            overrides.append(Override(k, "scale", float(v)))
        out.append(Scenario(d["name"], d["design"], d.get("workload", "typical"), d.get("operating_point"),
                            overrides, d.get("frequency_ghz"), d.get("voltage_v")))
    return out


def render_rows(rows: list[ExploreRow], model_key: str, title: str) -> str:
    has_t = any(np.isfinite(r.fmax_ghz) for r in rows)
    hdr = ["Scenario", "Changes", "f GHz", "V", "BE power", "90% interval", "Gops/s", "pJ/op", "dP", "dPerf", "dE/op", "Pareto"] + (["Fmax", "Timing"] if has_t else [])
    body = []
    for r in rows:
        extra = ([f"{r.fmax_ghz:.2f}" if np.isfinite(r.fmax_ghz) else "n/a", "ok" if r.feasible else f"VIOLATES ({r.critical_partition})"] if has_t else [])
        body.append([r.scenario, r.changes, f"{r.frequency_ghz:.2f}" if np.isfinite(r.frequency_ghz) else "",
                     f"{r.voltage_v:.3f}" if np.isfinite(r.voltage_v) else "", fmt_mw(r.power_mw),
                     f"{fmt_mw(r.power_lo)}..{fmt_mw(r.power_hi)}" if np.isfinite(r.power_lo) else "",
                     f"{r.throughput_gops:.1f}" if np.isfinite(r.throughput_gops) else "n/a",
                     f"{r.energy_pj:.2f}" if np.isfinite(r.energy_pj) else "n/a",
                     fmt_pct(r.delta_power_pct, True) if r.scenario != "baseline" else "",
                     fmt_pct(r.delta_perf_pct, True) if r.scenario != "baseline" else "",
                     fmt_pct(r.delta_energy_pct, True) if r.scenario != "baseline" else "",
                     "*" if r.pareto else ""] + extra)
    out = [title, "", table(hdr, body, ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "l"] + (["r", "l"] if has_t else [])), ""]
    best = [r for r in rows if np.isfinite(r.energy_pj) and r.feasible]
    if best:
        b = min(best, key=lambda r: r.energy_pj)
        out.append(f"Lowest energy per op: {b.scenario} ({b.energy_pj:.2f} pJ/op, {fmt_mw(b.power_mw)}, {b.throughput_gops:.1f} Gops/s).")
    out.append(f"Power from {MODEL_NAMES[model_key]}; throughput from the per-workload f^b perf model; * = Pareto-optimal (power vs throughput).")
    if has_t:
        out.append("Timing feasibility from the per-partition delay ~ V^k model (PrimeTime): a point VIOLATES when f exceeds the worst partition's Fmax at that voltage.")
    out.append("Predictions only: they inherit the model's build-to-build error and the perf model's fit quality.")
    return "\n".join(out)
