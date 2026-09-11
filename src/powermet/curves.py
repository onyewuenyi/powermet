"""Design-level curve fits used by exploration and integration.

  PerfModel   : log(throughput) = a + b * log(f)  per (design, workload)   [b < 1 => memory-bound saturation]
  DvfsCurve   : V = c0 + c1 * f                   per design               [from measured operating points]
  TimingModel : log(delay_ps) = a + k * log(V)    per (design, partition)  [Fmax(V) = 1000 / delay(V)]

All are deliberately simple and report their fit quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.selection import latest_build
from powermet.textfmt import fmt_r, table


@dataclass
class PerfModel:
    """Per (design, workload) log-log fit of throughput vs frequency."""

    params: dict[tuple[str, str], tuple[float, float, float, int]] = field(default_factory=dict)  # (a, b, r2, n)

    def fit(self, perf: pd.DataFrame) -> "PerfModel":
        for (d, w), g in perf.groupby(["design", "workload"]):
            f = pd.to_numeric(g["frequency_ghz"], errors="coerce")
            t = pd.to_numeric(g["throughput_gops"], errors="coerce")
            ok = (f > 0) & (t > 0)
            if ok.sum() < 2 or f[ok].nunique() < 2:
                if ok.any():
                    # single frequency: assume linear scaling (b = 1) anchored on the measured point
                    a = float(np.log(t[ok].mean()) - np.log(f[ok].mean()))
                    self.params[(d, w)] = (a, 1.0, float("nan"), int(ok.sum()))
                continue
            x, y = np.log(f[ok].to_numpy()), np.log(t[ok].to_numpy())
            b, a = np.polyfit(x, y, 1)
            pred = a + b * x
            ss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else float("nan")
            self.params[(d, w)] = (float(a), float(b), r2, int(ok.sum()))
        return self

    def predict(self, design: str, workload: str, frequency_ghz: float) -> float:
        p = self.params.get((design, workload))
        if p is None or frequency_ghz <= 0:
            return float("nan")
        a, b, _, _ = p
        return float(np.exp(a + b * np.log(frequency_ghz)))

    def scaling_exponent(self, design: str, workload: str) -> float:
        p = self.params.get((design, workload))
        return p[1] if p else float("nan")

    def render(self) -> str:
        rows = [[d, w, f"{b:.2f}", fmt_r(r2), n] for (d, w), (a, b, r2, n) in sorted(self.params.items())]
        return ("Performance model: throughput ~ f^b per (design, workload)   [b=1: frequency-bound, b<1: memory-bound]\n\n"
                + table(["Design", "Workload", "b", "R^2", "n"], rows))


@dataclass
class DvfsCurve:
    """Per-design linear V(f) fitted from the measured operating points."""

    params: dict[str, tuple[float, float, int]] = field(default_factory=dict)   # (c0, c1, n)

    def fit(self, df: pd.DataFrame) -> "DvfsCurve":
        if "voltage_v" not in df.columns or "frequency_ghz" not in df.columns:
            return self
        pts = df[["design", "frequency_ghz", "voltage_v"]].dropna().drop_duplicates()
        for d, g in pts.groupby("design"):
            f, v = g["frequency_ghz"].to_numpy(float), g["voltage_v"].to_numpy(float)
            if len(np.unique(f)) < 2:
                self.params[d] = (float(v.mean()), 0.0, len(f))
                continue
            c1, c0 = np.polyfit(f, v, 1)
            self.params[d] = (float(c0), float(c1), len(f))
        return self

    def voltage(self, design: str, frequency_ghz: float) -> float:
        p = self.params.get(design)
        if p is None:
            return float("nan")
        return p[0] + p[1] * frequency_ghz

    def render(self) -> str:
        rows = [[d, f"{c0:.3f}", f"{c1:+.3f}", n] for d, (c0, c1, n) in sorted(self.params.items())]
        return "DVFS curve: V = c0 + c1 * f per design (from measured operating points)\n\n" + table(["Design", "c0 (V)", "c1 (V/GHz)", "points"], rows)


@dataclass
class TimingModel:
    """Per (design, partition) fit of path delay vs voltage from the measured operating points:
    log(delay_ps) = a + k * log(V); fmax(V) = 1000 / delay(V). Feasible f <= min over partitions."""

    params: dict[tuple[str, str], tuple[float, float, int]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "TimingModel":
        need = {"partition", "voltage_v", "clock_period_ps", "wns_ps"}
        if not need <= set(df.columns):
            return self
        d = df.dropna(subset=list(need)).copy()
        if not len(d):
            return self
        d["delay_ps"] = d["clock_period_ps"] - d["wns_ps"]
        latest = d[d["build"].astype(str) == latest_build(d)]
        for (des, part), g_all in d.groupby(["design", "partition"]):
            # the latest build describes the current design; earlier builds only supply the V
            # exponent when the latest build has fewer than two distinct voltages
            g = latest[(latest["design"] == des) & (latest["partition"] == part)]
            pts = g[["voltage_v", "delay_ps"]].drop_duplicates()
            if pts["voltage_v"].nunique() < 2:
                pts = g_all[["voltage_v", "delay_ps"]].drop_duplicates()
            v, dl = pts["voltage_v"].to_numpy(float), pts["delay_ps"].to_numpy(float)
            ok = (v > 0) & (dl > 0)
            if ok.sum() >= 2 and len(np.unique(v[ok])) >= 2:
                k, a = np.polyfit(np.log(v[ok]), np.log(dl[ok]), 1)
                if len(g) and g["voltage_v"].nunique() < 2:
                    # anchor the intercept on the latest build's delay at its measured voltage
                    v0, d0 = float(g["voltage_v"].iloc[0]), float(g["delay_ps"].iloc[0])
                    a = float(np.log(d0) - k * np.log(v0))
                self.params[(des, part)] = (float(a), float(k), int(ok.sum()))
            elif ok.any():
                self.params[(des, part)] = (float(np.log(dl[ok].mean())), 0.0, int(ok.sum()))
        return self

    def fmax(self, design: str, voltage_v: float) -> tuple[float, str]:
        """(min fmax over partitions at this voltage, critical partition)."""
        best, crit = float("nan"), ""
        for (des, part), (a, k, _) in self.params.items():
            if des != design or voltage_v <= 0:
                continue
            f = 1000.0 / float(np.exp(a + k * np.log(voltage_v)))
            if not np.isfinite(best) or f < best:
                best, crit = f, part
        return best, crit

    def render(self) -> str:
        rows = [[d, p, f"{k:+.2f}", f"{1000.0 / np.exp(a):.3f}", n] for (d, p), (a, k, n) in sorted(self.params.items())]
        return ("Timing model: delay ~ V^k per (design, partition); Fmax at 1 V shown for scale\n\n"
                + table(["Design", "Partition", "k", "Fmax@1V GHz", "points"], rows))
