"""Power-optimization techniques as a registry: what each solves, why, the trade-off, and an assessment
over the dataset that says where it applies and roughly what it is worth, with the assumptions stated.

A Technique is documentation plus a function. `assess()` never claims a saving as a measurement; it
ranks candidates and gives an order-of-magnitude estimate under explicit assumptions so a team can
decide what to try first. Techniques whose assessment needs data the pipeline does not carry say so
and name the source that would enable them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from powermet.schema import identity_key, label
from powermet.selection import DatasetSlice
from powermet.textfmt import fmt_mw, fmt_pct, table


@dataclass
class TechniqueResult:
    technique: str
    assessable: bool
    candidates: pd.DataFrame            # per FUB: metrics that justify the candidate + est_saving_mw
    est_saving_mw: float
    assumptions: list[str]
    missing_data: list[str] = field(default_factory=list)
    scope: str = ""
    est_dynamic_saving_mw: float = float("nan")     # share of the saving that lowers dynamic power (-> CdynTot)
    est_leakage_saving_mw: float = float("nan")     # share that lowers leakage (-> LkgPwr)

    def saving_for(self, component: str) -> float:
        """Saving attributable to a convergence component: dynamic | leakage | total (mW)."""
        if component == "total":
            return self.est_saving_mw
        v = self.est_dynamic_saving_mw if component == "dynamic" else self.est_leakage_saving_mw
        if np.isfinite(v):
            return v
        t = BY_KEY[self.technique]
        return self.est_saving_mw if t.reduces == component else 0.0


@dataclass(frozen=True)
class Technique:
    key: str
    name: str
    stage: str                          # architecture | rtl | synthesis | physical | signoff | runtime
    problem: str                        # what it solves
    why: str                            # mechanism
    tradeoff: str                       # what it costs
    considerations: str                 # when it applies / what to check
    data_needed: tuple[str, ...]        # dataset columns the assessment uses
    assess: Callable[[pd.DataFrame, dict], TechniqueResult]
    reduces: str = "dynamic"            # convergence component the saving lands on: dynamic (CdynTot) | leakage (LkgPwr) | corner (V/f only)


# ----------------------------------------------------------------------------- helpers

def _latest(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    return DatasetSlice(design=ctx.get("design"), workload=ctx.get("workload"), operating_point=ctx.get("operating_point")).apply(df)


def _have(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c not in df.columns or df[c].isna().all()]


LEAK_SOURCE = {"measured": "leakage from the PrimePower leakage column (be_leakage_mw)",
               "decomposition": "leakage share from the data-movement decomposition",
               "assumed": "leakage assumed to be 15% of BE power"}


def _dyn_leak(df: pd.DataFrame, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Dynamic and leakage power per row: the measured leakage column when present, else the data-movement
    model's leakage term if a decomposition is supplied, else a fixed 85/15 split. Records which in ctx["leak_source"]."""
    if "be_leakage_mw" in df.columns and df["be_leakage_mw"].notna().any():
        leak = pd.to_numeric(df["be_leakage_mw"], errors="coerce").fillna(df["be_mw"] * 0.15).clip(upper=df["be_mw"])
        ctx["leak_source"] = "measured"
        return df["be_mw"] - leak, leak
    dec = ctx.get("decomposition")
    if dec is not None and len(dec) and "leakage_mw" in dec.columns:
        key = identity_key(df)
        on = [c for c in (key, "design", "build", "workload", "operating_point") if c in df.columns and c in dec.columns]
        look = dec[on + ["leakage_mw"]].drop_duplicates(subset=on)
        leak = df[on].merge(look, on=on, how="left")["leakage_mw"].to_numpy()
        leak = pd.Series(np.where(np.isfinite(leak), leak, df["be_mw"].to_numpy() * 0.15), index=df.index)
        ctx["leak_source"] = "decomposition"
        return df["be_mw"] - leak, leak
    ctx["leak_source"] = "assumed"
    return df["be_mw"] * 0.85, df["be_mw"] * 0.15


def _leak_note(ctx: dict) -> str:
    return LEAK_SOURCE[ctx.get("leak_source", "assumed")]


# ----------------------------------------------------------------------------- assessments

def assess_clock_gating(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
    miss = _have(df, ["cg_efficiency", "be_mw"])
    if miss:
        return TechniqueResult("clock_gating", False, pd.DataFrame(), 0.0, [], [f"{m} (PPRTL ClockGatingEff column)" for m in miss])
    d = _latest(df, ctx)
    dyn, _ = _dyn_leak(d, ctx)
    reg_frac, cap = 0.35, 0.90          # share of dynamic power in registers/clock; achievable gating ceiling
    gain = dyn * reg_frac * np.clip(cap - d["cg_efficiency"], 0, None)
    out = d[[identity_key(d), "be_mw", "cg_efficiency"]].copy()
    out["dynamic_mw"], out["est_saving_mw"] = dyn, gain
    out = out[out["est_saving_mw"] > 0].sort_values("est_saving_mw", ascending=False)
    return TechniqueResult("clock_gating", True, out, float(out["est_saving_mw"].sum()),
                           [f"{reg_frac:.0%} of dynamic power is in registers and clock network", f"gating efficiency can reach {cap:.0%}",
                            _leak_note(ctx)],
                           scope=DatasetSlice(design=ctx.get("design")).describe(d), est_dynamic_saving_mw=float(out["est_saving_mw"].sum()),
                           est_leakage_saving_mw=0.0)


def assess_power_gating(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
    if "workload" not in df.columns or df["workload"].nunique() < 2:
        return TechniqueResult("power_gating", False, pd.DataFrame(), 0.0, [], ["an idle / low-activity workload in the dataset"])
    wls = list(df["workload"].astype(str).unique())
    idle = "idle" if "idle" in wls else min(wls, key=lambda w: df[df["workload"] == w]["be_mw"].sum())
    d = DatasetSlice(design=ctx.get("design"), workload=idle, operating_point=ctx.get("operating_point")).apply(df)
    _, leak = _dyn_leak(d, ctx)
    duty = float(ctx.get("idle_duty", 0.5))
    key = identity_key(d)
    out = d[[key, "be_mw", "activity"] if "activity" in d.columns else [key, "be_mw"]].copy()
    out["idle_power_mw"] = d["be_mw"]
    out["leak_mw"] = leak
    out["est_saving_mw"] = d["be_mw"] * duty * 0.9
    out["est_leak_saving_mw"] = leak * duty * 0.9
    out = out.sort_values("est_saving_mw", ascending=False)
    leak_sav = float(out["est_leak_saving_mw"].sum())
    return TechniqueResult("power_gating", True, out, float(out["est_saving_mw"].sum()),
                           [f"'{idle}' workload represents the gated state", f"blocks are off {duty:.0%} of the time (idle_duty)",
                            "90% of idle power is removed when off (retention / always-on kept)",
                            "wake-up latency, isolation and retention cost are not modelled", _leak_note(ctx),
                            "toward convergence only the leakage part counts (LkgPwr); the idle dynamic power it removes is not a CdynTot reduction at the target workload"],
                           scope=f"design={ctx.get('design') or 'all'}, workload={idle}",
                           est_dynamic_saving_mw=0.0, est_leakage_saving_mw=leak_sav)


def assess_dvfs(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
    miss = _have(df, ["voltage_v", "frequency_ghz"])
    if miss or "operating_point" not in df.columns or df["operating_point"].nunique() < 2:
        return TechniqueResult("dvfs", False, pd.DataFrame(), 0.0, [], miss or ["at least two operating points"])
    d = DatasetSlice(design=ctx.get("design"), workload=ctx.get("workload"), default_operating_point=False).apply(df)
    g = d.groupby(["design", "operating_point"]).agg(power_mw=("be_mw", "sum"), voltage_v=("voltage_v", "first"),
                                                    frequency_ghz=("frequency_ghz", "first")).reset_index()
    g["mw_per_ghz"] = g["power_mw"] / g["frequency_ghz"]
    g = g.sort_values(["design", "frequency_ghz"])
    best = g.loc[g.groupby("design")["mw_per_ghz"].idxmin()]
    worst = g.loc[g.groupby("design")["mw_per_ghz"].idxmax()]
    saving = float((worst.set_index("design")["power_mw"] - best.set_index("design")["power_mw"]).clip(lower=0).sum())
    return TechniqueResult("dvfs", True, g.rename(columns={"operating_point": "op"}), saving,
                           ["saving = power at the least efficient corner minus the most efficient one (mW per GHz), per design",
                            "assumes the workload tolerates the lower frequency; see `explore opmap` for energy per op with throughput"],
                           scope=f"design={ctx.get('design') or 'all'}")


def assess_wire_cap_reduction(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
    miss = _have(df, ["wire_cap_pf", "cell_cap_pf"])
    if miss:
        return TechniqueResult("wire_cap_reduction", False, pd.DataFrame(), 0.0, [], miss)
    d = _latest(df, ctx)
    dyn, _ = _dyn_leak(d, ctx)
    frac = d["wire_cap_pf"] / (d["wire_cap_pf"] + d["cell_cap_pf"]).where(lambda s: s > 0)
    pct = float(ctx.get("wire_cap_cut_pct", 10.0))
    key = identity_key(d)
    out = d[[key, "be_mw", "wire_cap_pf"]].copy()
    out["wire_cap_fraction"] = frac
    out["est_saving_mw"] = dyn * frac * pct / 100
    out = out.sort_values("est_saving_mw", ascending=False)
    return TechniqueResult("wire_cap_reduction", True, out, float(out["est_saving_mw"].sum()),
                           [f"placement / routing work removes {pct:g}% of wire cap on each block (wire_cap_cut_pct)",
                            "wire switching power scales with wire cap share of dynamic power",
                            "use `model predict --scale wire_cap_pf=0.9` for the model-based number with its interval", _leak_note(ctx)],
                           scope=DatasetSlice(design=ctx.get("design")).describe(d), est_dynamic_saving_mw=float(out["est_saving_mw"].sum()),
                           est_leakage_saving_mw=0.0)


def assess_vt_swap(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
    d = _latest(df, ctx)
    _, leak = _dyn_leak(d, ctx)
    if "wns_ps" not in d.columns or d["wns_ps"].isna().all():
        return TechniqueResult("vt_swap", False, pd.DataFrame(), 0.0, [], ["partition timing (wns_ps) to know which blocks have slack"])
    slack_ok = d["wns_ps"] > float(ctx.get("slack_margin_ps", 20.0))
    key = identity_key(d)
    out = d[[key, "be_mw", "wns_ps"]].copy()
    out["leak_mw"] = leak
    out["est_saving_mw"] = np.where(slack_ok, leak * 0.4, 0.0)
    out = out[out["est_saving_mw"] > 0].sort_values("est_saving_mw", ascending=False)
    return TechniqueResult("vt_swap", True, out, float(out["est_saving_mw"].sum()),
                           ["only blocks whose partition has > slack_margin_ps of positive slack can absorb slower cells",
                            "40% of leakage removed by moving non-critical cells to higher-Vt (typical LVT -> SVT/HVT range 30-60%)",
                            _leak_note(ctx)],
                           scope=DatasetSlice(design=ctx.get("design")).describe(d), est_dynamic_saving_mw=0.0,
                           est_leakage_saving_mw=float(out["est_saving_mw"].sum()))


def assess_not_assessable(key: str, needs: list[str]):
    def f(df: pd.DataFrame, ctx: dict) -> TechniqueResult:
        return TechniqueResult(key, False, pd.DataFrame(), 0.0, [], needs)
    return f


# ----------------------------------------------------------------------------- registry

TECHNIQUES: tuple[Technique, ...] = (
    Technique("clock_gating", "Clock gating (ICG insertion, enable coverage)", "rtl",
              "Registers and the clock tree toggle every cycle even when their data does not change; the clock network is often 20-40% of dynamic power.",
              "An integrated clock gate stops the clock to a register bank when its enable is false, removing the clock pin switching and the downstream toggles.",
              "Adds ICG cells and enable logic (area, a small timing hit on the enable path), can create clock-tree imbalance, and low-activity enables gain nothing.",
              "Look for blocks with high dynamic power and low ClockGatingEff; check enable coverage per register bank in the RTL power tool; verify with a workload that actually idles the block.",
              ("be_mw", "cg_efficiency"), assess_clock_gating, "dynamic"),
    Technique("power_gating", "Power gating (domain shutoff)", "physical",
              "Leakage and idle clocking burn power in blocks that are off for long stretches of a workload.",
              "A switched supply (header/footer cells) cuts the domain's rail; state is kept in retention flops or restored on wake.",
              "Wake-up latency and rush current, isolation cells on every boundary, retention or re-initialisation, always-on logic, verification burden in UPF.",
              "Needs long idle intervals (microseconds and up), a clean UPF domain boundary and an architectural owner of the on/off policy; assess with an idle workload and a duty cycle.",
              ("be_mw", "workload"), assess_power_gating, "leakage"),
    Technique("dvfs", "Dynamic voltage and frequency scaling / AVS", "runtime",
              "Running at the turbo corner when the workload does not need it wastes V^2 f power.",
              "Power scales roughly with V^2 f while performance scales with f, so a lower corner buys energy per op when throughput is not the bottleneck.",
              "Voltage regulator and clock infrastructure, timing closure at every corner, transition latency, and a control loop that must not oscillate.",
              "Use the energy-per-op view (`explore opmap`) not raw power; memory-bound workloads (throughput exponent b < 1) benefit most; check the timing model marks the corner feasible.",
              ("be_mw", "voltage_v", "frequency_ghz"), assess_dvfs, "corner"),
    Technique("wire_cap_reduction", "Wire capacitance reduction (placement, routing, buffering)", "physical",
              "Wire-dominated blocks spend their dynamic power charging interconnect rather than cells; FE estimates miss most of it.",
              "Tighter placement, shorter nets, fewer buffers and better layer assignment reduce the capacitance switched per toggle.",
              "Placement density and congestion, timing on long nets, and routing effort; the gain is per block and hard to predict before P&R.",
              "Rank by wire-cap fraction and dynamic power; validate with a what-if on wire cap using the fitted model and compare against the next build.",
              ("be_mw", "wire_cap_pf", "cell_cap_pf"), assess_wire_cap_reduction, "dynamic"),
    Technique("vt_swap", "Multi-Vt / Vt swap and cell downsizing", "physical",
              "Low-Vt, high-drive cells leak heavily; many sit on paths with timing slack that never needed them.",
              "Swap non-critical cells to higher-Vt or smaller drive; leakage drops exponentially with Vt, dynamic power drops with drive.",
              "Consumes timing slack, can move the critical path, and the saving is bounded by the leakage share of power.",
              "Only where the partition has positive slack; confirm leakage share from a leakage-aware model or a PrimePower leakage column.",
              ("be_mw", "wns_ps"), assess_vt_swap, "leakage"),
    Technique("operand_isolation", "Operand isolation / data gating", "rtl",
              "Datapath inputs toggle and propagate through arithmetic units whose results are discarded.",
              "Gate the operands (AND/latch) when the unit output is unused so the datapath does not switch.",
              "Extra gating logic on wide buses (area, delay) and the risk of gating a live path; benefit depends on how often results are discarded.",
              "Needs per-unit toggle data with a 'result used' signal; not derivable from block-level SAIF.",
              (), assess_not_assessable("operand_isolation", ["per-net toggle counts with datapath enable correlation (RTL power tool report)"]), "dynamic"),
    Technique("glitch_reduction", "Glitch power reduction", "synthesis",
              "Unequal arrival times cause spurious transitions in combinational logic; PrimePower with glitch analysis shows it as 5-20% of dynamic power in some datapaths.",
              "Balance path delays, restructure logic, insert selective gating so nets settle once per cycle.",
              "Costs area and design effort; needs a glitch-aware power run to even see it.",
              "Requires a glitch-mode power report per block; add it as a source and compare to the non-glitch run.",
              (), assess_not_assessable("glitch_reduction", ["glitch-aware PrimePower report (glitch power per hierarchy)"]), "dynamic"),
    Technique("memory_low_power", "Memory sleep modes and banking", "architecture",
              "SRAM leakage and peripheral clocking dominate in memory-heavy blocks even when the array is idle.",
              "Light-sleep / deep-sleep / shutdown modes on macros, bank-level enables, and smaller active banks per access.",
              "Wake-up latency per mode, control logic, and lost bandwidth if banking is too fine.",
              "Needs memory instances identified per FUB with their mode power from the memory compiler datasheet.",
              (), assess_not_assessable("memory_low_power", ["memory instance list per FUB", "macro mode power (compiler datasheet)"]), "leakage"),
)
BY_KEY = {t.key: t for t in TECHNIQUES}


def assess_all(df: pd.DataFrame, ctx: dict, keys: list[str] | None = None) -> list[TechniqueResult]:
    out = []
    for t in TECHNIQUES:
        if keys and t.key not in keys:
            continue
        try:
            out.append(t.assess(df, ctx))
        except ValueError as exc:
            out.append(TechniqueResult(t.key, False, pd.DataFrame(), 0.0, [], [str(exc)]))
    return out


def render_catalog() -> str:
    lines = []
    for t in TECHNIQUES:
        moves = {"dynamic": "dynamic power -> CdynTot", "leakage": "leakage -> LkgPwr", "corner": "operating corner (V, f); CdynTot unchanged"}[t.reduces]
        lines += [f"{t.key}  [{t.stage}]  {t.name}", f"  problem        {t.problem}", f"  mechanism      {t.why}",
                  f"  trade-off      {t.tradeoff}", f"  considerations {t.considerations}", f"  moves          {moves}",
                  f"  data           {', '.join(t.data_needed) or 'not assessable from the dataset'}", ""]
    return "\n".join(lines)


def render_results(results: list[TechniqueResult], top: int = 5) -> str:
    out = ["TECHNIQUE ASSESSMENT (order-of-magnitude estimates under stated assumptions; not measurements)", ""]
    rows = []
    for r in results:
        t = BY_KEY[r.technique]
        rows.append([t.name, t.stage, t.reduces, fmt_mw(r.est_saving_mw) if r.assessable else "n/a",
                     f"{len(r.candidates):,} candidates" if r.assessable else "needs: " + "; ".join(r.missing_data)])
    out.append(table(["Technique", "Stage", "Moves", "Est. saving", "Status"], rows, ["l", "l", "l", "r", "l"]))
    for r in results:
        if not r.assessable or not len(r.candidates):
            continue
        t = BY_KEY[r.technique]
        out += ["", f"{t.name}   ({r.scope})", ""]
        c = r.candidates.head(top)
        cols = [col for col in c.columns]
        body = []
        for _, row in c.iterrows():
            body.append([f"{row[col]:.3g}" if isinstance(row[col], (float, np.floating)) and col != "est_saving_mw" else
                         (fmt_mw(row[col]) if col == "est_saving_mw" else str(row[col])) for col in cols])
        out.append(table([label(col) if col not in ("est_saving_mw",) else "Est. saving" for col in cols], body,
                         ["l"] + ["r"] * (len(cols) - 1)))
        out.append("  assumptions: " + "; ".join(r.assumptions))
    return "\n".join(out)
