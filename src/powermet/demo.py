"""Synthetic demo data with built-in physical relationships.

The generator is a *test fixture*, not a physical model. It is structured so that
analysis produces meaningful results:

  BE power  = k * activity(workload) * (cell_cap + 0.7 * wire_cap) * V^2 * f
              + k_move * bits_per_cycle * avg_net_length * V^2 * f          (data-movement energy)
              + leakage(area, V) + noise
  FE logical  ignores wire cap (flat wire-load factor)          -> large, wire-cap-correlated error
  FE physical captures ~83% of wire cap from early placement    -> smaller residual, still wire-cap-correlated
  Later builds gain routing-driven wire cap that FE never saw   -> error persists/grows across builds
  Wire-dominated blocks accumulate more of that growth          -> wire_cap_fraction explains % error

V3 additions: several workloads (per-FUB activity multipliers), several operating points
(voltage/frequency), a design-level performance model, and FE/BE hierarchy names for lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet import __version__

FUB_NAMES = [
    "Scheduler", "Decoder", "Dispatch", "Fetch", "BranchPred", "L1I", "L1D", "L2Ctrl", "TLB", "LSU",
    "ALU0", "ALU1", "FPU", "SIMD", "Crypto", "IntRF", "FpRF", "ROB", "RenameMap", "IssueQ",
    "MemArb", "Prefetch", "Coherence", "SnoopFilter", "NoCRouter", "NoCLink", "DMA", "IOMMU", "PCIeCtl", "DDRPhy",
    "PowerMgr", "ClockGen", "DebugTrace", "PerfCnt", "Interrupt", "Timer", "SecureEnc", "HashUnit", "Compress", "Decomp",
    "TexUnit", "Raster", "ShaderCore", "GeomEng", "TileBuf", "ZBuffer", "Blend", "VideoDec", "VideoEnc", "DisplayPipe",
]
MEMORY_FUBS = {"L1I", "L1D", "L2Ctrl", "TLB", "LSU", "MemArb", "Prefetch", "Coherence", "SnoopFilter", "DDRPhy", "TileBuf"}

DESIGNS = ["GPU_A", "GPU_B", "CPU_C", "SOC_D", "NPU_E"]

# workload -> (activity multiplier, memory-boundedness 0..1, ipc multiplier)
WORKLOADS: dict[str, tuple[float, float, float]] = {
    "idle": (0.15, 0.0, 0.05),
    "typical": (1.00, 0.35, 1.00),
    "compute": (1.55, 0.10, 1.35),
    "memory": (0.85, 0.80, 0.55),
}
# operating point -> (delta V, delta f GHz)
OPERATING_POINTS: dict[str, tuple[float, float]] = {
    "eco": (-0.07, -0.5),
    "nom": (0.0, 0.0),
    "turbo": (+0.08, +0.4),
}


@dataclass(frozen=True)
class DemoSpec:
    n_designs: int = 5
    n_builds: int = 10
    n_fubs: int = 50
    seed: int = 42
    workloads: tuple[str, ...] = ("typical",)
    operating_points: tuple[str, ...] = ("nom",)


PARTITION_NAMES = ["PCORE0", "PCORE1", "MEMSS", "IOFAB", "GFX0", "GFX1", "NPU0", "MISC"]


@dataclass
class DemoData:
    measurements: pd.DataFrame     # wide FUB-level rows (timing columns inherited from the partition)
    hierarchy: pd.DataFrame        # design, fub, model_root, partition, fe_hier, synth_object, be_hier, n_instances
    performance: pd.DataFrame      # design, build, workload, operating_point, frequency_ghz, voltage_v, ipc, throughput_gops
    metadata: pd.DataFrame         # design, build, build_date, run_id, tool versions
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)   # design, build, partition, operating_point, period, wns, tns, endpoints
    traces: pd.DataFrame = field(default_factory=pd.DataFrame)   # design, interval, workload, operating_point, activity_scale, ops
    spec: DemoSpec = field(default_factory=DemoSpec)


def _fub_name(i: int) -> str:
    base = FUB_NAMES[i % len(FUB_NAMES)]
    return base if i < len(FUB_NAMES) else f"{base}_{i // len(FUB_NAMES)}"


def generate_all(spec: DemoSpec = DemoSpec()) -> DemoData:
    rng = np.random.default_rng(spec.seed)
    rows: list[dict] = []
    hier_rows: list[dict] = []
    perf_rows: list[dict] = []
    meta_rows: list[dict] = []
    timing_rows: list[dict] = []
    trace_rows: list[dict] = []
    base_date = pd.Timestamp("2026-03-02")

    for d in range(spec.n_designs):
        design = DESIGNS[d] if d < len(DESIGNS) else f"DESIGN_{d}"
        top = design.lower() + "_top"
        freq0 = float(rng.choice([2.0, 2.4, 2.5, 2.8, 3.0]))
        vdd0 = float(rng.uniform(0.72, 0.85))
        size_scale = float(rng.uniform(0.7, 1.4))
        ops_per_cycle = float(rng.uniform(8, 64))

        n_fubs = spec.n_fubs
        area = rng.lognormal(mean=np.log(1000 * size_scale), sigma=0.55, size=n_fubs)
        density = rng.uniform(30, 50, size=n_fubs)
        cell_count = area * density
        fanout = 3.0 + rng.lognormal(mean=np.log(2.5), sigma=0.4, size=n_fubs)
        cell_cap = cell_count * 1.5e-4 * rng.lognormal(0, 0.12, size=n_fubs)
        wire_cap = cell_cap * (1.4 + 0.30 * (fanout - 5.0)) * rng.lognormal(0, 0.45, size=n_fubs)
        wire_cap = np.clip(wire_cap, 0.15 * cell_cap, None)
        activity0 = rng.beta(2.0, 4.0, size=n_fubs) * 0.6 + 0.05
        act_bias_l = rng.normal(0, 0.12, size=n_fubs)
        act_bias_p = rng.normal(0, 0.04, size=n_fubs)
        wire_bias_p = rng.normal(0.83, 0.05, size=n_fubs)
        wire_frac = wire_cap / (wire_cap + cell_cap)
        names = [_fub_name(i) for i in range(n_fubs)]
        # post-layout data-movement quantities: routed wire length grows with area and fanout;
        # average net length is the distance proxy; bits/cycle is workload-specific traffic.
        net_count = cell_count * 0.36
        avg_net_len = 12.0 * np.sqrt(area / 1000.0) * (0.8 + 0.08 * (fanout - 5.0)) * rng.lognormal(0, 0.2, size=n_fubs)
        wire_length = avg_net_len * net_count
        bus_width = rng.choice([8, 16, 32, 64], size=n_fubs).astype(float)   # nets carrying data traffic (written to SAIF)
        cg_eff = np.clip(rng.beta(5, 2, size=n_fubs), 0.2, 0.98)                # clock-gating efficiency per FUB
        # partitions: contiguous groups of FUBs; timing is a partition attribute
        n_parts = max(1, min(len(PARTITION_NAMES), n_fubs // 6))
        part_of = [PARTITION_NAMES[(i * n_parts) // n_fubs] for i in range(n_fubs)]
        part_base_delay = {p: float(rng.uniform(0.80, 0.98)) for p in set(part_of)}   # fraction of nominal period at build 0

        # per (fub, workload) activity multiplier: memory FUBs are busier on memory workloads
        wl_mult: dict[str, np.ndarray] = {}
        for wl in spec.workloads:
            base_mult, mem_bound, _ = WORKLOADS.get(wl, (1.0, 0.3, 1.0))
            m = np.full(n_fubs, base_mult) * rng.lognormal(0, 0.15, size=n_fubs)
            is_mem = np.array([n.split("_")[0] in MEMORY_FUBS for n in names])
            m = np.where(is_mem, m * (0.6 + 1.2 * mem_bound), m * (1.2 - 0.5 * mem_bound))
            wl_mult[wl] = m

        for i in range(n_fubs):
            renamed = rng.random() < 0.08   # BE occasionally renames/uniquifies the instance
            hier_rows.append({
                "design": design, "fub": names[i],
                "model_root": f"{design}.{part_of[i]}.{names[i]}",
                "partition": part_of[i],
                "fe_hier": f"{top}/u_{names[i].lower()}",
                "synth_object": f"{names[i]}_{rng.integers(0, 3)}" if rng.random() < 0.3 else names[i],
                "be_hier": f"{top}/part_{part_of[i].lower()}/u_{names[i].lower()}" + ("_phys" if renamed else ""),
                "n_instances": int(cell_count[i]),
            })

        k_dyn = 9.0
        k_move = 0.11      # data-movement energy: bits x distance x V^2 x f; FE estimates capture only part of it
        leak_per_area = 0.012
        for b in range(spec.n_builds):
            build = f"B{b + 1:03d}"
            meta_rows.append({
                "design": design, "build": build,
                "build_date": (base_date + pd.Timedelta(days=12 * b + d)).strftime("%Y-%m-%d"),
                "run_id": f"{design.lower()}_{build.lower()}_r{rng.integers(1000, 9999)}",
                "pprtl_version": "R-2025.06-SP2", "primepower_version": "V-2024.09-SP3",
                "starrc_version": "V-2024.09", "fusion_version": "V-2024.09-SP4",
            })
            build_drift = 1.0 + 0.012 * b + rng.normal(0, 0.01)
            routing_growth = 1.0 + 0.004 * b + 0.018 * b * wire_frac
            # timing closure effort grows with build maturity: buffering/upsizing adds cell cap and area
            # (power up) while path delay comes down (timing up) -> the power/timing trade-off
            closure = b / max(spec.n_builds - 1, 1)
            jit = rng.lognormal(0, 0.03, size=n_fubs)
            buffering = 1.0 + 0.05 * closure * (0.5 + wire_frac)
            a, cc, ccap = area * jit * (1 + 0.02 * closure), cell_count * jit * buffering, cell_cap * jit * buffering
            wcap = wire_cap * jit * routing_growth
            wlen = wire_length * jit * routing_growth
            act_b = np.clip(activity0 * (1 + rng.normal(0, 0.03, size=n_fubs)), 0.01, 1.0)
            # partition path delay at nominal V (ps): base fraction of period, worse with wire dominance,
            # better with closure effort and slightly noisy per build
            part_delay_nom = {}
            for p in set(part_of):
                members = [i for i in range(n_fubs) if part_of[i] == p]
                wf = float(np.mean(wire_frac[members]))
                d0 = part_base_delay[p] * (1000.0 / freq0) * (1 + 0.20 * (wf - 0.5)) * (1 + 0.06 * b * 0.0)
                part_delay_nom[p] = d0 * (1 + 0.10 * b / max(spec.n_builds, 1)) * (1 - 0.16 * closure) * rng.lognormal(0, 0.01)

            # activity and data traffic are workload attributes (independent of operating point)
            wl_act = {wl: np.clip(act_b * wl_mult[wl], 0.005, 1.5) for wl in spec.workloads}
            wl_bits = {wl: bus_width * wl_act[wl] * rng.lognormal(0, 0.05, size=n_fubs) for wl in spec.workloads}
            for op in spec.operating_points:
                dv, df_ = OPERATING_POINTS.get(op, (0.0, 0.0))
                vdd, freq = vdd0 + dv, freq0 + df_
                leak = leak_per_area * a * (vdd / vdd0) ** 3
                period = 1000.0 / freq
                # timing per partition at this operating point: delay scales ~ V^-1.3
                timing_at = {}
                for p, d_nom in part_delay_nom.items():
                    delay = d_nom * (vdd0 / vdd) ** 1.3
                    wns = period - delay
                    members = [i for i in range(n_fubs) if part_of[i] == p]
                    endpoints = int(sum(cc[members]) * 0.05)
                    viol = int(endpoints * min(1.0, max(0.0, -wns) / (0.08 * period))) if wns < 0 else 0
                    tns = float(wns * viol * 0.35) if wns < 0 else 0.0
                    timing_at[p] = (period, wns, tns, viol, endpoints)
                    timing_rows.append({"design": design, "build": build, "partition": p, "operating_point": op,
                                        "clock_period_ps": round(period, 1), "wns_ps": round(wns, 1), "tns_ps": round(tns, 1),
                                        "violating_endpoints": viol, "endpoints": endpoints})
                for wl in spec.workloads:
                    act = wl_act[wl]
                    bits = wl_bits[wl]
                    move = k_move * bits * (wlen / (cc * 0.36)) * vdd ** 2 * freq
                    be = (k_dyn * act * (ccap + 0.7 * wcap) * vdd ** 2 * freq * build_drift + move + leak
                          + rng.normal(0, 1.0, size=n_fubs))
                    fe_l = (k_dyn * act * (1 + act_bias_l) * (ccap * 2.1) * vdd ** 2 * freq + 0.3 * move
                            + leak * 0.9 + rng.normal(0, 1.5, size=n_fubs))
                    fe_p = (k_dyn * act * (1 + act_bias_p) * (ccap + 0.7 * wire_cap * jit * wire_bias_p) * vdd ** 2 * freq
                            + 0.6 * move + leak + rng.normal(0, 1.0, size=n_fubs))
                    be, fe_l, fe_p = (np.clip(x, 0.5, None) for x in (be, fe_l, fe_p))
                    for i in range(n_fubs):
                        per, wns, tns, viol, _ = timing_at[part_of[i]]
                        rows.append({
                            "design": design, "build": build, "fub": names[i], "stage": "FE_BE",
                            "workload": wl, "operating_point": op,
                            "model_root": f"{design}.{part_of[i]}.{names[i]}", "partition": part_of[i],
                            "fe_logical_mw": round(float(fe_l[i]), 3),
                            "fe_physical_mw": round(float(fe_p[i]), 3),
                            "be_mw": round(float(be[i]), 3),
                            "wire_cap_pf": round(float(wcap[i]), 4),
                            "cell_cap_pf": round(float(ccap[i]), 4),
                            "area": round(float(a[i]), 1),
                            "cell_count": float(int(cc[i])),
                            "fanout": round(float(fanout[i]), 2),
                            "frequency_ghz": round(freq, 2),
                            "voltage_v": round(vdd, 3),
                            "activity": round(float(act[i]), 4),
                            "wire_length_um": round(float(wlen[i]), 1),
                            "avg_net_length_um": round(float(wlen[i] / (cc[i] * 0.36)), 3),
                            "bits_per_cycle": round(float(bits[i]), 3),
                            "cg_efficiency": round(float(cg_eff[i]), 2),
                            "clock_period_ps": round(per, 1), "wns_ps": round(wns, 1), "tns_ps": round(tns, 1),
                            "violating_endpoints": float(viol),
                            "fmax_ghz": round(1000.0 / (per - wns), 4),
                            "tool": "powermet-demo", "tool_version": __version__,
                        })
                    # design-level performance for this (build, workload, op)
                    _, mem_bound, ipc_mult = WORKLOADS.get(wl, (1.0, 0.3, 1.0))
                    stall = 1.0 + 0.6 * mem_bound * max(freq / freq0 - 1.0, 0.0)
                    ipc = ops_per_cycle * ipc_mult / stall * (1 + 0.004 * b)
                    perf_rows.append({
                        "design": design, "build": build, "workload": wl, "operating_point": op,
                        "frequency_ghz": round(freq, 2), "voltage_v": round(vdd, 3),
                        "ipc": round(float(ipc), 3),
                        "throughput_gops": round(float(ipc * freq), 3),
                    })

        # a workload trace for performance-tool integration: phases of different workloads/op points
        t = 0
        for k in range(12):
            wl = spec.workloads[k % len(spec.workloads)]
            op = spec.operating_points[(k // 2) % len(spec.operating_points)]
            dur = float(rng.integers(50, 200)) * 1e-6   # seconds
            _, _, ipc_mult = WORKLOADS.get(wl, (1.0, 0.3, 1.0))
            trace_rows.append({"design": design, "interval": k, "t_start_s": round(t, 6), "duration_s": round(dur, 6),
                               "workload": wl, "operating_point": op,
                               "activity_scale": round(float(rng.uniform(0.85, 1.15)), 3)})
            t += dur

    return DemoData(pd.DataFrame(rows), pd.DataFrame(hier_rows), pd.DataFrame(perf_rows),
                    pd.DataFrame(meta_rows), pd.DataFrame(timing_rows), pd.DataFrame(trace_rows), spec)


def generate(spec: DemoSpec = DemoSpec()) -> pd.DataFrame:
    """V0 entry point: the wide FUB-level measurement table."""
    return generate_all(spec).measurements


def make_dirty(df: pd.DataFrame, seed: int = 7, n_rows: int = 200) -> pd.DataFrame:
    """A small sample with injected defects, for demonstrating validation."""
    rng = np.random.default_rng(seed)
    sample = df.sample(n=min(n_rows, len(df)), random_state=seed).reset_index(drop=True).copy()
    idx = rng.permutation(len(sample))
    sample.loc[idx[0:5], "be_mw"] = np.nan
    sample.loc[idx[5:8], "wire_cap_pf"] = -abs(sample.loc[idx[5:8], "wire_cap_pf"])
    sample.loc[idx[8:9], "be_mw"] = 0.0
    sample.loc[idx[9:13], "fanout"] = np.nan
    sample.loc[idx[13:14], "fe_physical_mw"] = sample.loc[idx[13:14], "be_mw"] * 8
    dup = sample.iloc[idx[14:16]].copy()
    return pd.concat([sample, dup], ignore_index=True)
