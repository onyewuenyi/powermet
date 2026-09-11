"""Write mock raw EDA run directories from the synthetic generator.

Layout (one run directory per design x build):

    <root>/<design>/<build>/
        metadata.json
        mapping/fub_map.csv
        pprtl/<workload>_<op>/rtl_power.rpt
        pprtl/<workload>_<op>/physical_power.rpt
        primepower/<workload>_<op>/power_hier.rpt
        primetime/<op>/timing_summary.rpt          (partition-level timing)
        starrc/parasitics_summary.rpt
        implementation/qor_summary.rpt
        activity/<workload>.activity.rpt
        perf/<workload>_<op>.csv
    <root>/traces/<design>_phases.csv           (workload phase trace for performance-tool integration)

With defects=True a handful of realistic problems are injected so `powermet sanitize`
has something to find: alternate units (W, fF), a BE-renamed instance missing from the
map, StarRC rows missing for some FUBs, a duplicated report row, a negative value,
a near-zero BE value, and one superseded (stale) build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from powermet.demo import DemoData, DemoSpec, generate_all


@dataclass
class MockDefects:
    enabled: bool = True
    log: list[str] = field(default_factory=list)


def _fmt_table(header: list[str], rows: list[list[str]], widths: list[int]) -> list[str]:
    def line(cells):
        return "".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
    return [line(header), "-" * sum(widths)] + [line(r) for r in rows]


def write_mock_runs(root: str | Path, spec: DemoSpec | None = None, defects: bool = True) -> tuple[Path, DemoData, list[str]]:
    spec = spec or DemoSpec(workloads=("idle", "typical", "compute", "memory"), operating_points=("eco", "nom", "turbo"))
    data = generate_all(spec)
    root = Path(root)
    rng = np.random.default_rng(spec.seed + 1)
    log: list[str] = []
    meas, hier, perf, meta = data.measurements, data.hierarchy, data.performance, data.metadata
    timing = data.timing

    designs = list(meas["design"].unique())
    builds = list(meas["build"].unique())
    # defect targets (deterministic)
    d_units_pp = designs[1] if defects and len(designs) > 1 else None        # PrimePower in mW instead of W
    d_units_rc = designs[2] if defects and len(designs) > 2 else None        # StarRC in fF
    stale_build = (designs[0], builds[1]) if defects and len(builds) > 1 else None
    stale_signoff = (designs[-1], builds[0]) if defects else None      # one StarRC report from an older run

    for (design, build), mrows in meas.groupby(["design", "build"], sort=True):
        run_dir = root / design / build
        h = hier[hier["design"] == design].set_index("fub")
        m = meta[(meta["design"] == design) & (meta["build"] == build)].iloc[0]
        run_id = m["run_id"]
        wls = list(spec.workloads)
        ops = list(spec.operating_points)

        # ---- defects for this run
        renamed_unmapped: set[str] = set()
        missing_rc: set[str] = set()
        dup_pp: str | None = None
        neg_rc: str | None = None
        zero_pp: str | None = None
        if defects:
            fubs = list(h.index)
            if build == builds[-1]:   # newest build: BE renamed two FUBs after the map was made
                for f in rng.choice(fubs, size=2, replace=False):
                    renamed_unmapped.add(str(f))
            if build == builds[0] or build == builds[len(builds) // 2]:
                for f in rng.choice(fubs, size=3, replace=False):
                    missing_rc.add(str(f))
            if build == builds[2 % len(builds)]:
                dup_pp = str(rng.choice(fubs))
                neg_rc = str(rng.choice(fubs))
            if build == builds[3 % len(builds)]:
                zero_pp = str(rng.choice(fubs))

        # ---- metadata.json
        (run_dir).mkdir(parents=True, exist_ok=True)
        op_table = {}
        for op in ops:
            r = perf[(perf.design == design) & (perf.build == build) & (perf.operating_point == op)].iloc[0]
            op_table[op] = {"voltage_v": float(r["voltage_v"]), "frequency_ghz": float(r["frequency_ghz"])}
        status = "superseded" if stale_build == (design, build) else "current"
        json.dump({
            "schema_version": "1", "design": design, "build": build, "build_date": m["build_date"],
            "run_id": run_id, "status": status,
            "tools": {"pprtl": m["pprtl_version"], "primepower": m["primepower_version"],
                      "starrc": m["starrc_version"], "fusion": m["fusion_version"]},
            "workloads": wls, "operating_points": op_table,
        }, open(run_dir / "metadata.json", "w"), indent=2)
        if status == "superseded":
            log.append(f"{design}/{build}: metadata status=superseded (stale build)")

        # ---- mapping/fub_map.csv
        (run_dir / "mapping").mkdir(exist_ok=True)
        h.reset_index()[["fub", "model_root", "partition", "fe_hier", "synth_object", "be_hier"]].to_csv(
            run_dir / "mapping" / "fub_map.csv", index=False)

        def be_name(fub: str) -> str:
            name = h.loc[fub, "be_hier"]
            return name + "_r2" if fub in renamed_unmapped else name

        # ---- primepower per workload x op
        pp_unit = "mW" if design == d_units_pp else "W"
        for wl in wls:
            for op in ops:
                sub = mrows[(mrows.workload == wl) & (mrows.operating_point == op)]
                out = run_dir / "primepower" / f"{wl}_{op}"
                out.mkdir(parents=True, exist_ok=True)
                top = design.lower() + "_top"
                total = sub["be_mw"].sum()
                scale = 1.0 if pp_unit == "mW" else 1e-3
                lines = ["*" * 60, "Report : power -hierarchy", f"Design : {top}",
                         f"Version: {m['primepower_version']}", f"Date   : {m['build_date']}", f"Run    : {run_id}",
                         f"Scenario: {wl}@{op}", f"Power Units = 1{pp_unit}", "*" * 60,
                         f"{'':40}{'Int':>11}{'Switch':>11}{'Leak':>11}{'Total':>11}{'%':>7}",
                         f"{'Hierarchy':40}{'Power':>11}{'Power':>11}{'Power':>11}{'Power':>11}",
                         "-" * 91]
                lines.append(f"{top:40}{total*0.45*scale:11.4e}{total*0.45*scale:11.4e}{total*0.10*scale:11.4e}{total*scale:11.4e}{100.0:7.1f}")
                for _, r in sub.iterrows():
                    fub = r["fub"]
                    leaf = be_name(fub).split("/")[-1]
                    v = r["be_mw"]
                    if zero_pp == fub:
                        v = 0.0004
                    v *= scale
                    ref = h.loc[fub, "synth_object"]
                    row = f"  {leaf + ' (' + ref + ')':38}{v*0.45:11.4e}{v*0.45:11.4e}{v*0.10:11.4e}{v:11.4e}{(r['be_mw']/total*100):7.1f}"
                    lines.append(row)
                    if dup_pp == fub:
                        lines.append(row)
                (out / "power_hier.rpt").write_text("\n".join(lines) + "\n")
        if pp_unit != "W":
            log.append(f"{design}/{build}: PrimePower reported in mW (unit variant)")
        if renamed_unmapped:
            log.append(f"{design}/{build}: BE renamed instances not in map: {sorted(renamed_unmapped)}")
        if dup_pp:
            log.append(f"{design}/{build}: duplicated PrimePower row for {dup_pp}")
        if zero_pp:
            log.append(f"{design}/{build}: near-zero PrimePower value for {zero_pp}")

        # ---- primetime per op (partition-level, workload independent)
        top = design.lower() + "_top"
        for op in ops:
            t = timing[(timing.design == design) & (timing.build == build) & (timing.operating_point == op)]
            out = run_dir / "primetime" / op
            out.mkdir(parents=True, exist_ok=True)
            lines = ["*" * 60, "Report : timing summary -partition", f"Design : {top}", f"Version: {m['primepower_version']}",
                     f"Date   : {m['build_date']}", f"Run    : {run_id}", f"Scenario: {op}", "Time units: ps", "*" * 60,
                     f"{'Partition':32}{'Clock':>10}{'Period':>10}{'WNS':>10}{'TNS':>12}{'Violating':>11}{'Endpoints':>11}",
                     "-" * 96]
            for _, r in t.iterrows():
                lines.append(f"{top + '/part_' + r['partition'].lower():32}{'core_clk':>10}{r['clock_period_ps']:10.1f}"
                             f"{r['wns_ps']:10.1f}{r['tns_ps']:12.1f}{int(r['violating_endpoints']):11d}{int(r['endpoints']):11d}")
                if defects and build == builds[1 % len(builds)] and r["partition"] == sorted(t["partition"])[0]:
                    lines[-1] = lines[-1]   # keep
            (out / "timing_summary.rpt").write_text("\n".join(lines) + "\n")
        if defects and build == builds[-1]:
            # newest build: one partition's timing report is missing (timing not yet run)
            missing_part = sorted(timing[(timing.design == design) & (timing.build == build)]["partition"].unique())[-1]
            for op in ops:
                path = run_dir / "primetime" / op / "timing_summary.rpt"
                txt = path.read_text().splitlines()
                path.write_text("\n".join(l for l in txt if not l.startswith(f"{top}/part_{missing_part.lower()}")) + "\n")
            log.append(f"{design}/{build}: PrimeTime rows missing for partition {missing_part}")

        # ---- pprtl per workload x op (rtl + physical-aware)
        for wl in wls:
            for op in ops:
                sub = mrows[(mrows.workload == wl) & (mrows.operating_point == op)]
                out = run_dir / "pprtl" / f"{wl}_{op}"
                out.mkdir(parents=True, exist_ok=True)
                for mode, col, fname in (("rtl", "fe_logical_mw", "rtl_power.rpt"),
                                         ("physical-aware", "fe_physical_mw", "physical_power.rpt")):
                    lines = ["PPRTL Power Report", f"Tool: PowerPro-RTL  Version: {m['pprtl_version']}",
                             f"Design: {design}   Build: {build}   Run: {run_id}", f"Mode: {mode}",
                             f"Workload: {wl}   Operating point: {op}", "Power units: mW", "-" * 80,
                             f"{'Hierarchy':44}{'Internal':>10}{'Switching':>11}{'Leakage':>9}{'Total':>9}", "-" * 80]
                    for _, r in sub.iterrows():
                        v = r[col]
                        lines.append(f"{h.loc[r['fub'], 'fe_hier']:44}{v*0.5:10.3f}{v*0.42:11.3f}{v*0.08:9.3f}{v:9.3f}")
                    (out / fname).write_text("\n".join(lines) + "\n")

        # ---- starrc (build-level)
        rc_unit = "fF" if design == d_units_rc else "pF"
        rc_scale = 1e3 if rc_unit == "fF" else 1.0
        first = mrows[(mrows.workload == wls[0]) & (mrows.operating_point == ops[0])]
        (run_dir / "starrc").mkdir(exist_ok=True)
        lines = ["StarRC Parasitic Summary", f"Version: {m['starrc_version']}",
                 f"Design: {design.lower()}_top   Corner: typical_rc", f"Run: {run_id}",
                 f"Capacitance units: {rc_unit}",
                 f"{'Instance':44}{'Nets':>8}{'TotalCap':>12}{'WireCap':>12}{'PinCap':>12}"]
        for _, r in first.iterrows():
            fub = r["fub"]
            if fub in missing_rc:
                continue
            w, c = r["wire_cap_pf"] * rc_scale, r["cell_cap_pf"] * rc_scale
            if neg_rc == fub:
                w = -w
            lines.append(f"{be_name(fub):44}{int(r['cell_count'] * 0.36):8d}{w + c:12.4f}{w:12.4f}{c:12.4f}")
        if stale_signoff == (design, build):
            lines[3] = f"Run: {run_id}_old"
            log.append(f"{design}/{build}: StarRC report carries a different run id (stale signoff artifact)")
        (run_dir / "starrc" / "parasitics_summary.rpt").write_text("\n".join(lines) + "\n")
        if rc_unit != "pF":
            log.append(f"{design}/{build}: StarRC reported in fF (unit variant)")
        if missing_rc:
            log.append(f"{design}/{build}: StarRC rows missing for {sorted(missing_rc)}")
        if neg_rc:
            log.append(f"{design}/{build}: negative StarRC wire cap for {neg_rc}")

        # ---- implementation (build-level)
        (run_dir / "implementation").mkdir(exist_ok=True)
        nom = op_table.get("nom") or next(iter(op_table.values()))
        lines = ["Fusion Compiler QoR Summary", f"Version: {m['fusion_version']}",
                 f"Design: {design.lower()}_top   Build: {build}   Run: {run_id}",
                 f"Clock: core_clk   Frequency: {nom['frequency_ghz']*1000:.0f} MHz   Voltage: {nom['voltage_v']:.2f} V",
                 "Area units: um^2   Length units: um",
                 f"{'Hierarchy':44}{'CellArea':>12}{'CellCount':>12}{'AvgFanout':>12}{'Utilization':>13}{'WireLength':>14}{'AvgNetLen':>12}"]
        for _, r in first.iterrows():
            lines.append(f"{be_name(r['fub']):44}{r['area']:12.1f}{int(r['cell_count']):12d}{r['fanout']:12.2f}{0.55 + 0.3 * rng.random():13.2f}"
                         f"{r['wire_length_um']:14.1f}{r['avg_net_length_um']:12.3f}")
        (run_dir / "implementation" / "qor_summary.rpt").write_text("\n".join(lines) + "\n")

        # ---- activity per workload (op-independent)
        (run_dir / "activity").mkdir(exist_ok=True)
        for wl in wls:
            sub = mrows[(mrows.workload == wl) & (mrows.operating_point == ops[0])]
            lines = ["Activity Summary", "Tool: saif_summary  Version: 1.2", f"Workload: {wl}",
                     f"{'Hierarchy':44}{'AvgToggleRate':>15}{'NetCount':>10}{'BitsPerCycle':>14}"]
            for _, r in sub.iterrows():
                lines.append(f"{h.loc[r['fub'], 'fe_hier']:44}{r['activity']:15.4f}{int(r['cell_count'] * 0.36):10d}{r['bits_per_cycle']:14.3f}")
            (run_dir / "activity" / f"{wl}.activity.rpt").write_text("\n".join(lines) + "\n")

        # ---- perf per workload x op
        (run_dir / "perf").mkdir(exist_ok=True)
        for wl in wls:
            for op in ops:
                r = perf[(perf.design == design) & (perf.build == build) & (perf.workload == wl) & (perf.operating_point == op)].iloc[0]
                pd.DataFrame([{"metric": "ipc", "value": r["ipc"], "unit": "ops/cycle"},
                              {"metric": "throughput_gops", "value": r["throughput_gops"], "unit": "Gops/s"}]
                             ).to_csv(run_dir / "perf" / f"{wl}_{op}.csv", index=False)

    if len(data.traces):
        (root / "traces").mkdir(exist_ok=True)
        for design, t in data.traces.groupby("design"):
            t.to_csv(root / "traces" / f"{design}_phases.csv", index=False)
    return root, data, log
