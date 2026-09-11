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
        activity/<workload>.saif                 (SAIF in the physical hierarchy, as written by the FSDB -> SAIF flow)
        perf/<workload>_<op>.csv
        voltus/<workload>_<op>/power_hier.rpt       (alternate signoff engine, only for some builds)
        intent/<design>.upf                         (power intent: domains per partition, supply states)
    <root>/traces/<design>_phases.csv           (workload phase trace for performance-tool integration)
    <root>/budgets.toml                         (power budgets per design / partition with milestone tolerances)

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
    vectorless_design = designs[1] if defects and len(designs) > 1 else None   # PrimePower for 'idle' run vectorless
    voltus_builds = set(builds[-2:])                                     # alternate engine only on recent builds
    upf_missing_part = (designs[0], builds[-1]) if defects else None     # newest build's UPF forgets one partition
    upf_v_mismatch = designs[2] if defects and len(designs) > 2 else None  # UPF turbo state disagrees with metadata

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
            if build == builds[-1] and design == designs[0]:   # newest build of one design: BE renamed two FUBs after the map
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
        nom_op = op_table.get("nom") or next(iter(op_table.values()))
        sim_period_ps = round(1000.0 / nom_op["frequency_ghz"], 1)
        b_idx = builds.index(build)
        milestone = ["rtl", "synthesis", "placement", "route", "signoff"][min(4, round(b_idx * 4 / max(len(builds) - 1, 1)))]
        json.dump({
            "schema_version": "1", "design": design, "design_type": "cpu", "build": build, "build_date": m["build_date"],
            "run_id": run_id, "status": status, "milestone": milestone,
            "tools": {"pprtl": m["pprtl_version"], "primepower": m["primepower_version"],
                      "starrc": m["starrc_version"], "fusion": m["fusion_version"], "verdi": "V-2024.09"},
            "workloads": wls, "operating_points": op_table,
            "activity_flow": {
                "tool": "Verdi", "hierarchy": "be", "sim_clock_period_ps": sim_period_ps,
                "source_fsdb": {wl: f"/sim/{design.lower()}/{build.lower()}/{wl}/rtl.fsdb" for wl in wls},
                "core": design.lower() + "_top", "mapping": "mapping/fub_map.csv",
                "partition_list": sorted(h["partition"].unique()),
            },
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

        # ---- primepower per workload x op (physical hierarchy: top -> partition -> block)
        pp_unit = "mW" if design == d_units_pp else "W"
        parts = sorted(h["partition"].unique())
        for wl in wls:
            for op in ops:
                sub = mrows[(mrows.workload == wl) & (mrows.operating_point == op)].set_index("fub")
                out = run_dir / "primepower" / f"{wl}_{op}"
                out.mkdir(parents=True, exist_ok=True)
                top = design.lower() + "_top"
                total = sub["be_mw"].sum()
                scale = 1.0 if pp_unit == "mW" else 1e-3
                vectorless = (design == vectorless_design and wl == "idle")
                lines = ["*" * 60, "Report : power -hierarchy", f"Design : {top}",
                         f"Version: {m['primepower_version']}", f"Date   : {m['build_date']}", f"Run    : {run_id}",
                         f"Scenario: {wl}@{op}", f"Activity: {'vectorless' if vectorless else 'SAIF'}", f"Power Units = 1{pp_unit}", "*" * 60,
                         f"{'':40}{'Int':>11}{'Switch':>11}{'Leak':>11}{'Total':>11}{'%':>7}",
                         f"{'Hierarchy':40}{'Power':>11}{'Power':>11}{'Power':>11}{'Power':>11}",
                         "-" * 91]

                def prow(indent, label, v, pct):
                    return f"{' ' * indent}{label:{40 - indent}}{v*0.45:11.4e}{v*0.45:11.4e}{v*0.10:11.4e}{v:11.4e}{pct:7.1f}"

                lines.append(prow(0, top, total * scale, 100.0))
                for part in parts:
                    members = [f for f in h[h["partition"] == part].index if f in sub.index]
                    ptotal = float(sub.loc[members, "be_mw"].sum())
                    lines.append(prow(2, f"part_{part.lower()} (PART_{part})", ptotal * scale, ptotal / total * 100))
                    for fub in members:
                        r = sub.loc[fub]
                        leaf = be_name(fub).split("/")[-1]
                        v = r["be_mw"]
                        if vectorless:
                            v = v * float(rng.uniform(0.85, 1.25))     # default-activity estimate: biased and noisier
                        if zero_pp == fub:
                            v = 0.0004
                        row = prow(4, f"{leaf} ({h.loc[fub, 'synth_object']})", v * scale, r["be_mw"] / total * 100)
                        lines.append(row)
                        if dup_pp == fub:
                            lines.append(row)
                (out / "power_hier.rpt").write_text("\n".join(lines) + "\n")
        if pp_unit != "W":
            log.append(f"{design}/{build}: PrimePower reported in mW (unit variant)")
        if design == vectorless_design and "idle" in wls and build == builds[0]:
            log.append(f"{design}/*: PrimePower 'idle' scenario is vectorless (default activity), values perturbed")
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
                    cg = mode == "physical-aware"
                    lines = ["PPRTL Power Report", f"Tool: PowerPro-RTL  Version: {m['pprtl_version']}",
                             f"Design: {design}   Build: {build}   Run: {run_id}", f"Mode: {mode}",
                             f"Workload: {wl}   Operating point: {op}", "Activity: saif", "Power units: mW", "-" * 96,
                             f"{'Hierarchy':44}{'Internal':>10}{'Switching':>11}{'Leakage':>9}{'Total':>9}" + ("{:>16}".format("ClockGatingEff") if cg else ""),
                             "-" * 96]
                    for _, r in sub.iterrows():
                        v = r[col]
                        line = f"{h.loc[r['fub'], 'fe_hier']:44}{v*0.5:10.3f}{v*0.42:11.3f}{v*0.08:9.3f}{v:9.3f}"
                        if cg:
                            line += f"{r['cg_efficiency']:16.2f}"
                        lines.append(line)
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

        # ---- SAIF per workload (physical hierarchy, op-independent): output of the FSDB -> SAIF flow
        (run_dir / "activity").mkdir(exist_ok=True)
        duration_ps = int(sim_period_ps * 20000)          # 20k simulated cycles
        cycles = duration_ps / sim_period_ps
        parts = sorted(h["partition"].unique())
        for wl in wls:
            sub = mrows[(mrows.workload == wl) & (mrows.operating_point == ops[0])].set_index("fub")
            lines = ["(SAIFILE", '(SAIFVERSION "2.0")', '(DIRECTION "backward")', f'(DESIGN "{top}")',
                     f'(DATE "{m["build_date"]}")', '(VENDOR "Synopsys")', '(PROGRAM_NAME "Verdi")', '(VERSION "V-2024.09")',
                     "(DIVIDER / )", "(TIMESCALE 1 ps)", f"(DURATION {duration_ps})", f"(INSTANCE {top}"]
            for part in parts:
                lines.append(f"  (INSTANCE part_{part.lower()}")
                for fub in h[h["partition"] == part].index:
                    if fub not in sub.index:
                        continue
                    r = sub.loc[fub]
                    leaf = be_name(fub).split("/")[-1]
                    n_nets = int(round(r["bits_per_cycle"] / max(r["activity"], 1e-6)))
                    n_nets = max(1, n_nets)
                    tc = int(round(r["activity"] * cycles))
                    lines.append(f"    (INSTANCE {leaf}")
                    lines.append("      (NET")
                    for k in range(n_nets):
                        jitter = int(rng.integers(-tc // 10 - 1, tc // 10 + 2)) if tc > 10 else 0
                        t1 = int(duration_ps * rng.uniform(0.3, 0.7))
                        lines.append(f"        (d\\[{k}\\] (T0 {duration_ps - t1}) (T1 {t1}) (TX 0) (TC {max(tc + jitter, 0)}) (IG 0))")
                    lines.append("      )")
                    lines.append("    )")
                lines.append("  )")
            lines.append("))")
            (run_dir / "activity" / f"{wl}.saif").write_text("\n".join(lines) + "\n")

        # ---- voltus (alternate signoff engine) on recent builds: small systematic bias vs PrimePower
        if build in voltus_builds:
            for wl in wls:
                for op in ops:
                    sub = mrows[(mrows.workload == wl) & (mrows.operating_point == op)]
                    out = run_dir / "voltus" / f"{wl}_{op}"
                    out.mkdir(parents=True, exist_ok=True)
                    lines = ["Cadence Voltus Power Report", "Version: 23.10", f"Design: {top}   Run: {run_id}   Date: {m['build_date']}",
                             f"Activity: SAIF   Scenario: {wl}@{op}", "Units: mW",
                             f"{'Instance':44}{'Internal':>10}{'Switching':>11}{'Leakage':>9}{'Total':>9}"]
                    for _, r in sub.iterrows():
                        v = r["be_mw"] * 0.97 * float(rng.lognormal(0, 0.03))
                        lines.append(f"{be_name(r['fub']):44}{v*0.5:10.2f}{v*0.42:11.2f}{v*0.08:9.2f}{v:9.2f}")
                    (out / "power_hier.rpt").write_text("\n".join(lines) + "\n")

        # ---- UPF power intent: one domain per partition, supply states per operating point
        (run_dir / "intent").mkdir(exist_ok=True)
        upf = [f"# UPF power intent for {design} {build}", "upf_version 2.1", "create_supply_port VSS", "create_supply_net VSS"]
        skip_part = sorted(h["partition"].unique())[-1] if upf_missing_part == (design, build) else None
        for part in sorted(h["partition"].unique()):
            if part == skip_part:
                continue
            net = f"VDD_{part}"
            upf += [f"create_supply_port {net}", f"create_supply_net {net}",
                    f"create_power_domain PD_{part} -elements {{{top}/part_{part.lower()}}}",
                    f"set_domain_supply_net PD_{part} -primary_power_net {net} -primary_ground_net VSS"]
            states = " ".join(f"-state {{{op} {vals['voltage_v'] + (0.05 if (design == upf_v_mismatch and op == 'turbo' and part == sorted(h['partition'].unique())[0]) else 0.0):.3f}}}"
                              for op, vals in op_table.items())
            upf.append(f"add_port_state {net} {states} -state {{off off}}")
        (run_dir / "intent" / f"{design.lower()}.upf").write_text("\n".join(upf) + "\n")
        if skip_part:
            log.append(f"{design}/{build}: UPF omits partition {skip_part} (FUBs without a power domain)")
        if design == upf_v_mismatch and build == builds[0]:
            log.append(f"{design}/*: UPF turbo state voltage differs from metadata for the first partition")

        # ---- perf per workload x op
        (run_dir / "perf").mkdir(exist_ok=True)
        for wl in wls:
            for op in ops:
                r = perf[(perf.design == design) & (perf.build == build) & (perf.workload == wl) & (perf.operating_point == op)].iloc[0]
                pd.DataFrame([{"metric": "ipc", "value": r["ipc"], "unit": "ops/cycle"},
                              {"metric": "throughput_gops", "value": r["throughput_gops"], "unit": "Gops/s"}]
                             ).to_csv(run_dir / "perf" / f"{wl}_{op}.csv", index=False)

    # budgets: design totals at ~2% under the final build (so late builds sit at risk) and partition budgets
    lines = ["# power budgets: scope = design | partition:<name> | fub:<name>; tolerance per milestone (%)",
             "[defaults]", 'workload = "typical"', 'operating_point = "nom"',
             "tolerance_pct = { rtl = 25, synthesis = 15, placement = 10, route = 5, signoff = 0 }", ""]
    last = builds[-1]
    wl0 = "typical" if "typical" in spec.workloads else spec.workloads[0]
    op0 = "nom" if "nom" in spec.operating_points else spec.operating_points[0]
    lines[2:4] = [f'workload = "{wl0}"', f'operating_point = "{op0}"']
    for design in designs:
        sub = meas[(meas.design == design) & (meas.build == last) & (meas.workload == wl0) & (meas.operating_point == op0)]
        lines += ["[[budget]]", f'design = "{design}"', 'scope = "design"', f"be_mw = {sub['be_mw'].sum() * 0.98:.0f}", 'owner = "power lead"', ""]
        hh = hier[hier.design == design]
        for i, part in enumerate(sorted(hh["partition"].unique())):
            fubs = hh[hh.partition == part]["fub"]
            p_mw = sub[sub.fub.isin(fubs)]["be_mw"].sum()
            factor = 1.06 if i % 3 == 0 else (0.95 if i % 3 == 1 else 1.0)
            lines += ["[[budget]]", f'design = "{design}"', f'scope = "partition:{part}"', f"be_mw = {p_mw * factor:.0f}", ""]
    (root / "budgets.toml").write_text("\n".join(lines) + "\n")

    if len(data.traces):
        (root / "traces").mkdir(exist_ok=True)
        for design, t in data.traces.groupby("design"):
            t.to_csv(root / "traces" / f"{design}_phases.csv", index=False)
    return root, data, log
