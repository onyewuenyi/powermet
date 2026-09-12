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


BE_LAYOUTS = ("separate", "same_hierarchy", "replicated", "merged", "split", "mixed")
EXTENSIVE = ("be_mw", "be_leakage_mw", "fe_leakage_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "wire_length_um", "bits_per_cycle")


def be_layout(h: pd.DataFrame, methodology: str, top: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decide how each FUB appears in back-end reports under a methodology.

    Returns (objects, fub_map):
      objects: one row per BE object: fub, path, fraction (share of the FUB's extensive metrics), partition,
               members (for merged blocks: the FUBs whose values are summed into the object)
      fub_map: the six columns plus be_share; several rows per FUB (split) or a shared be_hier (merged) or a
               glob be_hier (replicated) as the methodology requires
    """
    if methodology not in BE_LAYOUTS:
        raise ValueError(f"methodology must be one of {BE_LAYOUTS}")
    fubs = list(h.index)
    parts = sorted(h["partition"].unique())
    objs, rows = [], []
    merged_with: dict[str, str] = {}
    for i, fub in enumerate(fubs):
        part = h.loc[fub, "partition"]
        fe = h.loc[fub, "fe_hier"]
        leaf = f"u_{fub.lower()}"
        base = {"fub": fub, "model_root": h.loc[fub, "model_root"], "partition": part, "fe_hier": fe, "synth_object": h.loc[fub, "synth_object"]}
        kind = "separate"
        if methodology == "same_hierarchy":
            kind = "same"
        elif methodology == "replicated" and i % 4 == 1:
            kind = "replicated"
        elif methodology == "merged" and i % 6 in (2, 3) and i + 1 < len(fubs):
            kind = "merged"
        elif methodology == "split" and i % 5 == 4 and len(parts) > 1:
            kind = "split"
        elif methodology == "mixed":
            kind = {1: "replicated", 2: "merged", 3: "merged", 4: "split"}.get(i % 9, "separate") if len(parts) > 1 else "separate"
        if kind == "same":
            objs.append({"fub": fub, "path": fe, "fraction": 1.0, "partition": part, "members": [fub]})
            rows.append({**base, "be_hier": fe, "be_share": 1.0})
        elif kind == "replicated":
            fr = (0.55, 0.45)
            for k, f in enumerate(fr):
                objs.append({"fub": fub, "path": f"{top}/part_{part.lower()}/{leaf}_{k}", "fraction": f, "partition": part, "members": [fub]})
            rows.append({**base, "be_hier": f"{top}/part_{part.lower()}/{leaf}_*", "be_share": 1.0})
        elif kind == "merged":
            # pair consecutive FUBs of the same partition into one ungrouped BE block
            j = i + 1 if (i % 6 == 2 or i % 9 == 2) else i - 1
            partner = fubs[j] if 0 <= j < len(fubs) and h.loc[fubs[j], "partition"] == part else None
            if partner is None:
                objs.append({"fub": fub, "path": f"{top}/part_{part.lower()}/{leaf}", "fraction": 1.0, "partition": part, "members": [fub]})
                rows.append({**base, "be_hier": f"{top}/part_{part.lower()}/{leaf}", "be_share": 1.0})
                continue
            grp = f"{top}/part_{part.lower()}/u_grp_{min(i, j)}"
            merged_with[fub] = partner
            share = float(h.loc[fub, "n_instances"]) / float(h.loc[fub, "n_instances"] + h.loc[partner, "n_instances"])
            rows.append({**base, "be_hier": grp, "be_share": round(share, 4)})
            if not any(o["path"] == grp for o in objs):
                objs.append({"fub": fub, "path": grp, "fraction": 1.0, "partition": part, "members": [fub, partner]})
        elif kind == "split":
            other = parts[(parts.index(part) + 1) % len(parts)]
            objs.append({"fub": fub, "path": f"{top}/part_{part.lower()}/{leaf}", "fraction": 0.6, "partition": part, "members": [fub]})
            objs.append({"fub": fub, "path": f"{top}/part_{other.lower()}/{leaf}_split", "fraction": 0.4, "partition": other, "members": [fub]})
            rows.append({**base, "be_hier": f"{top}/part_{part.lower()}/{leaf}", "be_share": 1.0})
            rows.append({**base, "partition": other, "be_hier": f"{top}/part_{other.lower()}/{leaf}_split", "be_share": 1.0})
        else:
            objs.append({"fub": fub, "path": f"{top}/part_{part.lower()}/{leaf}", "fraction": 1.0, "partition": part, "members": [fub]})
            rows.append({**base, "be_hier": f"{top}/part_{part.lower()}/{leaf}", "be_share": 1.0})
    return pd.DataFrame(objs), pd.DataFrame(rows)


def object_values(objs: pd.DataFrame, sub: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Per BE object values from per-FUB values: extensive columns scaled by fraction and summed over
    members (merged blocks), intensive columns averaged."""
    out = []
    for _, o in objs.iterrows():
        members = [m for m in o["members"] if m in sub.index]
        if not members:
            continue
        row = {"fub": o["fub"], "path": o["path"], "partition": o["partition"]}
        for c in cols:
            if c not in sub.columns:
                continue
            vals = sub.loc[members, c].astype(float)
            row[c] = float(vals.sum()) * o["fraction"] if c in EXTENSIVE else float(vals.mean())
        out.append(row)
    return pd.DataFrame(out)


def hierarchical_rows(paths_values: list[tuple[str, float]]) -> list[tuple[int, str, float]]:
    """(depth, leaf, value) rows for a PrimePower-style indented report from full paths, with
    aggregate rows for every ancestor (value = sum of descendants)."""
    tree: dict = {}
    for path, v in paths_values:
        node = tree
        for seg in path.split("/"):
            node = node.setdefault(seg, {"__v": 0.0, "__c": {}})
            node["__v"] += v
            node = node["__c"]
    out: list[tuple[int, str, float]] = []

    def walk(children: dict, depth: int):
        for name, node in children.items():          # first-seen order, but every subtree stays contiguous
            out.append((depth, name, node["__v"]))
            walk(node["__c"], depth + 1)

    walk(tree, 0)
    return out


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
            "methodology": spec.methodology,
            "activity_flow": {
                "tool": "Verdi", "hierarchy": "be", "sim_clock_period_ps": sim_period_ps,
                "source_fsdb": {wl: f"/sim/{design.lower()}/{build.lower()}/{wl}/rtl.fsdb" for wl in wls},
                "core": design.lower() + "_top", "mapping": "mapping/fub_map.csv",
                "partition_list": sorted(h["partition"].unique()),
            },
        }, open(run_dir / "metadata.json", "w"), indent=2)
        if status == "superseded":
            log.append(f"{design}/{build}: metadata status=superseded (stale build)")

        # ---- mapping/fub_map.csv (shape depends on the BE methodology) and the BE object layout
        top = design.lower() + "_top"
        objs, fmap = be_layout(h, spec.methodology, top)
        (run_dir / "mapping").mkdir(exist_ok=True)
        fmap[["fub", "model_root", "partition", "fe_hier", "synth_object", "be_hier", "be_share"]].to_csv(
            run_dir / "mapping" / "fub_map.csv", index=False)
        if renamed_unmapped:          # BE renamed these after the map was exported: their objects will not resolve
            objs = objs.copy()
            objs["path"] = [p_ + "_r2" if f_ in renamed_unmapped else p_ for f_, p_ in zip(objs["fub"], objs["path"])]

        # ---- primepower per workload x op (hierarchical report over the BE object layout)
        pp_unit = "mW" if design == d_units_pp else "W"
        for wl in wls:
            for op in ops:
                sub = mrows[(mrows.workload == wl) & (mrows.operating_point == op)].set_index("fub")
                ov = object_values(objs, sub, ("be_mw", "be_leakage_mw"))
                out = run_dir / "primepower" / f"{wl}_{op}"
                out.mkdir(parents=True, exist_ok=True)
                total = float(ov["be_mw"].sum())
                scale = 1.0 if pp_unit == "mW" else 1e-3
                vectorless = (design == vectorless_design and wl == "idle")
                lines = ["*" * 60, "Report : power -hierarchy", f"Design : {top}",
                         f"Version: {m['primepower_version']}", f"Date   : {m['build_date']}", f"Run    : {run_id}",
                         f"Scenario: {wl}@{op}", f"Activity: {'vectorless' if vectorless else 'SAIF'}", f"Power Units = 1{pp_unit}", "*" * 60,
                         f"{'':40}{'Int':>11}{'Switch':>11}{'Leak':>11}{'Total':>11}{'%':>7}",
                         f"{'Hierarchy':40}{'Power':>11}{'Power':>11}{'Power':>11}{'Power':>11}",
                         "-" * 91]

                def prow(indent, label, v, pct, lk):
                    dyn = max(v - lk, 0.0)
                    return f"{' ' * indent}{label:{40 - indent}}{dyn*0.52:11.4e}{dyn*0.48:11.4e}{lk:11.4e}{v:11.4e}{pct:7.1f}"

                pv, leak_of = [], {}
                for _, o in ov.iterrows():
                    v = o["be_mw"]
                    lk = float(o["be_leakage_mw"])
                    if vectorless:
                        v = v * float(rng.uniform(0.85, 1.25))     # default-activity estimate: biased and noisier
                    if zero_pp == o["fub"]:
                        v, lk = 0.0004, 0.0001
                    pv.append((o["path"], v))
                    leak_of[o["path"]] = min(lk, v)
                leaves = {p_: v for p_, v in pv}
                ref = {o["path"]: h.loc[o["fub"], "synth_object"] for _, o in ov.iterrows()}
                for depth, leaf, v in hierarchical_rows(pv):
                    key = None
                    for p_ in leaves:
                        if p_.endswith("/" + leaf) or p_ == leaf:
                            key = p_
                    label = leaf if depth == 0 else (f"{leaf} ({ref[key]})" if key in ref and leaves.get(key) == v else f"{leaf} ({leaf.upper()})")
                    lk = leak_of[key] if key in leaves and leaves.get(key) == v else sum(l_ for p_, l_ in leak_of.items() if ("/" + leaf + "/") in p_ or p_.startswith(leaf + "/"))
                    row = prow(depth * 2, label, v * scale, v / total * 100 if total else 0.0, lk * scale)
                    lines.append(row)
                    if dup_pp and key in leaves and leaves[key] == v and ov[ov["path"] == key]["fub"].iloc[0] == dup_pp:
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
                        lk = min(float(r["fe_leakage_mw"]) * (1.0 if cg else 0.9), v)
                        line = f"{h.loc[r['fub'], 'fe_hier']:44}{(v-lk)*0.54:10.3f}{(v-lk)*0.46:11.3f}{lk:9.3f}{v:9.3f}"
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
        ov = object_values(objs, first.set_index("fub"), ("wire_cap_pf", "cell_cap_pf", "cell_count"))
        for _, r in ov.iterrows():
            fub = r["fub"]
            if fub in missing_rc:
                continue
            w, c = r["wire_cap_pf"] * rc_scale, r["cell_cap_pf"] * rc_scale
            if neg_rc == fub:
                w = -w
            lines.append(f"{r['path']:44}{int(r['cell_count'] * 0.36):8d}{w + c:12.4f}{w:12.4f}{c:12.4f}")
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
        ov = object_values(objs, first.set_index("fub"), ("area", "cell_count", "fanout", "wire_length_um", "avg_net_length_um"))
        for _, r in ov.iterrows():
            lines.append(f"{r['path']:44}{r['area']:12.1f}{int(r['cell_count']):12d}{r['fanout']:12.2f}{0.55 + 0.3 * rng.random():13.2f}"
                         f"{r['wire_length_um']:14.1f}{r['avg_net_length_um']:12.3f}")
        (run_dir / "implementation" / "qor_summary.rpt").write_text("\n".join(lines) + "\n")

        # ---- SAIF per workload (physical hierarchy, op-independent): output of the FSDB -> SAIF flow
        (run_dir / "activity").mkdir(exist_ok=True)
        duration_ps = int(sim_period_ps * 20000)          # 20k simulated cycles
        cycles = duration_ps / sim_period_ps
        parts = sorted(h["partition"].unique())
        for wl in wls:
            sub = mrows[(mrows.workload == wl) & (mrows.operating_point == ops[0])].set_index("fub")
            ov = object_values(objs, sub, ("activity", "bits_per_cycle"))
            lines = ["(SAIFILE", '(SAIFVERSION "2.0")', '(DIRECTION "backward")', f'(DESIGN "{top}")',
                     f'(DATE "{m["build_date"]}")', '(VENDOR "Synopsys")', '(PROGRAM_NAME "Verdi")', '(VERSION "V-2024.09")',
                     "(DIVIDER / )", "(TIMESCALE 1 ps)", f"(DURATION {duration_ps})"]
            # nest instances by path; every object is a leaf instance with its own NET block
            tree: dict = {}
            for _, o in ov.iterrows():
                node = tree
                for seg in o["path"].split("/"):
                    node = node.setdefault(seg, {})
                node["__obj__"] = o

            def emit(name, node, depth):
                ind = "  " * depth
                lines.append(f"{ind}(INSTANCE {name}")
                o = node.get("__obj__")
                if o is not None:
                    n_nets = max(1, int(round(o["bits_per_cycle"] / max(o["activity"], 1e-6))))
                    tc = int(round(o["activity"] * cycles))
                    lines.append(f"{ind}  (NET")
                    for k in range(n_nets):
                        jitter = int(rng.integers(-tc // 10 - 1, tc // 10 + 2)) if tc > 10 else 0
                        t1 = int(duration_ps * rng.uniform(0.3, 0.7))
                        lines.append(f"{ind}    (d\\[{k}\\] (T0 {duration_ps - t1}) (T1 {t1}) (TX 0) (TC {max(tc + jitter, 0)}) (IG 0))")
                    lines.append(f"{ind}  )")
                for child, sub_node in node.items():
                    if child != "__obj__":
                        emit(child, sub_node, depth + 1)
                lines.append(f"{ind})")

            for name, node in tree.items():
                emit(name, node, 0)
            lines.append(")")
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
                    ov = object_values(objs, sub.set_index("fub"), ("be_mw", "be_leakage_mw"))
                    for _, r in ov.iterrows():
                        v = r["be_mw"] * 0.97 * float(rng.lognormal(0, 0.03))
                        lk = min(float(r["be_leakage_mw"]) * 0.95, v)
                        lines.append(f"{r['path']:44}{(v-lk)*0.54:10.2f}{(v-lk)*0.46:11.2f}{lk:9.2f}{v:9.2f}")
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
                    f"create_power_domain PD_{part} -elements {{{(' '.join(sorted(set(objs[objs['partition'] == part]['path']))) if spec.methodology == 'same_hierarchy' else top + '/part_' + part.lower())}}}",
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
    lines = ["# power budgets and convergence targets: scope = design | partition:<name> | fub:<name>; tolerance per milestone (%)",
             "# metric = be_mw (default) | cdyn_pf (CdynTot, pF) | be_leakage_mw (LkgPwr) | be_dynamic_mw",
             "[defaults]", 'workload = "typical"', 'operating_point = "nom"',
             "tolerance_pct = { rtl = 25, synthesis = 15, placement = 10, route = 5, signoff = 0 }", ""]
    last = builds[-1]
    wl0 = "typical" if "typical" in spec.workloads else spec.workloads[0]
    op0 = "nom" if "nom" in spec.operating_points else spec.operating_points[0]
    i0 = lines.index("[defaults]") + 1
    lines[i0:i0 + 2] = [f'workload = "{wl0}"', f'operating_point = "{op0}"']
    for design in designs:
        sub = meas[(meas.design == design) & (meas.build == last) & (meas.workload == wl0) & (meas.operating_point == op0)]
        lines += ["[[budget]]", f'design = "{design}"', 'scope = "design"', f"be_mw = {sub['be_mw'].sum() * 0.98:.0f}", 'owner = "power lead"', ""]
        dyn = (sub["be_mw"] - sub["be_leakage_mw"]).clip(lower=0)
        cdyn = float((dyn / (sub["voltage_v"] ** 2 * sub["frequency_ghz"])).sum())
        lines += ["[[budget]]", f'design = "{design}"', 'scope = "design"', 'metric = "cdyn_pf"', f"target = {cdyn * 0.96:.1f}",
                  'note = "CdynTot: effective switched capacitance target, V/f independent"', ""]
        lines += ["[[budget]]", f'design = "{design}"', 'scope = "design"', 'metric = "be_leakage_mw"', f"target = {sub['be_leakage_mw'].sum() * 1.03:.1f}",
                  'note = "LkgPwr at the nominal corner"', ""]
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
