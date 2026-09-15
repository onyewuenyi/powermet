"""Power-groups adapter: BE power split by cell group (clock network, registers, combinational, memory)
per BE hierarchy. Optional source; it tells *where* the dynamic power sits, which is what an analysis
engineer needs to turn a hotspot into a power bug (clock network dominant, memory not sleeping, ...).

Representative format of a `report_power -hierarchy -groups {clock_network register combinational memory}`
style report (PrimePower; Voltus `report_power -format detailed -hierarchy` prints the same categories).
Full hierarchy paths are printed per row:

    ****************************************
    Report : power -hierarchy -groups
    Design : gpu_a_top
    Version: V-2024.09-SP3
    Date   : 2026-03-02
    Run    : gpu_a_b001_r1234
    Scenario: typical@nom
    Power Units = 1mW
    ****************************************
    Hierarchy                              clock_network     register  combinational       memory        Total
    --------------------------------------------------------------------------------------------------------
    gpu_a_top/part_pcore0/u_scheduler             12.345       20.100         30.200        0.000       62.645

Group columns are read by name from the header line, so a report with extra groups (io, black_box) or a
different order still parses; only the four canonical groups become metrics. Rows whose groups do not
sum to Total within COMPONENT_TOL are counted in the report notes.

EDA requirement: PrimePower (averaged or time-based mode) with the hierarchical group report enabled,
one report per workload x operating point, same run as the `power_hier.rpt` it accompanies. Without
this report the group columns stay empty and the clock-dominance rule in `analyze anomalies` is skipped.
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (BE_HIER, Located, ParsedReport, SourceInputs, convert_unit, locate, make_records, parse_header,
                                   read_lines, to_float, tool_name, units_from_text)

SOURCE = "power_groups"
STAGE = "BE"
OPTIONAL = True
TOOL_FAMILY = "Synopsys PrimePower / Cadence Voltus (hierarchical power by cell group)"
SUPPORTED_VERSIONS = ("V-2023.12", "V-2024.09")
DESCRIPTION = "BE power per hierarchy split into clock network, register, combinational and memory groups"
DEFAULT_PATTERN = "primepower/{workload}_{operating_point}/power_groups.rpt"
OBJECT_KIND = BE_HIER
GROUP_METRIC = {"clock_network": "be_clock_mw", "register": "be_register_mw", "sequential": "be_register_mw",
                "combinational": "be_comb_mw", "memory": "be_memory_mw"}
COMPONENT_TOL = 0.05


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Hierarchy")
    unit = units_from_text(hdr.get("power units", "mW"), "mW")
    head_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("Hierarchy")), None)
    if head_idx is None:
        raise ValueError(f"{path}: no 'Hierarchy' column header")
    cols = lines[head_idx].split()[1:]
    if "Total" not in cols:
        raise ValueError(f"{path}: group report has no Total column")
    total_i = cols.index("Total")
    rows, n_mismatch = [], 0
    for line in lines[head_idx + 1:]:
        parts = line.split()
        if len(parts) < len(cols) + 1 or parts[0].startswith(("-", "=")):
            continue
        obj, nums = parts[0], parts[1:1 + len(cols)]
        vals = {c: to_float(x) for c, x in zip(cols, nums)}
        total = vals["Total"]
        group_sum = sum(v for c, v in vals.items() if c != "Total")
        if total > 0 and abs(group_sum - total) > COMPONENT_TOL * total:
            n_mismatch += 1
        for c, metric in GROUP_METRIC.items():
            if c in vals:
                val, cu = convert_unit(vals[c], unit, metric)
                rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": metric,
                             "value": val, "unit": cu, "unit_original": unit})
    notes = [f"power converted from {unit} to mW"] if unit != "mW" else []
    if n_mismatch:
        notes.append(f"{n_mismatch} rows where the groups differ from Total by > {COMPONENT_TOL:.0%}")
    scenario = hdr.get("scenario", "")
    wl, _, op = scenario.partition("@")
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "PrimePower"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"),
        workload=context.get("workload") or (wl or None), operating_point=context.get("operating_point") or (op or None),
        activity_mode=(hdr.get("activity") or "").lower() or None, notes=notes,
    )
