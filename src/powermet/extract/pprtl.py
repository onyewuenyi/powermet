"""PPRTL (RTL power) adapter: FE logical and FE physical-aware power per FE hierarchy.

Representative format (one file per workload x operating point; Mode header selects the metric):

    PPRTL Power Report
    Tool: PowerPro-RTL  Version: R-2025.06-SP2
    Design: GPU_A   Build: B001   Run: gpu_a_b001_r1234
    Mode: rtl                          (or: physical-aware)
    Workload: typical   Operating point: nom
    Activity: saif                     (or: vectorless -> lower-trust estimate)
    Power units: mW
    ------------------------------------------------------------------
    Hierarchy                    Internal   Switching   Leakage    Total   [ClockGatingEff]
    ------------------------------------------------------------------
    gpu_a_top/u_scheduler          40.123      35.111     5.223   80.457    0.82

ClockGatingEff (optional) is the fraction of register clock pins gated, emitted as `cg_efficiency`.
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (FE_HIER, tool_name, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, to_float, units_from_text)

SOURCE = "pprtl"
STAGE = "FE"
TOOL_FAMILY = "RTL power: Synopsys PrimePower RTL / Siemens PowerPro / Keysight (ex-Ansys) PowerArtist / Cadence Joules"
SUPPORTED_VERSIONS = ('R-2024.09', 'R-2025.06')      # versions the representative parser was written against
DESCRIPTION = "FE logical and physical-aware power per RTL hierarchy"
DEFAULT_PATTERN = "pprtl/{workload}_{operating_point}/*_power.rpt"
OBJECT_KIND = FE_HIER
MODE_METRIC = {"rtl": "fe_logical_mw", "logical": "fe_logical_mw",
               "physical-aware": "fe_physical_mw", "physical": "fe_physical_mw"}


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Hierarchy")
    mode = hdr.get("mode", "rtl").lower()
    metric = MODE_METRIC.get(mode)
    if metric is None:
        raise ValueError(f"{path}: unknown PPRTL mode '{mode}'")
    unit = units_from_text(hdr.get("power units", "mW"), "mW")
    start = find_table_start(lines, "Hierarchy")
    rows, notes = [], []
    has_cg = "ClockGatingEff" in "".join(lines[max(start - 3, 0):start])
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5 or parts[0].startswith(("-", "=")):
            continue
        val, cu = convert_unit(to_float(parts[4]), unit, metric)          # columns: hier, int, sw, leak, total[, cg]
        rows.append({"object": parts[0], "object_kind": OBJECT_KIND, "metric": metric,
                     "value": val, "unit": cu, "unit_original": unit})
        if has_cg and len(parts) >= 6 and metric == "fe_physical_mw":
            rows.append({"object": parts[0], "object_kind": OBJECT_KIND, "metric": "cg_efficiency",
                         "value": to_float(parts[5]), "unit": "ratio", "unit_original": "ratio"})
    if unit != "mW":
        notes.append(f"power converted from {unit} to mW")
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=hdr.get("tool", "PPRTL"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"), build=hdr.get("build"),
        workload=context.get("workload") or hdr.get("workload"),
        operating_point=context.get("operating_point") or hdr.get("operating point"),
        activity_mode=(hdr.get("activity") or "").lower() or None, notes=notes,
    )
