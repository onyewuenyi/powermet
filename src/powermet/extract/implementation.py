"""Implementation (Fusion Compiler / BE) adapter: area, cell count, fanout per BE hierarchy.

Representative QoR summary format:

    Fusion Compiler QoR Summary
    Version: V-2024.09-SP4
    Design: gpu_a_top   Build: B001   Run: gpu_a_b001_r1234
    Clock: core_clk   Frequency: 2500 MHz   Voltage: 0.80 V
    Area units: um^2
    Hierarchy                       CellArea      CellCount    AvgFanout    Utilization   WireLength   AvgNetLen
    gpu_a_top/u_scheduler           1240.5        50000        8.02         0.71          182034.2     10.113

WireLength / AvgNetLen (um) are optional columns: the data-movement distance proxies.
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (BE_HIER, tool_name, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, to_float, units_from_text)

SOURCE = "implementation"
TOOL_FAMILY = "Synopsys Fusion Compiler (or Cadence Innovus QoR)"
SUPPORTED_VERSIONS = ('V-2023.12', 'V-2024.09')      # versions the representative parser was written against
DESCRIPTION = "Area, cell count, fanout, wire length per physical hierarchy"
DEFAULT_PATTERN = "implementation/qor_summary.rpt"
OBJECT_KIND = BE_HIER


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Hierarchy")
    unit = units_from_text(hdr.get("area units", "um2").split()[0], "um2")
    start = find_table_start(lines, "Hierarchy")
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        obj = parts[0]
        area, cu = convert_unit(to_float(parts[1]), unit, "area")
        rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": "area", "value": area, "unit": cu, "unit_original": unit})
        rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": "cell_count", "value": to_float(parts[2]), "unit": "count", "unit_original": "count"})
        rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": "fanout", "value": to_float(parts[3]), "unit": "count", "unit_original": "count"})
        if len(parts) >= 7:
            rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": "wire_length_um", "value": to_float(parts[5]), "unit": "um", "unit_original": "um"})
            rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": "avg_net_length_um", "value": to_float(parts[6]), "unit": "um", "unit_original": "um"})
    notes = [f"area converted from {unit} to um2"] if unit != "um2" else []
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "FusionCompiler"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"), build=hdr.get("build"), notes=notes,
    )
