"""StarRC adapter: parasitic capacitance per BE hierarchy (wire cap and pin cap).

Representative summary format:

    StarRC Parasitic Summary
    Version: V-2024.09
    Design: gpu_a_top   Corner: typical_rc
    Capacitance units: pF
    Instance                        Nets     TotalCap     WireCap      PinCap
    gpu_a_top/u_scheduler           12034    25.3456      18.2000      7.1456
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (BE_HIER, tool_name, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, to_float, units_from_text)

SOURCE = "starrc"
TOOL_FAMILY = "Synopsys StarRC"
SUPPORTED_VERSIONS = ('V-2023.12', 'V-2024.09')      # versions the representative parser was written against
DESCRIPTION = "Parasitic wire and pin capacitance per physical hierarchy"
DEFAULT_PATTERN = "starrc/parasitics_summary.rpt"
OBJECT_KIND = BE_HIER


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Instance")
    unit = units_from_text(hdr.get("capacitance units", "pF"), "pF")
    start = find_table_start(lines, "Instance")
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        obj = parts[0]
        for metric, tok in (("wire_cap_pf", parts[3]), ("cell_cap_pf", parts[4])):
            val, cu = convert_unit(to_float(tok), unit, metric)
            rows.append({"object": obj, "object_kind": OBJECT_KIND, "metric": metric,
                         "value": val, "unit": cu, "unit_original": unit})
    notes = [f"capacitance converted from {unit} to pF"] if unit != "pF" else []
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "StarRC"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"), notes=notes,
    )
