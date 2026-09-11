"""Cadence Voltus adapter: an alternate BE signoff power engine, per physical hierarchy.

Kept as its own metric (`be_voltus_mw`) rather than overwriting `be_mw` so the two engines can be
qualified against each other with `powermet qualify --a be_mw --b be_voltus_mw`. Representative
`report_power -hierarchy` style:

    Cadence Voltus Power Report
    Version: 23.10
    Design: gpu_a_top   Run: gpu_a_b001_r1234   Date: 2026-03-02
    Activity: SAIF   Scenario: typical@nom
    Units: mW
    Instance                                Internal   Switching   Leakage   Total
    gpu_a_top/part_pcore0/u_scheduler         38.10       33.20      5.10   76.40
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (BE_HIER, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, record, to_float, tool_name, units_from_text)

SOURCE = "voltus"
OPTIONAL = True          # absent files are not an ingest error
TOOL_FAMILY = "Cadence Voltus (alternate signoff engine)"
SUPPORTED_VERSIONS = ("23.10",)
DESCRIPTION = "BE power from a second signoff engine, for engine-to-engine qualification"
DEFAULT_PATTERN = "voltus/{workload}_{operating_point}/power_hier.rpt"
OBJECT_KIND = BE_HIER
METRIC = "be_voltus_mw"


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Instance")
    unit = units_from_text(hdr.get("units", "mW"), "mW")
    start = find_table_start(lines, "Instance")
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        val, cu = convert_unit(to_float(parts[-1]), unit, "be_mw")
        rows.append(record(parts[0], OBJECT_KIND, METRIC, val, cu, unit))
    wl, _, op = hdr.get("scenario", "").partition("@")
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "Voltus"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"),
        workload=context.get("workload") or (wl or None), operating_point=context.get("operating_point") or (op or None),
        activity_mode=hdr.get("activity", "").lower() or None,
    )
