"""PrimeTime adapter: partition-level timing per operating point.

PrimeTime does not report per FUB; the physical hierarchy is partition-based, so timing is a
partition attribute that every FUB in the partition inherits (see lineage). Representative
`report_timing -summary` per partition/block:

    ****************************************
    Report : timing summary -partition
    Design : gpu_a_top
    Version: V-2024.09-SP3
    Date   : 2026-03-02
    Run    : gpu_a_b001_r1234
    Scenario: nom
    Time units: ps
    ****************************************
    Partition                       Clock     Period       WNS         TNS  Violating  Endpoints
    gpu_a_top/part_pcore0        core_clk      400.0     -12.3     -1450.2        118      52034
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (PARTITION, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, record, to_float, tool_name)

SOURCE = "primetime"
TOOL_FAMILY = "Synopsys PrimeTime"
SUPPORTED_VERSIONS = ('V-2023.12', 'V-2024.09')      # versions the representative parser was written against
DESCRIPTION = "Partition timing: period, WNS, TNS, violating endpoints"
DEFAULT_PATTERN = "primetime/{operating_point}/timing_summary.rpt"
OBJECT_KIND = PARTITION


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Partition")
    unit = hdr.get("time units", "ps").strip()
    start = find_table_start(lines, "Partition")
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        obj = parts[0]
        for metric, tok in (("clock_period_ps", parts[2]), ("wns_ps", parts[3]), ("tns_ps", parts[4])):
            val, cu = convert_unit(to_float(tok), unit, metric)
            rows.append(record(obj, OBJECT_KIND, metric, val, cu, unit))
        rows.append(record(obj, OBJECT_KIND, "violating_endpoints", to_float(parts[5]), "count"))
    notes = [f"time converted from {unit} to ps"] if unit != "ps" else []
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "PrimeTime"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), run_id=hdr.get("run"), report_date=hdr.get("date"),
        operating_point=context.get("operating_point") or hdr.get("scenario"), notes=notes,
    )
