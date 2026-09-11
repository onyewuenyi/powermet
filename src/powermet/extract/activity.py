"""Activity adapter (V3): per-FE-hierarchy toggle activity per workload (SAIF/FSDB summary).

    Activity Summary
    Tool: saif_summary  Version: 1.2
    Workload: compute
    Hierarchy                     AvgToggleRate    NetCount    BitsPerCycle
    gpu_a_top/u_scheduler         0.2134           18234       63.8

BitsPerCycle is optional: average data bits moved per cycle (data-movement traffic proxy).
"""

from __future__ import annotations

from pathlib import Path

from powermet.extract.base import (FE_HIER, tool_name, Located, ParsedReport, SourceInputs, find_table_start, locate, make_records,
                                   parse_header, read_lines, to_float)

SOURCE = "activity"
DEFAULT_PATTERN = "activity/{workload}.activity.rpt"
OBJECT_KIND = FE_HIER


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Hierarchy")
    start = find_table_start(lines, "Hierarchy")
    rows = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append({"object": parts[0], "object_kind": OBJECT_KIND, "metric": "activity",
                     "value": to_float(parts[1]), "unit": "ratio", "unit_original": "ratio"})
        if len(parts) >= 4:
            rows.append({"object": parts[0], "object_kind": OBJECT_KIND, "metric": "bits_per_cycle",
                         "value": to_float(parts[3]), "unit": "bits", "unit_original": "bits"})
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=hdr.get("tool", "activity"), tool_version=hdr.get("version", "?"),
        records=make_records(rows), workload=context.get("workload") or hdr.get("workload"),
    )
