"""Performance adapter (V3): design-level performance per workload x operating point.

    perf/{workload}_{operating_point}.csv
    metric,value,unit
    ipc,42.1,ops/cycle
    throughput_gops,105.3,Gops/s
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from powermet.extract.base import DESIGN, tool_name, Located, ParsedReport, SourceInputs, convert_unit, locate, make_records

SOURCE = "perf"
OPTIONAL = True          # absent files are not an ingest error
TOOL_FAMILY = "Performance model / simulator export"
SUPPORTED_VERSIONS = ('csv-1',)      # versions the representative parser was written against
DESCRIPTION = "Design-level IPC and throughput per workload x operating point"
DEFAULT_PATTERN = "perf/{workload}_{operating_point}.csv"
OBJECT_KIND = DESIGN


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        unit = str(r.get("unit", "")) or "ratio"
        val, cu = convert_unit(float(r["value"]), unit, str(r["metric"]))
        rows.append({"object": "*", "object_kind": OBJECT_KIND, "metric": str(r["metric"]),
                     "value": val, "unit": cu, "unit_original": unit})
    return ParsedReport(
        source=SOURCE, path=Path(path), tool="perf-model", tool_version="1",
        records=make_records(rows), workload=context.get("workload"),
        operating_point=context.get("operating_point"),
    )
