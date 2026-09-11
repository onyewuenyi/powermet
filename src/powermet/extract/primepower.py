"""PrimePower adapter: BE signoff power per BE hierarchy.

Representative `report_power -hierarchy` format. Hierarchy is printed indented with only
the leaf instance name and its reference cell in parentheses; the parser rebuilds the
full path from indentation.

    ****************************************
    Report : power -hierarchy
    Design : gpu_a_top
    Version: V-2024.09-SP3
    Date   : 2026-03-02
    Run    : gpu_a_b001_r1234
    Scenario: typical@nom
    Activity: SAIF                     (or: vectorless)
    Power Units = 1W
    ****************************************
                                     Int      Switch   Leak     Total
    Hierarchy                        Power    Power    Power    Power    %
    --------------------------------------------------------------------------
    gpu_a_top                        ...
      u_scheduler (Scheduler)        4.01e-02 3.51e-02 5.22e-03 8.05e-02  4.1
"""

from __future__ import annotations

import re
from pathlib import Path

from powermet.extract.base import (BE_HIER, tool_name, Located, ParsedReport, SourceInputs, convert_unit, find_table_start, locate,
                                   make_records, parse_header, read_lines, to_float, units_from_text)

SOURCE = "primepower"
STAGE = "BE"
TOOL_FAMILY = "Synopsys PrimePower"
SUPPORTED_VERSIONS = ('V-2023.12', 'V-2024.09')      # versions the representative parser was written against
DESCRIPTION = "BE signoff power per physical hierarchy"
DEFAULT_PATTERN = "primepower/{workload}_{operating_point}/power_hier.rpt"
OBJECT_KIND = BE_HIER
METRIC = "be_mw"
_ROW_RE = re.compile(r"^(?P<indent>\s*)(?P<name>\S+)(?:\s+\((?P<ref>[^)]*)\))?\s+(?P<nums>[-+\d.eE\s]+)$")


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    hdr = parse_header(lines, stop_at="Hierarchy")
    unit = units_from_text(hdr.get("power units", "W"), "W")
    start = find_table_start(lines, "Hierarchy")
    rows, refs = [], {}
    stack: list[tuple[int, str]] = []     # (indent, full path)
    for line in lines[start:]:
        if not line.strip() or set(line.strip()) <= set("-="):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        indent = len(m.group("indent"))
        name = m.group("name")
        nums = m.group("nums").split()
        if len(nums) < 4:
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = f"{stack[-1][1]}/{name}" if stack else name
        stack.append((indent, full))
        if not stack[:-1]:
            continue    # top-level total row: not a FUB object
        total = to_float(nums[3])
        val, cu = convert_unit(total, unit, METRIC)
        rows.append({"object": full, "object_kind": OBJECT_KIND, "metric": METRIC,
                     "value": val, "unit": cu, "unit_original": unit})
        if m.group("ref"):
            refs[full] = m.group("ref")
    rec = make_records(rows)
    rec["reference"] = [refs.get(o) for o in rec["object"]] if len(rec) else []
    scenario = hdr.get("scenario", "")
    wl, _, op = scenario.partition("@")
    notes = [f"power converted from {unit} to mW"] if unit != "mW" else []
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "PrimePower"), tool_version=hdr.get("version", "?"),
        records=rec, run_id=hdr.get("run"), report_date=hdr.get("date"),
        workload=context.get("workload") or (wl or None),
        operating_point=context.get("operating_point") or (op or None),
        activity_mode=(hdr.get("activity") or "").lower() or None, notes=notes,
    )
