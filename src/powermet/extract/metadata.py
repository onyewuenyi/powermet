"""Build metadata source: metadata.json in the run directory.

Provides design/build identity, dates, tool versions, workloads, operating-point
voltage/frequency (as design-level records that apply to every FUB), and the activity-flow
provenance block (`activity_flow`: source FSDB, core, mapping data, partition list, tool,
output hierarchy, simulation clock period) that the SAIF adapter needs.

One file spans all operating points, so this adapter sets a per-record `operating_point`
column on `records` instead of the report-level ParsedReport.operating_point (see base.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from powermet.extract.base import DESIGN, tool_name, Located, ParsedReport, SourceInputs, locate, make_records

SOURCE = "metadata"
STAGE = "DESIGN"
TOOL_FAMILY = "Run metadata (metadata.json)"
SUPPORTED_VERSIONS = ('1',)      # versions the representative parser was written against
DESCRIPTION = "Identity, dates, tool versions, operating points, activity-flow provenance"
DEFAULT_PATTERN = "metadata.json"
OBJECT_KIND = DESIGN


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def parse(path: Path, **context) -> ParsedReport:
    meta = load(path)
    rows = []
    for op, vals in (meta.get("operating_points") or {}).items():
        for metric, unit in (("frequency_ghz", "GHz"), ("voltage_v", "V")):
            if metric in vals:
                rows.append({"object": "*", "object_kind": OBJECT_KIND, "metric": metric,
                             "value": float(vals[metric]), "unit": unit, "unit_original": unit,
                             "operating_point": op})
    rec = make_records(rows)
    if len(rec):
        rec["operating_point"] = [r["operating_point"] for r in rows]
    return ParsedReport(
        source=SOURCE, path=Path(path), tool="metadata", tool_version=str(meta.get("schema_version", "1")),
        records=rec, run_id=meta.get("run_id"), report_date=meta.get("build_date"),
        notes=[f"status={meta.get('status', 'current')}"],
    )


def inputs_from_metadata(run_dir: Path, patterns: dict[str, str] | None = None) -> tuple[SourceInputs, dict]:
    path = Path(run_dir) / DEFAULT_PATTERN
    if not path.exists():
        raise FileNotFoundError(f"no metadata.json in {run_dir}")
    meta = load(path)
    ops = meta.get("operating_points") or {}
    flow = dict(meta.get("activity_flow") or {})
    sim_period = flow.get("sim_clock_period_ps")
    if not sim_period:
        nom = ops.get("nom") or (next(iter(ops.values())) if ops else {})
        if nom.get("frequency_ghz"):
            sim_period = 1000.0 / float(nom["frequency_ghz"])
    meta.setdefault("design_type", "unspecified")     # cpu | gpu | asic | ai_accelerator | soc
    inputs = SourceInputs(
        design=str(meta["design"]), build=str(meta["build"]), run_dir=Path(run_dir),
        workloads=list(meta.get("workloads") or []),
        operating_points=list(ops.keys()),
        patterns=dict(patterns or {}),
        context={"sim_clock_period_ps": sim_period, "activity_hierarchy": flow.get("hierarchy", "be"), "activity_flow": flow},
    )
    return inputs, meta
