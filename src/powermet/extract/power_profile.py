"""Time-based power profile adapter: design-level power per time window for one workload x operating point.

Where the hierarchical report gives one averaged number per scenario, the profile says how power moves
inside the workload: the peak window (thermal / IR signoff vector), the peak-to-average ratio, the largest
step between windows (a di/dt proxy) and the energy of the whole run. On GPU and AI workloads the average
hides most of what matters.

Representative CSV, one file per workload x operating point (PrimePower time-based mode with a waveform
interval, or an emulator power profile such as Palladium DPA / ZeBu / Veloce exported per window):

    # Tool: PrimePower  Version: V-2024.09-SP3  Run: gpu_a_b001_r1234  Scenario: compute@nom
    # Power units: mW   Time units: ns   Interval: 100
    t_start_ns,t_end_ns,total_mw,dynamic_mw,leakage_mw
    0,100,5123.4,4850.1,273.3

Records carry `t_start_ns` / `t_end_ns` per row; the pipeline keeps them in `power_profile.parquet`
(design, build, workload, operating_point, window, power) rather than in the FUB dataset, because the
object is the design over time, not a FUB.

EDA requirement: PrimePower `set_power_analysis_mode time_based` with `-waveform_interval` set so the
profile is a table, not a waveform; the FSDB/VCD of the workload; the same netlist and parasitics as the
averaged run so the profile average reconciles with `power_hier.rpt` (the pipeline checks that). For
workloads too long for gate-level simulation, an emulator power profile in the same shape is accepted;
record the tool in the header so provenance says which.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from powermet.extract.base import DESIGN, Located, ParsedReport, SourceInputs, convert_unit, locate, parse_header, read_lines, tool_name

SOURCE = "power_profile"
STAGE = "PROFILE"
OPTIONAL = True
TOOL_FAMILY = "PrimePower time-based mode / emulator power profile (Palladium DPA, ZeBu, Veloce)"
SUPPORTED_VERSIONS = ("csv-1",)
DESCRIPTION = "Design power per time window for one workload x operating point"
DEFAULT_PATTERN = "primepower/{workload}_{operating_point}/power_profile.csv"
OBJECT_KIND = DESIGN
COLUMN_METRIC = {"total_mw": "profile_total_mw", "dynamic_mw": "profile_dynamic_mw", "leakage_mw": "profile_leakage_mw"}
_TIME_FACTOR = {"ps": 1e-3, "ns": 1.0, "us": 1e3, "ms": 1e6}


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


def parse(path: Path, **context) -> ParsedReport:
    lines = read_lines(path)
    comments = [l.lstrip("# ").strip() for l in lines if l.startswith("#")]
    hdr = parse_header(comments)
    unit = "mW"
    m = re.search(r"power units\s*[:=]\s*(\w+)", "\n".join(comments), re.I)
    if m:
        unit = m.group(1)
    tunit = "ns"
    m = re.search(r"time units\s*[:=]\s*(\w+)", "\n".join(comments), re.I)
    if m:
        tunit = m.group(1)
    if tunit not in _TIME_FACTOR:
        raise ValueError(f"{path}: unknown time unit '{tunit}'")
    df = pd.read_csv(path, comment="#")
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol_s = next((c for c in df.columns if c.startswith("t_start")), None)
    tcol_e = next((c for c in df.columns if c.startswith("t_end")), None)
    if tcol_s is None or tcol_e is None:
        raise ValueError(f"{path}: need t_start_* and t_end_* columns")
    rows = []
    for _, r in df.iterrows():
        t0 = float(r[tcol_s]) * _TIME_FACTOR[tunit]
        t1 = float(r[tcol_e]) * _TIME_FACTOR[tunit]
        for col, metric in COLUMN_METRIC.items():
            if col in df.columns and pd.notna(r[col]):
                val, cu = convert_unit(float(r[col]), unit, metric)
                rows.append({"object": "*", "object_kind": OBJECT_KIND, "metric": metric, "value": val, "unit": cu,
                             "unit_original": unit, "t_start_ns": t0, "t_end_ns": t1})
    rec = pd.DataFrame(rows, columns=["object", "object_kind", "metric", "value", "unit", "unit_original", "t_start_ns", "t_end_ns"])
    scenario = hdr.get("scenario", "")
    wl, _, op = scenario.partition("@")
    notes = [f"power converted from {unit} to mW"] if unit != "mW" else []
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=tool_name(hdr, "PrimePower"), tool_version=hdr.get("version", "?"),
        records=rec, run_id=hdr.get("run"), report_date=hdr.get("date"),
        workload=context.get("workload") or (wl or None), operating_point=context.get("operating_point") or (op or None),
        notes=notes,
    )
