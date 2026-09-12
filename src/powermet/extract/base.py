"""Shared contract for EDA source adapters.

Every source module exposes:

    SOURCE            short name used in provenance ("primepower", "starrc", ...)
    DEFAULT_PATTERN   relative path pattern under the run directory; may contain
                      {workload} and {operating_point} placeholders and glob wildcards
    get_files(inputs) -> list[Located]      locate the file(s) for a run   <- change here when the
                                                                             real data location is known
    parse(path, **context) -> ParsedReport  parse one file into long-format records

Records are *long*: one row per (object, metric) with the value in canonical units.
The pipeline maps objects to FUBs via the model root (identity.ModelRoot) and pivots to the wide schema.

Contract details every adapter must follow:
  * `object_kind` is one of OBJECT_KINDS (FE_HIER, BE_HIER, PARTITION, DESIGN); lineage resolves
    objects to FUBs by kind, so a new kind needs a resolver in identity.ModelRoot.resolve().
  * Units: convert with `convert_unit(value, unit, metric)`; keep the original unit in `unit_original`.
  * `tool`: prefer the report header value, fall back to the adapter's fixed name (`tool_name(hdr, default)`).
  * `workload` / `operating_point`: normally report-level fields on ParsedReport (from the file's
    context or header). An adapter whose single file spans several operating points / workloads may
    instead put a per-record `operating_point` / `workload` column on `records`; the pipeline only
    fills the report-level value where the column is absent (see metadata.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ["object", "object_kind", "metric", "value", "unit", "unit_original"]

# object kinds (what a report's object names refer to)
FE_HIER = "fe_hier"
BE_HIER = "be_hier"
PARTITION = "partition"
DESIGN = "design"
OBJECT_KINDS = (FE_HIER, BE_HIER, PARTITION, DESIGN)

# canonical units per metric
CANONICAL_UNITS = {
    "be_mw": "mW", "fe_logical_mw": "mW", "fe_physical_mw": "mW",
    "wire_cap_pf": "pF", "cell_cap_pf": "pF",
    "area": "um2", "cell_count": "count", "fanout": "count",
    "frequency_ghz": "GHz", "voltage_v": "V", "activity": "ratio",
    "ipc": "ops/cycle", "throughput_gops": "Gops/s",
    "wire_length_um": "um", "avg_net_length_um": "um", "bits_per_cycle": "bits", "net_count": "count",
    "be_voltus_mw": "mW", "cg_efficiency": "ratio", "be_leakage_mw": "mW", "fe_leakage_mw": "mW",
    "clock_period_ps": "ps", "wns_ps": "ps", "tns_ps": "ps", "violating_endpoints": "count",
}

# multiplicative factors to canonical: (from_unit, to_unit) -> factor
UNIT_FACTORS = {
    ("W", "mW"): 1e3, ("mW", "mW"): 1.0, ("uW", "mW"): 1e-3, ("nW", "mW"): 1e-6,
    ("F", "pF"): 1e12, ("pF", "pF"): 1.0, ("fF", "pF"): 1e-3, ("nF", "pF"): 1e3,
    ("um2", "um2"): 1.0, ("mm2", "um2"): 1e6, ("nm2", "um2"): 1e-6,
    ("Hz", "GHz"): 1e-9, ("kHz", "GHz"): 1e-6, ("MHz", "GHz"): 1e-3, ("GHz", "GHz"): 1.0,
    ("V", "V"): 1.0, ("mV", "V"): 1e-3,
    ("count", "count"): 1.0, ("ratio", "ratio"): 1.0, ("%", "ratio"): 0.01,
    ("ops/cycle", "ops/cycle"): 1.0, ("Gops/s", "Gops/s"): 1.0, ("Mops/s", "Gops/s"): 1e-3,
    ("um", "um"): 1.0, ("mm", "um"): 1e3, ("nm", "um"): 1e-3, ("bits", "bits"): 1.0,
    ("ps", "ps"): 1.0, ("ns", "ps"): 1e3, ("us", "ps"): 1e6,
}


class ParseError(ValueError):
    pass


@dataclass
class Located:
    path: Path
    context: dict = field(default_factory=dict)   # e.g. {"workload": "typical", "operating_point": "nom"}


@dataclass
class ParsedReport:
    source: str
    path: Path
    tool: str
    tool_version: str
    records: pd.DataFrame
    run_id: str | None = None
    report_date: str | None = None
    build: str | None = None           # build named in the report header, when the tool prints one
    activity_mode: str | None = None   # "saif" / "fsdb" (vector-based) or "vectorless" when the tool states it
    workload: str | None = None
    operating_point: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.records)


@dataclass
class SourceInputs:
    """The common inputs every adapter receives. Extend when real sources need more (NFS root, DB, ...)."""

    design: str
    build: str
    run_dir: Path
    workloads: list[str] = field(default_factory=list)
    operating_points: list[str] = field(default_factory=list)
    patterns: dict[str, str] = field(default_factory=dict)   # source -> pattern override
    context: dict = field(default_factory=dict)              # extra parse context from metadata (e.g. sim clock period)

    def pattern_for(self, source: str, default: str) -> str:
        return self.patterns.get(source, default)


def convert_unit(value: float, unit: str, metric: str) -> tuple[float, str]:
    """Convert value in `unit` to the canonical unit of `metric`. Raises on unknown units."""
    target = CANONICAL_UNITS.get(metric)
    if target is None:
        return value, unit
    key = (unit, target)
    if key not in UNIT_FACTORS:
        raise ParseError(f"cannot convert unit '{unit}' to '{target}' for metric {metric}")
    return value * UNIT_FACTORS[key], target


def tool_name(hdr: dict, default: str) -> str:
    """Adapter policy for the `tool` field: header value when present, else the adapter's fixed name."""
    return hdr.get("tool") or default


def record(obj: str, kind: str, metric: str, value: float, unit: str, unit_original: str | None = None) -> dict:
    """One long record; `unit` is canonical, `unit_original` defaults to `unit`."""
    if kind not in OBJECT_KINDS:
        raise ValueError(f"unknown object_kind '{kind}'")
    return {"object": obj, "object_kind": kind, "metric": metric, "value": value,
            "unit": unit, "unit_original": unit if unit_original is None else unit_original}


def make_records(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    df = pd.DataFrame(rows)
    for c in RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[RECORD_COLUMNS]


def locate(inputs: SourceInputs, source: str, default_pattern: str) -> list[Located]:
    """Expand {workload}/{operating_point} placeholders and glob under the run directory.

    This is the single place to change when real source data lives somewhere else
    (a different tree, an NFS share, a database dump): replace the glob with whatever
    fetch is needed and return Located(path, context) entries.
    """
    pattern = inputs.pattern_for(source, default_pattern)
    combos: list[dict] = []
    wls = inputs.workloads or [None]
    ops = inputs.operating_points or [None]
    need_wl = "{workload}" in pattern
    need_op = "{operating_point}" in pattern
    for wl in (wls if need_wl else [None]):
        for op in (ops if need_op else [None]):
            ctx = dict(inputs.context)
            if need_wl:
                ctx["workload"] = wl
            if need_op:
                ctx["operating_point"] = op
            combos.append(ctx)
    out: list[Located] = []
    for ctx in combos:
        rel = pattern.format(**{k: (v or "*") for k, v in ctx.items()})
        for p in sorted(inputs.run_dir.glob(rel)):
            if p.is_file():
                out.append(Located(p, dict(ctx)))
    return out


# ----------------------------------------------------------------------------- text helpers

_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/-]*?)\s*[:=]\s*(.+?)\s*$")


def read_lines(path: Path) -> list[str]:
    return Path(path).read_text(errors="replace").splitlines()


def parse_header(lines: list[str], stop_at: str | None = None) -> dict[str, str]:
    """Collect 'Key: Value' / 'Key = Value' pairs from the report preamble."""
    hdr: dict[str, str] = {}
    for line in lines:
        if stop_at and stop_at in line:
            break
        if set(line.strip()) <= set("*-=#"):
            continue
        # several "Key: Value" pairs may share a line separated by 2+ spaces; split only where a
        # new key starts, so aligned "Run    : r1" stays one pair
        for seg in re.split(r"\s{2,}(?=[A-Za-z][A-Za-z0-9 _/-]{0,39}\s*[:=]\s*\S)", line.strip()):
            m = _HEADER_RE.match(seg)
            if m and len(m.group(1)) <= 40:
                hdr[m.group(1).strip().lower()] = m.group(2).strip()
    return hdr


def find_table_start(lines: list[str], first_col: str) -> int:
    """Index of the first data line after a column-header line starting with `first_col`."""
    for i, line in enumerate(lines):
        if line.strip().startswith(first_col):
            j = i + 1
            while j < len(lines) and set(lines[j].strip()) <= set("-=") and lines[j].strip():
                j += 1
            return j
    raise ParseError(f"table header starting with '{first_col}' not found")


def to_float(tok: str) -> float:
    tok = tok.replace(",", "")
    if tok in ("-", "--", "n/a", "N/A", ""):
        return float("nan")
    return float(tok)


def units_from_text(text: str, default: str) -> str:
    """'Power Units = 1mW' / 'units: fF' / '(W)' -> unit token."""
    m = re.search(r"\b1?\s*(nW|uW|mW|W|fF|pF|nF|F|um\^?2|mm\^?2|MHz|GHz|kHz|Hz|mV|V)\b", text)
    if not m:
        return default
    return m.group(1).replace("^", "")
