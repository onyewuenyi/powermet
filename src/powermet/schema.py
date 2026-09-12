"""Canonical FUB-level measurement schema.

The schema is a plain table of ColumnSpec so it can be extended without
touching pipeline code. Unknown extra columns are always passed through.
"""

from __future__ import annotations

from dataclasses import dataclass

IDENTITY = "identity"
POWER = "power"
FEATURE = "feature"
TIMING = "timing"
PROVENANCE = "provenance"
DERIVED = "derived"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str            # "str" | "float"
    role: str
    required: bool = False
    non_negative: bool = False
    description: str = ""


COLUMNS: tuple[ColumnSpec, ...] = (
    # identity
    ColumnSpec("design", "str", IDENTITY, required=True, description="Design / product name"),
    ColumnSpec("build", "str", IDENTITY, required=True, description="Build or drop identifier"),
    ColumnSpec("fub", "str", IDENTITY, required=True, description="Functional unit block (analytical dimension)"),
    ColumnSpec("stage", "str", IDENTITY, required=True, description="Row kind; the pipeline always writes FE_BE (paired FE/BE row). Reserved for other pairings."),
    ColumnSpec("workload", "str", IDENTITY, description="Workload / vector name"),
    ColumnSpec("operating_point", "str", IDENTITY, description="Voltage/frequency corner"),
    ColumnSpec("model_root", "str", IDENTITY, description="Canonical FUB identity across FE/BE (e.g. GPU_A.PCORE0.Scheduler)"),
    ColumnSpec("partition", "str", IDENTITY, description="Physical partition (BE block) the FUB is implemented in"),
    # power
    ColumnSpec("fe_logical_mw", "float", POWER, required=True, non_negative=True, description="FE logical power estimate (mW)"),
    ColumnSpec("fe_physical_mw", "float", POWER, required=True, non_negative=True, description="FE physically-aware power estimate (mW)"),
    ColumnSpec("be_mw", "float", POWER, required=True, non_negative=True, description="BE signoff power (mW)"),
    ColumnSpec("fe_leakage_mw", "float", POWER, non_negative=True, description="FE physically-aware leakage power estimate (mW)"),
    ColumnSpec("be_leakage_mw", "float", POWER, non_negative=True, description="BE signoff leakage power (mW); BE dynamic = be_mw - be_leakage_mw"),
    ColumnSpec("be_voltus_mw", "float", FEATURE, non_negative=True, description="BE power from the alternate signoff engine (mW), for qualification"),
    ColumnSpec("cg_efficiency", "float", FEATURE, non_negative=True, description="Clock-gating efficiency (fraction of register clock pins gated)"),
    # physical features
    ColumnSpec("wire_cap_pf", "float", FEATURE, non_negative=True, description="Total wire capacitance (pF)"),
    ColumnSpec("cell_cap_pf", "float", FEATURE, non_negative=True, description="Total cell/pin capacitance (pF)"),
    ColumnSpec("area", "float", FEATURE, non_negative=True, description="Placed area (um^2 or design units)"),
    ColumnSpec("cell_count", "float", FEATURE, non_negative=True, description="Standard cell count"),
    ColumnSpec("fanout", "float", FEATURE, non_negative=True, description="Average net fanout"),
    ColumnSpec("frequency_ghz", "float", FEATURE, non_negative=True, description="Clock frequency (GHz)"),
    ColumnSpec("voltage_v", "float", FEATURE, non_negative=True, description="Supply voltage (V)"),
    ColumnSpec("activity", "float", FEATURE, non_negative=True, description="Toggles per cycle per net from SAIF (activity factor)"),
    ColumnSpec("net_count", "float", FEATURE, non_negative=True, description="Nets with activity data in the SAIF instance"),
    ColumnSpec("wire_length_um", "float", FEATURE, non_negative=True, description="Total routed wire length (um)"),
    ColumnSpec("avg_net_length_um", "float", FEATURE, non_negative=True, description="Average net length (um) - data-movement distance proxy"),
    ColumnSpec("bits_per_cycle", "float", FEATURE, non_negative=True, description="Bits switched per cycle = sum of net toggles / cycles (data-movement traffic proxy)"),
    # timing (partition-level, inherited by every FUB in the partition)
    ColumnSpec("clock_period_ps", "float", TIMING, non_negative=True, description="Target clock period (ps)"),
    ColumnSpec("wns_ps", "float", TIMING, description="Worst negative slack of the partition (ps; negative = violating)"),
    ColumnSpec("tns_ps", "float", TIMING, description="Total negative slack of the partition (ps)"),
    ColumnSpec("violating_endpoints", "float", TIMING, non_negative=True, description="Number of violating endpoints"),
    ColumnSpec("fmax_ghz", "float", TIMING, non_negative=True, description="Achievable frequency = 1000 / (period - WNS) (GHz)"),
    # provenance (added on import; may be present in source)
    ColumnSpec("source_file", "str", PROVENANCE, description="File the row was imported from"),
    ColumnSpec("tool", "str", PROVENANCE, description="Tool that produced the measurement"),
    ColumnSpec("tool_version", "str", PROVENANCE, description="Tool version"),
    ColumnSpec("imported_at", "str", PROVENANCE, description="Import timestamp (ISO 8601)"),
    ColumnSpec("run_id", "str", PROVENANCE, description="EDA run identifier"),
    ColumnSpec("build_date", "str", PROVENANCE, description="Build date (YYYY-MM-DD)"),
    ColumnSpec("design_type", "str", PROVENANCE, description="cpu | gpu | asic | ai_accelerator | soc (from metadata.json)"),
    ColumnSpec("milestone", "str", PROVENANCE, description="Design milestone of the build (rtl, synthesis, placement, route, signoff)"),
    ColumnSpec("be_activity_mode", "str", PROVENANCE, description="Activity source of the BE power number: saif/fsdb (vector-based) or vectorless"),
    ColumnSpec("power_domain", "str", PROVENANCE, description="UPF power domain the FUB belongs to"),
)

DERIVED_COLUMNS: tuple[str, ...] = (
    "be_dynamic_mw",
    "cdyn_pf",
    "fe_cdyn_pf",
    "leakage_fraction",
    "logical_error_mw",
    "logical_error_pct",
    "physical_error_mw",
    "physical_error_pct",
    "logical_to_be_ratio",
    "physical_to_be_ratio",
)

BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in COLUMNS}

REQUIRED_COLUMNS = tuple(c.name for c in COLUMNS if c.required)
REQUIRED_POWER_COLUMNS = tuple(c.name for c in COLUMNS if c.required and c.role == POWER)
FE_ESTIMATE_COLUMNS = ("fe_logical_mw", "fe_physical_mw")
# Power-convergence metrics: what a design signs up to hit. Cdyn (effective switched capacitance, pF) is
# dynamic power with V^2 f divided out, so it is the design-owned quantity that is comparable across corners,
# builds and FE/BE; leakage is tracked separately because its levers (Vt, gating, area) and its corner
# sensitivity (V^3, temperature) differ from the dynamic ones.
CONVERGENCE_METRICS: dict[str, tuple[str, str, str]] = {          # metric -> (short name, unit, component)
    "cdyn_pf": ("CdynTot", "pF", "dynamic"),
    "be_leakage_mw": ("LkgPwr", "mW", "leakage"),
    "be_dynamic_mw": ("DynPwr", "mW", "dynamic"),
    "be_mw": ("TotPwr", "mW", "total"),
}
METRIC_UNITS: dict[str, str] = {"be_mw": "mW", "be_dynamic_mw": "mW", "be_leakage_mw": "mW", "fe_leakage_mw": "mW",
                                "fe_logical_mw": "mW", "fe_physical_mw": "mW", "cdyn_pf": "pF", "fe_cdyn_pf": "pF",
                                "wire_cap_pf": "pF", "cell_cap_pf": "pF"}
IDENTITY_COLUMNS = tuple(c.name for c in COLUMNS if c.role == IDENTITY)
POWER_COLUMNS = tuple(c.name for c in COLUMNS if c.role == POWER)
FEATURE_COLUMNS = tuple(c.name for c in COLUMNS if c.role == FEATURE)
TIMING_COLUMNS = tuple(c.name for c in COLUMNS if c.role == TIMING)
PROVENANCE_COLUMNS = tuple(c.name for c in COLUMNS if c.role == PROVENANCE)
NUMERIC_COLUMNS = tuple(c.name for c in COLUMNS if c.dtype == "float")
NON_NEGATIVE_COLUMNS = tuple(c.name for c in COLUMNS if c.non_negative)

# Columns that identify a unique measurement row.
KEY_COLUMNS = ("design", "build", "fub", "stage", "workload", "operating_point")

TARGET = "be_mw"
PAIRED_STAGE = "FE_BE"
# Metrics the ingest pipeline pivots into the wide FUB dataset (fmax_ghz is derived, not ingested).
FUB_METRICS = tuple(c for c in POWER_COLUMNS + FEATURE_COLUMNS + TIMING_COLUMNS if c != "fmax_ghz")


def identity_key(df) -> str:
    """Column that identifies a FUB in a frame: model_root when present, else fub."""
    return "model_root" if "model_root" in df.columns else "fub"

# Human labels for CLI/report output.
LABELS: dict[str, str] = {
    "fe_logical_mw": "FE Logical Power",
    "fe_physical_mw": "FE Physical Power",
    "be_mw": "BE Power",
    "be_leakage_mw": "BE Leakage",
    "be_dynamic_mw": "BE Dynamic Power",
    "fe_leakage_mw": "FE Leakage",
    "cdyn_pf": "CdynTot",
    "fe_cdyn_pf": "FE CdynTot",
    "leakage_fraction": "Leakage Fraction",
    "wire_cap_pf": "Wire Cap",
    "cell_cap_pf": "Cell Cap",
    "area": "Area",
    "cell_count": "Cell Count",
    "fanout": "Fanout",
    "frequency_ghz": "Frequency",
    "wire_cap_fraction": "Wire Cap Fraction",
    "voltage_v": "Voltage",
    "activity": "Activity",
    "throughput_gops": "Throughput",
    "total_cap_pf": "Total Cap",
    "wire_length_um": "Wire Length",
    "avg_net_length_um": "Avg Net Length",
    "bits_per_cycle": "Bits switched / cycle",
    "net_count": "Net Count",
    "be_voltus_mw": "BE Power (Voltus)",
    "cg_efficiency": "Clock-gating Efficiency",
    "power_density": "Power Density",
    "clock_period_ps": "Clock Period",
    "wns_ps": "WNS",
    "tns_ps": "TNS",
    "violating_endpoints": "Violating Endpoints",
    "fmax_ghz": "Fmax",
    "cell_dyn_term": "Cell switching term",
    "wire_dyn_term": "Wire switching term",
    "move_term": "Data-movement term",
    "compute_mw": "Compute power",
    "wire_mw": "Wire switching power",
    "movement_mw": "Data-movement power",
    "leakage_mw": "Leakage power",
    "dyn_term": "CV^2f term",
    "leak_term": "Leakage term",
    "energy_pj_per_op": "Energy / op",
    "logical_error_mw": "FE Logical -> BE Error (mW)",
    "logical_error_pct": "FE Logical -> BE Error (%)",
    "physical_error_mw": "FE Physical -> BE Error (mW)",
    "physical_error_pct": "FE Physical -> BE Error (%)",
}


def label(col: str) -> str:
    return LABELS.get(col, col)
