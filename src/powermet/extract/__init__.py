"""EDA source adapters. Each module: SOURCE, DEFAULT_PATTERN, OBJECT_KIND, get_files(), parse(),
plus TOOL_FAMILY, SUPPORTED_VERSIONS, DESCRIPTION, STAGE and optional OPTIONAL. `SOURCE_SPECS` is the
declarative view of that contract, built from the modules so nothing is declared twice."""

from __future__ import annotations

from dataclasses import dataclass

from powermet.extract import implementation, metadata, perf, pprtl, primepower, primetime, saif, starrc, voltus

_MODULES = (metadata, pprtl, primepower, primetime, starrc, implementation, saif, perf, voltus)
SOURCES = {m.SOURCE: m for m in _MODULES}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    tool_family: str
    supported_versions: tuple[str, ...]
    description: str
    object_kind: str
    default_pattern: str
    stage: str                   # FE | BE | PHYS | TIMING | ACTIVITY | DESIGN | PERF
    optional: bool
    metrics: tuple[str, ...]


def _spec(m) -> SourceSpec:
    return SourceSpec(m.SOURCE, getattr(m, "TOOL_FAMILY", "?"), tuple(getattr(m, "SUPPORTED_VERSIONS", ())),
                      getattr(m, "DESCRIPTION", ""), m.OBJECT_KIND, m.DEFAULT_PATTERN, getattr(m, "STAGE", "OTHER"),
                      bool(getattr(m, "OPTIONAL", False)), tuple(SOURCE_METRICS.get(m.SOURCE, ())))

# metrics each source is expected to deliver (used by lineage/sanitize to report gaps)
SOURCE_METRICS = {
    "pprtl": ("fe_logical_mw", "fe_physical_mw", "cg_efficiency"),
    "primepower": ("be_mw",),
    "voltus": ("be_voltus_mw",),
    "starrc": ("wire_cap_pf", "cell_cap_pf"),
    "implementation": ("area", "cell_count", "fanout", "wire_length_um", "avg_net_length_um"),
    "saif": ("activity", "bits_per_cycle", "net_count"),
    "primetime": ("clock_period_ps", "wns_ps", "tns_ps", "violating_endpoints"),
    "metadata": ("frequency_ghz", "voltage_v"),
    "perf": ("ipc", "throughput_gops"),
}

# how a metric varies: which keys (besides design/build/object) it is specific to
METRIC_SCOPE = {
    "fe_logical_mw": ("workload", "operating_point"),
    "fe_physical_mw": ("workload", "operating_point"),
    "be_mw": ("workload", "operating_point"),
    "be_voltus_mw": ("workload", "operating_point"),
    "cg_efficiency": ("workload", "operating_point"),
    "activity": ("workload",),
    "bits_per_cycle": ("workload",),
    "net_count": (),
    "wire_length_um": (), "avg_net_length_um": (),
    "clock_period_ps": ("operating_point",), "wns_ps": ("operating_point",), "tns_ps": ("operating_point",),
    "violating_endpoints": ("operating_point",),
    "frequency_ghz": ("operating_point",),
    "voltage_v": ("operating_point",),
    "wire_cap_pf": (), "cell_cap_pf": (), "area": (), "cell_count": (), "fanout": (),
    "ipc": ("workload", "operating_point"),
    "throughput_gops": ("workload", "operating_point"),
}

SOURCE_SPECS: dict[str, SourceSpec] = {m.SOURCE: _spec(m) for m in _MODULES}
STAGE_OF_SOURCE = {name: sp.stage for name, sp in SOURCE_SPECS.items()}

PERF_METRICS = tuple(SOURCE_METRICS["perf"])
DESIGN_LEVEL_METRICS = tuple(SOURCE_METRICS["metadata"])


def metrics_with_scope(*keys: str) -> tuple[str, ...]:
    """Metrics whose scope is exactly `keys` (e.g. () for build-level, ("workload",) for workload-level)."""
    want = tuple(keys)
    return tuple(m for m, scope in METRIC_SCOPE.items() if tuple(scope) == want)


__all__ = ["SOURCES", "SOURCE_SPECS", "SourceSpec", "STAGE_OF_SOURCE", "SOURCE_METRICS", "METRIC_SCOPE", "PERF_METRICS",
           "DESIGN_LEVEL_METRICS", "metrics_with_scope"]
