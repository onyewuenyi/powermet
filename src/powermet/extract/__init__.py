"""EDA source adapters. Each module: SOURCE, DEFAULT_PATTERN, OBJECT_KIND, get_files(), parse()."""

from __future__ import annotations

from powermet.extract import activity, implementation, metadata, perf, pprtl, primepower, primetime, starrc

SOURCES = {m.SOURCE: m for m in (metadata, pprtl, primepower, primetime, starrc, implementation, activity, perf)}

# metrics each source is expected to deliver (used by lineage/sanitize to report gaps)
SOURCE_METRICS = {
    "pprtl": ("fe_logical_mw", "fe_physical_mw"),
    "primepower": ("be_mw",),
    "starrc": ("wire_cap_pf", "cell_cap_pf"),
    "implementation": ("area", "cell_count", "fanout", "wire_length_um", "avg_net_length_um"),
    "activity": ("activity", "bits_per_cycle"),
    "primetime": ("clock_period_ps", "wns_ps", "tns_ps", "violating_endpoints"),
    "metadata": ("frequency_ghz", "voltage_v"),
    "perf": ("ipc", "throughput_gops"),
}

# how a metric varies: which keys (besides design/build/object) it is specific to
METRIC_SCOPE = {
    "fe_logical_mw": ("workload", "operating_point"),
    "fe_physical_mw": ("workload", "operating_point"),
    "be_mw": ("workload", "operating_point"),
    "activity": ("workload",),
    "bits_per_cycle": ("workload",),
    "wire_length_um": (), "avg_net_length_um": (),
    "clock_period_ps": ("operating_point",), "wns_ps": ("operating_point",), "tns_ps": ("operating_point",),
    "violating_endpoints": ("operating_point",),
    "frequency_ghz": ("operating_point",),
    "voltage_v": ("operating_point",),
    "wire_cap_pf": (), "cell_cap_pf": (), "area": (), "cell_count": (), "fanout": (),
    "ipc": ("workload", "operating_point"),
    "throughput_gops": ("workload", "operating_point"),
}

PERF_METRICS = tuple(SOURCE_METRICS["perf"])
DESIGN_LEVEL_METRICS = tuple(SOURCE_METRICS["metadata"])


def metrics_with_scope(*keys: str) -> tuple[str, ...]:
    """Metrics whose scope is exactly `keys` (e.g. () for build-level, ("workload",) for workload-level)."""
    want = tuple(keys)
    return tuple(m for m, scope in METRIC_SCOPE.items() if tuple(scope) == want)


__all__ = ["SOURCES", "SOURCE_METRICS", "METRIC_SCOPE", "PERF_METRICS", "DESIGN_LEVEL_METRICS", "metrics_with_scope"]
