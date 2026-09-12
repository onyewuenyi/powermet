"""Lineage: model root -> FUB -> FE hierarchy -> synthesis object -> BE hierarchy -> partition -> measurement.

`ModelRoot` (identity.py) owns the FUB list and object-name resolution; this module applies it to parsed
records, records where each metric came from, and flags mismatches between what the map expects and
what the reports contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from powermet.extract.base import BE_HIER, DESIGN, FE_HIER, PARTITION
from powermet.identity import ModelRoot

LINEAGE_LEVELS = ("model_root", "fub", "fe_hier", "synth_object", "be_hier", "partition", "physical_instances", "measurement")
DESIGN_FUB = "*"      # sentinel fub value for design-level records

# lineage issue codes
FE_NOT_IN_REPORTS = "fe_hier_not_in_reports"
BE_NOT_IN_REPORTS = "be_hier_not_in_reports"
INCOMPLETE_PHYSICAL = "incomplete_physical_mapping"
NO_PARTITION = "no_partition"
TIMING_MISSING = "partition_timing_missing"
BLOCKING_ISSUES = (FE_NOT_IN_REPORTS, BE_NOT_IN_REPORTS, INCOMPLETE_PHYSICAL)
TIMING_ISSUES = (NO_PARTITION, TIMING_MISSING)

FE_METRICS = ("fe_logical_mw", "fe_physical_mw")
PHYSICAL_REQUIRED = ("wire_cap_pf", "cell_cap_pf", "area")


@dataclass
class LineageResult:
    mapped: pd.DataFrame          # long records + fub column (resolved)
    unmapped: pd.DataFrame        # records whose object is not in the map
    lineage: pd.DataFrame         # one row per fub: chain + per-metric source summary + flags
    flags: list[str] = field(default_factory=list)


def load_fub_map(path: str | Path) -> pd.DataFrame:
    """Backwards-compatible frame view of the FUB map (validated through ModelRoot)."""
    return ModelRoot.load(path, design="").to_frame()


def resolve_objects(records: pd.DataFrame, model: ModelRoot) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach `fub` to each record via the model root. Partition records fan out to member FUBs.

    Returns (mapped, unmapped). Design-level records get fub = DESIGN_FUB. Hierarchy rows that sit
    above mapped FUBs (partition / design aggregates in a BE report) are neither: they are dropped
    as aggregates so they do not show up as unmapped objects.
    """
    rec = records.copy()
    fubs: list[list[str]] = []
    weights: list[list[float]] = []
    instances: list[list[str | None]] = []
    aggregate: list[bool] = []
    for obj, kind in zip(rec["object"].astype(str), rec["object_kind"].astype(str)):
        agg = False
        if kind == DESIGN:
            hit, w, inst = [DESIGN_FUB], [1.0], [None]
        elif kind in (FE_HIER, BE_HIER, PARTITION):
            ms = model.resolve(obj, kind)
            hit = [m.spec.fub + (f"@{m.instance}" if m.instance else "") for m in ms]
            w = [m.weight for m in ms]
            inst = [m.instance for m in ms]
            agg = not hit and kind in (FE_HIER, BE_HIER) and model.is_ancestor(obj)
        else:
            hit, w, inst = [], [], []
        fubs.append(hit)
        weights.append(w)
        instances.append(inst)
        aggregate.append(agg)
    rec["fub"] = fubs
    rec["weight"] = weights
    rec["instance"] = instances
    rec["_aggregate"] = aggregate
    n = rec["fub"].str.len()
    unmapped = rec[(n == 0) & ~rec["_aggregate"]].drop(columns=["fub", "weight", "instance", "_aggregate"]).assign(fub=None)
    mapped = rec[n > 0].drop(columns=["_aggregate"]).explode(["fub", "weight", "instance"]).reset_index(drop=True)
    mapped["weight"] = mapped["weight"].astype(float)
    return mapped, unmapped


def lineage_table(mapped: pd.DataFrame, model: ModelRoot, design: str, build: str,
                  has_timing_source: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """One row per FUB: chain, metrics present, source per metric, issue flags."""
    fub_rec = mapped[mapped["fub"] != DESIGN_FUB] if len(mapped) else mapped
    pres = fub_rec.groupby(["fub", "metric"]).size().unstack(fill_value=0) if len(fub_rec) else pd.DataFrame()
    src_of = (fub_rec.groupby(["fub", "metric"])["source"].first().unstack()
              if "source" in fub_rec.columns and len(fub_rec) else pd.DataFrame())
    inst_count = fub_rec[fub_rec["metric"] == "cell_count"].groupby("fub")["value"].sum() if len(fub_rec) else pd.Series(dtype=float)
    rows, flags = [], []
    rel = model.relationships()
    specs = []
    for spec in model:
        insts = sorted(model.instances.get(spec.fub, ()))
        if insts:
            for i in insts:
                specs.append((f"{spec.fub}@{i}", spec, i))
        else:
            specs.append((spec.fub, spec, None))
    for f, spec, inst in specs:
        have = set(pres.columns[pres.loc[f] > 0]) if len(pres) and f in pres.index else set()
        issues = []
        if not (set(FE_METRICS) & have):
            issues.append(FE_NOT_IN_REPORTS)
        if "be_mw" not in have:
            issues.append(BE_NOT_IN_REPORTS)
        if not set(PHYSICAL_REQUIRED) <= have:
            issues.append(INCOMPLETE_PHYSICAL)
        if spec.partition is None:
            issues.append(NO_PARTITION)
        elif has_timing_source and "wns_ps" not in have:
            issues.append(TIMING_MISSING)
        row = spec.as_row()
        row["fub"] = f
        if inst is not None:
            row["model_root"] = f"{spec.model_root}@{inst}"
        row["relationship"] = ";".join(k for k, v in rel.items() if spec.fub in v) or ("replicated" if inst else "one-to-one")
        rows.append({
            "design": design, "build": build, **row,
            "physical_instances": float(inst_count.get(f, float("nan"))),
            "metrics_present": ",".join(sorted(have)),
            "sources": ",".join(f"{k}:{v}" for k, v in (src_of.loc[f].dropna().items() if len(src_of) and f in src_of.index else [])),
            "lineage_ok": not issues, "lineage_issues": ";".join(issues),
        })
        flags.extend(f"{f}: {i}" for i in issues)
    cols = ["design", "build", "fub", "model_root", "partition", "fe_hier", "synth_object", "be_hier", "relationship",
            "physical_instances", "metrics_present", "sources", "lineage_ok", "lineage_issues"]
    return pd.DataFrame(rows, columns=cols), flags


def resolve(records: pd.DataFrame, model: ModelRoot | pd.DataFrame, design: str, build: str,
            sources_present: set[str] | None = None) -> LineageResult:
    """Full lineage pass: resolve objects, build the lineage table, collect flags."""
    if isinstance(model, pd.DataFrame):
        model = ModelRoot.from_frame(model, design)
    mapped, unmapped = resolve_objects(records, model)
    has_timing = bool((records["object_kind"] == PARTITION).any()) or ("primetime" in (sources_present or set()))
    table, flags = lineage_table(mapped, model, design, build, has_timing)
    n_un = int((unmapped["object_kind"] != DESIGN).sum()) if len(unmapped) else 0
    if n_un:
        objs = sorted(unmapped["object"].astype(str).unique())[:5]
        flags.append(f"{n_un} report records reference objects not in the FUB map (e.g. {', '.join(objs)})")
    return LineageResult(mapped, unmapped, table, flags)


def render_chain(row: pd.Series, measurements: pd.DataFrame | None = None) -> str:
    """Text rendering of one FUB's lineage chain for `powermet lineage show`."""
    inst = row.get("physical_instances")
    inst_s = f"{int(inst):,} cells" if pd.notna(inst) else "unknown"
    lines = [
        f"MODEL ROOT          {row.get('model_root', row['fub'])}   ({row['design']} / {row['build']})",
        f"  |-- FUB               {row['fub']}",
        f"  |-- FE hierarchy      {row['fe_hier']}",
        f"  |-- synthesis object  {row['synth_object']}",
        f"  |-- BE hierarchy      {row['be_hier']}",
        f"  |-- partition         {row.get('partition') or 'unknown'}   (timing is a partition attribute)",
        f"  |-- relationship      {row.get('relationship', 'one-to-one')}",
        f"  |-- physical instances {inst_s}",
    ]
    if row.get("sources"):
        lines.append("  '-- measurements")
        for item in str(row["sources"]).split(","):
            metric, _, src = item.partition(":")
            lines.append(f"        {metric:<16} <- {src}")
    if not row.get("lineage_ok", True):
        lines.append(f"  LINEAGE ISSUES: {row['lineage_issues']}")
    if measurements is not None and len(measurements):
        lines.append("")
        cols = [c for c in ("workload", "operating_point", "fe_logical_mw", "fe_physical_mw", "be_mw", "wns_ps", "fmax_ghz", "run_id") if c in measurements.columns]
        lines.append(measurements[cols].to_string(index=False))
    return "\n".join(lines)
