"""Ingest pipeline: run directory -> extraction -> normalization -> lineage -> FUB dataset -> store.

Every stage is profiled. Outputs under .powermet/data/processed/:
    measurements.parquet      wide FUB-level dataset (one row per design/build/fub/workload/op)
    measurements_long.parquet long records with full provenance (source, file, tool, version, run_id, unit)
    lineage.parquet           FUB lineage chain per design/build with mismatch flags
    unmapped.parquet          report objects that could not be mapped to a FUB
    performance.parquet       design-level performance per build/workload/op (V3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from powermet.config import Config, Project
from powermet.extract import DESIGN_LEVEL_METRICS, METRIC_SCOPE, PERF_METRICS, SOURCES
from powermet.extract.base import ParseError, ParsedReport
from powermet.extract.metadata import inputs_from_metadata
from powermet.identity import ModelRoot
from powermet.ingest import file_sha256, write_table
from powermet.lineage import DESIGN_FUB, resolve
from powermet.metrics import add_derived_metrics, fmax_from_timing
from powermet.profiling import Profiler
from powermet.schema import FUB_METRICS, KEY_COLUMNS, PAIRED_STAGE, POWER_COLUMNS
from powermet.storage import add_analysis_features, load_dataset, register_views
from powermet.validation import validate

PROV_COLS = ["source", "source_file", "tool", "tool_version", "run_id", "report_date"]
DEFAULT_SCOPE_VALUE = "default"      # workload / operating_point value for metrics that do not vary by them


@dataclass
class RunResult:
    design: str
    build: str
    run_dir: Path
    reports: list[ParsedReport]
    long: pd.DataFrame
    wide: pd.DataFrame
    perf: pd.DataFrame
    lineage: pd.DataFrame
    unmapped: pd.DataFrame
    flags: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "current"


@dataclass
class IngestSummary:
    runs: list[RunResult]
    dataset_path: Path
    profiler: Profiler
    n_rows: int
    n_rejected: int


# ----------------------------------------------------------------------------- extraction

def verify_run_consistency(reports: list[ParsedReport], meta: dict) -> list[str]:
    """Every report in a run directory must describe the same build / signoff run as metadata.json.

    Checks the headers each adapter surfaces (run id, build) against the run metadata; a report that
    disagrees is a stale or misplaced artifact and is reported as an error so FE and BE numbers are
    never paired across different signoff runs.
    """
    problems = []
    want_run, want_build = meta.get("run_id"), str(meta.get("build", ""))
    for rep in reports:
        if want_run and rep.run_id and str(rep.run_id) != str(want_run):
            problems.append(f"{rep.source}: {rep.path.name}: run id '{rep.run_id}' != metadata run id '{want_run}' (stale or misplaced report)")
        rep_build = getattr(rep, "build", None)
        if want_build and rep_build and str(rep_build) != want_build:
            problems.append(f"{rep.source}: {rep.path.name}: build '{rep_build}' != metadata build '{want_build}'")
    return problems


def extract_run(run_dir: Path, cfg: Config, prof: Profiler | None = None) -> RunResult:
    prof = prof or Profiler("extract")
    inputs, meta = inputs_from_metadata(run_dir, cfg.source_patterns)
    design, build = inputs.design, inputs.build
    reports: list[ParsedReport] = []
    errors: list[str] = []

    with prof.stage("extraction", detail=f"{design}/{build}"):
        for name, mod in SOURCES.items():
            if name in cfg.disabled_sources:
                continue
            located = mod.get_files(inputs)
            if not located:
                errors.append(f"{name}: no files matched pattern '{inputs.pattern_for(name, mod.DEFAULT_PATTERN)}'")
                continue
            for loc in located:
                try:
                    reports.append(mod.parse(loc.path, **loc.context))
                except (ParseError, ValueError, KeyError) as exc:
                    errors.append(f"{name}: {loc.path.name}: {exc}")
        inconsistent = verify_run_consistency(reports, meta)
        errors.extend(inconsistent)
        if inconsistent and cfg.strict_consistency:
            bad = {p.split(":")[1].strip() for p in inconsistent}
            reports = [r for r in reports if r.path.name not in bad]

    with prof.stage("normalization", detail=f"{design}/{build}") as st:
        long = _normalize(reports, design, build, meta)
        st.rows = len(long)

    with prof.stage("lineage mapping", detail=f"{design}/{build}") as st:
        map_path = run_dir / cfg.fub_map_pattern
        if not map_path.exists():
            raise FileNotFoundError(f"FUB map not found: {map_path}")
        model = ModelRoot.load(map_path, design, model_version=meta.get("model_version"))
        lin = resolve(long, model, design, build, sources_present={r.source for r in reports})
        st.rows = len(lin.lineage)

    with prof.stage("pivot to FUB dataset", detail=f"{design}/{build}") as st:
        wide = pivot_fub_records(lin.mapped)
        wide = attach_identity(wide, model)
        wide["fmax_ghz"] = fmax_from_timing(wide["clock_period_ps"], wide["wns_ps"])
        wide = stamp_provenance(wide, design, build, meta, run_dir)
        perf = pivot_perf_records(lin.mapped, design, build)
        st.rows = len(wide)

    return RunResult(design, build, run_dir, reports, lin.mapped, wide, perf, lin.lineage, lin.unmapped,
                     lin.flags, errors, str(meta.get("status", "current")))


def _normalize(reports: list[ParsedReport], design: str, build: str, meta: dict) -> pd.DataFrame:
    frames = []
    for rep in reports:
        r = rep.records.copy()
        if not len(r):
            continue
        r["design"], r["build"] = design, build
        if "workload" not in r.columns:
            r["workload"] = rep.workload
        if "operating_point" not in r.columns:
            r["operating_point"] = rep.operating_point
        r["source"], r["source_file"] = rep.source, str(rep.path)
        r["tool"], r["tool_version"] = rep.tool, rep.tool_version
        r["run_id"] = rep.run_id or meta.get("run_id")
        r["report_date"] = rep.report_date or meta.get("build_date")
        frames.append(r)
    if not frames:
        return pd.DataFrame(columns=["object", "object_kind", "metric", "value", "unit", "unit_original",
                                     "design", "build", "workload", "operating_point", *PROV_COLS])
    long = pd.concat(frames, ignore_index=True)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    # metric scope: drop keys a metric does not vary by, so joins broadcast correctly
    for metric, scope in METRIC_SCOPE.items():
        sel = long["metric"] == metric
        if "workload" not in scope:
            long.loc[sel, "workload"] = None
        if "operating_point" not in scope:
            long.loc[sel, "operating_point"] = None
    return long


def join_metric(wide: pd.DataFrame, records: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Left-join every metric in `records` onto `wide`, honouring each metric's scope (METRIC_SCOPE).

    `keys` are the object keys (["fub"] for FUB-level records, [] for design-level records that
    broadcast to every row). Duplicate (key, scope) rows keep the first value; sanitize counts them.
    """
    out = wide
    for metric, grp in records.groupby("metric"):
        scope = [k for k in ("workload", "operating_point") if k in METRIC_SCOPE.get(metric, ())]
        cols = keys + scope
        if cols:
            g = grp.drop_duplicates(subset=cols, keep="first")[cols + ["value"]].rename(columns={"value": metric}).copy()
            for k in scope:
                g[k] = g[k].fillna(DEFAULT_SCOPE_VALUE)
            out = out.merge(g, on=cols, how="left")
        else:
            out = out.assign(**{metric: float(grp["value"].iloc[0])})   # scope-less design-level metric broadcasts
    return out


def pivot_fub_records(mapped: pd.DataFrame, metrics: tuple[str, ...] = FUB_METRICS) -> pd.DataFrame:
    """Long records -> one wide row per (fub, workload, operating_point) for the given metrics.

    The row set is defined by the power metrics (they carry the workload x operating-point grid);
    build-level and workload-level metrics are broadcast onto it.
    """
    fub_rec = mapped[(mapped["fub"] != DESIGN_FUB) & mapped["metric"].isin(metrics)]
    des_rec = mapped[(mapped["fub"] == DESIGN_FUB) & mapped["metric"].isin(metrics)]
    power = fub_rec[fub_rec["metric"].isin(POWER_COLUMNS)]
    if len(power):
        base = power[["fub", "workload", "operating_point"]].drop_duplicates().reset_index(drop=True)
    else:
        base = fub_rec[["fub"]].drop_duplicates().assign(workload=None, operating_point=None)
    base["workload"] = base["workload"].fillna(DEFAULT_SCOPE_VALUE)
    base["operating_point"] = base["operating_point"].fillna(DEFAULT_SCOPE_VALUE)
    wide = join_metric(base, fub_rec, ["fub"])
    if len(des_rec):
        wide = join_metric(wide, des_rec, [])
    for m in metrics:
        if m not in wide.columns:
            wide[m] = float("nan")
    return wide


def attach_identity(wide: pd.DataFrame, model: ModelRoot) -> pd.DataFrame:
    """Add model_root / partition from the model root, placed right after fub."""
    ident = model.to_frame()[["fub", "model_root", "partition"]]
    out = wide.merge(ident, on="fub", how="left")
    cols = [c for c in out.columns if c not in ("model_root", "partition")]
    i = cols.index("fub") + 1
    return out[cols[:i] + ["model_root", "partition"] + cols[i:]]


def stamp_provenance(wide: pd.DataFrame, design: str, build: str, meta: dict, run_dir: Path) -> pd.DataFrame:
    out = wide.copy()
    out.insert(0, "design", design)
    out.insert(1, "build", build)
    out.insert(out.columns.get_loc("fub") + 1, "stage", PAIRED_STAGE)
    out["run_id"] = meta.get("run_id")
    out["build_date"] = meta.get("build_date")
    out["build_status"] = meta.get("status", "current")
    out["design_type"] = meta.get("design_type", "unspecified")
    out["source_file"] = str(run_dir)
    out["tool"] = "pipeline"
    out["tool_version"] = ";".join(f"{k}={v}" for k, v in (meta.get("tools") or {}).items())
    out["imported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def pivot_perf_records(mapped: pd.DataFrame, design: str, build: str) -> pd.DataFrame:
    """Design-level performance metrics per (workload, operating_point), with the op's V and f attached."""
    rec = mapped[(mapped["fub"] == DESIGN_FUB) & mapped["metric"].isin(PERF_METRICS)]
    if not len(rec):
        return pd.DataFrame(columns=["design", "build", "workload", "operating_point", *PERF_METRICS])
    p = rec.pivot_table(index=["workload", "operating_point"], columns="metric", values="value", aggfunc="first").reset_index()
    p.columns.name = None
    op_rec = mapped[(mapped["fub"] == DESIGN_FUB) & mapped["metric"].isin(DESIGN_LEVEL_METRICS)]
    if len(op_rec):
        op = op_rec.pivot_table(index="operating_point", columns="metric", values="value", aggfunc="first").reset_index()
        op.columns.name = None
        p = p.merge(op, on="operating_point", how="left")
    p.insert(0, "design", design)
    p.insert(1, "build", build)
    return p


# ----------------------------------------------------------------------------- orchestration

def find_runs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/*/metadata.json"))


def ingest_runs(project: Project, run_dirs: list[Path], cfg: Config | None = None, replace: bool = True,
                prof: Profiler | None = None) -> IngestSummary:
    cfg = project.init(cfg)
    prof = prof or Profiler("ingest")
    results: list[RunResult] = []
    for rd in run_dirs:
        results.append(extract_run(Path(rd), cfg, prof))

    with prof.stage("validation") as st:
        wide = pd.concat([r.wide for r in results], ignore_index=True) if results else pd.DataFrame()
        long = pd.concat([r.long for r in results], ignore_index=True) if results else pd.DataFrame()
        perf = pd.concat([r.perf for r in results], ignore_index=True) if results else pd.DataFrame()
        lineage = pd.concat([r.lineage for r in results], ignore_index=True) if results else pd.DataFrame()
        unmapped = pd.concat([r.unmapped for r in results], ignore_index=True) if results else pd.DataFrame()
        report = validate(wide) if len(wide) else None
        st.rows = len(wide)
        if report is not None:
            wide = add_analysis_features(add_derived_metrics(wide))
            err = pd.Series(report.error_mask, index=wide.index)
            rejected = wide[err.to_numpy()].copy()
            if len(rejected):
                rejected["reject_reason"] = report.reject_reasons().reindex(rejected.index).to_numpy()
            wide = wide[~err.to_numpy()].reset_index(drop=True)
        else:
            rejected = pd.DataFrame()

    with prof.stage("store (Parquet + DuckDB)", rows=len(wide)) as st:
        pdir = project.processed_dir
        if not replace and project.dataset_path(cfg).exists():
            # always merge into the RAW dataset (never the sanitized view) so flagged rows are never lost
            old = load_dataset(project, cfg, raw=True)
            wide = pd.concat([old, wide], ignore_index=True).drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
            for name, new in (("measurements_long", long), ("lineage", lineage), ("unmapped", unmapped), ("performance", perf)):
                old_p = pdir / f"{name}.parquet"
                if old_p.exists():
                    prev = pd.read_parquet(old_p)
                    # a re-ingested (design, build) replaces its previous rows in every auxiliary table
                    key = new[["design", "build"]].drop_duplicates()
                    prev = prev.merge(key.assign(_new=True), on=["design", "build"], how="left")
                    prev = prev[prev["_new"].isna()].drop(columns="_new")
                    new = pd.concat([prev, new], ignore_index=True)
                if name == "measurements_long":
                    long = new
                elif name == "lineage":
                    lineage = new
                elif name == "unmapped":
                    unmapped = new
                else:
                    perf = new
        ds_path = write_table(wide, project.root / cfg.dataset)
        write_table(long, pdir / "measurements_long.parquet")
        write_table(lineage, pdir / "lineage.parquet")
        write_table(unmapped, pdir / "unmapped.parquet")
        write_table(perf, pdir / "performance.parquet")
        if len(rejected):
            write_table(rejected, project.root / cfg.rejected)
        # any previous sanitized dataset is now stale
        san = pdir / "measurements_sanitized.parquet"
        if san.exists():
            san.unlink()
        register_views(project, {
            "measurements": ds_path, "measurements_long": pdir / "measurements_long.parquet",
            "lineage": pdir / "lineage.parquet", "unmapped": pdir / "unmapped.parquet",
            "performance": pdir / "performance.parquet",
        })
        _log_imports(project, results, ds_path)

    prof.save(project)
    return IngestSummary(results, ds_path, prof, len(wide), len(rejected))


def _log_imports(project: Project, results: list[RunResult], ds_path: Path) -> None:
    from functools import lru_cache

    from powermet.catalog import record_build, record_import, record_source_files
    from powermet.extract.metadata import load as load_meta

    sha_of = lru_cache(maxsize=None)(lambda p: file_sha256(p))
    for r in results:
        meta = load_meta(r.run_dir / "metadata.json")
        record_build(project, r.design, r.build, meta, r.run_dir)
        record_source_files(project, r.design, r.build, r.reports, sha_of)
        record_import(project, {
            "source_file": str(r.run_dir), "sha256": sha_of(r.run_dir / "metadata.json"),
            "rows_imported": int(len(r.wide)), "rows_rejected": 0, "dataset": str(ds_path),
            "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "design": r.design, "build": r.build, "n_reports": len(r.reports), "errors": r.errors,
        })
