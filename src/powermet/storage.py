"""Local storage: processed Parquet (CSV fallback) + embedded DuckDB for queries/provenance.

Architecture:  raw CSV/Parquet -> validated -> processed Parquet -> DuckDB view over the Parquet.
DuckDB is optional; without it every reader falls back to pandas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from powermet.config import Config, Project
from powermet.deps import available
from powermet.ingest import file_sha256, normalize, read_table, write_table
from powermet.metrics import add_derived_metrics
from powermet.validation import ValidationReport, validate


@dataclass
class ImportResult:
    source: Path
    sha256: str
    report: ValidationReport
    dataset_path: Path
    rejected_path: Path | None
    n_imported: int
    n_rejected: int


def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived physical features used by analysis/models (see powermet.features)."""
    from powermet.features import add_engineered_features
    from powermet.metrics import add_convergence_metrics

    out = add_engineered_features(df)
    if "cdyn_pf" not in out.columns:            # datasets ingested before the convergence metrics existed
        add_convergence_metrics(out)
    return out


def import_file(project: Project, source: str | Path, config: Config | None = None,
                replace: bool = True) -> ImportResult:
    """Validate, normalize, and store a measurement file. Source is never modified.

    Rows with validation errors are written to the rejected dataset with a
    reject_reason column instead of being dropped silently.
    """
    cfg = project.init(config)
    src = Path(source)
    raw = read_table(src)
    report = validate(raw)

    norm = normalize(raw, source_file=str(src))
    norm = add_derived_metrics(norm)
    norm = add_analysis_features(norm)
    err = pd.Series(report.error_mask, index=norm.index)
    clean = norm[~err.to_numpy()].reset_index(drop=True)
    rejected = norm[err.to_numpy()].copy()
    rejected_path: Path | None = None
    if len(rejected):
        rejected["reject_reason"] = report.reject_reasons().reindex(rejected.index).to_numpy()
        rejected_path = write_table(rejected.reset_index(drop=True), project.root / cfg.rejected)

    ds_path = project.root / cfg.dataset
    if not replace:
        existing = project.dataset_path(cfg)
        if existing.exists():
            clean = pd.concat([read_table(existing), clean], ignore_index=True)
    ds_path = write_table(clean, ds_path)
    if ds_path.suffix == ".csv" and cfg.dataset.endswith(".parquet"):
        cfg.dataset = str(Path(cfg.dataset).with_suffix(".csv"))
        project.save_config(cfg)

    sha = file_sha256(src)
    _record_import(project, src, sha, len(clean), len(rejected), ds_path)
    return ImportResult(src, sha, report, ds_path, rejected_path, len(clean), len(rejected))


# ----------------------------------------------------------------------------- DuckDB layer

def _record_import(project: Project, src: Path, sha: str, n_ok: int, n_rej: int, ds_path: Path) -> None:
    """Record provenance in DuckDB if available, always in a JSON log as well."""
    entry = {
        "source_file": str(src.resolve()),
        "sha256": sha,
        "rows_imported": n_ok,
        "rows_rejected": n_rej,
        "dataset": str(ds_path),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    from powermet.catalog import record_import

    record_import(project, entry)

    if not available("duckdb"):
        return
    import duckdb

    con = duckdb.connect(str(project.duckdb_path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS imports (source_file VARCHAR, sha256 VARCHAR, rows_imported BIGINT, "
            "rows_rejected BIGINT, dataset VARCHAR, imported_at VARCHAR)"
        )
        con.execute("INSERT INTO imports VALUES (?, ?, ?, ?, ?, ?)", list(entry.values()))
        reader = "read_parquet" if ds_path.suffix == ".parquet" else "read_csv_auto"
        con.execute(f"CREATE OR REPLACE VIEW measurements AS SELECT * FROM {reader}('{ds_path.resolve()}')")
    finally:
        con.close()


def register_views(project: Project, tables: dict[str, Path]) -> None:
    """(Re)create DuckDB views over processed Parquet/CSV files. No-op without duckdb."""
    if not available("duckdb"):
        return
    import duckdb

    project.cache_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(project.duckdb_path))
    try:
        for name, path in tables.items():
            path = Path(path)
            if not path.exists():
                continue
            reader = "read_parquet" if path.suffix == ".parquet" else "read_csv_auto"
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {reader}('{path.resolve()}')")
    finally:
        con.close()


def query(project: Project, sql: str) -> pd.DataFrame:
    """Run SQL against the embedded DuckDB (view `measurements`). Requires duckdb."""
    if not available("duckdb"):
        raise RuntimeError("duckdb is not installed; SQL queries are unavailable")
    import duckdb

    con = duckdb.connect(str(project.duckdb_path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def sanitized_path(project: Project) -> Path:
    return project.processed_dir / "measurements_sanitized.parquet"


def load_dataset(project: Project, config: Config | None = None, raw: bool = False) -> pd.DataFrame:
    """Load the FUB-level dataset (sanitized version when present, unless raw=True)."""
    cfg = config or project.load_config()
    path = project.dataset_path(cfg)
    san = sanitized_path(project)
    if not raw and cfg.use_sanitized and san.exists():
        path = san
    if not path.exists():
        raise FileNotFoundError(
            f"no processed dataset at {path}; run `powermet data import <file>` first"
        )
    if available("duckdb"):
        import duckdb

        reader = "read_parquet" if path.suffix == ".parquet" else "read_csv_auto"
        df = duckdb.sql(f"SELECT * FROM {reader}('{path.resolve()}')").df()
    else:
        df = read_table(path)
    # engineered features are always recomputed from raw columns so an older dataset stays usable
    return add_analysis_features(df)


def load_table(project: Project, name: str) -> pd.DataFrame:
    """Load an auxiliary processed table (measurements_long, lineage, unmapped, performance)."""
    for ext in (".parquet", ".csv"):
        p = project.processed_dir / f"{name}{ext}"
        if p.exists():
            return read_table(p)
    return pd.DataFrame()


def import_history(project: Project) -> list[dict]:
    from powermet.catalog import import_history as _hist

    return _hist(project)
