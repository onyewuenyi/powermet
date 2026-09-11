"""SQLite metadata catalog: builds, source files, imports, quality runs, models, profiles.

Measurements stay in Parquet (queried by DuckDB); this database holds the *metadata* that
must be queryable and durable: what was ingested, from where, with which tool versions, what
quality it had, and which models were trained on it. It is the laptop stand-in for a shared
PostgreSQL catalog; the schema is plain SQL so it ports without changes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_NAME = "metrology.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS build (
    design TEXT NOT NULL, build TEXT NOT NULL, build_date TEXT, run_id TEXT, status TEXT,
    tools TEXT, run_dir TEXT, ingested_at TEXT,
    PRIMARY KEY (design, build)
);
CREATE TABLE IF NOT EXISTS source_file (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design TEXT, build TEXT, source TEXT, path TEXT, sha256 TEXT, tool TEXT, tool_version TEXT,
    run_id TEXT, report_date TEXT, workload TEXT, operating_point TEXT, n_records INTEGER, ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS import_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT, sha256 TEXT, rows_imported INTEGER, rows_rejected INTEGER, dataset TEXT,
    imported_at TEXT, design TEXT, build TEXT, n_reports INTEGER, errors TEXT
);
CREATE TABLE IF NOT EXISTS quality_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT, n_rows INTEGER, n_usable INTEGER, usable_pct REAL, counts TEXT, metric_quality TEXT
);
CREATE TABLE IF NOT EXISTS model (
    model_file TEXT PRIMARY KEY, created_at TEXT, best_model TEXT, best_mape REAL, baseline_mape REAL,
    test_builds TEXT, dataset_sha256 TEXT, metrics TEXT
);
CREATE TABLE IF NOT EXISTS profile_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT, started_at TEXT, total_wall_s REAL, peak_rss_mb REAL, python TEXT, stages TEXT
);
"""


def db_path(project) -> Path:
    return Path(project.root) / DB_NAME


@contextmanager
def connect(project):
    path = db_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- writers

def record_build(project, design: str, build: str, meta: dict, run_dir: Path) -> None:
    with connect(project) as con:
        con.execute(
            "INSERT OR REPLACE INTO build VALUES (?,?,?,?,?,?,?,?)",
            (design, build, meta.get("build_date"), meta.get("run_id"), meta.get("status", "current"),
             json.dumps(meta.get("tools") or {}), str(run_dir), _now()),
        )


def record_source_files(project, design: str, build: str, reports, sha_of) -> None:
    rows = [(design, build, r.source, str(r.path), sha_of(r.path), r.tool, r.tool_version, r.run_id, r.report_date,
             r.workload, r.operating_point, int(r.n_records), _now()) for r in reports]
    with connect(project) as con:
        con.execute("DELETE FROM source_file WHERE design = ? AND build = ?", (design, build))
        con.executemany("INSERT INTO source_file (design, build, source, path, sha256, tool, tool_version, run_id, report_date, "
                        "workload, operating_point, n_records, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def record_import(project, entry: dict) -> None:
    with connect(project) as con:
        con.execute(
            "INSERT INTO import_run (source_file, sha256, rows_imported, rows_rejected, dataset, imported_at, design, build, n_reports, errors) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry.get("source_file"), entry.get("sha256"), entry.get("rows_imported"), entry.get("rows_rejected"),
             entry.get("dataset"), entry.get("imported_at", _now()), entry.get("design"), entry.get("build"),
             entry.get("n_reports"), json.dumps(entry.get("errors") or [])),
        )


def record_quality(project, rep, metric_quality: pd.DataFrame | None = None) -> None:
    with connect(project) as con:
        con.execute("INSERT INTO quality_run (run_at, n_rows, n_usable, usable_pct, counts, metric_quality) VALUES (?,?,?,?,?,?)",
                    (_now(), rep.n_rows, rep.n_usable, rep.usable_pct, json.dumps(rep.counts),
                     metric_quality.to_json(orient="records") if metric_quality is not None else None))


def record_model(project, meta: dict, best: str) -> None:
    """`best` is the winning model key (computed by the caller so this module stays dependency-free)."""
    with connect(project) as con:
        con.execute("INSERT OR REPLACE INTO model VALUES (?,?,?,?,?,?,?,?)",
                    (meta["model_file"], meta["created_at"], best, meta["metrics"][best].get("mape"),
                     meta["metrics"]["baseline"].get("mape"), json.dumps(meta["test_builds"]), meta.get("dataset_sha256"),
                     json.dumps(meta["metrics"])))


def record_profile(project, d: dict) -> None:
    with connect(project) as con:
        con.execute("INSERT INTO profile_run (command, started_at, total_wall_s, peak_rss_mb, python, stages) VALUES (?,?,?,?,?,?)",
                    (d["command"], d["started_at"], d["total_wall_s"], d["peak_rss_mb"], d["python"], json.dumps(d["stages"])))


# ----------------------------------------------------------------------------- readers

def query(project, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db_path(project).exists():
        return pd.DataFrame()
    con = sqlite3.connect(db_path(project))
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def tables(project) -> list[str]:
    df = query(project, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return list(df["name"]) if len(df) else []


def import_history(project) -> list[dict]:
    df = query(project, "SELECT * FROM import_run ORDER BY id")
    if not len(df):
        return []
    out = df.to_dict(orient="records")
    for e in out:
        e["errors"] = json.loads(e.get("errors") or "[]")
    return out


def profiles(project, last: int = 5) -> list[dict]:
    df = query(project, "SELECT * FROM profile_run ORDER BY id DESC LIMIT ?", (last,))
    out = []
    for e in reversed(df.to_dict(orient="records")) if len(df) else []:
        e["stages"] = json.loads(e["stages"])
        out.append(e)
    return out
