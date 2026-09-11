"""Read CSV/Parquet into the canonical schema. Source files are never modified."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from powermet.deps import available
from powermet.schema import COLUMNS, NUMERIC_COLUMNS, PROVENANCE_COLUMNS

SUPPORTED = (".csv", ".parquet", ".pq")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet file with column names normalized (lowercase, stripped)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in (".parquet", ".pq"):
        if not available("pyarrow"):
            raise RuntimeError("reading Parquet requires pyarrow (missing); convert to CSV or install pyarrow")
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"unsupported input type '{suffix}'; expected one of {SUPPORTED}")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write CSV or Parquet by extension. Falls back to CSV if pyarrow is missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        if available("pyarrow"):
            df.to_parquet(p, index=False)
            return p
        p = p.with_suffix(".csv")
    df.to_csv(p, index=False)
    return p


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(df: pd.DataFrame, source_file: str | Path | None = None) -> pd.DataFrame:
    """Coerce canonical dtypes and add provenance columns. Extra columns pass through.

    Non-numeric values in numeric columns become NaN here; validation reports
    them before this step so nothing is lost silently.
    """
    out = df.copy()
    for spec in COLUMNS:
        if spec.name not in out.columns:
            continue
        if spec.dtype == "float":
            out[spec.name] = pd.to_numeric(out[spec.name], errors="coerce").astype(float)
        else:
            s = out[spec.name]
            out[spec.name] = s.where(s.isna(), s.astype(str).str.strip()).astype(object)
    if source_file is not None:
        out["source_file"] = str(source_file)
    for col in PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out["imported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def numeric_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in NUMERIC_COLUMNS if c in df.columns]
