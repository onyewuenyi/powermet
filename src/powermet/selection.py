"""One place for dataset slicing: latest build, default workload / operating point, FUB selection.

Every analysis module used to re-implement "pick the latest build", "default to `nom`", and
"filter by design/fub". Use `DatasetSlice` (or the functions) instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DEFAULT_WORKLOAD = "typical"
DEFAULT_OPERATING_POINT = "nom"


def natural_key(text: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(text))]


def build_order(builds: pd.Series | list) -> list[str]:
    """Builds sorted naturally (B2 < B10)."""
    return sorted({str(b) for b in pd.Series(list(builds)).dropna().unique()}, key=natural_key)


def latest_build(df: pd.DataFrame) -> str:
    order = build_order(df["build"])
    if not order:
        raise ValueError("no builds in frame")
    return order[-1]


def latest_build_per_design(df: pd.DataFrame) -> dict[str, str]:
    return {str(d): latest_build(g) for d, g in df.groupby("design")}


def default_value(df: pd.DataFrame, col: str, preferred: str) -> str | None:
    """Preferred value if present in df[col], else the first sorted value; None if column absent."""
    if col not in df.columns or not len(df):
        return None
    vals = df[col].dropna().astype(str)
    if not len(vals):
        return None
    return preferred if (vals == preferred).any() else sorted(vals.unique())[0]


@dataclass(frozen=True)
class DatasetSlice:
    """Declarative selection over the wide FUB dataset. None = don't filter (except build: None = latest)."""

    design: str | None = None
    build: str | None = None           # None -> latest build (per design when design is None)
    fub: str | None = None
    model_root: str | None = None
    partition: str | None = None
    workload: str | None = None        # None -> DEFAULT_WORKLOAD when present, else first
    operating_point: str | None = None  # None -> DEFAULT_OPERATING_POINT when present, else first
    latest_only: bool = True
    default_workload: bool = True
    default_operating_point: bool = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        sel = df
        for col, val in (("design", self.design), ("fub", self.fub), ("model_root", self.model_root), ("partition", self.partition)):
            if val is not None:
                if col not in sel.columns:
                    raise ValueError(f"dataset has no '{col}' column")
                sel = sel[sel[col].astype(str) == str(val)]
        if not len(sel):
            raise ValueError("no rows match " + ", ".join(f"{k}={v}" for k, v in
                                                          (("design", self.design), ("fub", self.fub), ("model_root", self.model_root),
                                                           ("partition", self.partition)) if v is not None))
        if self.build is not None:
            sel = sel[sel["build"].astype(str) == str(self.build)]
        elif self.latest_only:
            latest = latest_build_per_design(sel)
            sel = sel[[latest[str(d)] == str(b) for d, b in zip(sel["design"], sel["build"])]]
        for col, val, use_default, pref in (("workload", self.workload, self.default_workload, DEFAULT_WORKLOAD),
                                            ("operating_point", self.operating_point, self.default_operating_point, DEFAULT_OPERATING_POINT)):
            if col not in sel.columns:
                continue
            if val is None and use_default:
                val = default_value(sel, col, pref)
            if val is not None:
                sel = sel[sel[col].astype(str) == str(val)]
        if not len(sel):
            raise ValueError(f"no rows match build={self.build or 'latest'} workload={self.workload or 'default'} "
                             f"operating_point={self.operating_point or 'default'}")
        return sel.reset_index(drop=True)

    def describe(self, df: pd.DataFrame | None = None) -> str:
        parts = []
        for k in ("design", "build", "fub", "model_root", "partition", "workload", "operating_point"):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={v}")
        if df is not None and len(df):
            for k in ("build", "workload", "operating_point"):
                if getattr(self, k) is None and k in df.columns and df[k].nunique() == 1:
                    parts.append(f"{k}={df[k].iloc[0]}")
        return ", ".join(parts)
