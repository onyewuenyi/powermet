"""Measurement-level access: one atomic record per (identity, stage, metric) with full provenance.

    Measurement(model_root, fub, design, build, stage, metric, value, unit, workload, operating_point,
                hierarchy, source, source_file, tool, tool_version, run_id)

`MeasurementStore` wraps the long provenance table written by the pipeline and answers
    store.get(fub="Scheduler", build="B003", stage="FE", metric="fe_physical_mw")
without any caller knowing how the table is laid out. Stages are derived from the source adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

STAGE_OF_SOURCE = {
    "pprtl": "FE", "saif": "ACTIVITY",
    "primepower": "BE", "primetime": "TIMING", "starrc": "PHYS", "implementation": "PHYS",
    "metadata": "DESIGN", "perf": "PERF",
}
STAGES = tuple(sorted(set(STAGE_OF_SOURCE.values())))
KEY = ("design", "build", "fub", "stage", "metric", "workload", "operating_point")


@dataclass(frozen=True)
class Measurement:
    model_root: str
    fub: str
    design: str
    build: str
    stage: str
    metric: str
    value: float
    unit: str
    workload: str | None = None
    operating_point: str | None = None
    hierarchy: str | None = None       # report object the value came from (FE/BE path or partition)
    source: str | None = None
    source_file: str | None = None
    tool: str | None = None
    tool_version: str | None = None
    run_id: str | None = None

    @property
    def key(self) -> tuple:
        return (self.design, self.build, self.fub, self.stage, self.metric, self.workload, self.operating_point)


class MeasurementStore:
    """Query interface over long-format measurement records."""

    def __init__(self, long: pd.DataFrame):
        df = long.copy()
        if "stage" not in df.columns:
            df["stage"] = df["source"].map(STAGE_OF_SOURCE).fillna("OTHER") if "source" in df.columns else "OTHER"
        if "model_root" not in df.columns:
            df["model_root"] = df["fub"]
        if "hierarchy" not in df.columns and "object" in df.columns:
            df["hierarchy"] = df["object"]
        for c in ("workload", "operating_point"):
            if c not in df.columns:
                df[c] = None
        self.df = df

    @classmethod
    def from_project(cls, project) -> "MeasurementStore":
        from powermet.storage import load_table

        long = load_table(project, "measurements_long")
        if not len(long):
            raise FileNotFoundError("no measurements_long table; run `powermet ingest` first")
        lin = load_table(project, "lineage")
        if len(lin) and "model_root" in lin.columns and "model_root" not in long.columns:
            ident = lin[["design", "build", "fub", "model_root"]].drop_duplicates()
            long = long.merge(ident, on=["design", "build", "fub"], how="left")
        return cls(long)

    # ---- queries
    def select(self, *, fub: str | None = None, model_root: str | None = None, design: str | None = None,
               build: str | None = None, stage: str | None = None, metric: str | None = None,
               workload: str | None = None, operating_point: str | None = None, source: str | None = None) -> pd.DataFrame:
        m = pd.Series(True, index=self.df.index)
        for col, val in (("fub", fub), ("model_root", model_root), ("design", design), ("build", build), ("stage", stage),
                         ("metric", metric), ("workload", workload), ("operating_point", operating_point), ("source", source)):
            if val is not None and col in self.df.columns:
                m &= self.df[col].astype(str) == str(val)
        return self.df[m]

    def get(self, **filters) -> Measurement:
        """Exactly one measurement, or KeyError / ValueError (ambiguous)."""
        sel = self.select(**filters)
        if not len(sel):
            raise KeyError(f"no measurement matches {filters}")
        if len(sel) > 1:
            distinct = sel.drop_duplicates(subset=[c for c in KEY if c in sel.columns])
            if len(distinct) > 1:
                raise ValueError(f"{len(distinct)} measurements match {filters}; add workload/operating_point/build")
        return self._row(sel.iloc[0])

    def get_value(self, default: float | None = None, **filters) -> float | None:
        try:
            return self.get(**filters).value
        except KeyError:
            return default

    def find(self, **filters) -> list[Measurement]:
        return [self._row(r) for _, r in self.select(**filters).iterrows()]

    def iter(self, **filters) -> Iterator[Measurement]:
        for _, r in self.select(**filters).iterrows():
            yield self._row(r)

    def metrics(self, stage: str | None = None) -> list[str]:
        sel = self.select(stage=stage)
        return sorted(sel["metric"].astype(str).unique())

    def fubs(self, design: str | None = None, build: str | None = None) -> list[str]:
        sel = self.select(design=design, build=build)
        return sorted(f for f in sel["fub"].astype(str).unique() if f != "*")

    def pivot(self, index=("design", "build", "fub", "workload", "operating_point")) -> pd.DataFrame:
        """Wide table (one column per metric) from the selected records; first value wins on duplicates."""
        idx = [c for c in index if c in self.df.columns]
        d = self.df.copy()
        for c in ("workload", "operating_point"):
            if c in d.columns:
                d[c] = d[c].fillna("default")
        return d.pivot_table(index=idx, columns="metric", values="value", aggfunc="first").reset_index()

    def _row(self, r: pd.Series) -> Measurement:
        return Measurement(
            model_root=str(r.get("model_root", r["fub"])), fub=str(r["fub"]), design=str(r["design"]), build=str(r["build"]),
            stage=str(r["stage"]), metric=str(r["metric"]), value=float(r["value"]), unit=str(r.get("unit", "")),
            workload=_opt(r.get("workload")), operating_point=_opt(r.get("operating_point")),
            hierarchy=_opt(r.get("hierarchy")), source=_opt(r.get("source")), source_file=_opt(r.get("source_file")),
            tool=_opt(r.get("tool")), tool_version=_opt(r.get("tool_version")), run_id=_opt(r.get("run_id")),
        )


def _opt(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def get_measurement(project, **filters) -> Measurement:
    """Convenience: `get_measurement(project, fub="A", build="B003", stage="FE", metric="fe_physical_mw")`."""
    return MeasurementStore.from_project(project).get(**filters)
