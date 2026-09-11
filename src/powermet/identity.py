"""Design identity: the model root is the source of truth for FUB identity and FUB -> partition mapping.

    ModelRoot
      +-- FubSpec (one per FUB): model_root id, partition, FE hierarchy, synthesis object, BE hierarchy

Everything downstream (lineage, pivot, what-if, compact model) keys on `model_root`; `fub` is its short
name. This module owns the parsing of the FUB map and every "which FUB does this object belong to"
question so no other module re-implements name matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

FUB_MAP_REQUIRED = ("fub", "fe_hier", "synth_object", "be_hier")
FUB_MAP_OPTIONAL = ("model_root", "partition")
FUB_MAP_COLUMNS = FUB_MAP_REQUIRED + FUB_MAP_OPTIONAL

# How a partition object in a BE/timing report is named relative to the partition id in the map.
# Override PARTITION_OBJECT_PATTERNS when the real convention is known; each pattern must expose
# a `name` group.
PARTITION_OBJECT_PATTERNS: tuple[str, ...] = (
    r"^(?:.*/)?part_(?P<name>[^/]+)$",      # top/part_pcore0
    r"^(?:.*/)?(?P<name>[^/]+)$",           # top/pcore0 or bare pcore0
)


@dataclass(frozen=True)
class FubSpec:
    fub: str
    model_root: str
    partition: str | None
    fe_hier: str
    synth_object: str
    be_hier: str

    def as_row(self) -> dict:
        return {"fub": self.fub, "model_root": self.model_root, "partition": self.partition,
                "fe_hier": self.fe_hier, "synth_object": self.synth_object, "be_hier": self.be_hier}


@dataclass
class ModelRoot:
    """FUB list + FUB -> partition / hierarchy mapping for one design (and optionally one model version)."""

    design: str
    fubs: list[FubSpec]
    model_version: str | None = None
    source: str | None = None
    _by_fub: dict[str, FubSpec] = field(default_factory=dict, repr=False)
    _by_fe: dict[str, FubSpec] = field(default_factory=dict, repr=False)
    _by_be: dict[str, FubSpec] = field(default_factory=dict, repr=False)
    _by_partition: dict[str, list[FubSpec]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_fub = {f.fub: f for f in self.fubs}
        self._by_fe = {f.fe_hier: f for f in self.fubs}
        self._by_be = {f.be_hier: f for f in self.fubs}
        self._by_partition = {}
        for f in self.fubs:
            if f.partition:
                self._by_partition.setdefault(_norm(f.partition), []).append(f)

    # ---- construction
    @classmethod
    def from_frame(cls, df: pd.DataFrame, design: str, model_version: str | None = None, source: str | None = None) -> "ModelRoot":
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in FUB_MAP_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"fub map{f' {source}' if source else ''} missing columns: {missing}")
        fubs = []
        for r in df.itertuples(index=False):
            d = r._asdict()
            fub = str(d["fub"]).strip()
            root = d.get("model_root")
            part = d.get("partition")
            fubs.append(FubSpec(
                fub=fub,
                model_root=str(root).strip() if _present(root) else fub,
                partition=str(part).strip() if _present(part) else None,
                fe_hier=str(d["fe_hier"]).strip(),
                synth_object=str(d["synth_object"]).strip(),
                be_hier=str(d["be_hier"]).strip(),
            ))
        dup = pd.Series([f.fub for f in fubs]).duplicated()
        if dup.any():
            raise ValueError(f"fub map{f' {source}' if source else ''} has duplicate FUB names: "
                             + ", ".join(sorted({f.fub for f, d in zip(fubs, dup) if d})))
        return cls(design=design, fubs=fubs, model_version=model_version, source=source)

    @classmethod
    def load(cls, path: str | Path, design: str, model_version: str | None = None) -> "ModelRoot":
        return cls.from_frame(pd.read_csv(path), design, model_version, source=str(path))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_row() for f in self.fubs], columns=list(FUB_MAP_COLUMNS))

    # ---- identity queries
    def __len__(self) -> int:
        return len(self.fubs)

    def __iter__(self) -> Iterable[FubSpec]:
        return iter(self.fubs)

    def fub(self, name: str) -> FubSpec:
        return self._by_fub[name]

    def get(self, name: str) -> FubSpec | None:
        return self._by_fub.get(name)

    @property
    def fub_names(self) -> list[str]:
        return [f.fub for f in self.fubs]

    @property
    def partitions(self) -> list[str]:
        return sorted({f.partition for f in self.fubs if f.partition})

    def partition_of(self, fub: str) -> str | None:
        return self._by_fub[fub].partition

    def fubs_in(self, partition: str) -> list[FubSpec]:
        return list(self._by_partition.get(_norm(partition), []))

    def has_partitions(self) -> bool:
        return bool(self._by_partition)

    # ---- object resolution (report object name -> FUB(s))
    def resolve_fe(self, hier: str) -> FubSpec | None:
        return self._by_fe.get(hier)

    def resolve_be(self, hier: str) -> FubSpec | None:
        return self._by_be.get(hier)

    def resolve_partition(self, obj: str) -> list[FubSpec]:
        """Map a partition object name from a report to the FUBs implemented in it ([] if unknown)."""
        for pat in PARTITION_OBJECT_PATTERNS:
            m = re.match(pat, obj)
            if m and _norm(m.group("name")) in self._by_partition:
                return list(self._by_partition[_norm(m.group("name"))])
        return []

    def resolve(self, obj: str, object_kind: str) -> list[FubSpec]:
        """Generic entry point used by lineage: returns the FUB(s) an object belongs to."""
        if object_kind == "fe_hier":
            f = self.resolve_fe(obj)
            return [f] if f else []
        if object_kind == "be_hier":
            f = self.resolve_be(obj)
            return [f] if f else []
        if object_kind == "partition":
            return self.resolve_partition(obj)
        if object_kind == "design":
            return list(self.fubs)
        raise ValueError(f"unknown object_kind '{object_kind}'")


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() not in ("", "nan", "None")


def _norm(name: str) -> str:
    return str(name).strip().lower()
