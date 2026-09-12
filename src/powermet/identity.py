"""Design identity: the model root is the source of truth for FUB identity and FUB -> partition mapping.

    ModelRoot
      +-- FubSpec  (one per FUB): model_root id, partition, FE hierarchy, synthesis object, BE hierarchy
      +-- MapEntry (one per map row): a FUB may have several BE rows (split), several FUBs may share
                    one BE object (merged), and a BE path may be a glob (replicated instances)
      +-- IdentityStrategy: how report object names are matched to FUBs and how many-to-one /
                    one-to-many relationships are weighted

The FE <-> BE relationship differs between companies and even between blocks of one design:

    same hierarchy       BE keeps the RTL hierarchy; be_hier == fe_hier               -> strategy kind "same_hierarchy"
    separate hierarchies BE paths differ; the map carries both                        -> "explicit_map" (default)
    renamed / uniquified synthesis adds _0/_1, BE adds _phys/_r2, dividers differ    -> NameRules normalise names first
    replicated units     one RTL module instantiated N times in BE (cores, SMs, PEs)  -> be_hier glob + replica_policy
    split FUB            one FUB implemented across several BE blocks / partitions   -> several map rows, summed
    merged / flattened   several FUBs ungrouped into one BE block                    -> shared be_hier, apportioned by be_share
    ancestor rows        partition / top totals in BE reports                        -> aggregates, never unmapped

Everything downstream keys on `model_root`; `fub` is its short name. This module owns every
"which FUB does this object belong to, and with what weight" question.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

FUB_MAP_REQUIRED = ("fub", "fe_hier", "synth_object", "be_hier")
FUB_MAP_OPTIONAL = ("model_root", "partition", "be_share", "fe_share")
FUB_MAP_COLUMNS = FUB_MAP_REQUIRED + FUB_MAP_OPTIONAL

# How a partition object in a BE/timing report is named relative to the partition id in the map.
PARTITION_OBJECT_PATTERNS: tuple[str, ...] = (
    r"^(?:.*/)?part_(?P<name>[^/]+)$",      # top/part_pcore0
    r"^(?:.*/)?(?P<name>[^/]+)$",           # top/pcore0 or bare pcore0
)

REPLICA_POLICIES = ("sum", "per_instance", "mean")
IDENTITY_KINDS = ("explicit_map", "same_hierarchy")


# ----------------------------------------------------------------------------- name normalisation

@dataclass
class NameRules:
    """Regex rewrites applied to report object names (and map paths) before matching.

    Typical rules: drop a top-level instance name that only the BE tool prints, strip synthesis
    uniquification suffixes (_0, _1), strip BE rename suffixes (_phys, _r2), unescape Verilog
    identifiers, normalise dividers. Order matters; rules apply in sequence.
    """

    rules: list[tuple[str, str]] = field(default_factory=list)
    divider: str = "/"            # '.' or '/' in reports -> canonical '/'
    lowercase: bool = False
    strip_top: bool = False       # drop the first path element (top module name)
    unescape: bool = True         # remove Verilog escape backslashes: \\bus[3] -> bus[3]

    def __post_init__(self) -> None:
        self._compiled = [(re.compile(p), r) for p, r in self.rules]

    def normalize(self, name: str) -> str:
        n = str(name).strip()
        if self.unescape:
            n = n.replace("\\", "")
        if self.divider != "/":
            n = n.replace(self.divider, "/")
        if self.strip_top and "/" in n:
            n = n.split("/", 1)[1]
        for rx, repl in self._compiled:
            n = rx.sub(repl, n)
        if self.lowercase:
            n = n.lower()
        return n

    @classmethod
    def from_config(cls, d: dict | None) -> "NameRules":
        d = d or {}
        rules = [tuple(r) for r in d.get("name_rules", [])]
        return cls(rules, d.get("divider", "/"), bool(d.get("lowercase", False)), bool(d.get("strip_top", False)),
                   bool(d.get("unescape", True)))


@dataclass
class IdentityStrategy:
    kind: str = "explicit_map"          # explicit_map | same_hierarchy
    replica_policy: str = "sum"         # sum | per_instance | mean  (for be_hier globs matching several instances)
    merge_basis: str = "share"          # share (be_share column) | equal   (several FUBs on one BE object)
    name_rules: NameRules = field(default_factory=NameRules)

    def __post_init__(self) -> None:
        if self.kind not in IDENTITY_KINDS:
            raise ValueError(f"identity kind must be one of {IDENTITY_KINDS}, got '{self.kind}'")
        if self.replica_policy not in REPLICA_POLICIES:
            raise ValueError(f"replica_policy must be one of {REPLICA_POLICIES}, got '{self.replica_policy}'")

    @classmethod
    def from_config(cls, d: dict | None) -> "IdentityStrategy":
        d = d or {}
        return cls(d.get("kind", "explicit_map"), d.get("replica_policy", "sum"), d.get("merge_basis", "share"),
                   NameRules.from_config(d))


# ----------------------------------------------------------------------------- map entries

@dataclass(frozen=True)
class FubSpec:
    fub: str
    model_root: str
    partition: str | None
    fe_hier: str
    synth_object: str
    be_hier: str                 # primary BE path (first map row); see ModelRoot.entries for all rows

    def as_row(self) -> dict:
        return {"fub": self.fub, "model_root": self.model_root, "partition": self.partition,
                "fe_hier": self.fe_hier, "synth_object": self.synth_object, "be_hier": self.be_hier}


@dataclass(frozen=True)
class MapEntry:
    spec: FubSpec
    fe_hier: str
    be_hier: str
    be_share: float = 1.0
    fe_share: float = 1.0
    partition: str | None = None


@dataclass(frozen=True)
class Match:
    spec: FubSpec
    weight: float = 1.0
    instance: str | None = None   # replica instance label when replica_policy == per_instance


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip() not in ("", "nan", "None")


def _norm_part(name: str) -> str:
    return str(name).strip().lower()


def _is_glob(p: str) -> bool:
    return any(c in p for c in "*?[")


# ----------------------------------------------------------------------------- model root

@dataclass
class ModelRoot:
    """FUB list + FUB -> partition / hierarchy mapping for one design, under an identity strategy."""

    design: str
    fubs: list[FubSpec]
    entries: list[MapEntry] = field(default_factory=list)
    strategy: IdentityStrategy = field(default_factory=IdentityStrategy)
    model_version: str | None = None
    source: str | None = None
    instances: dict[str, set[str]] = field(default_factory=dict)          # fub -> replica instances seen (per_instance)
    _by_fub: dict[str, FubSpec] = field(default_factory=dict, repr=False)
    _fe_exact: dict[str, list[MapEntry]] = field(default_factory=dict, repr=False)
    _be_exact: dict[str, list[MapEntry]] = field(default_factory=dict, repr=False)
    _fe_glob: list[MapEntry] = field(default_factory=list, repr=False)
    _be_glob: list[MapEntry] = field(default_factory=list, repr=False)
    _by_partition: dict[str, list[FubSpec]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_fub = {f.fub: f for f in self.fubs}
        if not self.entries:
            self.entries = [MapEntry(f, f.fe_hier, f.be_hier, 1.0, 1.0, f.partition) for f in self.fubs]
        nr = self.strategy.name_rules
        self._fe_exact, self._be_exact, self._fe_glob, self._be_glob = {}, {}, [], []
        for e in self.entries:
            fe, be = nr.normalize(e.fe_hier), nr.normalize(e.be_hier)
            e2 = MapEntry(e.spec, fe, be, e.be_share, e.fe_share, e.partition)
            (self._fe_glob.append(e2) if _is_glob(fe) else self._fe_exact.setdefault(fe, []).append(e2))
            (self._be_glob.append(e2) if _is_glob(be) else self._be_exact.setdefault(be, []).append(e2))
        self._by_partition = {}
        for f in self.fubs:
            if f.partition:
                self._by_partition.setdefault(_norm_part(f.partition), []).append(f)
        for e in self.entries:                      # split rows may sit in other partitions
            if e.partition and e.spec not in self._by_partition.get(_norm_part(e.partition), []):
                self._by_partition.setdefault(_norm_part(e.partition), []).append(e.spec)

    # ---- construction
    @classmethod
    def from_frame(cls, df: pd.DataFrame, design: str, model_version: str | None = None, source: str | None = None,
                   strategy: IdentityStrategy | None = None) -> "ModelRoot":
        strategy = strategy or IdentityStrategy()
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if strategy.kind == "same_hierarchy":
            if "be_hier" not in df.columns and "fe_hier" in df.columns:
                df["be_hier"] = df["fe_hier"]
            if "fe_hier" not in df.columns and "be_hier" in df.columns:
                df["fe_hier"] = df["be_hier"]
        if "synth_object" not in df.columns and "fub" in df.columns:
            df["synth_object"] = df["fub"]
        missing = [c for c in FUB_MAP_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"fub map{f' {source}' if source else ''} missing columns: {missing}")
        specs: dict[str, FubSpec] = {}
        entries: list[MapEntry] = []
        for r in df.itertuples(index=False):
            d = r._asdict()
            fub = str(d["fub"]).strip()
            root = d.get("model_root")
            part = d.get("partition")
            fe, be = str(d["fe_hier"]).strip(), str(d["be_hier"]).strip()
            if fub not in specs:
                specs[fub] = FubSpec(fub, str(root).strip() if _present(root) else fub,
                                     str(part).strip() if _present(part) else None, fe, str(d["synth_object"]).strip(), be)
            elif _present(root) and specs[fub].model_root != str(root).strip():
                raise ValueError(f"FUB {fub} has conflicting model_root values in the map")
            if any(e.spec.fub == fub and e.fe_hier == fe and e.be_hier == be for e in entries):
                raise ValueError(f"fub map{f' {source}' if source else ''} has duplicate rows for FUB {fub} ({be})")
            entries.append(MapEntry(specs[fub], fe, be,
                                    float(d["be_share"]) if _present(d.get("be_share")) else 1.0,
                                    float(d["fe_share"]) if _present(d.get("fe_share")) else 1.0,
                                    str(part).strip() if _present(part) else None))
        return cls(design=design, fubs=list(specs.values()), entries=entries, strategy=strategy,
                   model_version=model_version, source=source)

    @classmethod
    def load(cls, path: str | Path, design: str, model_version: str | None = None,
             strategy: IdentityStrategy | None = None) -> "ModelRoot":
        return cls.from_frame(pd.read_csv(path), design, model_version, source=str(path), strategy=strategy)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            row = e.spec.as_row()
            row.update({"fe_hier": e.fe_hier, "be_hier": e.be_hier, "be_share": e.be_share, "fe_share": e.fe_share,
                        "partition": e.partition or e.spec.partition})
            rows.append(row)
        return pd.DataFrame(rows, columns=list(FUB_MAP_COLUMNS))

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
        names = {f.partition for f in self.fubs if f.partition} | {e.partition for e in self.entries if e.partition}
        return sorted(names)

    def partition_of(self, fub: str) -> str | None:
        return self._by_fub[fub].partition

    def fubs_in(self, partition: str) -> list[FubSpec]:
        return list(self._by_partition.get(_norm_part(partition), []))

    def has_partitions(self) -> bool:
        return bool(self._by_partition)

    def relationships(self) -> dict[str, list[str]]:
        """Which FUBs are split (several BE rows), merged (share a BE object) or replicated (glob BE path)."""
        split = [f for f, n in pd.Series([e.spec.fub for e in self.entries]).value_counts().items() if n > 1]
        merged = sorted({e.spec.fub for be, es in self._be_exact.items() if len(es) > 1 for e in es})
        replicated = sorted({e.spec.fub for e in self._be_glob})
        return {"split": sorted(split), "merged": merged, "replicated": replicated}

    # ---- object resolution (report object name -> FUB matches with weights)
    def _matches(self, exact: dict, globs: list[MapEntry], name: str, share_attr: str) -> list[Match]:
        n = self.strategy.name_rules.normalize(name)
        hits = list(exact.get(n, []))
        instance = None
        if not hits:
            for e in globs:
                if fnmatch.fnmatchcase(n, getattr(e, "be_hier" if share_attr == "be_share" else "fe_hier")):
                    hits.append(e)
                    instance = _instance_label(n, getattr(e, "be_hier" if share_attr == "be_share" else "fe_hier"))
        if not hits:
            return []
        if len(hits) == 1:
            e = hits[0]
            w = 1.0
            if instance is not None and self.strategy.replica_policy == "per_instance":
                self.instances.setdefault(e.spec.fub, set()).add(instance)
                return [Match(e.spec, 1.0, instance)]
            return [Match(e.spec, w)]
        # several FUBs on one object: apportion
        if self.strategy.merge_basis == "equal":
            return [Match(e.spec, 1.0 / len(hits)) for e in hits]
        total = sum(getattr(e, share_attr) for e in hits) or 1.0
        return [Match(e.spec, getattr(e, share_attr) / total) for e in hits]

    def resolve_fe(self, hier: str) -> list[Match]:
        return self._matches(self._fe_exact, self._fe_glob, hier, "fe_share")

    def resolve_be(self, hier: str) -> list[Match]:
        return self._matches(self._be_exact, self._be_glob, hier, "be_share")

    def resolve_partition(self, obj: str) -> list[Match]:
        """Map a partition object name from a report to the FUBs implemented in it ([] if unknown)."""
        for pat in PARTITION_OBJECT_PATTERNS:
            m = re.match(pat, self.strategy.name_rules.normalize(obj))
            if m and _norm_part(m.group("name")) in self._by_partition:
                return [Match(f, 1.0) for f in self._by_partition[_norm_part(m.group("name"))]]
        return []

    def is_ancestor(self, obj: str) -> bool:
        """True when `obj` is a hierarchy level above at least one mapped path (an aggregate row)."""
        n = self.strategy.name_rules.normalize(obj)
        prefix = n.rstrip("/") + "/"
        for e in self.entries:
            for path in (e.be_hier, e.fe_hier):
                p = self.strategy.name_rules.normalize(path)
                if p.startswith(prefix) or (_is_glob(p) and fnmatch.fnmatchcase(p.split("*")[0].rstrip("/"), n.rstrip("/"))):
                    return True
        return False

    def resolve(self, obj: str, object_kind: str) -> list[Match]:
        """Generic entry point used by lineage: FUB matches (with apportion weights) for one object."""
        if object_kind == "fe_hier":
            return self.resolve_fe(obj)
        if object_kind == "be_hier":
            return self.resolve_be(obj)
        if object_kind == "partition":
            return self.resolve_partition(obj)
        if object_kind == "design":
            return [Match(f, 1.0) for f in self.fubs]
        raise ValueError(f"unknown object_kind '{object_kind}'")


def _instance_label(name: str, pattern: str) -> str:
    """The part of `name` that the glob's wildcard matched (e.g. 'u_sm_3' for 'top/u_sm_*')."""
    head = pattern.split("*")[0]
    tail = name[len(head):] if name.startswith(head) else name
    return tail.split("/")[0] or tail or name
