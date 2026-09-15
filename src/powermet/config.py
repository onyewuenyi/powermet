"""Project directory layout and config.toml (read via tomllib, written by a tiny flat writer)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_DIRNAME = ".powermet"
ENV_VAR = "POWERMET_HOME"

SUBDIRS = ("data/raw", "data/processed", "models", "reports", "cache")


@dataclass
class Config:
    """Analysis settings stored in .powermet/config.toml. Keep flat and simple."""

    dataset: str = "data/processed/measurements.parquet"   # relative to project dir
    rejected: str = "data/processed/rejected.parquet"
    target: str = "be_mw"
    baseline_feature: str = "fe_physical_mw"
    linear_features: list[str] = field(default_factory=lambda: ["fe_physical_mw", "wire_cap_pf", "cell_cap_pf"])
    tree_features: list[str] = field(
        default_factory=lambda: ["fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "fanout",
                                 "frequency_ghz", "voltage_v", "activity", "wire_cap_fraction", "dyn_term", "leak_term"]
    )
    physics_features: list[str] = field(default_factory=lambda: ["dyn_term", "move_term", "leak_term", "fe_physical_mw"])
    datamove_features: list[str] = field(default_factory=lambda: ["cell_dyn_term", "wire_dyn_term", "move_term", "leak_term"])
    whatif_model: str = "datamove"         # what-if model: datamove/physics/linear extrapolate, tree does not
    cv_min_builds: int = 3
    test_fraction: float = 0.2
    min_builds_warn: int = 5
    seed: int = 42
    top_n_errors: int = 10
    # V1 extraction
    fub_map_pattern: str = "mapping/fub_map.csv"
    # identity strategy: how FE/BE report object names map to FUBs (see identity.IdentityStrategy, docs/methodology-variants.md)
    identity: dict = field(default_factory=lambda: {"kind": "explicit_map", "replica_policy": "sum", "merge_basis": "share",
                                                    "name_rules": [], "divider": "/", "lowercase": False, "strip_top": False})
    profile: str = ""                    # methodology profile the config was initialised from (informational)
    source_patterns: dict[str, str] = field(default_factory=dict)   # per-source override of DEFAULT_PATTERN
    disabled_sources: list[str] = field(default_factory=list)
    strict_consistency: bool = True     # drop reports whose run id / build disagree with metadata.json
    upf_pattern: str = "intent/*.upf"   # power intent under the run directory ("" to skip)
    budgets_file: str = "budgets.toml"  # relative to the project root's parent (the working dir) unless absolute
    qualify_tolerance_pct: float = 5.0
    # scheduler fan-out: `ingest plan` wraps each job in this template ({design}, {build}, {cmd}); "" = bare command
    submit_cmd: str = ""
    partition_dir: str = "data/processed/runs"   # relative to the project root
    # V1 sanitization
    stale_days: int = 120
    near_zero_mw: float = 0.01
    leakage_workload_tol_pct: float = 5.0   # leakage should not depend on the workload; larger spread per FUB/op flags a component mismatch
    group_sum_tol_pct: float = 5.0          # clock + register + combinational + memory should reconstruct be_mw within this
    # comparative analysis (analyze anomalies): thresholds per rule; see anomalies.RULES and docs/power-analysis.md
    idle_workload: str = "idle"
    anomaly_idle_ratio: float = 0.35        # idle dynamic power / reference-workload dynamic power above this = clocks not gated at idle
    anomaly_eff_ratio: float = 2.0          # dynamic power per unit activity*C*V^2f above this multiple of the design median
    anomaly_regression_pct: float = 10.0    # build-to-build dynamic growth beyond what activity x capacitance explains, above this = unexplained
    anomaly_clock_fraction: float = 0.5     # clock-network share of dynamic power above this = clock dominant
    anomaly_creep_builds: int = 3           # consecutive rising builds ...
    anomaly_creep_pct: float = 15.0         # ... with this cumulative growth = creeping
    anomaly_replica_ratio: float = 1.3      # max/min dynamic power across replica instances above this = divergent replicas
    anomaly_profile_tol_pct: float = 5.0    # time-based profile average vs averaged report beyond this = profile mismatch
    unit_magnitude_ratio: float = 50.0
    use_sanitized: bool = True

    def to_toml(self) -> str:
        lines = ["# powermet project configuration", ""]
        for k, v in asdict(self).items():
            lines.append(f"{k} = {_toml_value(v)}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"unsupported config value type: {type(v).__name__}")


class Project:
    """Handle to a .powermet/ directory. Deleting the directory resets everything."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get(ENV_VAR) or Path.cwd() / PROJECT_DIRNAME
        self.root = Path(root)

    # ---- paths
    @property
    def config_path(self) -> Path:
        return self.root / "config.toml"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def duckdb_path(self) -> Path:
        return self.cache_dir / "powermet.duckdb"

    # ---- lifecycle
    def exists(self) -> bool:
        return self.config_path.exists()

    def init(self, config: Config | None = None) -> Config:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        if config is None:
            config = self.load_config() if self.config_path.exists() else Config()
        self.save_config(config)
        return config

    def load_config(self) -> Config:
        if not self.config_path.exists():
            return Config()
        with open(self.config_path, "rb") as fh:
            return Config.from_dict(tomllib.load(fh))

    def save_config(self, config: Config) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(config.to_toml())

    def dataset_path(self, config: Config | None = None) -> Path:
        cfg = config or self.load_config()
        p = self.root / cfg.dataset
        if not p.exists() and p.suffix == ".parquet" and p.with_suffix(".csv").exists():
            return p.with_suffix(".csv")
        return p
