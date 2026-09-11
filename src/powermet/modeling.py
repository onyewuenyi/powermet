"""BE-power models with build-based validation (V0 -> V2).

Models (all predict be_mw):
  baseline : be = fe_physical_mw                              (the number to beat)
  scaled   : be = a * fe_physical_mw + b                      (bias correction only)
  linear   : OLS on fe_physical, wire cap, cell cap           (numpy)
  physics  : OLS on CV^2f dynamic term, leakage term, FE phys (physically-structured; extrapolates
                                                               sensibly for what-if on V/f/cap/activity)
  datamove : OLS on cell-switching, wire-switching, data-movement (bits x distance x V^2 x f) and
             leakage terms WITHOUT FE power: a compact analytical energy model whose fitted terms
             decompose predicted power into compute / wire / data-movement / leakage
  tree     : HistGradientBoostingRegressor with monotonic     (non-linear check; cannot extrapolate)
             constraints on power-increasing features

Validation never splits rows at random:
  holdout : latest builds held out ("predict the next build")
  lobo    : leave-one-build-out cross-validation -> per-build metrics and empirical prediction intervals
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from powermet.config import Config, Project
from powermet.deps import available
from powermet.metrics import prediction_metrics
from powermet.schema import label
from powermet.textfmt import fmt_pct, fmt_r, table

# ----------------------------------------------------------------------------- model registry

@dataclass(frozen=True)
class ModelSpec:
    """One entry in the model registry. Adding a model kind = adding one ModelSpec."""

    key: str
    name: str
    family: str                       # "baseline" | "linear" | "tree"
    features: str | None = None       # Config attribute holding the feature list (None -> baseline feature)
    extrapolates: bool = True         # safe for what-if beyond the training range
    min_features: int = 1
    optional_features: tuple[str, ...] = ()   # missing ones are not reported as notes
    requires: str | None = None       # optional dependency module


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("baseline", "FE Physical Baseline", "baseline", extrapolates=True),
    ModelSpec("scaled", "FE Physical (scaled)", "linear", features=None),
    ModelSpec("linear", "Linear Regression", "linear", features="linear_features"),
    ModelSpec("physics", "Physics-structured OLS", "linear", features="physics_features", optional_features=("move_term",)),
    ModelSpec("datamove", "Data-movement decomposition", "linear", features="datamove_features", min_features=2),
    ModelSpec("tree", "Gradient Boosting", "tree", features="tree_features", extrapolates=False, requires="sklearn"),
)
SPEC_BY_KEY = {m.key: m for m in MODEL_REGISTRY}
MODEL_NAMES = {m.key: m.name for m in MODEL_REGISTRY}
MODEL_ORDER = tuple(m.key for m in MODEL_REGISTRY)
EXTRAPOLATING = tuple(m.key for m in MODEL_REGISTRY if m.extrapolates and m.key != "baseline")
WHATIF_CHOICES = tuple(k for k in MODEL_ORDER if k != "baseline")
# preference order when a requested model kind is unavailable in an artifact
PREFERRED_MODEL_KEYS = ("datamove", "physics", "linear", "scaled")


def pick_model_key(models: dict, requested: str | None = None, allow_tree: bool = False) -> str:
    """The requested key if present, else the first available preferred extrapolating model."""
    if requested and requested in models:
        return requested
    for k in PREFERRED_MODEL_KEYS + (("tree",) if allow_tree else ()):
        if k in models:
            return k
    raise ValueError("no extrapolating model available in the saved artifact")
DATAMOVE_TERMS = {"cell_dyn_term": "compute_mw", "wire_dyn_term": "wire_mw", "move_term": "movement_mw", "leak_term": "leakage_mw"}
MONOTONE_INCREASING = {"fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "dyn_term", "leak_term",
                       "total_cap_pf", "activity", "frequency_ghz", "voltage_v", "wire_cap_fraction",
                       "cell_dyn_term", "wire_dyn_term", "move_term", "bits_per_cycle", "wire_length_um", "avg_net_length_um"}


# ----------------------------------------------------------------------------- split

@dataclass
class Split:
    strategy: str
    train_builds: list[str]
    test_builds: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"Split strategy:  {self.strategy}",
            f"Training builds: {len(self.train_builds)}  ({', '.join(self.train_builds)})",
            f"Test builds:     {len(self.test_builds)}  ({', '.join(self.test_builds)})",
            f"Training rows:   {len(self.train_idx):,}",
            f"Test rows:       {len(self.test_idx):,}",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


from powermet.selection import build_order  # noqa: E402  (re-exported for backwards compatibility)


def split_by_build(df: pd.DataFrame, test_fraction: float = 0.2, min_builds_warn: int = 5,
                   test_builds: list[str] | None = None) -> Split:
    builds = build_order(df["build"])
    warnings: list[str] = []
    if len(builds) < 2:
        raise ValueError(
            f"build-based split needs at least 2 distinct builds; found {len(builds)}. "
            "Import more builds before training."
        )
    if test_builds is None:
        n_test = max(1, int(math.ceil(len(builds) * test_fraction)))
        n_test = min(n_test, len(builds) - 1)
        test_builds = builds[-n_test:]
    train_builds = [b for b in builds if b not in set(test_builds)]
    if len(builds) < min_builds_warn:
        warnings.append(
            f"only {len(builds)} builds available; held-out metrics on {len(test_builds)} build(s) "
            "will be noisy and may not generalize."
        )
    b = df["build"].astype(str)
    train_idx = np.flatnonzero(b.isin(train_builds).to_numpy())
    test_idx = np.flatnonzero(b.isin(test_builds).to_numpy())
    return Split("BUILD", train_builds, list(test_builds), train_idx, test_idx, warnings)


# ----------------------------------------------------------------------------- models

class BaselineModel:
    kind = "baseline"

    def __init__(self, feature: str):
        self.features = [feature]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaselineModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(X[self.features[0]], errors="coerce").to_numpy(dtype=float)


class LinearModel:
    """OLS with intercept via numpy lstsq. Missing feature values -> training medians."""

    kind = "linear"

    def __init__(self, features: list[str], kind: str = "linear"):
        self.features = list(features)
        self.kind = kind
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fill_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in self.features:
            v = pd.to_numeric(X[f], errors="coerce").astype(float)
            cols.append(v.fillna(self.fill_.get(f, 0.0)).to_numpy())
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearModel":
        for f in self.features:
            v = pd.to_numeric(X[f], errors="coerce").astype(float)
            self.fill_[f] = float(v.median()) if v.notna().any() else 0.0
            self.scale_[f] = float(v.std()) if v.notna().sum() > 1 else 1.0
        A = self._matrix(X)
        A1 = np.column_stack([A, np.ones(len(A))])
        beta, *_ = np.linalg.lstsq(A1, np.asarray(y, dtype=float), rcond=None)
        self.coef_ = beta[:-1]
        self.intercept_ = float(beta[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._matrix(X) @ self.coef_ + self.intercept_

    def coefficients(self) -> dict[str, float]:
        return {f: float(c) for f, c in zip(self.features, self.coef_)}

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row contribution of each term (coef * value) plus intercept, in mW."""
        A = self._matrix(X)
        out = pd.DataFrame(A * self.coef_, columns=self.features, index=X.index)
        out["intercept"] = self.intercept_
        return out

    def standardized_coefficients(self) -> dict[str, float]:
        """coef * std(feature): contribution of one standard deviation of each feature, in mW."""
        return {f: float(c * self.scale_.get(f, 1.0)) for f, c in zip(self.features, self.coef_)}


class TreeModel:
    kind = "tree"

    def __init__(self, features: list[str], seed: int = 42, monotone: bool = True):
        self.features = list(features)
        self.seed = seed
        self.monotone = monotone
        self.model = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float) for f in self.features])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TreeModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        cst = [1 if (self.monotone and f in MONOTONE_INCREASING) else 0 for f in self.features]
        self.model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=0.1, random_state=self.seed, monotonic_cst=cst,
            early_stopping=True, n_iter_no_change=25, validation_fraction=0.15,
        )
        with _thread_limit(len(X)):
            self.model.fit(self._matrix(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        with _thread_limit(len(X)):
            return self.model.predict(self._matrix(X))


SMALL_DATA_ROWS = 50_000


def _thread_limit(n_rows: int):
    """OpenMP threading in sklearn's histogram GBDT costs more than it saves on FUB-sized data
    (thousands of rows): fit is ~10x faster single-threaded. Limit threads below SMALL_DATA_ROWS."""
    from contextlib import nullcontext

    if n_rows >= SMALL_DATA_ROWS:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, seed: int = 0, n_repeats: int = 5) -> dict[str, float]:
    """Increase in MAPE when each feature is shuffled (model-agnostic, on the given rows), normalized to shares."""
    rng = np.random.default_rng(seed)
    base = prediction_metrics(y, model.predict(X))["mape"]
    raw: dict[str, float] = {}
    for f in model.features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(prediction_metrics(y, model.predict(Xp))["mape"] - base)
        raw[f] = max(float(np.mean(drops)), 0.0)
    total = sum(raw.values())
    if total <= 0:
        return {f: 0.0 for f in raw}
    return {f: v / total for f, v in sorted(raw.items(), key=lambda kv: -kv[1])}


# ----------------------------------------------------------------------------- training

def _available_features(df: pd.DataFrame, wanted: list[str]) -> tuple[list[str], list[str]]:
    have = [f for f in wanted if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().any()]
    return have, [f for f in wanted if f not in have]


def build_models(df: pd.DataFrame, cfg: Config, notes: list[str] | None = None,
                registry: tuple[ModelSpec, ...] = MODEL_REGISTRY) -> dict[str, object]:
    """Instantiate (unfitted) models from the registry with the features available in df."""
    notes = notes if notes is not None else []
    models: dict[str, object] = {}
    for spec in registry:
        if spec.requires and not available(spec.requires):
            notes.append(f"{spec.name} skipped: {spec.requires} not installed.")
            continue
        wanted = [cfg.baseline_feature] if spec.features is None else list(getattr(cfg, spec.features))
        feats, missing = _available_features(df, wanted)
        if set(missing) - set(spec.optional_features):
            notes.append(f"{spec.name} skipped features not in dataset: " + ", ".join(missing))
        if len(feats) < spec.min_features:
            notes.append(f"{spec.name} skipped: needs at least {spec.min_features} feature(s), found {len(feats)}.")
            continue
        if spec.family == "baseline":
            models[spec.key] = BaselineModel(feats[0])
        elif spec.family == "linear":
            models[spec.key] = LinearModel(feats, kind=spec.key)
        elif spec.family == "tree":
            models[spec.key] = TreeModel(feats, seed=cfg.seed)
        else:
            raise ValueError(f"unknown model family '{spec.family}'")
    return models


def fit_all(models: dict[str, object], X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    return {k: m.fit(X, y) for k, m in models.items()}


@dataclass
class TrainResult:
    split: Split
    models: dict
    features: dict[str, list[str]]
    metrics_test: dict[str, dict]
    metrics_train: dict[str, dict]
    importance: dict[str, dict[str, float]]
    notes: list[str]
    cv: dict | None = None
    model_path: Path | None = None
    meta_path: Path | None = None


def train(df: pd.DataFrame, cfg: Config, cv: bool = False) -> TrainResult:
    notes: list[str] = []
    df = df.reset_index(drop=True)
    y = pd.to_numeric(df[cfg.target], errors="coerce").to_numpy(dtype=float)
    split = split_by_build(df, cfg.test_fraction, cfg.min_builds_warn)
    tr, te = split.train_idx, split.test_idx
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]
    models = fit_all(build_models(df, cfg, notes), Xtr, ytr)
    metrics_test = {k: prediction_metrics(yte, m.predict(Xte)) for k, m in models.items()}
    metrics_train = {k: prediction_metrics(ytr, m.predict(Xtr)) for k, m in models.items()}
    importance = {k: permutation_importance(models[k], Xte, yte, seed=cfg.seed)
                  for k in ("tree", "physics", "datamove", "linear") if k in models}
    features = {k: list(m.features) for k, m in models.items()}
    res = TrainResult(split, models, features, metrics_test, metrics_train, importance, notes)
    if cv:
        res.cv = cross_validate_builds(df, cfg)
    # final models for deployment/what-if are refit on ALL builds (metrics above stay held-out)
    res.models = fit_all(build_models(df, cfg), df, y)
    return res


def cross_validate_builds(df: pd.DataFrame, cfg: Config) -> dict:
    """Leave-one-build-out CV. Returns per-fold metrics and empirical relative-error quantiles."""
    df = df.reset_index(drop=True)
    builds = build_order(df["build"])
    y = pd.to_numeric(df[cfg.target], errors="coerce").to_numpy(dtype=float)
    b = df["build"].astype(str).to_numpy()
    folds: list[dict] = []
    resid: dict[str, list[np.ndarray]] = {}
    if len(builds) < cfg.cv_min_builds:
        return {"strategy": "leave-one-build-out", "folds": [], "summary": {}, "intervals": {},
                "note": f"need at least {cfg.cv_min_builds} builds for cross-validation; found {len(builds)}"}
    for hold in builds:
        te = np.flatnonzero(b == hold)
        tr = np.flatnonzero(b != hold)
        models = fit_all(build_models(df, cfg), df.iloc[tr], y[tr])
        fold = {"build": hold, "n_test": int(len(te)), "metrics": {}}
        for k, m in models.items():
            pred = m.predict(df.iloc[te])
            fold["metrics"][k] = prediction_metrics(y[te], pred)
            ok = np.isfinite(pred) & np.isfinite(y[te]) & (y[te] != 0)
            resid.setdefault(k, []).append((pred[ok] - y[te][ok]) / y[te][ok] * 100.0)
        folds.append(fold)
    summary: dict[str, dict] = {}
    intervals: dict[str, dict] = {}
    for k in folds[0]["metrics"]:
        mapes = np.array([f["metrics"][k]["mape"] for f in folds])
        r2s = np.array([f["metrics"][k]["r2"] for f in folds])
        summary[k] = {"mape_mean": float(mapes.mean()), "mape_std": float(mapes.std(ddof=0)),
                      "mape_worst": float(mapes.max()), "r2_mean": float(np.nanmean(r2s)), "n_folds": len(folds)}
        r = np.concatenate(resid[k])
        intervals[k] = {"p05": float(np.percentile(r, 5)), "p50": float(np.percentile(r, 50)),
                        "p95": float(np.percentile(r, 95)), "n": int(len(r))}
    return {"strategy": "leave-one-build-out", "folds": folds, "summary": summary, "intervals": intervals}


def save(project: Project, result: TrainResult, cfg: Config, dataset_sha: str = "") -> tuple[Path, Path]:
    project.models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"models": result.models, "features": result.features, "target": cfg.target}
    if available("joblib"):
        import joblib

        mpath = project.models_dir / f"model_{stamp}.joblib"
        joblib.dump(payload, mpath)
    else:
        mpath = project.models_dir / f"model_{stamp}.pkl"
        with open(mpath, "wb") as fh:
            pickle.dump(payload, fh)
    coefs = {k: m.coefficients() for k, m in result.models.items() if isinstance(m, LinearModel)}
    std_coefs = {k: m.standardized_coefficients() for k, m in result.models.items() if isinstance(m, LinearModel)}
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_file": mpath.name,
        "model_types": {k: type(m).__name__ for k, m in result.models.items()},
        "features": result.features,
        "target": cfg.target,
        "split_strategy": result.split.strategy,
        "train_builds": result.split.train_builds,
        "test_builds": result.split.test_builds,
        "n_train": int(len(result.split.train_idx)),
        "n_test": int(len(result.split.test_idx)),
        "metrics": result.metrics_test,
        "metrics_train": result.metrics_train,
        "feature_importance": result.importance,
        "coefficients": coefs,
        "linear_std_coefficients": std_coefs.get("linear", {}),
        "std_coefficients": std_coefs,
        "cv": result.cv,
        "final_fit": "all builds",
        "config": asdict(cfg),
        "dataset_sha256": dataset_sha,
        "notes": result.notes,
    }
    jpath = project.models_dir / f"model_{stamp}.json"
    jpath.write_text(json.dumps(meta, indent=2))
    result.model_path, result.meta_path = mpath, jpath
    from powermet.catalog import record_model

    record_model(project, meta, best=best_model_key(meta["metrics"]))
    return mpath, jpath


def list_models(project: Project) -> list[Path]:
    if not project.models_dir.exists():
        return []
    return sorted(project.models_dir.glob("model_*.json"))


def latest_model_metadata(project: Project) -> dict | None:
    metas = list_models(project)
    if not metas:
        return None
    return json.loads(metas[-1].read_text())


def load(project: Project, meta_path: str | Path | None = None) -> tuple[dict, dict]:
    metas = list_models(project)
    if meta_path is None:
        if not metas:
            raise FileNotFoundError("no trained model found; run `powermet model train` first")
        meta_path = metas[-1]
    meta_path = Path(meta_path)
    if meta_path.suffix != ".json":
        meta_path = meta_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    mpath = meta_path.parent / meta["model_file"]
    if mpath.suffix == ".joblib":
        import joblib

        payload = joblib.load(mpath)
    else:
        with open(mpath, "rb") as fh:
            payload = pickle.load(fh)
    return payload, meta


def evaluate(df: pd.DataFrame, payload: dict, meta: dict) -> dict[str, dict]:
    """Re-score saved models on the saved test builds of the given dataset.

    Note: saved models are refit on all builds, so this is an in-sample score for the test
    builds; the held-out numbers live in meta['metrics'] and meta['cv'].
    """
    df = df.reset_index(drop=True)
    y = pd.to_numeric(df[meta["target"]], errors="coerce").to_numpy(dtype=float)
    mask = df["build"].astype(str).isin(meta["test_builds"]).to_numpy()
    if not mask.any():
        raise ValueError("none of the model's test builds are present in the current dataset: "
                         + ", ".join(meta["test_builds"]))
    Xte, yte = df[mask], y[mask]
    return {k: prediction_metrics(yte, m.predict(Xte)) for k, m in payload["models"].items()}


# ----------------------------------------------------------------------------- rendering

def render_model_comparison(meta: dict, metrics: dict | None = None, title: str | None = None) -> str:
    metrics = metrics or meta["metrics"]
    rows = []
    for k in MODEL_ORDER:
        if k in metrics:
            m = metrics[k]
            rows.append([MODEL_NAMES[k], fmt_pct(m.get("mape")), fmt_pct(m.get("p50_ape")),
                         fmt_pct(m.get("p95_ape")), f"{m.get('mae', float('nan')):.1f}",
                         f"{m.get('rmse', float('nan')):.1f}", fmt_r(m.get("r2"))])
    hdr = ["Model", "MAPE", "P50", "P95", "MAE mW", "RMSE mW", "R^2"]
    title = title or ("Model comparison (held-out builds: " + ", ".join(meta["test_builds"]) + ")")
    return title + "\n\n" + table(hdr, rows)


def render_cv(cv: dict | None) -> str:
    if not cv:
        return ""
    if not cv.get("folds"):
        return f"Cross-validation: {cv.get('note', 'not run')}"
    out = [f"Cross-validation: {cv['strategy']} ({len(cv['folds'])} folds)", ""]
    rows = []
    for k in MODEL_ORDER:
        if k in cv["summary"]:
            s, iv = cv["summary"][k], cv["intervals"][k]
            rows.append([MODEL_NAMES[k], fmt_pct(s["mape_mean"]), fmt_pct(s["mape_std"]), fmt_pct(s["mape_worst"]),
                         fmt_r(s["r2_mean"]), f"{iv['p05']:+.1f}% .. {iv['p95']:+.1f}%"])
    out.append(table(["Model", "MAPE mean", "MAPE sd", "MAPE worst", "R^2 mean", "90% error interval"], rows))
    out.append("")
    rows = []
    for f in cv["folds"]:
        rows.append([f["build"], f"{f['n_test']:,}"] + [fmt_pct(f["metrics"][k]["mape"]) for k in MODEL_ORDER if k in f["metrics"]])
    out.append("Per-build MAPE (each build predicted from all others):")
    out.append("")
    out.append(table(["Held-out build", "n"] + [MODEL_NAMES[k] for k in MODEL_ORDER if k in cv["folds"][0]["metrics"]], rows))
    return "\n".join(out)


def render_importance(meta: dict) -> str:
    imp = meta.get("feature_importance", {})
    key = "tree" if "tree" in imp else "physics" if "physics" in imp else "linear" if "linear" in imp else None
    if key is None:
        return ""
    rows = [[label(f), f"{v:.2f}"] for f, v in imp[key].items()]
    out = [f"Feature importance ({MODEL_NAMES[key]}, permutation on held-out builds; share of MAPE increase)", "",
           table(["Feature", "Share"], rows)]
    phys = {f: v for f, v in imp[key].items() if f != "fe_physical_mw"}
    tot = sum(phys.values())
    if tot > 0 and len(phys) > 1:
        rows = [[label(f), f"{v / tot:.2f}"] for f, v in phys.items()]
        out += ["", "Share among physical features only (FE physical power excluded)", "",
                table(["Feature", "Share"], rows)]
    if meta.get("std_coefficients", {}).get("physics"):
        rows = [[label(f), f"{v:+.1f}"] for f, v in sorted(meta["std_coefficients"]["physics"].items(), key=lambda kv: -abs(kv[1]))]
        out += ["", "Physics-structured model: mW change in predicted BE per one standard deviation of each term", "",
                table(["Term", "mW / sd"], rows)]
    if meta.get("linear_std_coefficients"):
        rows = [[label(f), f"{v:+.1f}"] for f, v in sorted(meta["linear_std_coefficients"].items(), key=lambda kv: -abs(kv[1]))]
        out += ["", "Linear model: mW change in predicted BE per one standard deviation of each feature", "",
                table(["Feature", "mW / sd"], rows)]
    out.append("")
    out.append("Importance is model-specific evidence, not proof of physical cause.")
    return "\n".join(out)


def improvement_pct(base: dict, other: dict, key: str = "mape") -> float:
    b, o = base.get(key, float("nan")), other.get(key, float("nan"))
    if not (np.isfinite(b) and np.isfinite(o)) or b == 0:
        return float("nan")
    return (b - o) / b * 100.0


def best_model_key(metrics: dict) -> str:
    return min(metrics, key=lambda k: metrics[k].get("mape", float("inf")))
