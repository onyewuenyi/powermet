import numpy as np
import pandas as pd
import pytest

from powermet.config import Config, Project
from powermet.deps import available
from powermet.ingest import write_table
from powermet.modeling import LinearModel, evaluate, load, save, split_by_build, train
from powermet.storage import add_analysis_features, import_file, load_dataset
from powermet.metrics import add_derived_metrics


def test_split_by_build_holds_out_later_builds(demo_df):
    s = split_by_build(demo_df, test_fraction=0.2, min_builds_warn=5)
    assert s.strategy == "BUILD"
    assert s.test_builds == ["B006"] or s.test_builds == ["B005", "B006"]
    assert set(s.train_builds).isdisjoint(s.test_builds)
    assert len(s.train_idx) + len(s.test_idx) == len(demo_df)
    assert not s.warnings


def test_split_natural_order_and_warning():
    df = pd.DataFrame({"build": ["B2", "B10", "B1"] * 4, "be_mw": 1.0})
    s = split_by_build(df, 0.2, min_builds_warn=5)
    assert s.test_builds == ["B10"]
    assert s.warnings


def test_split_refuses_single_build():
    df = pd.DataFrame({"build": ["B1"] * 5, "be_mw": 1.0})
    with pytest.raises(ValueError):
        split_by_build(df)


def test_linear_model_recovers_relationship():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = 2 * X["a"] - 3 * X["b"] + 1
    m = LinearModel(["a", "b"]).fit(X, y.to_numpy())
    assert np.allclose(m.coef_, [2, -3], atol=1e-6) and abs(m.intercept_ - 1) < 1e-6


def test_train_beats_baseline_and_roundtrips(tmp_path, demo_df):
    df = add_analysis_features(add_derived_metrics(demo_df))
    cfg = Config()
    res = train(df, cfg)
    base = res.metrics_test["baseline"]["mape"]
    assert res.metrics_test["linear"]["mape"] < base
    if available("sklearn"):
        assert "tree" in res.models
        assert res.metrics_test["tree"]["mape"] < base
        assert abs(sum(res.importance["tree"].values()) - 1) < 1e-6
    proj = Project(tmp_path / ".powermet")
    proj.init(cfg)
    mpath, jpath = save(proj, res, cfg)
    payload, meta = load(proj)
    assert meta["split_strategy"] == "BUILD" and meta["test_builds"] == res.split.test_builds
    again = evaluate(df, payload, meta)
    assert again["baseline"]["mape"] == pytest.approx(base)
