import numpy as np
import pandas as pd

from powermet.demo import DemoSpec, generate, make_dirty
from powermet.metrics import add_derived_metrics, prediction_metrics
from powermet.schema import REQUIRED_COLUMNS
from powermet.validation import validate


def test_demo_is_deterministic():
    a = generate(DemoSpec(n_designs=1, n_builds=2, n_fubs=5, seed=3))
    b = generate(DemoSpec(n_designs=1, n_builds=2, n_fubs=5, seed=3))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 10


def test_demo_has_structure(demo_df):
    assert set(REQUIRED_COLUMNS) <= set(demo_df.columns)
    d = add_derived_metrics(demo_df)
    # Physical estimate should be closer to BE than logical
    assert d["physical_error_pct"].abs().mean() < d["logical_error_pct"].abs().mean()
    # Physical error should be associated with wire cap
    r = np.corrcoef(d["physical_error_mw"], d["wire_cap_pf"])[0, 1]
    assert r > 0.3
    assert (demo_df["be_mw"] > 0).all()


def test_demo_validates_clean(demo_df):
    rep = validate(demo_df)
    assert rep.ok, rep.render()
    assert rep.n_valid == len(demo_df)


def test_dirty_data_is_reported(demo_df):
    dirty = make_dirty(demo_df, n_rows=60)
    rep = validate(dirty)
    codes = {i.code for i in rep.issues}
    assert "missing_value:be_mw" in codes
    assert "negative:wire_cap_pf" in codes
    assert "zero_be" in codes
    assert "duplicate_key" in codes
    assert "missing_feature" in codes
    assert not rep.ok
    assert rep.n_errors > 0 and rep.n_valid < len(dirty)
    txt = rep.render()
    assert "Errors:" in txt and "FAIL" in txt


def test_missing_required_column():
    df = pd.DataFrame({"design": ["x"], "build": ["b"], "fub": ["f"], "stage": ["s"],
                       "fe_logical_mw": [1.0], "fe_physical_mw": [1.0]})
    rep = validate(df)
    assert not rep.ok and rep.n_valid == 0


def test_non_numeric_reported():
    df = pd.DataFrame({"design": ["x"], "build": ["b"], "fub": ["f"], "stage": ["s"],
                       "fe_logical_mw": ["abc"], "fe_physical_mw": [1.0], "be_mw": [2.0]})
    rep = validate(df)
    assert any(i.code == "non_numeric:fe_logical_mw" for i in rep.issues)


def test_safe_division():
    df = pd.DataFrame({"fe_logical_mw": [10.0, 10.0], "fe_physical_mw": [10.0, 10.0], "be_mw": [0.0, np.nan]})
    d = add_derived_metrics(df)
    assert d["logical_error_pct"].isna().all()
    assert len(d) == 2


def test_prediction_metrics_identity():
    m = prediction_metrics([1, 2, 3], [1, 2, 3])
    assert m["mape"] == 0 and m["r2"] == 1 and m["n"] == 3
