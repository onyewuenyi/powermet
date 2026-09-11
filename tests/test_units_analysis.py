"""Focused unit tests for sanitize checks, metric trust, models, curves, what-if, exploration, integration, catalog, profiling."""

import json
import platform

import numpy as np
import pandas as pd
import pytest

from powermet.catalog import db_path, profiles, query, tables
from powermet.config import Config, Project
from powermet.curves import DvfsCurve, PerfModel, TimingModel
from powermet.decomposition import decompose
from powermet.deltas import classify
from powermet.explore import ExploreRow, mark_pareto
from powermet.features import FE_RESCALE_TRIGGERS, add_engineered_features, rescale_fe_physical
from powermet.frontier import FrontierPoint, frontier
from powermet.integrate import CompactPowerModel, run_trace
from powermet.modeling import (MODEL_ORDER, MODEL_REGISTRY, WHATIF_CHOICES, LinearModel, ModelSpec, build_models, evaluate,
                               permutation_importance, pick_model_key, split_by_build)
from powermet.profiling import Profiler, peak_rss_mb
from powermet.sanitize import CHECKS, metric_quality, sanitize
from powermet.whatif import Override, parse_override


# ---------------------------------------------------------------- sanitize, one check at a time

def _base(n=12):
    return pd.DataFrame({
        "design": "D", "build": ["B1"] * (n // 2) + ["B2"] * (n - n // 2), "fub": [f"F{i}" for i in range(n)], "stage": "FE_BE",
        "workload": "typical", "operating_point": "nom",
        "fe_logical_mw": np.linspace(10, 20, n), "fe_physical_mw": np.linspace(11, 21, n), "be_mw": np.linspace(12, 22, n),
        "wire_cap_pf": 1.0, "cell_cap_pf": 1.0, "area": 10.0, "build_status": "current",
        "build_date": ["2026-05-01"] * (n // 2) + ["2026-06-01"] * (n - n // 2),
        "clock_period_ps": 400.0, "wns_ps": 10.0,
    })


def _only(rep, check):
    return {c for c, n in rep.counts.items() if n} == ({check} if check else set())


def test_sanitize_clean_frame_has_no_flags():
    rep = sanitize(_base(), Config())
    assert rep.n_usable == rep.n_rows and _only(rep, None) and rep.usable_pct == 100.0


@pytest.mark.parametrize("mutate,check,blocks", [
    (lambda d: d.assign(be_mw=[np.nan] + list(d["be_mw"][1:])), "missing_metric", True),
    (lambda d: d.assign(area=[np.nan] + list(d["area"][1:])), "missing_physical", True),
    (lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "duplicate", True),
    (lambda d: d.assign(wire_cap_pf=[-1.0] + list(d["wire_cap_pf"][1:])), "negative", True),
    (lambda d: d.assign(be_mw=[0.001] + list(d["be_mw"][1:])), "near_zero_be", True),
    (lambda d: d.assign(be_mw=[5000.0] + list(d["be_mw"][1:])), "unit_suspect", True),
    (lambda d: d.assign(build_status=["superseded"] + list(d["build_status"][1:])), "stale_build", True),
    (lambda d: d.assign(wns_ps=[500.0] + list(d["wns_ps"][1:])), "timing_suspect", False),
    (lambda d: d.assign(clock_period_ps=[0.0] + list(d["clock_period_ps"][1:])), "timing_suspect", False),
])
def test_sanitize_single_check(mutate, check, blocks):
    df = mutate(_base())
    rep = sanitize(df, Config())
    assert rep.counts[check] >= 1, rep.counts
    others = {c for c, n in rep.counts.items() if n and c not in (check, "outlier")}
    assert not others, others
    assert CHECKS[check][1] == blocks
    assert (rep.n_usable < rep.n_rows) == blocks
    assert check in rep.flags.iloc[0]


def test_sanitize_stale_by_age_and_near_zero_not_double_counted():
    df = _base()
    df.loc[df["build"] == "B1", "build_date"] = "2024-01-01"
    rep = sanitize(df, Config(stale_days=120))
    assert rep.counts["stale_build"] == 6 and "D/B1" in rep.details["stale_build"]
    df = _base()
    df.loc[0, "be_mw"] = 1e-4        # would also trip the magnitude ratio
    rep = sanitize(df, Config())
    assert rep.counts["near_zero_be"] == 1 and rep.counts["unit_suspect"] == 0


def test_sanitize_lineage_split_and_long_table_info():
    df = _base()
    lineage = pd.DataFrame([
        {"design": "D", "build": "B1", "fub": "F0", "partition": "P", "lineage_ok": False, "lineage_issues": "incomplete_physical_mapping"},
        {"design": "D", "build": "B1", "fub": "F1", "partition": "P", "lineage_ok": False, "lineage_issues": "partition_timing_missing"},
    ])
    long = pd.DataFrame({"design": "D", "build": "B1", "object": ["o", "o", "p"], "fub": ["F0", "F0", "F1"], "metric": ["be_mw"] * 3,
                         "workload": None, "operating_point": None, "source_file": "f", "source": "primepower",
                         "unit": ["mW", "mW", "mW"], "unit_original": ["W", "mW", "mW"]})
    unmapped = pd.DataFrame({"design": ["D"], "build": ["B1"], "object": ["top/x"], "object_kind": ["be_hier"]})
    rep = sanitize(df, Config(), long=long, lineage=lineage, unmapped=unmapped)
    assert rep.counts["lineage_mismatch"] == 1 and rep.counts["timing_missing"] == 1
    assert rep.n_unit_conversions == 1 and rep.n_source_duplicates == 1 and rep.n_unmapped_objects == 1
    txt = rep.render()
    assert "Timing missing" in txt and "info" in txt and "primepower: W -> mW" in txt


def test_metric_quality_verdicts():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({"build": ["B1"] * 40 + ["B2"] * 40, "be_mw": rng.uniform(10, 100, n)})
    df["fe_physical_mw"] = df["be_mw"] * 0.9                                      # trusted
    df["wire_cap_pf"] = np.where(np.arange(n) < 30, df["be_mw"], np.nan)          # coverage 37% -> unusable
    df["area"] = np.where(np.arange(n) < 70, rng.uniform(1, 5, n), np.nan)        # 87% -> partial
    df["fanout"] = np.concatenate([df["be_mw"][:40], -df["be_mw"][40:]])            # sign flips by build -> unstable
    mq = metric_quality(df).set_index("metric")
    assert mq.loc["fe_physical_mw", "verdict"] == "trusted"
    assert mq.loc["wire_cap_pf", "verdict"] == "unusable"
    assert mq.loc["area", "verdict"] == "partial"
    assert mq.loc["fanout", "verdict"] == "unstable" and "varies" in mq.loc["fanout", "notes"]


# ---------------------------------------------------------------- modeling

def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"build": [f"B{i % 4 + 1}" for i in range(n)], "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = 2 * df["a"] - 3 * df["b"] + 1
    return df


def test_linear_contributions_sum_to_prediction():
    df = _toy()
    m = LinearModel(["a", "b"]).fit(df, df["y"].to_numpy())
    c = m.contributions(df)
    assert np.allclose(c.sum(axis=1), m.predict(df)) and np.allclose(c["intercept"], m.intercept_)
    assert set(m.standardized_coefficients()) == {"a", "b"}


def test_permutation_importance_degenerate_and_normalized():
    df = _toy()
    m = LinearModel(["a", "b"]).fit(df, df["y"].to_numpy())
    imp = permutation_importance(m, df, df["y"].to_numpy(), seed=1)
    assert abs(sum(imp.values()) - 1) < 1e-9 and imp["b"] > imp["a"]
    const = LinearModel(["a"]).fit(df, np.full(len(df), 5.0))       # prediction independent of a
    assert permutation_importance(const, df, np.full(len(df), 5.0)) == {"a": 0.0}


def test_split_explicit_test_builds():
    df = _toy()
    s = split_by_build(df, test_builds=["B2"])
    assert s.test_builds == ["B2"] and s.train_builds == ["B1", "B3", "B4"]
    assert set(df.iloc[s.test_idx]["build"]) == {"B2"}


def test_registry_drives_build_models_and_choices():
    assert MODEL_ORDER == tuple(m.key for m in MODEL_REGISTRY) and "datamove" in WHATIF_CHOICES and "baseline" not in WHATIF_CHOICES
    cfg = Config()
    df = pd.DataFrame({"build": ["B1", "B2"] * 5, "fe_physical_mw": np.arange(10.0), "be_mw": np.arange(10.0) * 1.1})
    notes = []
    models = build_models(df, cfg, notes)
    assert set(models) == {"baseline", "scaled", "linear", "physics", "tree"}      # datamove needs >= 2 of its terms
    assert any("Data-movement decomposition skipped" in n for n in notes)
    custom = (ModelSpec("only", "Only", "linear", features="linear_features"),)
    assert list(build_models(df, cfg, registry=custom)) == ["only"]
    with pytest.raises(ValueError):
        build_models(df, cfg, registry=(ModelSpec("x", "X", "unknown-family", features="linear_features"),))


def test_pick_model_key():
    models = {"baseline": 1, "linear": 2, "tree": 3}
    assert pick_model_key(models, "linear") == "linear" and pick_model_key(models, "physics") == "linear"
    assert pick_model_key({"tree": 1}, None, allow_tree=True) == "tree"
    with pytest.raises(ValueError):
        pick_model_key({"baseline": 1}, "datamove")


def test_evaluate_rejects_dataset_without_test_builds():
    df = _toy()
    m = LinearModel(["a", "b"]).fit(df, df["y"].to_numpy())
    payload, meta = {"models": {"linear": m}}, {"target": "y", "test_builds": ["B9"]}
    with pytest.raises(ValueError, match="test builds"):
        evaluate(df, payload, meta)


# ---------------------------------------------------------------- features / what-if

def test_engineered_feature_defaults_and_guards():
    df = pd.DataFrame({"wire_cap_pf": [2.0, 0.0], "cell_cap_pf": [2.0, 0.0], "area": [10.0, 10.0], "bits_per_cycle": [8.0, np.nan]})
    f = add_engineered_features(df)
    assert f["dyn_term"].iloc[0] == 4.0 and f["leak_term"].iloc[0] == 10.0        # act = V = f = 1 by default
    assert np.isnan(f["wire_cap_fraction"].iloc[1]) and f["move_term"].isna().all()   # no distance column


def test_rescale_fe_physical_guard():
    cur = add_engineered_features(pd.DataFrame({"fe_physical_mw": [10.0], "wire_cap_pf": [1.0], "cell_cap_pf": [1.0], "area": [1.0]}))
    prop = add_engineered_features(Override("wire_cap_pf", "scale", 0.5).apply(cur))
    out, applied = rescale_fe_physical(cur, prop, {"wire_cap_pf"})
    assert applied and out["fe_physical_mw"].iloc[0] == pytest.approx(7.5)
    out, applied = rescale_fe_physical(cur, prop, {"wire_cap_pf", "fe_physical_mw"})
    assert not applied and out["fe_physical_mw"].iloc[0] == 10.0
    out, applied = rescale_fe_physical(cur, prop, {"fanout"})
    assert not applied
    assert "voltage_v" in FE_RESCALE_TRIGGERS


@pytest.mark.parametrize("text,feature", [
    ("wire-cap=1", "wire_cap_pf"), ("wire_cap=1", "wire_cap_pf"), ("cell-cap=1", "cell_cap_pf"), ("freq=2", "frequency_ghz"),
    ("frequency=2", "frequency_ghz"), ("vdd=0.8", "voltage_v"), ("voltage=0.8", "voltage_v"), ("fe-physical=3", "fe_physical_mw"),
    ("anything_else=4", "anything_else"),
])
def test_parse_override_aliases(text, feature):
    o = parse_override(text, "set")
    assert o.feature == feature and o.mode == "set"


def test_override_errors():
    with pytest.raises(ValueError):
        parse_override("no-equals", "set")
    with pytest.raises(ValueError, match="unknown feature"):
        Override("nope", "set", 1.0).apply(pd.DataFrame({"a": [1]}))


# ---------------------------------------------------------------- deltas / frontier / explore

def test_classify_boundaries():
    assert classify(0.5, 1.0) == "neutral"                       # exactly at tolerance = flat
    assert classify(0.51, 1.01) == "power-for-performance trade"
    assert classify(-0.6, 0.0) == "pareto improvement"
    assert classify(0.0, -1.5) == "regression"
    assert classify(3.0, 0.0, tol=5.0) == "neutral"
    assert classify(np.nan, 1.0) == "power unknown"


def test_frontier_ties_and_dominance():
    df = pd.DataFrame({"design": "D", "build": ["B1", "B2", "B3"], "fub": "F", "workload": "typical", "operating_point": "nom",
                       "be_mw": [100.0, 100.0, 120.0], "fmax_ghz": [2.0, 2.5, 2.4], "wns_ps": [0.0, 20.0, 15.0]})
    pts = frontier(df, "D")
    assert [p.pareto for p in pts] == [False, True, False]        # B2 dominates both (equal power, higher fmax)
    assert pts[1].status == "pareto improvement" and pts[2].status == "regression"


def _row(name, power, thr, feasible=True):
    return ExploreRow(name, "", 2.0, 0.8, power, np.nan, np.nan, thr, power / thr if thr else np.nan, 0, 0, 0, feasible=feasible)


def test_mark_pareto_with_infeasible_and_ties():
    rows = [_row("a", 100, 50), _row("b", 100, 50), _row("c", 90, 60, feasible=False), _row("d", 200, 40)]
    mark_pareto(rows)
    assert [r.pareto for r in rows] == [True, True, False, False]   # ties keep both, infeasible never Pareto
    rows = [_row("x", 1, 1, feasible=False)]
    mark_pareto(rows)
    assert rows[0].pareto is False


# ---------------------------------------------------------------- curves

def test_dvfs_single_point_and_perf_fallback():
    df = pd.DataFrame({"design": ["D"], "frequency_ghz": [2.0], "voltage_v": [0.8]})
    dv = DvfsCurve().fit(df)
    assert dv.params["D"] == (0.8, 0.0, 1) and dv.voltage("D", 3.0) == 0.8 and np.isnan(dv.voltage("Z", 1.0))
    perf = pd.DataFrame({"design": ["D"], "workload": ["w"], "frequency_ghz": [2.0], "throughput_gops": [50.0]})
    pm = PerfModel().fit(perf)
    assert pm.scaling_exponent("D", "w") == 1.0 and pm.predict("D", "w", 2.0) == pytest.approx(50.0)
    assert np.isnan(pm.predict("D", "w", 0.0)) and np.isnan(pm.predict("D", "zz", 2.0))


def test_timing_model_uses_history_when_latest_build_has_one_voltage():
    rows = []
    for b, v, d in (("B1", 0.7, 500.0), ("B1", 0.9, 400.0), ("B2", 0.8, 420.0)):
        rows.append({"design": "D", "build": b, "partition": "P", "voltage_v": v, "clock_period_ps": 450.0, "wns_ps": 450.0 - d})
    tm = TimingModel().fit(pd.DataFrame(rows))
    a, k, n = tm.params[("D", "P")]
    assert k < 0 and n == 3                                          # exponent from all history points
    assert 1000.0 / np.exp(a + k * np.log(0.8)) == pytest.approx(1000.0 / 420.0)   # anchored on the latest build
    assert tm.fmax("D", 0.9)[0] > tm.fmax("D", 0.7)[0]
    assert np.isnan(tm.fmax("Z", 0.8)[0]) and np.isnan(tm.fmax("D", 0.0)[0])


# ---------------------------------------------------------------- compact model / trace

def _compact_doc():
    return {
        "format": "powermet-compact-power-model", "version": 1, "model_kind": "linear",
        "terms": {"features": ["dyn_term", "leak_term"], "coefficients": {"dyn_term": 2.0, "leak_term": -50.0}, "intercept": 1.0, "fill": {}},
        "error_interval_pct": None,
        "designs": {"D": {
            "latest_build": "B1", "operating_points": {"nom": {"voltage_v": 1.0, "frequency_ghz": 1.0}},
            "fubs": {"D.P.A": {"fub": "A", "partition": "P", "physical": {"wire_cap_pf": 1.0, "cell_cap_pf": 1.0, "area": 1.0},
                               "workloads": {"w": {"activity": 0.5}}}},
            "dvfs": {"c0": 0.5, "c1": 0.25}, "perf": {"w": {"a": np.log(10.0), "b": 1.0, "r2": 1.0}},
            "timing": {"P": {"a": np.log(400.0), "k": -1.0}},
        }},
    }


def test_compact_model_matches_linear_and_does_not_clip():
    m = CompactPowerModel(_compact_doc())
    # dyn = 0.5 * 2 * 1 * 1 = 1 -> 2*1 - 50*1 + 1 = -47 : negative stays negative (LinearModel semantics)
    assert m.power("D", "w", 1.0, 1.0) == pytest.approx(-47.0)
    lin = LinearModel(["dyn_term", "leak_term"])
    lin.coef_, lin.intercept_ = np.array([2.0, -50.0]), 1.0
    assert lin.predict(m.rows("D", "w", 1.0, 1.0))[0] == pytest.approx(m.power("D", "w", 1.0, 1.0))
    assert m.voltage_for("D", 2.0) == 1.0 and m.throughput("D", "w", 2.0) == pytest.approx(20.0) and np.isnan(m.throughput("D", "w", 0))
    assert m.fmax("D", 1.0) == (pytest.approx(2.5), "P") and np.isnan(m.fmax("D", 0.0)[0]) and np.isnan(m.fmax("D", np.nan)[0])
    with pytest.raises(KeyError):
        m.operating_point("D", "turbo")


def test_run_trace_with_frequency_voltage_rows():
    m = CompactPowerModel(_compact_doc())
    trace = pd.DataFrame([
        {"design": "D", "interval": 0, "duration_s": 1e-3, "workload": "w", "frequency_ghz": 2.0, "voltage_v": 1.0},
        {"design": "D", "interval": 1, "duration_s": 1e-3, "workload": "w", "frequency_ghz": 4.0},    # voltage from DVFS (1.5 V -> Fmax 3.75)
        {"design": "D", "interval": 2, "duration_s": 1e-3, "workload": "w", "operating_point": "nom", "activity_scale": 2.0},
    ])
    res = run_trace(m, trace)
    tl = res.timeline
    assert tl["voltage_v"].tolist() == [1.0, 1.5, 1.0]
    assert tl["timing_ok"].tolist() == [True, False, True] and tl["critical_partition"].iloc[1] == "P"
    assert res.totals["timing_violations"] == 1 and res.totals["duration_s"] == pytest.approx(3e-3)


# ---------------------------------------------------------------- decomposition unit math

def test_pj_per_bit_mm_math():
    df = pd.DataFrame({"design": "D", "build": "B1", "fub": "A", "workload": "w", "operating_point": "o", "be_mw": [10.0],
                       "wire_cap_pf": [1.0], "cell_cap_pf": [1.0], "area": [1.0], "activity": [1.0], "voltage_v": [1.0],
                       "frequency_ghz": [1.0], "bits_per_cycle": [10.0], "avg_net_length_um": [1000.0]})
    m = LinearModel(["cell_dyn_term", "wire_dyn_term", "move_term", "leak_term"])
    m.coef_, m.intercept_ = np.array([0.0, 0.0, 1.0, 0.0]), 0.0     # power = move_term = 10 * 1000 * 1 * 1 = 1e4 mW
    dec = decompose(df, m)
    # 1e4 mW over 10 bits/cycle * 1 GHz * 1 mm = 1e-3*1e4 J/s / (1e10 bit/s * 1 mm) = 1e-9 J/(bit mm) = 1000 pJ/(bit mm)
    assert dec["pj_per_bit_mm"].iloc[0] == pytest.approx(1000.0)
    assert dec["movement_mw_share"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------- catalog / profiling

def test_catalog_empty_project(tmp_path):
    proj = Project(tmp_path / ".powermet")
    assert not db_path(proj).exists()
    assert query(proj, "SELECT 1").empty and tables(proj) == [] and profiles(proj) == []


def test_profiler_heap_and_linux_rss(monkeypatch):
    prof = Profiler("t", trace_heap=True)
    with prof.stage("alloc"):
        _ = [0] * 1_000_000
    assert prof.stages[0].heap_peak_mb is not None and prof.stages[0].heap_peak_mb > 1.0
    import resource

    class RU:
        ru_maxrss = 2048   # KB on Linux, bytes on macOS
    monkeypatch.setattr(resource, "getrusage", lambda _: RU())
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert peak_rss_mb() == 2.0
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert peak_rss_mb() == pytest.approx(2048 / 1024 / 1024)
