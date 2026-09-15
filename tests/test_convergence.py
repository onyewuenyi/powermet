"""Power convergence: leakage component through the adapters, Cdyn / leakage power derived metrics, targets on any
convergence metric, trend / projection verdicts, and closure plans that only count the techniques that move
the target's component."""

import numpy as np
import pandas as pd
import pytest

from powermet.budgets import Budget, check_budgets, load_budgets
from powermet.config import Config, Project
from powermet.convergence import converge, plan, render_convergence, render_plan, saving_in_metric
from powermet.demo import DemoSpec
from powermet.metrics import add_convergence_metrics, cdyn_pf
from powermet.mockdata import write_mock_runs
from powermet.pipeline import find_runs, ingest_runs
from powermet.sanitize import run_sanitize, sanitize
from powermet.storage import load_dataset, load_table
from powermet.techniques import BY_KEY, assess_all

SPEC = DemoSpec(n_designs=2, n_builds=5, n_fubs=10, seed=33, workloads=("idle", "typical"), operating_points=("nom", "turbo"))


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    root, data, log = write_mock_runs(root, SPEC, defects=False)
    proj = Project(tmp_path_factory.mktemp("proj") / ".powermet")
    ingest_runs(proj, find_runs(root))
    run_sanitize(proj)
    return proj, root, data


def test_cdyn_is_pf_exactly_and_nan_safe():
    # 100 mW dynamic at 1 V, 1 GHz -> 100 pF; V^2 f divides out the corner
    assert cdyn_pf([100.0], [1.0], [1.0]).iloc[0] == pytest.approx(100.0)
    assert cdyn_pf([100.0], [0.5], [2.0]).iloc[0] == pytest.approx(200.0)
    assert np.isnan(cdyn_pf([100.0], [0.0], [2.0]).iloc[0])
    df = pd.DataFrame({"be_mw": [10.0, 8.0], "be_leakage_mw": [2.0, np.nan], "fe_physical_mw": [9.0, 9.0], "fe_leakage_mw": [1.5, np.nan],
                       "voltage_v": [0.8, 0.8], "frequency_ghz": [2.0, 2.0]})
    add_convergence_metrics(df)
    assert df["be_dynamic_mw"].tolist()[0] == pytest.approx(8.0) and np.isnan(df["be_dynamic_mw"].iloc[1])
    assert df["cdyn_pf"].iloc[0] == pytest.approx(8.0 / (0.64 * 2.0)) and df["leakage_fraction"].iloc[0] == pytest.approx(0.2)
    assert df["fe_cdyn_pf"].iloc[0] == pytest.approx(7.5 / 1.28)


def test_leakage_component_round_trips_through_the_pipeline(project):
    proj, _, data = project
    df = load_dataset(proj, raw=True)
    assert {"be_leakage_mw", "fe_leakage_mw", "be_dynamic_mw", "cdyn_pf", "fe_cdyn_pf"} <= set(df.columns)
    gen = data.measurements.merge(df, on=["design", "build", "fub", "workload", "operating_point"], suffixes=("_gen", ""))
    assert len(gen) == len(df)
    assert np.allclose(gen["be_leakage_mw_gen"], gen["be_leakage_mw"], rtol=2e-3)
    assert (df["be_leakage_mw"] <= df["be_mw"] * 1.001).all()
    # leakage does not depend on the workload at a fixed corner -> the sanitize check stays quiet
    rep = sanitize(df, Config())
    assert rep.counts["leakage_suspect"] == 0
    # Cdyn is corner independent by construction: nom and turbo agree per FUB far better than raw power does
    piv = df[df.workload == "typical"].pivot_table(index=["design", "build", "fub"], columns="operating_point", values=["cdyn_pf", "be_mw"])
    cd = (piv["cdyn_pf"]["turbo"] / piv["cdyn_pf"]["nom"]).dropna()
    pw = (piv["be_mw"]["turbo"] / piv["be_mw"]["nom"]).dropna()
    assert abs(cd.median() - 1) < abs(pw.median() - 1)


def test_sanitize_flags_leakage_that_varies_with_workload(project):
    proj, _, _ = project
    df = load_dataset(proj, raw=True).copy()
    key = (df.design == df.design.iloc[0]) & (df.build == df.build.iloc[0]) & (df.fub == df.fub.iloc[0]) & (df.operating_point == "nom")
    idx = df.index[key & (df.workload == "idle")]
    df.loc[idx, "be_leakage_mw"] = df.loc[idx, "be_leakage_mw"] * 1.5           # same FUB, same corner, different leakage
    rep = sanitize(df, Config())
    assert rep.counts["leakage_suspect"] == int(key.sum())
    df.loc[idx, "be_leakage_mw"] = df.loc[idx, "be_mw"] * 1.2                    # leakage above the total is always wrong
    rep = sanitize(df, Config())
    assert rep.counts["leakage_suspect"] >= len(idx)


def test_targets_on_any_convergence_metric(tmp_path, project):
    toml = tmp_path / "t.toml"
    toml.write_text('[defaults]\nworkload = "typical"\noperating_point = "nom"\n'
                    '[[budget]]\ndesign = "GPU_A"\nscope = "design"\nbe_mw = 1.0\n'                      # legacy shorthand
                    '[[budget]]\ndesign = "GPU_A"\nscope = "design"\nmetric = "cdyn_pf"\ntarget = 1e9\n'
                    '[[budget]]\ndesign = "GPU_A"\nscope = "design"\nbe_leakage_mw = 1.0\n')             # metric-name shorthand
    budgets = load_budgets(toml)
    assert [b.metric for b in budgets] == ["be_mw", "cdyn_pf", "be_leakage_mw"] and budgets[0].target == 1.0 and budgets[0].be_mw == 1.0
    assert budgets[1].short_metric == "Cdyn" and budgets[1].unit == "pF"
    (tmp_path / "bad.toml").write_text('[[budget]]\ndesign="X"\nmetric="area"\ntarget=1\n')
    with pytest.raises(ValueError, match="not a convergence metric"):
        load_budgets(tmp_path / "bad.toml")
    proj, _, _ = project
    df = load_dataset(proj, raw=True)
    st = check_budgets(df, budgets, lineage=load_table(proj, "lineage"))
    assert [s.status for s in st] == ["OVER", "ON TRACK", "OVER"]
    sub = df[(df.design == "GPU_A") & (df.build == st[1].build) & (df.workload == "typical") & (df.operating_point == "nom")]
    assert st[1].actual == pytest.approx(sub["cdyn_pf"].sum()) and st[2].actual == pytest.approx(sub["be_leakage_mw"].sum())


def test_convergence_verdicts_and_projection(project):
    proj, _, _ = project
    df = load_dataset(proj, raw=True)
    b_cd = Budget("GPU_A", "design", 1e9, "typical", "nom", metric="cdyn_pf")           # far under -> converged
    b_lk = Budget("GPU_A", "design", 1.0, "typical", "nom", metric="be_leakage_mw")     # unreachable -> flat or diverging
    items = converge(df, [b_cd, b_lk])
    assert items[0].verdict == "CONVERGED" and items[0].gap < 0 and items[0].builds_to_target == 0
    assert items[1].verdict in ("FLAT", "DIVERGING", "CONVERGING") and items[1].gap > 0 and items[1].required_cut_pct > 99
    if items[1].verdict == "CONVERGING":
        assert items[1].builds_to_target == pytest.approx(items[1].gap / -items[1].trend_per_build)
    else:
        assert not np.isfinite(items[1].builds_to_target)
    text = render_convergence(items)
    assert "Cdyn" in text and "Leakage" in text and "never at this trend" in text or "builds" in text
    # a synthetic history that decreases steadily is CONVERGING with a finite projection
    hist = pd.DataFrame({"build": ["B001", "B002", "B003"], "milestone": ["signoff"] * 3, "actual": [130.0, 120.0, 110.0],
                         "tolerance_pct": [0.0] * 3, "margin_pct": [0.0] * 3, "status": ["OVER"] * 3})
    from powermet.budgets import BudgetStatus
    from powermet.convergence import Convergence, _verdict
    assert _verdict(10.0, -10.0, -10.0) == "CONVERGING" and _verdict(10.0, 0.0, -10.0) == "FLAT" and _verdict(10.0, 3.0, -10.0) == "DIVERGING"
    assert _verdict(-5.0, 3.0, 5.0) == "CONVERGED"


def test_plan_counts_only_the_techniques_that_move_the_component(project):
    proj, _, _ = project
    df = load_dataset(proj)
    raw = load_dataset(proj, raw=True)
    sub = raw[(raw.design == "GPU_A") & (raw.workload == "typical") & (raw.operating_point == "nom")]
    last = sorted(sub.build.unique())[-1]
    cd_now = float(sub[sub.build == last]["cdyn_pf"].sum())
    lk_now = float(sub[sub.build == last]["be_leakage_mw"].sum())
    budgets = [Budget("GPU_A", "design", cd_now * 0.9, "typical", "nom", metric="cdyn_pf"),
               Budget("GPU_A", "design", lk_now * 0.9, "typical", "nom", metric="be_leakage_mw"),
               Budget("GPU_A", "design", float(sub[sub.build == last]["be_mw"].sum()) * 0.9, "typical", "nom", metric="be_mw")]
    items = converge(raw, budgets)
    assert all(c.gap > 0 for c in items)
    ctx = {"design": "GPU_A", "workload": "typical", "operating_point": "nom"}
    results = assess_all(df, ctx)
    by = {r.technique: r for r in results}
    assert by["clock_gating"].saving_for("dynamic") > 0 and by["clock_gating"].saving_for("leakage") == 0
    assert by["vt_swap"].saving_for("leakage") > 0 and by["vt_swap"].saving_for("dynamic") == 0
    assert by["power_gating"].saving_for("dynamic") == 0 and by["power_gating"].saving_for("leakage") > 0
    assert ctx["leak_source"] == "measured" and "PrimePower leakage column" in by["vt_swap"].assumptions[-1]

    p_cd = plan(df, items[0], results, ctx)
    keys = [l.technique for l in p_cd.lines]
    assert "clock_gating" in keys and "wire_cap_reduction" in keys
    assert "vt_swap" not in keys and "dvfs" not in keys and "power_gating" not in keys
    v, f = p_cd.voltage_v, p_cd.frequency_ghz
    assert p_cd.lines[0].saving == pytest.approx(saving_in_metric(max(r.saving_for("dynamic") for r in results if r.assessable), "cdyn_pf", v, f))
    assert p_cd.lines[-1].cumulative == pytest.approx(sum(l.saving for l in p_cd.lines))
    assert p_cd.covered_pct == pytest.approx(min(100.0, p_cd.lines[-1].cumulative / items[0].gap * 100))
    assert any("changes the corner" in s for s in p_cd.not_applicable)

    p_lk = plan(df, items[1], results, ctx)
    keys = [l.technique for l in p_lk.lines]
    assert set(keys) <= {"vt_swap", "power_gating", "memory_low_power"} and "clock_gating" not in keys
    assert all(l.saving == r.saving_for("leakage") for l, r in ((l, by[l.technique]) for l in p_lk.lines))   # mW stays mW

    p_tot = plan(df, items[2], results, ctx)
    assert "dvfs" not in [l.technique for l in p_tot.lines]          # the target is at a fixed corner
    for p in (p_cd, p_lk, p_tot):
        txt = render_plan(p)
        assert "Closure plan" in txt and ("cover" in txt or "remainder" in txt)
    assert "mW -> pF" in render_plan(p_cd)


def test_techniques_catalog_states_what_each_moves():
    assert BY_KEY["clock_gating"].reduces == "dynamic" and BY_KEY["dvfs"].reduces == "corner" and BY_KEY["vt_swap"].reduces == "leakage"
    from powermet.techniques import render_catalog
    assert "Cdyn" in render_catalog() and "Leakage" in render_catalog()
