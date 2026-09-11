"""V2: models, CV, what-if. V3: workload, perf, exploration."""

import numpy as np
import pandas as pd
import pytest

from powermet.config import Config, Project
from powermet.demo import DemoSpec, generate_all
from powermet.explore import Explorer, Scenario, load_scenarios
from powermet.features import add_engineered_features
from powermet.metrics import add_derived_metrics
from powermet.modeling import cross_validate_builds, save, train
from powermet.whatif import Override, parse_override, run_whatif, select_rows
from powermet.curves import DvfsCurve, PerfModel
from powermet.workload import energy_pj_per_op, summarize_workloads

SPEC = DemoSpec(n_designs=2, n_builds=5, n_fubs=15, seed=9, workloads=("idle", "typical", "compute"), operating_points=("eco", "nom", "turbo"))


@pytest.fixture(scope="module")
def data():
    d = generate_all(SPEC)
    d.measurements = add_engineered_features(add_derived_metrics(d.measurements))
    return d


@pytest.fixture(scope="module")
def trained(data):
    cfg = Config()
    res = train(data.measurements, cfg, cv=True)
    return cfg, res


def test_physics_model_beats_baseline_out_of_build(trained):
    cfg, res = trained
    mt = res.metrics_test
    assert mt["physics"]["mape"] < mt["baseline"]["mape"]
    assert mt["physics"]["mape"] < mt["scaled"]["mape"]
    assert "tree" in mt


def test_cv_has_folds_and_intervals(trained):
    cfg, res = trained
    cv = res.cv
    assert cv["strategy"] == "leave-one-build-out" and len(cv["folds"]) == SPEC.n_builds
    iv = cv["intervals"]["physics"]
    assert iv["p05"] < iv["p50"] < iv["p95"]
    assert cv["summary"]["physics"]["mape_mean"] < cv["summary"]["baseline"]["mape_mean"]


def test_cv_refuses_too_few_builds(data):
    df = data.measurements[data.measurements["build"].isin(["B001", "B002"])]
    cv = cross_validate_builds(df, Config())
    assert cv["folds"] == [] and "note" in cv


def test_tree_is_monotone_in_wire_cap(data, trained):
    cfg, res = trained
    rows = select_rows(data.measurements, "GPU_A", None, None, "typical", "nom").head(20)
    lo = add_engineered_features(Override("wire_cap_pf", "scale", 0.8).apply(rows))
    hi = add_engineered_features(Override("wire_cap_pf", "scale", 1.2).apply(rows))
    tree = res.models["tree"]
    assert (tree.predict(hi) >= tree.predict(lo) - 1e-9).all()


def test_whatif_direction_and_interval(tmp_path, data, trained):
    cfg, res = trained
    proj = Project(tmp_path / ".powermet")
    proj.init(cfg)
    save(proj, res, cfg)
    from powermet.modeling import load
    payload, meta = load(proj)
    rows = select_rows(data.measurements, "GPU_A", None, "Scheduler", "typical", "nom")
    assert rows["build"].iloc[0] == "B005"
    r = run_whatif(rows, [parse_override("wire-cap=0.8", "scale")], payload, meta, "physics")
    assert r.delta_pct < 0 and r.interval is not None
    r2 = run_whatif(rows, [Override("frequency_ghz", "set", float(rows["frequency_ghz"].iloc[0]) * 1.2)], payload, meta, "physics")
    assert r2.delta_pct > 0


def test_energy_and_perf_model(data):
    e = energy_pj_per_op([1000.0, 500.0, 10.0], [100.0, 0.0, np.nan])
    assert e[0] == 10.0 and np.isnan(e[1]) and np.isnan(e[2])
    pm = PerfModel().fit(data.performance)
    b_mem = pm.scaling_exponent("GPU_A", "compute")
    assert 0.5 < b_mem <= 1.05
    dv = DvfsCurve().fit(data.measurements)
    assert dv.voltage("GPU_A", 2.0) > dv.voltage("GPU_A", 1.5)


def test_workload_summary(data):
    ws = summarize_workloads(data.measurements, data.performance)
    assert ws.latest["energy_pj_per_op"].notna().all()
    assert len(ws.sensitivity) and (ws.sensitivity["dynamic_range"] >= 1).all()
    # idle should be the cheapest, compute the most expensive per design
    lat = ws.latest[ws.latest.operating_point == "nom"].groupby(["design", "workload"])["be_mw"].sum().unstack()
    assert (lat["idle"] < lat["typical"]).all() and (lat["typical"] < lat["compute"]).all()


def test_explore_sweep_and_scenarios(tmp_path, data, trained):
    cfg, res = trained
    proj = Project(tmp_path / ".powermet")
    proj.init(cfg)
    save(proj, res, cfg)
    from powermet.modeling import load
    payload, meta = load(proj)
    ex = Explorer(data.measurements, data.performance, payload, meta, "physics")
    f0 = float(ex.base_rows("GPU_A", "typical", None)["frequency_ghz"].iloc[0])
    rows = ex.sweep("GPU_A", "typical", "frequency_ghz", [round(f0 * 0.8, 2), round(f0 * 1.2, 2)])
    assert rows[0].scenario == "baseline" and len(rows) == 3
    assert rows[1].power_mw < rows[0].power_mw < rows[2].power_mw
    assert rows[1].throughput_gops < rows[2].throughput_gops
    assert rows[1].voltage_v < rows[2].voltage_v          # DVFS followed
    assert any(r.pareto for r in rows)
    toml = tmp_path / "s.toml"
    toml.write_text('[defaults]\ndesign = "GPU_A"\n[[scenario]]\nname = "wc"\nscale = { wire_cap_pf = 0.8 }\n[[scenario]]\nname = "f"\nfrequency_ghz = 2.2\nvoltage_v = 0.8\n')
    scs = load_scenarios(toml)
    assert len(scs) == 2 and scs[0].overrides[0].feature == "wire_cap_pf"
    out = ex.scenarios(scs)
    assert len(out) == 3 and out[1].delta_power_pct < 0
