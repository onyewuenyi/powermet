"""Timing (PrimeTime at partition level), energy decomposition, deltas, frontier, catalog, integration."""

import json

import numpy as np
import pandas as pd
import pytest

from powermet.catalog import db_path, import_history, profiles, query, tables
from powermet.config import Config, Project
from powermet.demo import DemoSpec, generate_all
from powermet.decomposition import decompose
from powermet.deltas import build_deltas, classify
from powermet.frontier import frontier
from powermet.explore import Explorer
from powermet.extract import SOURCES
from powermet.features import add_engineered_features
from powermet.integrate import CompactPowerModel, export_compact, run_trace
from powermet.lineage import load_fub_map, resolve
from powermet.metrics import add_derived_metrics
from powermet.mockdata import write_mock_runs
from powermet.modeling import load, save, train
from powermet.pipeline import find_runs, ingest_runs
from powermet.sanitize import metric_quality, run_sanitize
from powermet.storage import load_dataset, load_table
from powermet.curves import TimingModel

SPEC = DemoSpec(n_designs=2, n_builds=5, n_fubs=16, seed=11, workloads=("typical", "compute"), operating_points=("eco", "nom", "turbo"))


@pytest.fixture(scope="module")
def data():
    d = generate_all(SPEC)
    d.measurements = add_engineered_features(add_derived_metrics(d.measurements))
    return d


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    root, data, _ = write_mock_runs(root, SPEC, defects=True)
    proj = Project(tmp_path_factory.mktemp("proj") / ".powermet")
    ingest_runs(proj, find_runs(root))
    run_sanitize(proj)
    cfg = proj.load_config()
    df = load_dataset(proj, cfg)
    res = train(df, cfg, cv=True)
    save(proj, res, cfg)
    return proj, root


def test_primetime_parser_and_partition_fanout(project):
    proj, root = project
    run = root / "GPU_A" / "B001"
    rep = SOURCES["primetime"].parse(run / "primetime" / "nom" / "timing_summary.rpt", operating_point="nom")
    assert rep.tool == "PrimeTime" and set(rep.records["metric"]) == {"clock_period_ps", "wns_ps", "tns_ps", "violating_endpoints"}
    assert rep.records["object"].str.contains("/part_").all()
    fub_map = load_fub_map(run / "mapping" / "fub_map.csv")
    assert {"model_root", "partition"} <= set(fub_map.columns)
    rec = rep.records.copy()
    rec["source"] = "primetime"
    res = resolve(rec, fub_map, "GPU_A", "B001")
    # every FUB received the timing of its partition
    assert set(res.mapped["fub"]) == set(fub_map["fub"])
    per_fub = res.mapped[res.mapped["metric"] == "wns_ps"].groupby("fub")["value"].nunique()
    assert (per_fub == 1).all()


def test_dataset_has_timing_and_identity(project):
    proj, _ = project
    df = load_dataset(proj)
    for c in ("model_root", "partition", "wns_ps", "clock_period_ps", "fmax_ghz", "bits_per_cycle", "avg_net_length_um", "move_term"):
        assert c in df.columns, c
    assert df["model_root"].str.count(r"\.").eq(2).all()
    assert (df["fmax_ghz"].dropna() > 0).all()
    lin = load_table(proj, "lineage")
    assert "partition_timing_missing" in ";".join(lin["lineage_issues"].astype(str))


def test_catalog_records(project):
    proj, _ = project
    assert db_path(proj).exists()
    assert {"build", "source_file", "import_run", "quality_run", "model", "profile_run"} <= set(tables(proj))
    b = query(proj, "SELECT * FROM build")
    assert len(b) == SPEC.n_designs * SPEC.n_builds and (b["status"] == "superseded").sum() == 1
    sf = query(proj, "SELECT source, COUNT(*) n FROM source_file GROUP BY source")
    assert "primetime" in set(sf["source"])
    assert len(import_history(proj)) == SPEC.n_designs * SPEC.n_builds
    assert profiles(proj, last=10)
    assert len(query(proj, "SELECT * FROM model")) == 1
    q = query(proj, "SELECT * FROM quality_run")
    assert len(q) == 1 and json.loads(q["metric_quality"].iloc[0])


def test_metric_quality_table(project):
    proj, _ = project
    df = load_dataset(proj, raw=True)
    long = load_table(proj, "measurements_long")
    mq = metric_quality(df, long)
    assert {"metric", "coverage_pct", "r_target", "verdict"} <= set(mq.columns)
    fe = mq.set_index("metric").loc["fe_physical_mw"]
    assert fe["verdict"] == "trusted" and fe["r_target"] > 0.9
    wc = mq.set_index("metric").loc["wire_cap_pf"]
    assert wc["coverage_pct"] < 100          # injected missing StarRC rows


def test_datamove_model_and_decomposition(project):
    proj, _ = project
    payload, meta = load(proj)
    assert "datamove" in payload["models"]
    cv = meta["cv"]["summary"]
    assert cv["datamove"]["mape_mean"] < cv["baseline"]["mape_mean"]
    df = load_dataset(proj)
    dec = decompose(df[df["build"] == "B005"], payload["models"]["datamove"])
    shares = dec[["compute_mw_share", "wire_mw_share", "movement_mw_share", "leakage_mw_share"]].sum(axis=1)
    assert np.allclose(shares.dropna(), 1.0)
    assert dec["movement_mw_share"].mean() > 0.05
    assert np.allclose(dec["predicted_mw"], payload["models"]["datamove"].predict(add_engineered_features(df[df["build"] == "B005"])))


def test_classify_and_build_deltas(data):
    assert classify(-2.0, 5.0) == "pareto improvement"
    assert classify(3.0, -5.0) == "regression"
    assert classify(3.0, 5.0) == "power-for-performance trade"
    assert classify(-3.0, -5.0) == "performance-for-power trade"
    assert classify(0.1, 0.2) == "neutral"
    assert classify(1.0, float("nan")) == "timing unknown"
    deltas = build_deltas(data.measurements, "GPU_A", "typical", "nom")
    assert len(deltas) == SPEC.n_builds - 1
    d = deltas[-1]
    assert "wns_ps" in d.totals and len(d.partition_table) and len(d.fub_table)
    assert d.classification in {"pareto improvement", "regression", "power-for-performance trade", "performance-for-power trade", "neutral"}
    # closure effort in the generator: timing improves across builds overall
    first, last = deltas[0].totals["wns_ps"][0], deltas[-1].totals["wns_ps"][1]
    assert last > first


def test_frontier(data):
    pts = frontier(data.measurements, "GPU_A", "typical", "nom", data.performance)
    assert len(pts) == SPEC.n_builds and any(p.pareto for p in pts)
    assert all(np.isfinite(p.fmax_ghz) and np.isfinite(p.throughput_gops) for p in pts)


def test_timing_model_feasibility(data):
    tm = TimingModel().fit(data.measurements)
    assert tm.params
    f_lo, _ = tm.fmax("GPU_A", 0.70)
    f_hi, crit = tm.fmax("GPU_A", 0.90)
    assert f_hi > f_lo and crit


def test_explore_marks_timing_violation(project):
    proj, _ = project
    payload, meta = load(proj)
    df = load_dataset(proj)
    ex = Explorer(df, load_table(proj, "performance"), payload, meta, "datamove")
    f0 = float(ex.base_rows("GPU_A", "compute", "nom")["frequency_ghz"].iloc[0])
    rows = ex.sweep("GPU_A", "compute", "frequency_ghz", [f0 * 1.6], operating_point="nom", fixed_voltage=True)
    assert rows[0].feasible and not rows[1].feasible and rows[1].critical_partition
    assert not rows[1].pareto


def test_compact_export_and_trace(project, tmp_path):
    proj, root = project
    payload, meta = load(proj)
    df = load_dataset(proj)
    doc = export_compact(df, load_table(proj, "performance"), payload, meta, "datamove")
    path = tmp_path / "compact.json"
    path.write_text(json.dumps(doc))
    m = CompactPowerModel.load(path)
    f, v = m.operating_point("GPU_A", "nom")
    p_nom = m.power("GPU_A", "compute", f, v)
    p_scaled = m.power("GPU_A", "compute", f, v, activity_scale=1.2)
    assert p_scaled > p_nom > 0
    # compact model reproduces the full model on the same rows
    rows = df[(df.design == "GPU_A") & (df.build == "B005") & (df.workload == "compute") & (df.operating_point == "nom")]
    assert abs(p_nom - payload["models"]["datamove"].predict(add_engineered_features(rows)).sum()) / p_nom < 0.02
    trace = pd.read_csv(root / "traces" / "GPU_A_phases.csv")
    res = run_trace(m, trace)
    assert len(res.timeline) == len(trace) and res.totals["energy_mj"] > 0
    assert np.isfinite(res.totals["energy_pj_per_op"])
    tl = res.timeline
    # feasibility is consistent with the exported partition timing model (turbo phases may legitimately violate)
    expected = (~np.isfinite(tl["fmax_ghz"])) | (tl["frequency_ghz"] <= tl["fmax_ghz"] * 1.005)
    assert (tl["timing_ok"] == expected).all() and res.totals["timing_violations"] == int((~tl["timing_ok"]).sum())
