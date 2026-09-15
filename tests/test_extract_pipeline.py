"""V1: parsers, pipeline, lineage, sanitize, profiling."""

import numpy as np
import pandas as pd
import pytest

from powermet.config import Config, Project
from powermet.demo import DemoSpec
from powermet.extract import SOURCES
from powermet.extract.base import convert_unit, parse_header
from powermet.extract.metadata import inputs_from_metadata
from powermet.lineage import load_fub_map, resolve
from powermet.mockdata import write_mock_runs
from powermet.pipeline import extract_run, find_runs, ingest_runs
from powermet.profiling import Profiler, aggregate_stages
from powermet.sanitize import run_sanitize
from powermet.storage import load_dataset, load_table

SPEC = DemoSpec(n_designs=2, n_builds=4, n_fubs=10, seed=5, workloads=("typical", "compute"), operating_points=("nom", "turbo"))


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    root, data, log = write_mock_runs(root, SPEC, defects=True)
    return root, data, log


@pytest.fixture(scope="module")
def clean_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    root, data, log = write_mock_runs(root, SPEC, defects=False)
    return root, data


def test_unit_conversion():
    assert convert_unit(0.05, "W", "be_mw") == (50.0, "mW")
    assert convert_unit(1500, "fF", "wire_cap_pf")[0] == pytest.approx(1.5)
    assert convert_unit(2500, "MHz", "frequency_ghz")[0] == pytest.approx(2.5)


def test_header_multi_pair():
    h = parse_header(["Tool: PowerPro-RTL  Version: R-2025.06", "Mode: rtl"])
    assert h["tool"] == "PowerPro-RTL" and h["version"] == "R-2025.06" and h["mode"] == "rtl"


def test_every_parser_reads_mock(mock_root):
    root, data, _ = mock_root
    inputs, meta = inputs_from_metadata(root / "GPU_A" / "B001")
    for name, mod in SOURCES.items():
        files = mod.get_files(inputs)
        if not files and getattr(mod, "OPTIONAL", False):
            continue                       # e.g. voltus is only present on recent builds
        assert files, name
        rep = mod.parse(files[0].path, **files[0].context)
        assert rep.n_records > 0, name
        assert set(rep.records["unit"]) <= {"mW", "pF", "um2", "count", "ratio", "GHz", "V", "ops/cycle", "Gops/s", "um", "bits", "ps"}


def test_primepower_units_and_paths(mock_root):
    root, data, _ = mock_root
    # GPU_B is written in mW, GPU_A in W: both must land in mW and agree with the generator
    for design in ("GPU_A", "GPU_B"):
        run = root / design / "B001"
        rep = SOURCES["primepower"].parse(run / "primepower" / "typical_nom" / "power_hier.rpt")
        rec = rep.records
        assert set(rec["metric"]) == {"be_mw", "be_leakage_mw"}          # total and the leakage component
        leak = rec[rec["metric"] == "be_leakage_mw"].set_index("object")["value"]
        rec = rec[rec["metric"] == "be_mw"]
        assert rec["object"].str.startswith(design.lower() + "_top/").all()
        gen = data.measurements.query("design == @design and build == 'B001' and workload == 'typical' and operating_point == 'nom'")
        hier = data.hierarchy[data.hierarchy.design == design].set_index("fub")["be_hier"]
        merged = gen.assign(be_hier=gen["fub"].map(hier)).merge(rec, left_on="be_hier", right_on="object")
        assert len(merged) >= len(gen) - 3
        assert np.allclose(merged["be_mw"], merged["value"], rtol=2e-3)
        assert np.allclose(merged["be_leakage_mw"], merged["object"].map(leak), rtol=2e-3)   # leakage power round-trips too


def test_pipeline_roundtrip_clean(tmp_path, clean_root):
    root, data = clean_root
    proj = Project(tmp_path / ".powermet")
    summary = ingest_runs(proj, find_runs(root))
    assert summary.n_rejected == 0
    df = load_dataset(proj)
    assert len(df) == len(data.measurements)
    m = data.measurements.merge(df, on=["design", "build", "fub", "workload", "operating_point"], suffixes=("_gen", ""))
    for col in ("be_mw", "fe_logical_mw", "fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "frequency_ghz", "voltage_v",
                "wns_ps", "clock_period_ps", "avg_net_length_um"):
        assert np.allclose(m[col + "_gen"], m[col], rtol=5e-3), col
    # SAIF is written with per-net toggle jitter, so activity aggregates are close, not exact
    assert np.allclose(m["activity_gen"], m["activity"], rtol=0.06)
    assert np.allclose(m["bits_per_cycle_gen"], m["bits_per_cycle"], rtol=0.15)   # net count is rounded when writing
    assert (m["net_count"] >= 1).all()
    assert (m["model_root_gen"] == m["model_root"]).all() and (m["partition_gen"] == m["partition"]).all()
    assert np.allclose(m["fmax_ghz_gen"], m["fmax_ghz"], rtol=5e-3)
    assert df["run_id"].notna().all() and df["source_file"].notna().all()
    lin = load_table(proj, "lineage")
    assert lin["lineage_ok"].all()
    perf = load_table(proj, "performance")
    assert {"throughput_gops", "ipc", "frequency_ghz"} <= set(perf.columns)
    assert len(perf) == len(data.performance)


def test_pipeline_flags_defects_and_sanitize(tmp_path, mock_root):
    root, data, log = mock_root
    proj = Project(tmp_path / ".powermet")
    summary = ingest_runs(proj, find_runs(root))
    assert any(r.flags for r in summary.runs)
    rep, flagged = run_sanitize(proj)
    assert rep.n_usable < rep.n_rows
    assert rep.counts.get("stale_build", 0) > 0
    assert rep.counts.get("missing_physical", 0) > 0
    assert rep.counts.get("near_zero_be", 0) > 0
    assert rep.n_unit_conversions > 0
    assert rep.n_source_duplicates > 0
    assert rep.n_unmapped_objects > 0
    assert "Usable dataset" in rep.render()
    san = load_dataset(proj)
    assert len(san) == rep.n_usable
    raw = load_dataset(proj, raw=True)
    assert len(raw) == rep.n_rows
    # unmapped table holds the BE-renamed instances
    um = load_table(proj, "unmapped")
    assert um["object"].str.endswith("_r2").any()


def test_lineage_resolve_detects_missing():
    fub_map = pd.DataFrame({"fub": ["A", "B"], "fe_hier": ["top/a", "top/b"], "synth_object": ["A", "B"], "be_hier": ["top/a", "top/b"]})
    rec = pd.DataFrame({
        "object": ["top/a", "top/a", "top/b", "top/zzz"], "object_kind": ["fe_hier", "be_hier", "fe_hier", "be_hier"],
        "metric": ["fe_physical_mw", "be_mw", "fe_physical_mw", "be_mw"], "value": [1, 2, 3, 4],
        "unit": ["mW"] * 4, "unit_original": ["mW"] * 4, "source": ["pprtl", "primepower", "pprtl", "primepower"],
    })
    res = resolve(rec, fub_map, "D", "B1")
    assert len(res.unmapped) == 1
    b = res.lineage.set_index("fub").loc["B"]
    assert not b["lineage_ok"] and "be_hier_not_in_reports" in b["lineage_issues"]


def test_profiler_records_and_aggregates(tmp_path):
    prof = Profiler("t")
    with prof.stage("x", rows=10):
        sum(range(1000))
    with prof.stage("x", rows=5):
        pass
    d = prof.to_dict()
    assert d["stages"][0]["wall_s"] >= 0 and d["peak_rss_mb"] > 0
    agg = aggregate_stages(d["stages"])
    assert len(agg) == 1 and agg[0]["rows"] == 15 and agg[0]["calls"] == 2
    proj = Project(tmp_path / ".powermet")
    proj.init()
    assert prof.save(proj).exists()
