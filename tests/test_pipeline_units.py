"""Pipeline building blocks and append-mode regressions."""

import numpy as np
import pandas as pd
import pytest

from powermet.config import Config, Project
from powermet.demo import DemoSpec
from powermet.identity import ModelRoot
from powermet.lineage import (BE_NOT_IN_REPORTS, DESIGN_FUB, NO_PARTITION, TIMING_MISSING, lineage_table, load_fub_map,
                              render_chain, resolve, resolve_objects)
from powermet.metrics import add_derived_metrics, fmax_from_timing, prediction_metrics
from powermet.mockdata import write_mock_runs
from powermet.pipeline import (RunResult, attach_identity, extract_run, find_partitions, find_runs, ingest_runs, join_metric,
                               merge_partitions, pivot_fub_records, pivot_perf_records, plan_jobs, verify_run_consistency)
from powermet.sanitize import run_sanitize
from powermet.storage import load_dataset, load_table


def _records():
    rows = [
        ("top/u_a", "fe_hier", "fe_physical_mw", 10.0, "typical", "nom"),
        ("top/u_a", "be_hier", "be_mw", 12.0, "typical", "nom"),
        ("top/u_a", "be_hier", "be_mw", 99.0, "typical", "nom"),          # duplicate source row -> first wins
        ("top/u_a", "be_hier", "wire_cap_pf", 3.0, None, None),
        ("top/u_a", "be_hier", "cell_cap_pf", 1.0, None, None),
        ("top/u_a", "be_hier", "area", 5.0, None, None),
        ("top/u_a", "be_hier", "activity", 0.4, "typical", None),
        ("top/part_p0", "partition", "wns_ps", -5.0, None, "nom"),
        ("top/part_p0", "partition", "clock_period_ps", 400.0, None, "nom"),
        ("*", "design", "frequency_ghz", 2.5, None, "nom"),
        ("*", "design", "voltage_v", 0.8, None, "nom"),
        ("*", "design", "throughput_gops", 100.0, "typical", "nom"),
        ("top/u_zz", "be_hier", "be_mw", 1.0, "typical", "nom"),          # not in the map
    ]
    df = pd.DataFrame(rows, columns=["object", "object_kind", "metric", "value", "workload", "operating_point"])
    df["source"] = df["object_kind"].map({"fe_hier": "pprtl", "be_hier": "primepower", "partition": "primetime", "design": "metadata"})
    df.loc[df["metric"] == "activity", "source"] = "saif"
    df["unit"] = "x"
    return df


def _model():
    return ModelRoot.from_frame(pd.DataFrame([
        {"fub": "A", "model_root": "D.P0.A", "partition": "P0", "fe_hier": "top/u_a", "synth_object": "A", "be_hier": "top/u_a"},
        {"fub": "B", "model_root": "D.P1.B", "partition": "P1", "fe_hier": "top/u_b", "synth_object": "B", "be_hier": "top/u_b"},
    ]), "D")


def test_resolve_objects_fanout_and_unmapped():
    mapped, unmapped = resolve_objects(_records(), _model())
    assert set(unmapped["object"]) == {"top/u_zz"}
    assert (mapped[mapped["object_kind"] == "design"]["fub"] == DESIGN_FUB).all()
    assert mapped[mapped["metric"] == "wns_ps"]["fub"].tolist() == ["A"]      # P0 has one member


def test_lineage_table_issue_codes():
    mapped, _ = resolve_objects(_records(), _model())
    table, flags = lineage_table(mapped, _model(), "D", "B1", has_timing_source=True)
    t = table.set_index("fub")
    assert t.loc["A", "lineage_ok"] and t.loc["A", "sources"].startswith("activity:saif")
    assert BE_NOT_IN_REPORTS in t.loc["B", "lineage_issues"] and TIMING_MISSING in t.loc["B", "lineage_issues"]
    txt = render_chain(t.reset_index().iloc[1])
    assert "LINEAGE ISSUES" in txt and "MODEL ROOT          D.P1.B" in txt
    # no partition column at all -> NO_PARTITION, never TIMING_MISSING
    m2 = ModelRoot.from_frame(_model().to_frame().drop(columns=["partition", "model_root"]), "D")
    table2, _ = lineage_table(mapped, m2, "D", "B1", has_timing_source=True)
    assert all(NO_PARTITION in x and TIMING_MISSING not in x for x in table2["lineage_issues"])


def test_resolve_accepts_frame_and_flags_unmapped(tmp_path):
    frame = _model().to_frame().drop(columns=["model_root", "partition"])
    frame.to_csv(tmp_path / "map.csv", index=False)
    res = resolve(_records(), load_fub_map(tmp_path / "map.csv"), "D", "B1")
    assert any("not in the FUB map" in f for f in res.flags)
    assert (res.lineage["model_root"] == res.lineage["fub"]).all()


def test_pivot_and_join_semantics():
    mapped, _ = resolve_objects(_records(), _model())
    wide = pivot_fub_records(mapped)
    assert len(wide) == 1
    r = wide.iloc[0]
    assert r["be_mw"] == 12.0                                 # first duplicate kept
    assert r["wire_cap_pf"] == 3.0 and r["activity"] == 0.4   # build-level and workload-level broadcast
    assert r["frequency_ghz"] == 2.5 and r["voltage_v"] == 0.8  # design-level joined on operating point
    assert r["wns_ps"] == -5.0
    assert np.isnan(r["fanout"])                              # absent metric -> NaN column, not missing
    wide = attach_identity(wide, _model())
    assert list(wide.columns[:3]) == ["fub", "model_root", "partition"]
    assert fmax_from_timing(wide["clock_period_ps"], wide["wns_ps"]).iloc[0] == pytest.approx(1000 / 405)
    perf = pivot_perf_records(mapped, "D", "B1")
    assert perf.iloc[0]["throughput_gops"] == 100.0 and perf.iloc[0]["frequency_ghz"] == 2.5


def test_pivot_without_power_records_keeps_fubs():
    rec = _records()
    rec = rec[~rec["metric"].isin(["be_mw", "fe_physical_mw"])]
    mapped, _ = resolve_objects(rec, _model())
    wide = pivot_fub_records(mapped)
    assert wide["fub"].tolist() == ["A"] and np.isnan(wide.iloc[0]["be_mw"]) and wide.iloc[0]["wire_cap_pf"] == 3.0


def test_join_metric_scope_less_design_metric_broadcasts():
    base = pd.DataFrame({"fub": ["A", "B"], "workload": ["w", "w"], "operating_point": ["o", "o"]})
    rec = pd.DataFrame([{"fub": "*", "metric": "cell_count", "value": 5.0, "workload": None, "operating_point": None}])
    out = join_metric(base, rec, [])
    assert (out["cell_count"] == 5.0).all()


def test_metrics_helpers():
    df = pd.DataFrame({"fe_logical_mw": [1.0], "fe_physical_mw": [1.0], "be_mw": [2.0], "clock_period_ps": [400.0], "wns_ps": [-100.0]})
    assert add_derived_metrics(df)["fmax_ghz"].iloc[0] == pytest.approx(2.0)
    assert prediction_metrics([np.nan], [np.nan]) == {"n": 0}
    assert np.isnan(fmax_from_timing([400.0], [500.0]).iloc[0])


SPEC = DemoSpec(n_designs=1, n_builds=3, n_fubs=8, seed=3, workloads=("typical",), operating_points=("nom", "turbo"))


def test_append_is_idempotent_and_never_drops_raw_rows(tmp_path):
    root, data, _ = write_mock_runs(tmp_path / "runs", SPEC, defects=True)
    runs = find_runs(root)
    proj = Project(tmp_path / ".powermet")
    ingest_runs(proj, runs[:2])
    n_raw = len(load_dataset(proj, raw=True))
    n_long = len(load_table(proj, "measurements_long"))
    rep, _ = run_sanitize(proj)
    assert rep.n_usable < rep.n_rows                     # some rows are flagged
    # re-append an already ingested run: nothing grows
    ingest_runs(proj, runs[:1], replace=False)
    assert len(load_dataset(proj, raw=True)) == n_raw
    assert len(load_table(proj, "measurements_long")) == n_long
    assert len(load_table(proj, "lineage")) == len(load_table(proj, "lineage").drop_duplicates(["design", "build", "fub"]))
    # append a new build after sanitize: raw rows from before are all still there
    ingest_runs(proj, runs[2:], replace=False)
    raw = load_dataset(proj, raw=True)
    assert len(raw) > n_raw and set(raw["build"]) == {"B001", "B002", "B003"}
    assert len(raw[raw["build"] != "B003"]) == n_raw
    # the sanitized view is invalidated by an ingest
    assert len(load_dataset(proj)) == len(raw)
    # catalog stays idempotent on re-ingest of the same build
    from powermet.catalog import query
    assert len(query(proj, "SELECT * FROM build")) == 3
    sf = query(proj, "SELECT design, build, COUNT(*) n FROM source_file GROUP BY 1,2")
    first = query(proj, "SELECT COUNT(*) n FROM source_file WHERE build = 'B001'")["n"].iloc[0]
    assert first == sf[sf["build"] == "B001"]["n"].iloc[0]           # re-ingest did not duplicate B001's files


def test_verify_run_consistency():
    from powermet.extract.base import ParsedReport
    from pathlib import Path
    ok = ParsedReport("pprtl", Path("a.rpt"), "t", "1", pd.DataFrame(), run_id="r1", build="B1")
    bad_run = ParsedReport("starrc", Path("rc.rpt"), "t", "1", pd.DataFrame(), run_id="r0")
    bad_build = ParsedReport("implementation", Path("q.rpt"), "t", "1", pd.DataFrame(), build="B9")
    none = ParsedReport("perf", Path("p.csv"), "t", "1", pd.DataFrame())
    problems = verify_run_consistency([ok, bad_run, bad_build, none], {"run_id": "r1", "build": "B1"})
    assert len(problems) == 2 and "rc.rpt" in problems[0] and "q.rpt" in problems[1]
    assert verify_run_consistency([ok], {}) == []


def test_stale_signoff_report_is_excluded(tmp_path):
    root, data, log = write_mock_runs(tmp_path / "runs", SPEC, defects=True)
    assert any("different run id" in l for l in log)
    run = root / data.measurements["design"].iloc[0] / "B001"
    res = extract_run(run, Config())
    assert any("run id" in e for e in res.errors)
    assert not any(r.source == "starrc" for r in res.reports)      # dropped under strict_consistency
    assert res.wide["wire_cap_pf"].isna().all()
    res2 = extract_run(run, Config(strict_consistency=False))
    assert any(r.source == "starrc" for r in res2.reports) and res2.errors


def test_aggregate_rows_are_not_unmapped():
    rec = pd.DataFrame([
        {"object": "top", "object_kind": "be_hier", "metric": "be_mw", "value": 100.0},
        {"object": "top/part_p0", "object_kind": "be_hier", "metric": "be_mw", "value": 50.0},
        {"object": "top/u_zz", "object_kind": "be_hier", "metric": "be_mw", "value": 1.0},
    ])
    model = ModelRoot.from_frame(pd.DataFrame([{"fub": "A", "partition": "P0", "fe_hier": "top/u_a", "synth_object": "A",
                                                "be_hier": "top/part_p0/u_a"}]), "D")
    mapped, unmapped = resolve_objects(rec, model)
    assert len(mapped) == 0 and unmapped["object"].tolist() == ["top/u_zz"]


def test_partitioned_ingest_matches_serial(tmp_path):
    root, data, _ = write_mock_runs(tmp_path / "runs", SPEC, defects=True)
    runs = find_runs(root)
    serial = Project(tmp_path / "serial" / ".powermet")
    ingest_runs(serial, runs)
    a = load_dataset(serial, raw=True)
    # worker mode: one partition per run, no dataset or catalog touched
    fan = Project(tmp_path / "fan" / ".powermet")
    fan.init()
    pdir = fan.root / "data" / "processed" / "runs"
    jobs = plan_jobs(runs, pdir, submit_cmd="submit -J pm-{design}-{build} {cmd}")
    assert len(jobs) == len(runs) and jobs[0].startswith("submit -J pm-") and "--partition-dir" in jobs[0]
    for rd in runs:
        res = extract_run(rd, Config())
        res.save_partition(pdir)
    assert not fan.dataset_path().exists() and len(find_partitions(pdir)) == len(runs)
    back = RunResult.load_partition(find_partitions(pdir)[0])
    assert back.source_rows() and back.source_rows()[0]["sha256"]
    summary = merge_partitions(fan, pdir)
    b = load_dataset(fan, raw=True)
    cols = [c for c in a.columns if c != "imported_at"]
    key = ["design", "build", "fub", "workload", "operating_point"]
    pd.testing.assert_frame_equal(a[cols].sort_values(key).reset_index(drop=True), b[cols].sort_values(key).reset_index(drop=True))
    for name in ("measurements_long", "lineage", "unmapped", "performance", "power_intent"):
        assert len(load_table(serial, name)) == len(load_table(fan, name)), name
    from powermet.catalog import query
    assert len(query(serial, "SELECT * FROM source_file")) == len(query(fan, "SELECT * FROM source_file"))
    assert summary.n_rows == len(b)
