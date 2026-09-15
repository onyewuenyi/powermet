"""Comparative analysis (anomalies / power bugs), power groups, time-based profiles, ownership."""

import numpy as np
import pandas as pd
import pytest

from powermet.anomalies import RULES, anomalies, render_anomalies, render_rules
from powermet.config import Config, Project
from powermet.demo import DemoSpec
from powermet.extract import SOURCE_SPECS, power_groups, power_profile
from powermet.mockdata import write_mock_runs
from powermet.pipeline import extract_run, find_runs, ingest_runs
from powermet.sanitize import run_sanitize
from powermet.storage import load_dataset, load_table
from powermet.timeprofile import summarize_profile, render_profile


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    write_mock_runs(root / "mock", DemoSpec(n_designs=2, n_builds=6, n_fubs=12, workloads=("idle", "typical", "compute"),
                                            operating_points=("nom",)), defects=True)
    project = Project(root / ".powermet")
    cfg = project.init(Config())
    ingest_runs(project, find_runs(root / "mock"), cfg)
    run_sanitize(project, cfg)
    return project, cfg, load_dataset(project, cfg)


def test_power_groups_parser(tmp_path):
    txt = "\n".join([
        "*" * 40, "Report : power -hierarchy -groups", "Design : gpu_a_top", "Version: V-2024.09-SP3", "Run    : gpu_a_b001_r1",
        "Scenario: typical@nom", "Power Units = 1W", "*" * 40,
        "Hierarchy                       io  clock_network  register  combinational  memory  Total",
        "-" * 90,
        "gpu_a_top/part_p0/u_sched      0.0        0.012        0.020         0.030    0.000  0.062",
        "gpu_a_top/part_p0/u_l1         0.0        0.005        0.010         0.010    0.030  0.070",   # groups != total
    ])
    p = tmp_path / "power_groups.rpt"
    p.write_text(txt)
    rep = power_groups.parse(p)
    r = rep.records
    assert set(r["metric"]) == {"be_clock_mw", "be_register_mw", "be_comb_mw", "be_memory_mw"}
    assert r[(r.object == "gpu_a_top/part_p0/u_sched") & (r.metric == "be_clock_mw")]["value"].iloc[0] == pytest.approx(12.0)
    assert (r["unit"] == "mW").all() and (r["unit_original"] == "W").all()
    assert rep.workload == "typical" and rep.operating_point == "nom" and rep.run_id == "gpu_a_b001_r1"
    assert any("1 rows" in n for n in rep.notes)


def test_power_profile_parser(tmp_path):
    p = tmp_path / "power_profile.csv"
    p.write_text("\n".join(["# Tool: PrimePower  Version: V-2024.09  Run: r9  Scenario: compute@turbo",
                            "# Power units: W   Time units: us   Interval: 0.1",
                            "t_start_us,t_end_us,total_mw,dynamic_mw,leakage_mw",
                            "0,0.1,5.0,4.7,0.3", "0.1,0.2,6.0,5.7,0.3"]) + "\n")
    rep = power_profile.parse(p)
    r = rep.records
    assert rep.workload == "compute" and rep.operating_point == "turbo" and rep.run_id == "r9"
    assert set(r["metric"]) == {"profile_total_mw", "profile_dynamic_mw", "profile_leakage_mw"}
    tot = r[r.metric == "profile_total_mw"].sort_values("t_start_ns")
    assert list(tot["t_start_ns"]) == [0.0, 100.0] and list(tot["t_end_ns"]) == [100.0, 200.0]
    assert list(tot["value"]) == [5000.0, 6000.0]
    assert SOURCE_SPECS["power_profile"].optional and SOURCE_SPECS["power_groups"].optional


def test_owner_groups_and_profile_reach_the_dataset(project):
    _, cfg, df = project
    assert "owner" in df.columns and df["owner"].notna().all()
    assert df["owner"].str.startswith("rtl-").all()
    for c in ("be_clock_mw", "be_register_mw", "be_comb_mw", "be_memory_mw", "clock_fraction"):
        assert c in df.columns and df[c].notna().mean() > 0.9
    gsum = df[["be_clock_mw", "be_register_mw", "be_comb_mw", "be_memory_mw"]].sum(axis=1)
    # the vectorless idle scenario of the second design is a planted defect: its groups report came from the SAIF run
    ok = (df["be_mw"] > 1) & df["be_clock_mw"].notna() & (df["be_activity_mode"].astype(str) != "vectorless")
    assert np.allclose(gsum[ok], df.loc[ok, "be_mw"], rtol=0.02)
    vl = df["be_activity_mode"].astype(str) == "vectorless"
    assert vl.any() and (df.loc[vl, "quality_flags"].astype(str).str.contains("group_sum_mismatch")).mean() > 0.5
    prof = load_table(project[0], "power_profile")
    assert len(prof) and {"design", "build", "workload", "operating_point", "t_start_ns", "t_end_ns", "profile_total_mw", "run_id"} <= set(prof.columns)
    assert prof.groupby(["design", "build", "workload", "operating_point"]).size().min() == 40
    # the profile is not in the FUB dataset or the long table
    long = load_table(project[0], "measurements_long")
    assert not long["metric"].str.startswith("profile_").any()


def test_anomalies_find_planted_bugs(project):
    _, cfg, df = project
    d0, d1 = sorted(df["design"].unique())[:2]
    rep0 = anomalies(df, cfg, d0)
    rules0 = {f.rule for f in rep0.findings}
    assert "idle_dynamic" in rules0, rep0.findings
    idle = [f for f in rep0.findings if f.rule == "idle_dynamic"]
    assert all(f.owner and f.owner.startswith("rtl-") for f in idle) and all(f.technique == "clock_gating" for f in idle)
    assert any(f.rule == "clock_dominant" for f in rep0.findings)
    # the regression is planted from raw build index 2/3 (B005 of 6): new there, gone (cleared) at the next build
    at = anomalies(df, cfg, d0, build="B005")
    regs = [f for f in at.findings if f.rule == "unexplained_regression"]
    assert regs and regs[0].status == "new" and "vs " in regs[0].evidence and "explains" in regs[0].evidence
    after = anomalies(df, cfg, d0, build="B006")
    assert not [f for f in after.findings if f.rule == "unexplained_regression" and f.model_root == regs[0].model_root]
    assert any(f.model_root == regs[0].model_root and f.rule == "unexplained_regression" for f in after.cleared)
    rep1 = anomalies(df, cfg, d1)
    glitch = [f for f in rep1.findings if f.rule == "activity_power_mismatch"]
    assert len(glitch) == 1 and glitch[0].value > 2 and glitch[0].severity == "high"
    txt = render_anomalies(rep0)
    assert "By owner" in txt and "idle_dynamic" in txt
    only = anomalies(df, cfg, d0, rules=["leakage_share"])
    assert {f.rule for f in only.findings} <= {"leakage_share"}
    assert "activity_power_mismatch" in render_rules() and set(RULES) >= {"idle_dynamic", "creeping_growth", "replica_divergence"}


def test_anomaly_rules_skip_without_inputs(project):
    _, cfg, df = project
    d0 = sorted(df["design"].unique())[0]
    slim = df.drop(columns=["be_clock_mw", "be_register_mw", "be_comb_mw", "be_memory_mw", "clock_fraction"])
    rep = anomalies(slim, cfg, d0)
    assert "clock_dominant" in rep.skipped and "replica_divergence" in rep.skipped
    first = sorted(df[df.design == d0]["build"].unique())[0]
    rep = anomalies(df, cfg, d0, build=first)
    assert "unexplained_regression" in rep.skipped and "creeping_growth" in rep.skipped


def test_profile_summary_and_mismatch(project):
    proj, cfg, df = project
    prof = load_table(proj, "power_profile")
    perf = load_table(proj, "performance")
    designs = sorted(df["design"].unique())
    s = summarize_profile(prof, designs[0], wide=df, perf=perf)
    t = s.table
    assert set(t["workload"]) == {"idle", "typical", "compute"}
    comp = t[t.workload == "compute"].iloc[0]
    assert comp.peak_to_avg > 1.15 and comp.peak_mw > comp.avg_mw > comp.min_mw
    assert comp.energy_uj == pytest.approx(comp.avg_mw * comp.duration_ns * 1e-6)
    assert s.peak_workload == "compute"
    assert t["profile_ok"].all() and (t["avg_gap_pct"].abs() < 5).all()      # gap = FUBs the map lost, not the profile
    assert np.isfinite(comp.energy_pj_per_op)
    txt = render_profile(s)
    assert "Peak-power vector: compute" in txt


def test_profile_mismatch_flagged(tmp_path):
    root = tmp_path / "mock"
    write_mock_runs(root, DemoSpec(n_designs=3, n_builds=2, n_fubs=6, workloads=("typical",), operating_points=("nom",)), defects=True)
    project = Project(tmp_path / ".powermet")
    cfg = project.init(Config())
    ingest_runs(project, find_runs(root), cfg)
    df = load_dataset(project, cfg, raw=True)
    prof = load_table(project, "power_profile")
    d2 = sorted(df["design"].unique())[2]
    s = summarize_profile(prof, d2, wide=df)
    assert not s.table["profile_ok"].all()
    assert "MISMATCH" in render_profile(s)


def test_group_sum_mismatch_check(project):
    proj, cfg, df = project
    from powermet.sanitize import sanitize
    base = sanitize(df, cfg).counts["group_sum_mismatch"]
    bad = df.copy()
    idx = bad.index[bad["be_activity_mode"].astype(str) != "vectorless"][:5]
    bad.loc[idx, "be_clock_mw"] = bad.loc[idx, "be_mw"] * 0.9
    rep = sanitize(bad, cfg)
    assert rep.counts["group_sum_mismatch"] == base + 5
