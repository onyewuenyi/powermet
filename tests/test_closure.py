"""Power-closure infrastructure: budgets, hotspots, qualification, power intent (UPF), activity provenance."""

import numpy as np
import pandas as pd
import pytest

from powermet.budgets import Budget, check_budgets, classify_status, load_budgets
from powermet.config import Config, Project
from powermet.demo import DemoSpec
from powermet.extract import SOURCES, voltus, pprtl
from powermet.hotspots import hotspots
from powermet.identity import ModelRoot
from powermet.intent import intent_table, parse_upf
from powermet.mockdata import write_mock_runs
from powermet.pipeline import find_runs, ingest_runs
from powermet.qualify import qualify
from powermet.sanitize import run_sanitize
from powermet.storage import load_dataset, load_table

SPEC = DemoSpec(n_designs=2, n_builds=4, n_fubs=10, seed=21, workloads=("idle", "typical"), operating_points=("nom", "turbo"))


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    root, data, log = write_mock_runs(root, SPEC, defects=True)
    proj = Project(tmp_path_factory.mktemp("proj") / ".powermet")
    ingest_runs(proj, find_runs(root))
    run_sanitize(proj)
    return proj, root, log


UPF = """upf_version 2.1
create_supply_port VDD_A
create_supply_net VDD_A
create_power_domain PD_A -elements {top/part_a}
set_domain_supply_net PD_A -primary_power_net VDD_A \\
    -primary_ground_net VSS
add_port_state VDD_A -state {nom 0.80} -state {turbo 0.90} -state {off off}
create_power_domain PD_B -elements {top/part_b/u_x top/part_b/u_y}
# PD_B has no supply state on purpose
"""


def test_parse_upf_and_intent_table():
    pi = parse_upf(UPF)
    assert pi.domains["PD_A"].states == {"nom": 0.80, "turbo": 0.90} and pi.domains["PD_A"].supply_net == "VDD_A"
    assert pi.domain_of("top/part_a/u_q") == ["PD_A"] and pi.domain_of("top/part_b/u_x") == ["PD_B"] and pi.domain_of("top/part_c") == []
    model = ModelRoot.from_frame(pd.DataFrame([
        {"fub": "Q", "partition": "A", "fe_hier": "top/u_q", "synth_object": "Q", "be_hier": "top/part_a/u_q"},
        {"fub": "X", "partition": "B", "fe_hier": "top/u_x", "synth_object": "X", "be_hier": "top/part_b/u_x"},
        {"fub": "Z", "partition": "C", "fe_hier": "top/u_z", "synth_object": "Z", "be_hier": "top/part_c/u_z"},
    ]), "D")
    t = intent_table(pi, model, "D", "B1", {"nom": {"voltage_v": 0.80}, "turbo": {"voltage_v": 0.85}}).set_index("fub")
    assert t.loc["Q", "power_domain"] == "PD_A" and "domain_voltage_mismatch" in t.loc["Q", "intent_issues"] and "turbo" in t.loc["Q", "voltage_mismatch"]
    assert "domain_state_missing" in t.loc["X", "intent_issues"]        # PD_B has no supply states
    assert t.loc["Z", "intent_issues"] == "no_power_domain"


def test_intent_flows_through_pipeline_and_sanitize(project):
    proj, root, log = project
    intent = load_table(proj, "power_intent")
    assert len(intent) and {"power_domain", "intent_ok", "intent_issues"} <= set(intent.columns)
    assert (~intent["intent_ok"]).any()                                   # injected defects
    df = load_dataset(proj, raw=True)
    assert "power_domain" in df.columns and df["power_domain"].notna().any()
    flags = load_table(proj, "quality_flags")
    joined = ";".join(flags["quality_flags"].astype(str))
    assert "intent_missing" in joined or "intent_mismatch" in joined


VOLTUS = """Cadence Voltus Power Report
Version: 23.10
Design: top   Run: r1   Date: 2026-01-01
Activity: SAIF   Scenario: typical@nom
Units: mW
Instance                                Internal   Switching   Leakage   Total
top/part_a/u_q                             38.10       33.20      5.10   76.40
"""


def test_voltus_parser_and_activity_mode(tmp_path):
    p = tmp_path / "v.rpt"
    p.write_text(VOLTUS)
    rep = voltus.parse(p)
    assert rep.records.iloc[0]["metric"] == "be_voltus_mw" and rep.records.iloc[0]["value"] == pytest.approx(76.4)
    assert rep.activity_mode == "saif" and rep.workload == "typical" and rep.operating_point == "nom"
    assert getattr(voltus, "OPTIONAL", False) and "voltus" in SOURCES


PPRTL_CG = """PPRTL Power Report
Tool: PowerPro-RTL  Version: R-2025
Mode: physical-aware
Workload: typical   Operating point: nom
Activity: vectorless
Power units: mW
------
Hierarchy         Internal   Switching   Leakage    Total   ClockGatingEff
------
top/u_a           1.0        2.0         0.5        3.5     0.72
"""


def test_pprtl_clock_gating_and_vectorless(tmp_path):
    p = tmp_path / "p.rpt"
    p.write_text(PPRTL_CG)
    rep = pprtl.parse(p)
    r = rep.records.set_index("metric")["value"]
    assert r["fe_physical_mw"] == pytest.approx(3.5) and r["cg_efficiency"] == pytest.approx(0.72)
    assert rep.activity_mode == "vectorless"


def test_vectorless_provenance_reaches_dataset(project):
    proj, _, log = project
    df = load_dataset(proj, raw=True)
    assert "be_activity_mode" in df.columns
    modes = set(df["be_activity_mode"].dropna().astype(str))
    assert "saif" in modes and "vectorless" in modes
    flags = load_table(proj, "quality_flags")
    assert flags["quality_flags"].astype(str).str.contains("vectorless_power").any()
    assert df.loc[df["be_activity_mode"] == "vectorless", "workload"].eq("idle").all()


def test_qualify_engine_vs_engine(project):
    proj, _, _ = project
    df = load_dataset(proj)
    q = qualify(df, "be_mw", "be_voltus_mw", tolerance_pct=8.0)
    assert q.n > 0 and q.verdict == "PASS" and -8 < q.bias_pct < 0        # mock Voltus reads ~3% low
    assert len(q.by_partition) and len(q.worst)
    strict = qualify(df, "be_mw", "be_voltus_mw", tolerance_pct=1.0)
    assert strict.verdict == "FAIL"
    with pytest.raises(ValueError):
        qualify(df, "be_mw", "nope")


def test_hotspots_ranking_and_flags(project):
    proj, _, _ = project
    df = load_dataset(proj)
    design = sorted(df["design"].unique())[0]
    rep = hotspots(df, design)
    t = rep.table
    assert t["be_mw"].is_monotonic_decreasing and abs(t["share_pct"].sum() - 100) < 1e-6
    assert rep.prev_build is not None and t["delta_pct"].notna().any()
    assert "cg_efficiency" in t.columns and t["flags"].str.contains("hotspot").any()
    assert len(rep.partitions) == t["partition"].nunique()


def test_budget_classification_and_check(tmp_path, project):
    assert classify_status(100, 100, 0)[0] == "AT RISK"          # exactly on budget at signoff = no margin
    assert classify_status(95, 100, 0)[0] == "ON TRACK" and classify_status(101, 100, 0)[0] == "OVER"
    assert classify_status(110, 100, 25)[0] == "ON TRACK" and classify_status(120, 100, 25)[0] == "AT RISK"
    assert classify_status(126, 100, 25)[0] == "OVER"
    toml = tmp_path / "b.toml"
    toml.write_text('[defaults]\nworkload = "typical"\noperating_point = "nom"\n'
                    '[[budget]]\ndesign = "GPU_A"\nscope = "design"\nbe_mw = 1.0\n'
                    '[[budget]]\ndesign = "GPU_A"\nscope = "partition:PCORE0"\nbe_mw = 1e6\ntolerance_pct = { signoff = 10 }\n')
    budgets = load_budgets(toml)
    assert budgets[1].tolerance_pct["signoff"] == 10 and budgets[1].tolerance_pct["rtl"] == 25
    proj, _, _ = project
    df = load_dataset(proj, raw=True)
    st = check_budgets(df, budgets, lineage=load_table(proj, "lineage"))
    assert st[0].status == "OVER" and st[1].status == "ON TRACK"
    assert st[0].milestone == "signoff" and len(st[0].history) >= 3
    assert st[0].fubs_expected >= st[0].fubs_measured > 0
    assert not st[0].complete          # GPU_A's last build has renamed instances -> undercount flagged


def test_milestone_and_design_type_provenance(project):
    proj, _, _ = project
    df = load_dataset(proj, raw=True)
    assert set(df["milestone"].unique()) <= {"rtl", "synthesis", "placement", "route", "signoff"}
    assert df.groupby("build")["milestone"].nunique().eq(1).all()
