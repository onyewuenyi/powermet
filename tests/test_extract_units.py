"""Adapter contract tests on hand-written report strings (independent of the mock generator)."""

from pathlib import Path

import numpy as np
import pytest

from powermet.extract import SOURCES, SOURCE_METRICS, METRIC_SCOPE, metrics_with_scope
from powermet.extract import implementation, metadata, pprtl, primepower, primetime, saif, starrc
from powermet.extract.base import (BE_HIER, FE_HIER, OBJECT_KINDS, PARTITION, Located, ParseError, SourceInputs, convert_unit,
                                   find_table_start, locate, parse_header, record, to_float, tool_name, units_from_text)


# ---------------------------------------------------------------- base helpers

@pytest.mark.parametrize("value,unit,metric,expected", [
    (0.05, "W", "be_mw", 50.0), (1500, "fF", "wire_cap_pf", 1.5), (2500, "MHz", "frequency_ghz", 2.5),
    (1, "ns", "clock_period_ps", 1000.0), (2, "mm", "wire_length_um", 2000.0), (3, "mm2", "area", 3e6),
    (50, "%", "activity", 0.5), (0.5, "us", "wns_ps", 5e5), (7, "count", "cell_count", 7),
])
def test_convert_unit(value, unit, metric, expected):
    got, cu = convert_unit(value, unit, metric)
    assert got == pytest.approx(expected)


def test_convert_unit_unknown_raises_and_unknown_metric_passes_through():
    with pytest.raises(ParseError):
        convert_unit(1, "furlong", "area")
    assert convert_unit(4.0, "widgets", "not_a_metric") == (4.0, "widgets")


def test_parse_header_stop_divider_and_long_keys():
    lines = ["*****", "Tool: X  Version: 1.2", "-----", "a" * 45 + ": junk", "Hierarchy  Total", "Later: nope"]
    h = parse_header(lines, stop_at="Hierarchy")
    assert h == {"tool": "X", "version": "1.2"}


def test_find_table_start_skips_underline_and_fails_clearly():
    assert find_table_start(["x", "Hierarchy  Total", "-----", "row"], "Hierarchy") == 3
    with pytest.raises(ParseError, match="not found"):
        find_table_start(["a", "b"], "Hierarchy")


def test_to_float_and_units_from_text():
    assert to_float("1,234.5") == 1234.5 and np.isnan(to_float("--")) and np.isnan(to_float("n/a"))
    assert units_from_text("Power Units = 1mW", "W") == "mW" and units_from_text("nothing", "pF") == "pF"
    assert units_from_text("Area units: um^2", "x") == "um2"


def test_record_helper_and_object_kinds():
    r = record("top/a", FE_HIER, "activity", 0.5, "ratio")
    assert r["unit_original"] == "ratio" and r["object_kind"] == "fe_hier"
    with pytest.raises(ValueError):
        record("x", "bogus", "m", 1, "u")
    assert set(OBJECT_KINDS) == {"fe_hier", "be_hier", "partition", "design"}
    assert tool_name({"tool": "HeaderTool"}, "Default") == "HeaderTool" and tool_name({}, "Default") == "Default"


def test_registry_consistency():
    from powermet.extract import SOURCE_SPECS, STAGE_OF_SOURCE
    assert set(SOURCES) == set(SOURCE_METRICS) == set(SOURCE_SPECS)
    assert all(sp.stage != "OTHER" and sp.tool_family != "?" and sp.supported_versions for sp in SOURCE_SPECS.values())
    assert STAGE_OF_SOURCE["saif"] == "ACTIVITY" and SOURCE_SPECS["voltus"].optional
    for src, ms in SOURCE_METRICS.items():
        for m in ms:
            assert m in METRIC_SCOPE, f"{src}:{m} missing from METRIC_SCOPE"
    assert "wire_cap_pf" in metrics_with_scope() and "activity" in metrics_with_scope("workload")
    assert "be_mw" in metrics_with_scope("workload", "operating_point")


def test_locate_patterns_and_overrides(tmp_path):
    (tmp_path / "pp" / "typical_nom").mkdir(parents=True)
    (tmp_path / "pp" / "typical_nom" / "power.rpt").write_text("x")
    (tmp_path / "pp" / "compute_nom").mkdir()
    (tmp_path / "pp" / "compute_nom" / "power.rpt").write_text("x")
    (tmp_path / "alt").mkdir()
    (tmp_path / "alt" / "rc.txt").write_text("x")
    inputs = SourceInputs("D", "B1", tmp_path, workloads=["typical", "compute"], operating_points=["nom"],
                          patterns={"starrc": "alt/*.txt"})
    found = locate(inputs, "primepower", "pp/{workload}_{operating_point}/*.rpt")
    assert [(l.context["workload"], l.context["operating_point"]) for l in found] == [("typical", "nom"), ("compute", "nom")]
    # override replaces the default pattern for that source only
    assert [l.path.name for l in locate(inputs, "starrc", "starrc/none.rpt")] == ["rc.txt"]
    assert locate(inputs, "pprtl", "missing/{workload}.rpt") == []
    # placeholder-free pattern yields empty context
    assert locate(inputs, "starrc", "alt/rc.txt")[0].context == {}


# ---------------------------------------------------------------- adapters on literal reports

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


PRIMEPOWER = """****
Report : power -hierarchy
Design : top
Version: V-2024.09
Run    : r1
Scenario: typical@nom
Power Units = 1W
****
                          Int      Switch   Leak     Total
Hierarchy                 Power    Power    Power    Power    %
----------------------------------------------------------------
top                       1.0e+00 1.0e+00 1.0e-01 2.1e+00 100.0
  u_a (A)                 4.0e-01 4.0e-01 2.0e-02 8.2e-01  39.0
    u_a_sub (ASUB)        1.0e-01 1.0e-01 1.0e-02 2.1e-01  10.0
  u_b (B)                 5.0e-01 5.0e-01 5.0e-02 1.05e+00 50.0
  bad row without numbers
"""


def test_primepower_hierarchy_reconstruction_and_units(tmp_path):
    rep = primepower.parse(_write(tmp_path, "pp.rpt", PRIMEPOWER))
    tot = rep.records[rep.records["metric"] == "be_mw"]
    objs = dict(zip(tot["object"], tot["value"]))
    assert objs == pytest.approx({"top/u_a": 820.0, "top/u_a/u_a_sub": 210.0, "top/u_b": 1050.0})
    leak = rep.records[rep.records["metric"] == "be_leakage_mw"]
    assert dict(zip(leak["object"], leak["value"])) == pytest.approx({"top/u_a": 20.0, "top/u_a/u_a_sub": 10.0, "top/u_b": 50.0})
    assert (rep.records["unit"] == "mW").all() and (rep.records["unit_original"] == "W").all()
    assert rep.workload == "typical" and rep.operating_point == "nom" and rep.run_id == "r1"
    assert tot["reference"].tolist() == ["A", "ASUB", "B"]
    assert rep.notes == ["power converted from W to mW"]


PRIMETIME_NS = """Report : timing summary -partition
Version: X
Scenario: turbo
Time units: ns
Partition               Clock     Period       WNS         TNS  Violating  Endpoints
top/part_pcore0      core_clk      0.400     -0.012      -1.450        118      52034
top/part_memss       core_clk      0.400      0.030       0.000          0      10000
"""


def test_primetime_units_and_partition_kind(tmp_path):
    rep = primetime.parse(_write(tmp_path, "pt.rpt", PRIMETIME_NS), operating_point="turbo")
    r = rep.records.set_index(["object", "metric"])["value"]
    assert r[("top/part_pcore0", "wns_ps")] == pytest.approx(-12.0)
    assert r[("top/part_pcore0", "clock_period_ps")] == pytest.approx(400.0)
    assert r[("top/part_pcore0", "violating_endpoints")] == 118
    assert (rep.records["object_kind"] == PARTITION).all() and rep.operating_point == "turbo"
    assert set(rep.records["unit"]) == {"ps", "count"}
    with pytest.raises(ParseError):
        primetime.parse(_write(tmp_path, "bad.rpt", PRIMETIME_NS.replace("Time units: ns", "Time units: fs")))


STARRC_FF = """StarRC Parasitic Summary
Version: V-2024.09
Capacitance units: fF
Instance                        Nets     TotalCap     WireCap      PinCap
top/u_a                         100      2500.0       1800.0       700.0
short row
"""


def test_starrc_ff_conversion(tmp_path):
    rep = starrc.parse(_write(tmp_path, "rc.rpt", STARRC_FF))
    r = rep.records.set_index("metric")["value"]
    assert r["wire_cap_pf"] == pytest.approx(1.8) and r["cell_cap_pf"] == pytest.approx(0.7)
    assert rep.tool == "StarRC" and len(rep.records) == 2 and (rep.records["object_kind"] == BE_HIER).all()


IMPL = """Fusion Compiler QoR Summary
Version: V
Area units: mm^2   Length units: um
Hierarchy                       CellArea      CellCount    AvgFanout    Utilization   WireLength   AvgNetLen
top/u_a                         0.001         50000        8.02         0.71          182034.2     10.113
top/u_b                         0.002         60000        7.50         0.70
"""


def test_implementation_optional_length_columns_and_area_units(tmp_path):
    rep = implementation.parse(_write(tmp_path, "qor.rpt", IMPL))
    a = rep.records[rep.records["object"] == "top/u_a"].set_index("metric")["value"]
    b = rep.records[rep.records["object"] == "top/u_b"].set_index("metric")["value"]
    assert a["area"] == pytest.approx(1000.0) and a["wire_length_um"] == pytest.approx(182034.2) and a["avg_net_length_um"] == pytest.approx(10.113)
    assert "wire_length_um" not in b.index and b["cell_count"] == 60000


PPRTL = """PPRTL Power Report
Tool: PowerPro-RTL  Version: R-2025
Mode: logical
Workload: compute   Operating point: eco
Power units: mW
Hierarchy         Internal   Switching   Leakage    Total
top/u_a           1.0        2.0         0.5        3.5
"""


def test_pprtl_mode_aliases_and_errors(tmp_path):
    rep = pprtl.parse(_write(tmp_path, "p.rpt", PPRTL))
    assert rep.records["metric"].tolist() == ["fe_logical_mw"] and rep.workload == "compute" and rep.operating_point == "eco"
    assert rep.tool == "PowerPro-RTL" and (rep.records["object_kind"] == FE_HIER).all()
    rep2 = pprtl.parse(_write(tmp_path, "p2.rpt", PPRTL.replace("Mode: logical", "Mode: physical")))
    assert rep2.records["metric"].tolist() == ["fe_physical_mw", "fe_leakage_mw"]      # physical-aware mode carries the leakage estimate
    with pytest.raises(ValueError, match="unknown PPRTL mode"):
        pprtl.parse(_write(tmp_path, "p3.rpt", PPRTL.replace("Mode: logical", "Mode: bogus")))


SAIF = """(SAIFILE
(SAIFVERSION "2.0") (DIRECTION "backward") (DESIGN "top") (DATE "2026-03-02")
(VENDOR "Synopsys") (PROGRAM_NAME "Verdi") (VERSION "V-2024.09")
(DIVIDER / ) (TIMESCALE 1 ns) (DURATION 4000)
(INSTANCE top
  (PORT (clk (T0 2000000) (T1 2000000) (TC 20000)))
  (INSTANCE part_p0
    (INSTANCE u_a
      (NET
        (d\\[0\\] (T0 1000) (T1 3000) (TX 0) (TC 2000) (IG 0))
        (d\\[1\\] (T0 1000) (T1 3000) (TX 0) (TC 4000) (IG 0))
      )
      (INSTANCE u_a_sub (NET (x (T0 1) (T1 1) (TC 1000))))
    )
    (INSTANCE u_empty)
  )
))
"""


def test_saif_parser_aggregates_per_instance(tmp_path):
    p = _write(tmp_path, "typical.saif", SAIF)
    rep = saif.parse(p, workload="typical", sim_clock_period_ps=400.0)      # 4000 ns / 400 ps = 10,000 cycles
    r = rep.records.pivot(index="object", columns="metric", values="value")
    assert r.loc["top/part_p0/u_a", "net_count"] == 3                            # own nets + descendant
    assert r.loc["top/part_p0/u_a", "bits_per_cycle"] == pytest.approx(0.7)      # (2000+4000+1000)/10000
    assert r.loc["top/part_p0/u_a", "activity"] == pytest.approx(0.7 / 3)
    assert r.loc["top/part_p0/u_a/u_a_sub", "activity"] == pytest.approx(0.1)
    assert r.loc["top/part_p0", "net_count"] == 3 and "top" not in r.index and "top/part_p0/u_empty" not in r.index
    assert (rep.records["object_kind"] == BE_HIER).all() and rep.tool == "Verdi" and rep.workload == "typical"
    fe = saif.parse(p, activity_hierarchy="fe", sim_clock_period_ps=400.0)
    assert (fe.records["object_kind"] == FE_HIER).all()
    with pytest.raises(ParseError, match="clock period"):
        saif.parse(p)
    with pytest.raises(ParseError, match="DURATION"):
        saif.parse(_write(tmp_path, "d.saif", SAIF.replace("(DURATION 4000)", "(DURATION 0)")), sim_clock_period_ps=400.0)


def test_saif_context_from_metadata(tmp_path):
    import json
    (tmp_path / "activity").mkdir()
    _write(tmp_path / "activity", "gemm.saif", SAIF)
    _write(tmp_path, "metadata.json", json.dumps({
        "design": "D", "build": "B1", "workloads": ["gemm"], "operating_points": {"nom": {"frequency_ghz": 2.5, "voltage_v": 0.8}},
        "activity_flow": {"tool": "Verdi", "hierarchy": "be", "source_fsdb": {"gemm": "/sim/gemm.fsdb"}, "core": "top",
                          "mapping": "mapping/fub_map.csv", "partition_list": ["P0"]}}))
    inputs, meta = metadata.inputs_from_metadata(tmp_path)
    assert inputs.context["sim_clock_period_ps"] == pytest.approx(400.0)      # from the nominal operating point
    files = saif.get_files(inputs)
    assert files and files[0].context["workload"] == "gemm" and files[0].context["activity_flow"]["core"] == "top"
    rep = saif.parse(files[0].path, **files[0].context)
    assert any("flow inputs" in n and "/sim/gemm.fsdb" in n for n in rep.notes)


def test_metadata_without_operating_points_and_missing_file(tmp_path):
    import json
    p = _write(tmp_path, "metadata.json", json.dumps({"design": "D", "build": "B1", "run_id": "r"}))
    rep = metadata.parse(p)
    assert rep.n_records == 0 and rep.run_id == "r"
    with pytest.raises(FileNotFoundError):
        metadata.inputs_from_metadata(tmp_path / "nowhere")
    inputs, meta = metadata.inputs_from_metadata(tmp_path, patterns={"pprtl": "x"})
    assert inputs.design == "D" and inputs.pattern_for("pprtl", "default") == "x" and inputs.workloads == []
