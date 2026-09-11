"""End-to-end smoke test of the documented workflow in an isolated directory."""

import os
from pathlib import Path

import pytest

from powermet.cli import main


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POWERMET_HOME", raising=False)
    return tmp_path


def run(*argv):
    return main(list(argv))


def test_full_workflow(workdir, capsys):
    assert run("demo", "generate", "--designs", "2", "--builds", "6", "--fubs", "15", "--dirty") == 0
    files = list((workdir / "data" / "demo").glob("power_measurements.*"))
    assert files
    data = str(files[0])
    assert run("data", "validate", data) == 0
    assert run("data", "validate", "data/demo/power_measurements_dirty.csv") == 1
    assert run("data", "import", data) == 0
    assert run("analyze", "summary") == 0
    out = capsys.readouterr().out
    assert "FE Physical -> BE" in out and "Key finding" in out
    assert run("analyze", "correlation") == 0
    assert run("analyze", "errors", "--top", "3") == 0
    assert run("model", "train") == 0
    out = capsys.readouterr().out
    assert "Split strategy:  BUILD" in out
    assert run("model", "evaluate") == 0
    out = capsys.readouterr().out
    assert "Improvement over baseline" in out
    assert run("report") == 0
    report = workdir / ".powermet" / "reports" / "power_metrology_report.md"
    assert report.exists()
    text = report.read_text()
    for section in ("Dataset summary", "Data quality", "Model comparison", "Key findings", "Limitations"):
        assert section in text


def test_missing_dataset_is_clear_error(workdir, capsys):
    assert run("analyze", "summary") == 1
    assert "data import" in capsys.readouterr().err


def test_v1_v3_workflow(workdir, capsys):
    assert run("mock", "generate", "--designs", "2", "--builds", "4", "--fubs", "8",
               "--workloads", "typical,compute", "--operating-points", "nom,turbo") == 0
    assert run("ingest", "scan", "mock_runs") == 0
    assert run("sanitize") == 0
    out = capsys.readouterr().out
    assert "CORRELATION DATA QUALITY" in out and "Usable dataset" in out
    assert run("lineage", "show", "Scheduler", "--design", "GPU_A") == 0
    out = capsys.readouterr().out
    assert "FE hierarchy" in out and "primepower" in out
    assert run("--profile", "model", "train", "--cv") == 0
    out = capsys.readouterr().out
    assert "Physics-structured OLS" in out and "leave-one-build-out" in out and "Runtime profile" in out
    assert run("model", "validate") == 0
    assert run("model", "predict", "--design", "GPU_A", "--fub", "Scheduler", "--workload", "typical",
               "--operating-point", "nom", "--scale", "wire_cap_pf=0.8") == 0
    out = capsys.readouterr().out
    assert "WHAT IF" in out and "estimated reduction" in out
    assert run("workload", "summary") == 0
    out = capsys.readouterr().out
    assert "pJ/op" in out
    assert run("explore", "sweep", "--design", "GPU_A", "--workload", "compute", "--param", "frequency_ghz", "--values", "1.8,2.2") == 0
    assert run("explore", "opmap", "--design", "GPU_A", "--add", "v=0.8,f=2.3") == 0
    assert run("profile", "show") == 0
    assert run("report") == 0
    text = (workdir / ".powermet" / "reports" / "power_metrology_report.md").read_text()
    for section in ("13. Correlation data quality", "14. Provenance and lineage", "15. Runtime", "16. Build-based cross-validation",
                    "17. What-if", "18. Workload", "19. Design-space"):
        assert section in text


def test_timing_energy_integration_cli(workdir, capsys):
    assert run("mock", "generate", "--designs", "2", "--builds", "4", "--fubs", "10",
               "--workloads", "typical,compute", "--operating-points", "nom,turbo") == 0
    assert run("ingest", "scan", "mock_runs") == 0
    assert run("sanitize") == 0
    out = capsys.readouterr().out
    assert "METRIC QUALITY" in out and "Timing missing" in out
    assert run("model", "train") == 0
    assert run("analyze", "energy", "--design", "GPU_A") == 0
    assert "Data-movement power" in capsys.readouterr().out
    assert run("analyze", "deltas", "--design", "GPU_A") == 0
    out = capsys.readouterr().out
    assert "WNS (worst partition" in out and "Per partition" in out
    assert run("analyze", "frontier", "--design", "GPU_A", "--no-plots") == 0
    assert "Pareto-optimal builds" in capsys.readouterr().out
    assert run("model", "export") == 0
    assert run("integrate", "trace", "mock_runs/traces/GPU_A_phases.csv") == 0
    out = capsys.readouterr().out
    assert "energy per op" in out and "Energy by workload" in out
    assert run("measure", "get", "--fub", "Scheduler", "--design", "GPU_A", "--stage", "BE", "--metric", "be_mw", "--show-file") == 0
    out = capsys.readouterr().out
    assert "primepower" in out and "GPU_A.PCORE0.Scheduler" in out and "source file:" in out
    assert run("db", "tables") == 0
    assert run("db", "query", "SELECT COUNT(*) AS n FROM build") == 0
    assert run("report") == 0
    text = (workdir / ".powermet" / "reports" / "power_metrology_report.md").read_text()
    for section in ("20. Metric quality", "21. Build-to-build", "22. Power", "23. Energy decomposition", "24. Compact model"):
        assert section in text


def test_sources_and_design_type(workdir, capsys):
    assert run("sources") == 0
    out = capsys.readouterr().out
    assert "saif" in out and "SAIF 2.0" in out and "primetime" in out
    assert run("mock", "generate", "--designs", "1", "--builds", "2", "--fubs", "6", "--workloads", "typical", "--operating-points", "nom") == 0
    assert run("ingest", "scan", "mock_runs") == 0
    assert run("db", "query", "SELECT DISTINCT design_type FROM build") == 0
    assert "cpu" in capsys.readouterr().out
