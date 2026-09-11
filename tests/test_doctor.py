from powermet.cli import main
from powermet.deps import doctor_report


def test_doctor_report_lists_required():
    text, ready = doctor_report()
    assert "pandas" in text and "numpy" in text
    assert "Status:" in text


def test_cli_doctor_runs(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "PowerMet Doctor" in out
    assert rc in (0, 1)
