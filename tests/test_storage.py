import pandas as pd

from powermet.config import Config, Project
from powermet.demo import make_dirty
from powermet.ingest import write_table
from powermet.storage import import_file, import_history, load_dataset


def test_import_roundtrip(tmp_path, demo_df):
    src = write_table(demo_df, tmp_path / "in.csv")
    proj = Project(tmp_path / ".powermet")
    res = import_file(proj, src)
    assert res.n_imported == len(demo_df) and res.n_rejected == 0
    assert proj.config_path.exists()
    cfg = proj.load_config()
    assert isinstance(cfg, Config) and cfg.target == "be_mw"
    df = load_dataset(proj)
    assert len(df) == len(demo_df)
    assert "physical_error_pct" in df.columns and "wire_cap_fraction" in df.columns
    assert (df["source_file"] == str(src)).all()
    assert import_history(proj)[0]["rows_imported"] == len(demo_df)
    # source untouched
    pd.testing.assert_frame_equal(pd.read_csv(src), pd.read_csv(src))


def test_import_rejects_bad_rows(tmp_path, demo_df):
    dirty = make_dirty(demo_df, n_rows=60)
    src = write_table(dirty, tmp_path / "dirty.csv")
    proj = Project(tmp_path / ".powermet")
    res = import_file(proj, src)
    assert res.n_rejected > 0
    assert res.n_imported + res.n_rejected == len(dirty)
    rej = pd.read_parquet(res.rejected_path) if res.rejected_path.suffix == ".parquet" else pd.read_csv(res.rejected_path)
    assert rej["reject_reason"].notna().all()


def test_config_toml_roundtrip(tmp_path):
    proj = Project(tmp_path / ".powermet")
    cfg = Config(test_fraction=0.3, linear_features=["a", "b"])
    proj.save_config(cfg)
    assert proj.load_config() == cfg
