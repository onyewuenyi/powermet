"""Design-methodology variants: each BE layout round-trips through the pipeline with the matching strategy."""

import numpy as np
import pandas as pd
import pytest

from powermet.config import Config, Project
from powermet.demo import DemoSpec
from powermet.mockdata import BE_LAYOUTS, be_layout, hierarchical_rows, write_mock_runs
from powermet.pipeline import find_runs, ingest_runs
from powermet.profiles import list_profiles, load_profile, render_profile
from powermet.storage import load_dataset, load_table
from powermet.techniques import BY_KEY, TECHNIQUES, assess_all, render_catalog, render_results

KEY = ["design", "build", "fub", "workload", "operating_point"]


def _spec(meth):
    return DemoSpec(n_designs=1, n_builds=2, n_fubs=12, seed=4, workloads=("typical",), operating_points=("nom", "turbo"), methodology=meth)


@pytest.mark.parametrize("meth", BE_LAYOUTS)
def test_layout_round_trip(tmp_path, meth):
    root, data, _ = write_mock_runs(tmp_path / "runs", _spec(meth), defects=False)
    proj = Project(tmp_path / ".powermet")
    cfg = Config()
    if meth == "same_hierarchy":
        cfg.identity = {"kind": "same_hierarchy"}
    summary = ingest_runs(proj, find_runs(root), cfg)
    assert not any(r.errors for r in summary.runs)
    df = load_dataset(proj, raw=True)
    gen = data.measurements
    assert len(df) == len(gen) and len(load_table(proj, "unmapped")) == 0
    m = gen.merge(df, on=KEY, suffixes=("_g", ""))
    # totals are conserved in every layout
    for col in ("be_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count"):
        assert np.isclose(m[col + "_g"].sum(), m[col].sum(), rtol=1e-3), col
    lin = load_table(proj, "lineage")
    rels = set(lin["relationship"].str.split(";").explode())
    if meth in ("separate", "same_hierarchy"):
        assert rels == {"one-to-one"}
        assert np.allclose(m["be_mw_g"], m["be_mw"], rtol=1e-3)
    if meth == "replicated":
        assert "replicated" in rels and np.allclose(m["be_mw_g"], m["be_mw"], rtol=1e-3)     # replicas summed back
    if meth == "split":
        assert "split" in rels and np.allclose(m["be_mw_g"], m["be_mw"], rtol=1e-3)
        split = lin[lin["relationship"].str.contains("split")]
        assert split["fub"].nunique() >= 1 and df["wns_ps"].notna().all()
    if meth in ("merged", "mixed"):
        merged = set(lin.loc[lin["relationship"].str.contains("merged"), "fub"])
        assert merged
        g = m[m["fub"].isin(merged)]
        assert np.isclose(g["be_mw_g"].sum(), g["be_mw"].sum(), rtol=1e-3)                    # block total exact
        others = m[~m["fub"].isin(merged)]
        assert np.allclose(others["be_mw_g"], others["be_mw"], rtol=1e-3)                     # everything else exact


def test_same_hierarchy_without_profile_still_maps(tmp_path):
    # the mock writes be_hier == fe_hier explicitly, so the default strategy also resolves it
    root, data, _ = write_mock_runs(tmp_path / "runs", _spec("same_hierarchy"), defects=False)
    proj = Project(tmp_path / ".powermet")
    ingest_runs(proj, find_runs(root))
    assert len(load_table(proj, "unmapped")) == 0


def test_per_instance_replicas(tmp_path):
    root, data, _ = write_mock_runs(tmp_path / "runs", _spec("replicated"), defects=False)
    proj = Project(tmp_path / ".powermet")
    cfg = Config()
    cfg.identity = {"kind": "explicit_map", "replica_policy": "per_instance"}
    ingest_runs(proj, find_runs(root), cfg)
    df = load_dataset(proj, raw=True)
    inst = df[df["fub"].astype(str).str.contains("@")]
    assert len(inst) and inst["model_root"].str.contains("@").all()
    base = inst["fub"].str.split("@").str[0].unique()
    for b in base:
        rows = inst[inst["fub"].str.startswith(b + "@")]
        gen = data.measurements[data.measurements["fub"] == b]
        assert np.isclose(rows["be_mw"].sum(), gen["be_mw"].sum(), rtol=1e-3)


def test_be_layout_shapes():
    h = pd.DataFrame({"fub": [f"F{i}" for i in range(10)], "model_root": [f"D.P.F{i}" for i in range(10)],
                      "partition": ["P0"] * 5 + ["P1"] * 5, "fe_hier": [f"top/u_f{i}" for i in range(10)],
                      "synth_object": [f"F{i}" for i in range(10)], "be_hier": "", "n_instances": range(100, 1100, 100)}).set_index("fub")
    objs, fmap = be_layout(h, "mixed", "top")
    assert objs["fraction"].groupby(objs["fub"]).sum().round(6).le(1.0).all()
    assert fmap["be_hier"].str.contains(r"\*").any()                      # replicated glob
    assert fmap.groupby("be_hier").size().max() >= 2                         # merged share
    assert fmap.groupby("fub").size().max() >= 2                              # split rows
    with pytest.raises(ValueError):
        be_layout(h, "bogus", "top")
    rows = hierarchical_rows([("top/a/x", 1.0), ("top/b/y", 2.0), ("top/a/z", 3.0)])
    assert [r[1] for r in rows] == ["top", "a", "x", "z", "b", "y"] and rows[1][2] == 4.0   # subtrees stay contiguous


def test_profiles_apply_and_render():
    names = list_profiles()
    assert {"separate_hierarchies", "same_hierarchy", "replicated_units", "flattened_backend", "rtl_activity_only"} <= set(names)
    for n in names:
        pr = load_profile(n)
        cfg = pr.apply(Config())
        assert cfg.profile == n and cfg.identity["kind"] in ("explicit_map", "same_hierarchy")
        txt = render_profile(pr)
        assert "Problem" in txt and "Implication" in txt
    with pytest.raises(FileNotFoundError):
        load_profile("nope")


def test_techniques_catalog_and_assessment(tmp_path):
    assert len(TECHNIQUES) >= 8 and "clock_gating" in BY_KEY
    assert "trade-off" in render_catalog()
    spec = DemoSpec(n_designs=1, n_builds=2, n_fubs=12, seed=4, workloads=("idle", "typical"), operating_points=("nom", "turbo"))
    root, data, _ = write_mock_runs(tmp_path / "runs", spec, defects=False)
    proj = Project(tmp_path / ".powermet")
    ingest_runs(proj, find_runs(root))
    df = load_dataset(proj, raw=True)
    res = {r.technique: r for r in assess_all(df, {"design": "GPU_A"})}
    assert res["clock_gating"].assessable and res["clock_gating"].est_saving_mw > 0
    assert res["power_gating"].assessable and "idle" in res["power_gating"].scope
    assert res["dvfs"].assessable and res["wire_cap_reduction"].assessable and res["vt_swap"].assessable
    assert not res["operand_isolation"].assessable and res["operand_isolation"].missing_data
    txt = render_results(list(res.values()))
    assert "TECHNIQUE ASSESSMENT" in txt and "assumptions:" in txt and "needs:" in txt
    # with a decomposition supplied (several builds -> non-unique keys) the leakage lookup must still work
    dec = df[[c for c in ("model_root", "design", "build", "workload", "operating_point") if c in df.columns]].copy()
    dec["leakage_mw"] = df["be_mw"] * 0.12
    with_dec = {r.technique: r for r in assess_all(df, {"design": "GPU_A", "decomposition": dec}, ["clock_gating", "vt_swap"])}
    assert with_dec["clock_gating"].assessable and with_dec["vt_swap"].assessable
    single = assess_all(df, {"design": "GPU_A", "wire_cap_cut_pct": 20.0}, ["wire_cap_reduction"])
    assert len(single) == 1 and single[0].est_saving_mw == pytest.approx(res["wire_cap_reduction"].est_saving_mw * 2, rel=1e-6)
