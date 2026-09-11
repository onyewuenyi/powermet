"""Reusable abstractions: ModelRoot identity, MeasurementStore accessor, DatasetSlice selection."""

import numpy as np
import pandas as pd
import pytest

from powermet.identity import FubSpec, ModelRoot
from powermet.measurements import Measurement, MeasurementStore, STAGE_OF_SOURCE
from powermet.selection import DatasetSlice, build_order, default_value, latest_build, latest_build_per_design


def _map_frame(with_optional=True):
    rows = [
        {"fub": "A", "fe_hier": "top/u_a", "synth_object": "A", "be_hier": "top/u_a"},
        {"fub": "B", "fe_hier": "top/u_b", "synth_object": "B_1", "be_hier": "top/u_b_phys"},
        {"fub": "C", "fe_hier": "top/u_c", "synth_object": "C", "be_hier": "top/u_c"},
    ]
    if with_optional:
        for r, part in zip(rows, ("PCORE0", "PCORE0", "MEMSS")):
            r["model_root"] = f"D.{part}.{r['fub']}"
            r["partition"] = part
    return pd.DataFrame(rows)


class TestModelRoot:
    def test_from_frame_full(self):
        mr = ModelRoot.from_frame(_map_frame(), "D")
        assert len(mr) == 3 and mr.fub_names == ["A", "B", "C"]
        assert mr.partitions == ["MEMSS", "PCORE0"]
        assert mr.partition_of("B") == "PCORE0"
        assert [f.fub for f in mr.fubs_in("pcore0")] == ["A", "B"]      # case-insensitive
        assert mr.fub("B").model_root == "D.PCORE0.B"

    def test_optional_columns_fall_back(self):
        mr = ModelRoot.from_frame(_map_frame(with_optional=False), "D")
        assert mr.fub("A").model_root == "A" and mr.fub("A").partition is None
        assert not mr.has_partitions() and mr.resolve_partition("top/part_pcore0") == []

    def test_missing_required_column_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            ModelRoot.from_frame(pd.DataFrame({"fub": ["A"], "fe_hier": ["x"]}), "D")

    def test_duplicate_fub_raises(self):
        df = pd.concat([_map_frame(), _map_frame().iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate FUB"):
            ModelRoot.from_frame(df, "D")

    def test_resolve_by_kind(self):
        mr = ModelRoot.from_frame(_map_frame(), "D")
        assert mr.resolve("top/u_b", "fe_hier")[0].fub == "B"
        assert mr.resolve("top/u_b_phys", "be_hier")[0].fub == "B"
        assert mr.resolve("top/u_b", "be_hier") == []                         # BE path differs from FE path
        assert [f.fub for f in mr.resolve("top/part_pcore0", "partition")] == ["A", "B"]
        assert [f.fub for f in mr.resolve("PCORE0", "partition")] == ["A", "B"]  # bare name also accepted
        assert mr.resolve("top/part_nope", "partition") == []
        assert len(mr.resolve("*", "design")) == 3
        with pytest.raises(ValueError):
            mr.resolve("x", "bogus")

    def test_roundtrip_frame_and_load(self, tmp_path):
        p = tmp_path / "map.csv"
        _map_frame().to_csv(p, index=False)
        mr = ModelRoot.load(p, "D", model_version="v2")
        assert mr.model_version == "v2" and mr.source == str(p)
        pd.testing.assert_frame_equal(mr.to_frame(), ModelRoot.from_frame(mr.to_frame(), "D").to_frame())


def _long():
    rows = []
    for b in ("B1", "B2"):
        for fub, root in (("A", "D.P0.A"), ("B", "D.P0.B")):
            rows += [
                {"design": "D", "build": b, "fub": fub, "model_root": root, "object": f"top/u_{fub.lower()}", "source": "pprtl",
                 "metric": "fe_physical_mw", "value": 10.0, "unit": "mW", "workload": "typical", "operating_point": "nom",
                 "source_file": "pprtl.rpt", "tool": "PPRTL", "tool_version": "1", "run_id": f"r_{b}"},
                {"design": "D", "build": b, "fub": fub, "model_root": root, "object": f"top/u_{fub.lower()}", "source": "primepower",
                 "metric": "be_mw", "value": 12.0, "unit": "mW", "workload": "typical", "operating_point": "nom",
                 "source_file": "pp.rpt", "tool": "PrimePower", "tool_version": "2", "run_id": f"r_{b}"},
                {"design": "D", "build": b, "fub": fub, "model_root": root, "object": f"top/u_{fub.lower()}", "source": "starrc",
                 "metric": "wire_cap_pf", "value": 3.0, "unit": "pF", "workload": None, "operating_point": None,
                 "source_file": "rc.rpt", "tool": "StarRC", "tool_version": "3", "run_id": f"r_{b}"},
            ]
    return pd.DataFrame(rows)


class TestMeasurementStore:
    def test_stage_derivation_and_get(self):
        st = MeasurementStore(_long())
        m = st.get(fub="A", build="B2", stage="FE", metric="fe_physical_mw")
        assert isinstance(m, Measurement) and m.value == 10.0 and m.model_root == "D.P0.A"
        assert m.source == "pprtl" and m.tool_version == "1" and m.run_id == "r_B2" and m.hierarchy == "top/u_a"
        be = st.get(fub="A", build="B2", stage="BE", metric="be_mw")
        assert be.value - m.value == 2.0
        assert st.get(fub="B", build="B1", stage="PHYS", metric="wire_cap_pf").operating_point is None

    def test_missing_and_ambiguous(self):
        st = MeasurementStore(_long())
        with pytest.raises(KeyError):
            st.get(fub="Z", metric="be_mw")
        with pytest.raises(ValueError, match="add workload"):
            st.get(fub="A", metric="be_mw")          # two builds match
        assert st.get_value(fub="Z", metric="be_mw", default=np.nan) is not None

    def test_listing_and_pivot(self):
        st = MeasurementStore(_long())
        assert st.metrics(stage="FE") == ["fe_physical_mw"] and st.fubs(build="B1") == ["A", "B"]
        assert len(st.find(stage="BE")) == 4
        wide = st.pivot()
        assert set(wide.columns) >= {"be_mw", "fe_physical_mw", "wire_cap_pf"}
        assert len(wide) == 8   # 4 (fub x build) power rows + 4 build-level cap rows keyed workload/op = default

    def test_every_source_has_a_stage(self):
        from powermet.extract import SOURCES
        assert set(SOURCES) <= set(STAGE_OF_SOURCE)


class TestSelection:
    def _df(self):
        rows = []
        for d in ("X", "Y"):
            for b in ("B1", "B2", "B10"):
                for wl in ("typical", "compute"):
                    for op in ("eco", "nom"):
                        rows.append({"design": d, "build": b, "fub": "F", "model_root": f"{d}.P.F", "partition": "P",
                                     "workload": wl, "operating_point": op, "be_mw": 1.0})
        return pd.DataFrame(rows)

    def test_build_order_natural(self):
        assert build_order(["B10", "B2", "B1"]) == ["B1", "B2", "B10"]
        assert latest_build(self._df()) == "B10"
        assert latest_build_per_design(self._df()) == {"X": "B10", "Y": "B10"}

    def test_default_value(self):
        df = self._df()
        assert default_value(df, "operating_point", "nom") == "nom"
        assert default_value(df, "operating_point", "turbo") == "eco"     # preferred absent -> first sorted
        assert default_value(df, "missing", "x") is None

    def test_slice_defaults_to_latest_typical_nom(self):
        sel = DatasetSlice(design="X").apply(self._df())
        assert set(sel["build"]) == {"B10"} and set(sel["workload"]) == {"typical"} and set(sel["operating_point"]) == {"nom"}
        assert "design=X" in DatasetSlice(design="X").describe(sel) and "build=B10" in DatasetSlice(design="X").describe(sel)

    def test_slice_explicit_and_no_defaults(self):
        sel = DatasetSlice(design="Y", build="B2", workload="compute", operating_point="eco").apply(self._df())
        assert len(sel) == 1
        sel = DatasetSlice(design="Y", default_workload=False, default_operating_point=False).apply(self._df())
        assert len(sel) == 4                                       # all wl x op of the latest build
        sel = DatasetSlice(latest_only=False, default_workload=False, default_operating_point=False).apply(self._df())
        assert len(sel) == 24

    def test_slice_errors(self):
        with pytest.raises(ValueError, match="no rows match design=Z"):
            DatasetSlice(design="Z").apply(self._df())
        with pytest.raises(ValueError, match="build=B7"):
            DatasetSlice(design="X", build="B7").apply(self._df())
        with pytest.raises(ValueError, match="no 'design' column"):
            DatasetSlice(design="X").apply(self._df().drop(columns=["design"]))
