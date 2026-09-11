"""What-if prediction (V2): change a design parameter, re-derive engineered features, predict BE power.

Only extrapolating models (physics / linear / scaled) are used by default; a tree model is
reported alongside for reference when the override stays inside the training range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.features import add_engineered_features, rescale_fe_physical
from powermet.modeling import EXTRAPOLATING, MODEL_NAMES, pick_model_key
from powermet.selection import DatasetSlice
from powermet.schema import label
from powermet.textfmt import fmt_mw, fmt_pct, table

RAW_FEATURES = ("fe_physical_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "fanout",
                "frequency_ghz", "voltage_v", "activity")


@dataclass
class Override:
    feature: str
    mode: str      # "set" | "scale"
    value: float

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.feature not in out.columns:
            raise ValueError(f"unknown feature '{self.feature}'")
        cur = pd.to_numeric(out[self.feature], errors="coerce")
        out[self.feature] = self.value if self.mode == "set" else cur * self.value
        return out

    def describe(self) -> str:
        return f"{label(self.feature)} {'=' if self.mode == 'set' else 'x'} {self.value:g}"


@dataclass
class WhatIfResult:
    rows: pd.DataFrame            # selected rows with identity columns
    overrides: list[Override]
    model_key: str
    current: np.ndarray
    proposed: np.ndarray
    interval: dict | None = None  # relative-error quantiles from CV, if available
    others: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def delta_pct(self) -> float:
        c, p = float(np.nansum(self.current)), float(np.nansum(self.proposed))
        return (p - c) / c * 100.0 if c else float("nan")


def parse_override(text: str, mode: str) -> Override:
    """'wire_cap_pf=0.8' -> Override; accepts CLI aliases (wire-cap, cell-cap, ...)."""
    if "=" not in text:
        raise ValueError(f"expected feature=value, got '{text}'")
    name, _, val = text.partition("=")
    alias = {"wire-cap": "wire_cap_pf", "wire_cap": "wire_cap_pf", "cell-cap": "cell_cap_pf", "cell_cap": "cell_cap_pf",
             "frequency": "frequency_ghz", "freq": "frequency_ghz", "voltage": "voltage_v", "vdd": "voltage_v",
             "fe-physical": "fe_physical_mw", "fe_physical": "fe_physical_mw"}
    feat = alias.get(name.strip(), name.strip())
    return Override(feat, mode, float(val))


def select_rows(df: pd.DataFrame, design: str | None, build: str | None, fub: str | None,
                workload: str | None, operating_point: str | None) -> pd.DataFrame:
    """Rows for a what-if: latest build unless given; workload/operating point exactly as given (no defaults)."""
    return DatasetSlice(design=design, build=build, fub=fub, workload=workload, operating_point=operating_point,
                        default_workload=False, default_operating_point=False).apply(df)


def run_whatif(rows: pd.DataFrame, overrides: list[Override], payload: dict, meta: dict,
               model_key: str = "physics") -> WhatIfResult:
    models = payload["models"]
    model_key = pick_model_key(models, model_key, allow_tree=True)
    cur = add_engineered_features(rows)
    prop = cur
    for o in overrides:
        prop = o.apply(prop)
    prop = add_engineered_features(prop)
    warnings = []
    if model_key not in EXTRAPOLATING:
        warnings.append(f"{MODEL_NAMES[model_key]} cannot extrapolate beyond its training range; prefer physics/linear for what-if.")
    # FE physical power is itself a function of the changed parameters; when caps/V/f/activity
    # change we scale it with the CV^2f term so the model is not fed a stale FE estimate.
    prop, applied = rescale_fe_physical(cur, prop, {o.feature for o in overrides})
    if applied:
        warnings.append("FE physical power was rescaled with the CV^2f term to reflect the changed parameters.")
    res = WhatIfResult(rows, overrides, model_key, models[model_key].predict(cur), models[model_key].predict(prop))
    for k, m in models.items():
        if k != model_key and k != "baseline":
            res.others[k] = (m.predict(cur), m.predict(prop))
    iv = (meta.get("cv") or {}).get("intervals", {}).get(model_key)
    res.interval = iv
    res.warnings = warnings
    return res


def render_whatif(res: WhatIfResult) -> str:
    rows = res.rows
    ident = [c for c in ("design", "build", "fub", "workload", "operating_point") if c in rows.columns]
    multi = len(rows) > 1
    out = []
    head = ", ".join(f"{c}={rows[c].iloc[0]}" for c in ident if rows[c].nunique() == 1)
    out.append(f"What-if prediction  [{MODEL_NAMES[res.model_key]}]  {head}")
    if multi:
        out.append(f"  {len(rows)} rows aggregated (sum of predicted power)")
    out.append("")
    cur_sum, prop_sum = float(np.nansum(res.current)), float(np.nansum(res.proposed))
    feats = sorted({o.feature for o in res.overrides})
    cur_df = add_engineered_features(rows)
    prop_df = cur_df
    for o in res.overrides:
        prop_df = o.apply(prop_df)
    def _show(v: pd.Series) -> str:
        v = pd.to_numeric(v, errors="coerce")
        if not multi or v.nunique() == 1:
            return f"{float(v.iloc[0]):.4g}"
        return f"{v.min():.3g} .. {v.max():.3g} (sum {v.sum():.3g})"

    lines_cur = [("FE physical power", fmt_mw(float(cur_df["fe_physical_mw"].sum())))]
    for f in feats:
        lines_cur.append((label(f), _show(cur_df[f])))
    lines_cur.append(("predicted BE power", fmt_mw(cur_sum)))
    if "be_mw" in rows.columns and rows["be_mw"].notna().all():
        lines_cur.append(("actual BE power", fmt_mw(float(rows["be_mw"].sum()))))
    out.append("CURRENT")
    out += [f"  {k:<22}{v:>14}" for k, v in lines_cur]
    out.append("")
    out.append("WHAT IF   " + "; ".join(o.describe() for o in res.overrides))
    for f in feats:
        out.append(f"  {label(f):<22}{_show(prop_df[f]):>14}")
    out.append(f"  {'predicted BE power':<22}{fmt_mw(prop_sum):>14}")
    if res.interval:
        lo, hi = res.interval["p05"], res.interval["p95"]
        out.append(f"  {'90% interval':<22}{fmt_mw(prop_sum * (1 + lo / 100)):>14} .. {fmt_mw(prop_sum * (1 + hi / 100))}")
    out.append("")
    d = res.delta_pct
    word = "reduction" if d < 0 else "increase"
    out.append(f"estimated {word}   {fmt_pct(abs(d))}   ({fmt_mw(prop_sum - cur_sum)})")
    if res.others:
        out.append("")
        rows_t = [[MODEL_NAMES[k], fmt_mw(float(np.nansum(c))), fmt_mw(float(np.nansum(p))),
                   fmt_pct((np.nansum(p) - np.nansum(c)) / np.nansum(c) * 100, signed=True) if np.nansum(c) else "n/a"]
                  for k, (c, p) in res.others.items()]
        out.append(table(["Other models", "current", "what-if", "delta"], rows_t))
    for w in res.warnings:
        out.append(f"NOTE: {w}")
    out.append("This is a model prediction under the stated change; validate against a real build before acting on it.")
    return "\n".join(out)
