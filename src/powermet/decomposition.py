"""Data-movement energy decomposition: attribute predicted power to compute / wire / movement / leakage.

Uses the fitted `datamove` model: predicted power = sum(coef_k * term_k) + intercept, so each FUB's
predicted power splits into cell switching, wire switching, data movement (bits x distance) and leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powermet.features import add_engineered_features
from powermet.modeling import DATAMOVE_TERMS, LinearModel
from powermet.schema import identity_key, label
from powermet.textfmt import fmt_mw, fmt_pct, table



def decompose(df: pd.DataFrame, model: LinearModel) -> pd.DataFrame:
    X = add_engineered_features(df)
    c = model.contributions(X)
    out = df[[k for k in ("design", "build", "fub", "model_root", "partition", "workload", "operating_point", "be_mw") if k in df.columns]].copy()
    for term, col in DATAMOVE_TERMS.items():
        out[col] = c[term].clip(lower=0) if term in c.columns else 0.0
    out["intercept_mw"] = c["intercept"]
    out["predicted_mw"] = c.sum(axis=1)
    parts = [DATAMOVE_TERMS[t] for t in model.features if t in DATAMOVE_TERMS]
    tot = out[parts].sum(axis=1).where(lambda s: s > 0)
    for col in parts:
        out[col + "_share"] = out[col] / tot
    if "movement_mw" in out.columns and "bits_per_cycle" in df.columns and "avg_net_length_um" in df.columns and "frequency_ghz" in df.columns:
        bits_per_s = pd.to_numeric(df["bits_per_cycle"], errors="coerce") * pd.to_numeric(df["frequency_ghz"], errors="coerce") * 1e9
        dist_mm = pd.to_numeric(df["avg_net_length_um"], errors="coerce") / 1000.0
        # mW / (bits/s * mm) = 1e-3 J/s / (bit/s * mm) = 1e-3 J/(bit mm) -> pJ/(bit mm) = *1e9
        out["pj_per_bit_mm"] = (out["movement_mw"] * 1e-3) / (bits_per_s * dist_mm).where(lambda s: s > 0) * 1e12
    return out


def render_decomposition(dec: pd.DataFrame, model: LinearModel, top: int = 8) -> str:
    parts = [DATAMOVE_TERMS[t] for t in model.features if t in DATAMOVE_TERMS]
    out = ["Data-movement energy model (compact analytical form fitted to BE power)", ""]
    out.append("  P(FUB) = " + " + ".join(f"{model.coefficients()[t]:.3g} * {t}" for t in model.features) + f" + {model.intercept_:.3g}")
    out.append("")
    keys = [k for k in ("design", "workload", "operating_point") if k in dec.columns]
    g = dec.groupby(keys)[["predicted_mw", "be_mw"] + parts].sum().reset_index()
    rows = []
    for _, r in g.iterrows():
        tot = sum(r[p] for p in parts) or np.nan
        rows.append([r[k] for k in keys] + [fmt_mw(r["be_mw"]), fmt_mw(r["predicted_mw"])] + [fmt_pct(r[p] / tot * 100) for p in parts])
    out.append(table([k.capitalize() for k in keys] + ["BE power", "Predicted"] + [label(p) for p in parts], rows))
    if "pj_per_bit_mm" in dec.columns:
        d = dec.dropna(subset=["pj_per_bit_mm"])
        if len(d):
            out.append("")
            out.append(f"Data-movement energy coefficient: {d['pj_per_bit_mm'].median():.3f} pJ per bit-mm "
                       f"(constant within an operating point by construction: coef x V^2; compare across designs/op points).")
    if "movement_mw_share" in dec.columns:
        out.append("")
        out.append(f"FUBs with the largest data-movement share (top {top}):")
        out.append("")
        key = identity_key(dec)
        top_rows = dec.sort_values("movement_mw", ascending=False).drop_duplicates(subset=[key]).head(top)
        out.append(table([label(key) if key != "model_root" else "Model root", "Workload", "Predicted", "Movement", "Share", "pJ/bit-mm"],
                         [[r[key], r.get("workload", ""), fmt_mw(r["predicted_mw"]), fmt_mw(r["movement_mw"]),
                           fmt_pct(r["movement_mw_share"] * 100), f"{r['pj_per_bit_mm']:.3f}" if "pj_per_bit_mm" in dec.columns and pd.notna(r["pj_per_bit_mm"]) else "n/a"]
                          for _, r in top_rows.iterrows()]))
    out.append("")
    out.append("Terms are fitted coefficients on physically-motivated proxies; the split is a model attribution, not a measurement.")
    return "\n".join(out)
