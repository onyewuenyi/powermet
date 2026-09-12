"""Derived FE -> BE error metrics. Pure DataFrame -> DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from powermet.schema import DERIVED_COLUMNS


def fmax_from_timing(clock_period_ps, wns_ps) -> pd.Series:
    """fmax (GHz) = 1000 / (period - WNS); NaN where the path delay is not positive."""
    delay = pd.to_numeric(pd.Series(clock_period_ps), errors="coerce") - pd.to_numeric(pd.Series(wns_ps), errors="coerce")
    return 1000.0 / delay.where(delay > 0)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with NaN wherever den is 0, NaN, or non-finite."""
    n = pd.to_numeric(num, errors="coerce").astype(float)
    d = pd.to_numeric(den, errors="coerce").astype(float)
    ok = np.isfinite(d) & (d != 0)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    out[ok] = n[ok] / d[ok]
    return out


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with error/ratio columns added. Never drops rows."""
    out = df.copy()
    be = pd.to_numeric(out["be_mw"], errors="coerce").astype(float)
    fe_l = pd.to_numeric(out["fe_logical_mw"], errors="coerce").astype(float)
    fe_p = pd.to_numeric(out["fe_physical_mw"], errors="coerce").astype(float)

    out["logical_error_mw"] = be - fe_l
    out["physical_error_mw"] = be - fe_p
    out["logical_error_pct"] = _safe_div(be - fe_l, be) * 100.0
    out["physical_error_pct"] = _safe_div(be - fe_p, be) * 100.0
    out["logical_to_be_ratio"] = _safe_div(fe_l, be)
    out["physical_to_be_ratio"] = _safe_div(fe_p, be)
    add_convergence_metrics(out)
    if "clock_period_ps" in out.columns and "wns_ps" in out.columns and "fmax_ghz" not in out.columns:
        out["fmax_ghz"] = fmax_from_timing(out["clock_period_ps"], out["wns_ps"]).to_numpy()
    return out


def cdyn_pf(dynamic_mw, voltage_v, frequency_ghz) -> pd.Series:
    """Effective switched capacitance Cdyn = P_dyn / (V^2 f). mW / (V^2 GHz) = 1e-3 W / (V^2 1e9 Hz) = 1e-12 F = pF exactly."""
    v = pd.to_numeric(pd.Series(voltage_v), errors="coerce").astype(float)
    f = pd.to_numeric(pd.Series(frequency_ghz), errors="coerce").astype(float)
    return _safe_div(pd.Series(dynamic_mw), v ** 2 * f)


def add_convergence_metrics(out: pd.DataFrame) -> pd.DataFrame:
    """In place: be_dynamic_mw, cdyn_pf, fe_cdyn_pf, leakage_fraction (NaN where the inputs are absent)."""
    n = len(out)
    col = lambda c: pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns else pd.Series(np.nan, index=out.index, dtype=float)
    be, leak = col("be_mw"), col("be_leakage_mw")
    out["be_dynamic_mw"] = (be - leak).clip(lower=0) if n else pd.Series(dtype=float)
    out["leakage_fraction"] = _safe_div(leak, be) if n else pd.Series(dtype=float)
    if n and "voltage_v" in out.columns and "frequency_ghz" in out.columns:
        out["cdyn_pf"] = cdyn_pf(out["be_dynamic_mw"], out["voltage_v"], out["frequency_ghz"]).to_numpy()
        out["fe_cdyn_pf"] = cdyn_pf((col("fe_physical_mw") - col("fe_leakage_mw")).clip(lower=0), out["voltage_v"], out["frequency_ghz"]).to_numpy()
    else:
        out["cdyn_pf"] = np.nan
        out["fe_cdyn_pf"] = np.nan
    return out


def has_derived(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in DERIVED_COLUMNS)


def prediction_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard regression metrics; NaN pairs are excluded and counted."""
    yt = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    yp = np.asarray(pd.to_numeric(pd.Series(y_pred), errors="coerce"), dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    err = yp - yt
    abs_err = np.abs(err)
    nz = yt != 0
    ape = np.full_like(yt, np.nan)
    ape[nz] = abs_err[nz] / np.abs(yt[nz]) * 100.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": n,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.nanmean(ape)),
        "mean_error_pct": float(np.nanmean(np.where(nz, err / np.where(nz, yt, 1) * 100.0, np.nan))),
        "r2": r2,
        "p50_ape": float(np.nanpercentile(ape, 50)),
        "p95_ape": float(np.nanpercentile(ape, 95)),
    }
