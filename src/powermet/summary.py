"""Dataset-level engineering summary shared by `analyze summary` and the report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.errors import stage_metrics
from powermet.textfmt import fmt_int, fmt_mw, fmt_pct, fmt_r, heading, kv


@dataclass
class Summary:
    n_rows: int
    n_designs: int
    n_builds: int
    n_fubs: int
    designs: list[str]
    be_stats: dict[str, float]
    logical: dict[str, float]
    physical: dict[str, float]
    missing_features: dict[str, int] = field(default_factory=dict)


def summarize(df: pd.DataFrame) -> Summary:
    be = pd.to_numeric(df["be_mw"], errors="coerce")
    be_stats = {
        "mean": float(be.mean()),
        "median": float(be.median()),
        "p05": float(be.quantile(0.05)),
        "p95": float(be.quantile(0.95)),
        "min": float(be.min()),
        "max": float(be.max()),
        "total": float(be.sum()),
    }
    feats = ["wire_cap_pf", "cell_cap_pf", "area", "cell_count", "fanout", "frequency_ghz"]
    missing = {c: int(df[c].isna().sum()) for c in feats if c in df.columns and df[c].isna().any()}
    return Summary(
        n_rows=len(df),
        n_designs=int(df["design"].nunique()) if "design" in df else 0,
        n_builds=int(df.groupby("design")["build"].nunique().sum()) if "design" in df and "build" in df else 0,
        n_fubs=int(df.groupby("design")["fub"].nunique().sum()) if "design" in df and "fub" in df else 0,
        designs=sorted(map(str, df["design"].unique())) if "design" in df else [],
        be_stats=be_stats,
        logical=stage_metrics(df, "logical"),
        physical=stage_metrics(df, "physical"),
        missing_features=missing,
    )


def render_summary(s: Summary) -> str:
    out = [heading("Power Metrology Summary"), ""]
    out.append(kv([
        ("Designs", fmt_int(s.n_designs)),
        ("Builds", fmt_int(s.n_builds)),
        ("FUBs", fmt_int(s.n_fubs)),
        ("Measurements", fmt_int(s.n_rows)),
    ], indent=0))
    out += ["", "BE Power", kv([
        ("Mean", fmt_mw(s.be_stats["mean"])),
        ("Median", fmt_mw(s.be_stats["median"])),
        ("P95", fmt_mw(s.be_stats["p95"])),
        ("Range", f"{fmt_mw(s.be_stats['min'])} .. {fmt_mw(s.be_stats['max'])}"),
    ])]
    for name, m in (("FE Logical -> BE", s.logical), ("FE Physical -> BE", s.physical)):
        out += ["", name, kv([
            ("Mean Error", fmt_pct(-m.get("mean_error_pct", np.nan), signed=True) + "  (BE - FE, relative to BE)"),
            ("MAPE", fmt_pct(m.get("mape"))),
            ("P95 |Error|", fmt_pct(m.get("p95_ape"))),
            ("R^2", fmt_r(m.get("r2"))),
            ("n", fmt_int(m.get("n"))),
        ])]
    if s.missing_features:
        out += ["", "Missing feature values", kv([(k, fmt_int(v)) for k, v in s.missing_features.items()])]
    return "\n".join(out)


def compare_stages_text(s: Summary) -> str:
    lm, pm = s.logical.get("mape", np.nan), s.physical.get("mape", np.nan)
    if not (np.isfinite(lm) and np.isfinite(pm)):
        return "Stage comparison unavailable (insufficient data)."
    if pm < lm:
        rel = (lm - pm) / lm * 100 if lm else 0
        return (f"FE physical power is more predictive of BE power than FE logical power in this dataset "
                f"(MAPE {fmt_pct(pm)} vs {fmt_pct(lm)}, a {fmt_pct(rel)} relative reduction; "
                f"R^2 {fmt_r(s.physical.get('r2'))} vs {fmt_r(s.logical.get('r2'))}).")
    return (f"FE logical power is at least as predictive as FE physical power in this dataset "
            f"(MAPE {fmt_pct(lm)} vs {fmt_pct(pm)}). Check whether the physical estimate is being fed the same activity.")
