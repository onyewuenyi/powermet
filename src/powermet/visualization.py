"""Optional PNG charts (matplotlib). Every chart has a text interpretation elsewhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from powermet.correlation import ANALYSIS_FEATURES, correlation_matrix
from powermet.deps import available
from powermet.schema import label

CHARTS = ("fe_logical_vs_be.png", "fe_physical_vs_be.png", "error_vs_wire_cap.png", "feature_correlations.png")


def plots_available() -> bool:
    return available("matplotlib")


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # headless; never opens a window
    import matplotlib.pyplot as plt

    return plt


def _scatter_fe_be(plt, df: pd.DataFrame, fe_col: str, path: Path, r2: float | None = None) -> None:
    x = df[fe_col].to_numpy(dtype=float)
    y = df["be_mw"].to_numpy(dtype=float)
    lim = float(np.nanmax(np.concatenate([x, y]))) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(x, y, s=8, alpha=0.5, color="#3b6ea5", edgecolors="none")
    ax.plot([0, lim], [0, lim], color="#888", lw=1, ls="--", label="BE = FE")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"{label(fe_col)} (mW)")
    ax.set_ylabel("BE Power (mW)")
    title = f"{label(fe_col)} vs BE"
    if r2 is not None and np.isfinite(r2):
        title += f"  (R² = {r2:.2f})"
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _error_vs_feature(plt, df: pd.DataFrame, feat: str, err_col: str, path: Path, r: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df[feat], df[err_col], s=8, alpha=0.5, color="#c0603a", edgecolors="none")
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_xlabel(label(feat))
    ax.set_ylabel(label(err_col))
    title = f"{label(err_col)} vs {label(feat)}"
    if r is not None and np.isfinite(r):
        title += f"  (r = {r:.2f})"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _corr_heatmap(plt, df: pd.DataFrame, path: Path) -> None:
    cols = ["fe_logical_mw", "fe_physical_mw", "be_mw", "physical_error_pct", *ANALYSIS_FEATURES]
    m = correlation_matrix(df, cols)
    names = [label(c) for c in m.columns]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlations (association, not causation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_charts(df: pd.DataFrame, out_dir: str | Path, stats: dict | None = None) -> list[Path]:
    """Write the standard chart set. Returns written paths; [] if matplotlib is missing."""
    if not plots_available():
        return []
    plt = _plt()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = stats or {}
    written = []

    p = out / "fe_logical_vs_be.png"
    _scatter_fe_be(plt, df, "fe_logical_mw", p, stats.get("logical_r2"))
    written.append(p)
    p = out / "fe_physical_vs_be.png"
    _scatter_fe_be(plt, df, "fe_physical_mw", p, stats.get("physical_r2"))
    written.append(p)
    if "wire_cap_pf" in df.columns:
        p = out / "error_vs_wire_cap.png"
        _error_vs_feature(plt, df, "wire_cap_pf", "physical_error_pct", p, stats.get("wire_cap_r"))
        written.append(p)
    p = out / "feature_correlations.png"
    _corr_heatmap(plt, df, p)
    written.append(p)
    return written


def frontier_chart(points_by_design: dict, path: Path) -> Path:
    """Power vs Fmax per build, one panel per design; Pareto builds filled, others hollow."""
    plt = _plt()
    designs = list(points_by_design)
    n = len(designs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, d in zip(axes[0], designs):
        pts = [p for p in points_by_design[d] if np.isfinite(p.fmax_ghz)]
        for p in pts:
            ax.scatter(p.fmax_ghz, p.power_mw, s=40, color="#3b6ea5" if p.pareto else "white", edgecolors="#3b6ea5", zorder=3)
            ax.annotate(p.build, (p.fmax_ghz, p.power_mw), textcoords="offset points", xytext=(4, 4), fontsize=7)
        if len(pts) > 1:
            ax.plot([p.fmax_ghz for p in pts], [p.power_mw for p in pts], color="#bbb", lw=0.8, zorder=2)
        ax.set_title(f"{d}: power vs Fmax by build")
        ax.set_xlabel("Fmax of worst partition (GHz)")
        ax.set_ylabel("BE power (mW)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
