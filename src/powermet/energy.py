"""Compatibility facade: energy analysis now lives in decomposition.py, deltas.py and frontier.py."""

from powermet.decomposition import decompose, render_decomposition  # noqa: F401
from powermet.deltas import DELTA_METRICS, TIMING_METRICS, BuildDelta, build_deltas, classify, render_deltas  # noqa: F401
from powermet.frontier import FrontierPoint, ascii_scatter, frontier, render_frontier  # noqa: F401
