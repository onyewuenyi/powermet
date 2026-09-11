"""Plain-text formatting helpers for engineering CLI output. No color, no deps."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


def fmt_pct(x, signed: bool = False) -> str:
    if _isnan(x):
        return "n/a"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def fmt_mw(x) -> str:
    if _isnan(x):
        return "n/a"
    return f"{x:.1f} mW" if abs(x) < 1000 else f"{x:,.0f} mW"


def fmt_num(x, digits: int = 1) -> str:
    if _isnan(x):
        return "n/a"
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) >= 100):
        return f"{int(x):,}"
    return f"{x:,.{digits}f}"


def fmt_r(x) -> str:
    """Correlation coefficient / R^2 to two decimals."""
    if _isnan(x):
        return "n/a"
    return f"{x:.2f}"


def fmt_int(x) -> str:
    if _isnan(x):
        return "n/a"
    return f"{int(x):,}"


def heading(text: str, char: str = "=") -> str:
    return f"{text}\n{char * len(text)}"


def table(headers: Sequence[str], rows: Iterable[Sequence], align: Sequence[str] | None = None) -> str:
    """Fixed-width table. align: 'l' or 'r' per column (default: first left, rest right)."""
    rows = [[str(c) for c in r] for r in rows]
    ncol = len(headers)
    if align is None:
        align = ["l"] + ["r"] * (ncol - 1)
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells):
        return "  ".join(
            (c.ljust(widths[i]) if align[i] == "l" else c.rjust(widths[i])) for i, c in enumerate(cells)
        ).rstrip()

    out = [fmt_row(headers), "-" * (sum(widths) + 2 * (ncol - 1))]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def kv(pairs: Iterable[tuple[str, object]], indent: int = 2, width: int | None = None) -> str:
    pairs = list(pairs)
    if width is None:
        width = max((len(k) for k, _ in pairs), default=0) + 2
    pad = " " * indent
    return "\n".join(f"{pad}{k + ':':<{width}}{v}" for k, v in pairs)
