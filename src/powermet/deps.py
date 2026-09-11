"""Optional dependency detection.

Every optional import in powermet goes through this module so the CLI can
degrade gracefully and `powermet doctor` can report a consistent picture.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache

REQUIRED = "REQUIRED"
OPTIONAL = "OPTIONAL"


@dataclass(frozen=True)
class DepSpec:
    name: str          # display name
    module: str        # import name
    tier: str          # REQUIRED / OPTIONAL
    purpose: str


DEPS: tuple[DepSpec, ...] = (
    DepSpec("pandas", "pandas", REQUIRED, "tabular data"),
    DepSpec("numpy", "numpy", REQUIRED, "numerics"),
    DepSpec("pyarrow", "pyarrow", REQUIRED, "Parquet read/write"),
    DepSpec("duckdb", "duckdb", REQUIRED, "embedded analytical queries"),
    DepSpec("scipy", "scipy", REQUIRED, "correlation p-values"),
    DepSpec("scikit-learn", "sklearn", REQUIRED, "gradient-boosting model"),
    DepSpec("joblib", "joblib", REQUIRED, "model serialization"),
    DepSpec("matplotlib", "matplotlib", OPTIONAL, "PNG charts (else text only)"),
    DepSpec("psutil", "psutil", OPTIONAL, "current-RSS sampling in profiles"),
)

MIN_PYTHON = (3, 11)


@dataclass(frozen=True)
class DepStatus:
    spec: DepSpec
    available: bool
    version: str = ""
    error: str = ""


@lru_cache(maxsize=None)
def check(module: str) -> DepStatus:
    spec = next((d for d in DEPS if d.module == module), None)
    if spec is None:
        spec = DepSpec(module, module, OPTIONAL, "")
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # ImportError or a broken install
        return DepStatus(spec, False, error=f"{type(exc).__name__}: {exc}")
    return DepStatus(spec, True, version=str(getattr(mod, "__version__", "?")))


def available(module: str) -> bool:
    return check(module).available


def all_statuses() -> list[DepStatus]:
    return [check(d.module) for d in DEPS]


def python_ok() -> bool:
    return sys.version_info[:2] >= MIN_PYTHON


def missing_required() -> list[DepStatus]:
    return [s for s in all_statuses() if s.spec.tier == REQUIRED and not s.available]


def doctor_report() -> tuple[str, bool]:
    """Return (text, ready) for `powermet doctor`."""
    lines = ["PowerMet Doctor", ""]
    pyver = ".".join(map(str, sys.version_info[:3]))
    py_state = "OK" if python_ok() else f"TOO OLD (need >= {'.'.join(map(str, MIN_PYTHON))})"
    lines.append(f"{'Python':<14}{pyver:<12}{py_state}")
    ready = python_ok()
    for st in all_statuses():
        if st.available:
            state = "OK"
        elif st.spec.tier == REQUIRED:
            state = "MISSING (REQUIRED)"
            ready = False
        else:
            state = f"MISSING (OPTIONAL: {st.spec.purpose})"
        lines.append(f"{st.spec.name:<14}{st.version:<12}{state}")
    lines.append("")
    lines.append(f"Interpreter: {sys.executable}")
    lines.append("")
    if ready:
        degraded = [s.spec.name for s in all_statuses() if not s.available]
        if degraded:
            lines.append("Status: READY (degraded: " + ", ".join(degraded) + ")")
        else:
            lines.append("Status: READY")
    else:
        lines.append("Status: NOT READY")
    return "\n".join(lines), ready
