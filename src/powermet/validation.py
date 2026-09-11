"""Data validation. Reports problems; never mutates or drops the user's data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.schema import (
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
)
from powermet.textfmt import fmt_int

ERROR = "error"
WARNING = "warning"

RATIO_BOUNDS = (0.2, 5.0)
LOG_Z_LIMIT = 4.0


@dataclass
class Issue:
    level: str
    code: str
    message: str
    rows: list[int] = field(default_factory=list)   # positional row indices (0-based)
    column_level: bool = False                      # affects every row

    @property
    def count(self) -> int:
        return len(self.rows)


@dataclass
class ValidationReport:
    n_rows: int
    issues: list[Issue]
    error_mask: np.ndarray      # True where the row has at least one error
    warning_mask: np.ndarray

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def n_errors(self) -> int:
        return int(self.error_mask.sum())

    @property
    def n_warnings(self) -> int:
        return int(self.warning_mask.sum())

    @property
    def n_valid(self) -> int:
        return int((~self.error_mask).sum())

    @property
    def ok(self) -> bool:
        return not self.errors

    def reject_reasons(self) -> pd.Series:
        """Per-row ';'-joined error codes for rows with errors."""
        reasons: dict[int, list[str]] = {}
        for iss in self.errors:
            rows = range(self.n_rows) if iss.column_level else iss.rows
            for r in rows:
                reasons.setdefault(r, []).append(iss.code)
        return pd.Series({r: ";".join(v) for r, v in reasons.items()}, dtype=object)

    def render(self) -> str:
        lines = ["Validation Summary", ""]
        width = 22
        lines.append(f"{'Rows:':<{width}}{fmt_int(self.n_rows):>8}")
        lines.append(f"{'Valid:':<{width}}{fmt_int(self.n_valid):>8}")
        lines.append(f"{'Warnings:':<{width}}{fmt_int(self.n_warnings):>8}")
        lines.append(f"{'Errors:':<{width}}{fmt_int(self.n_errors):>8}")
        if self.errors:
            lines += ["", "Errors:"]
            for iss in self.errors:
                n = "all" if iss.column_level else fmt_int(iss.count)
                lines.append(f"  {n:>5} rows  {iss.message}")
        if self.warnings:
            lines += ["", "Warnings:"]
            for iss in self.warnings:
                lines.append(f"  {fmt_int(iss.count):>5} rows  {iss.message}")
        lines.append("")
        lines.append("Result: " + ("PASS" if self.ok else "FAIL (errors must be fixed or rows will be rejected on import)"))
        return "\n".join(lines)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def outlier_masks(be: pd.Series, fe_physical: pd.Series, ratio_bounds=RATIO_BOUNDS, log_z_limit=LOG_Z_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """(ratio outside bounds, |z| of log(be) beyond limit) as boolean arrays aligned to `be`.

    Shared by validation (warnings) and sanitize (quality flags) so the rule is defined once.
    """
    be = pd.to_numeric(be, errors="coerce")
    fe = pd.to_numeric(fe_physical, errors="coerce")
    ratio = fe / be.where(be > 0)
    lo, hi = ratio_bounds
    ratio_mask = ((ratio < lo) | (ratio > hi)).fillna(False).to_numpy()
    logz = np.zeros(len(be), dtype=bool)
    pos = be[be > 0]
    if len(pos) >= 10:
        logs = np.log(pos)
        z = (logs - logs.mean()) / (logs.std(ddof=0) or 1.0)
        far = pd.Series(False, index=be.index)
        far.loc[pos.index] = (z.abs() > log_z_limit).to_numpy()
        logz = far.to_numpy()
    return ratio_mask, logz


def validate(df: pd.DataFrame) -> ValidationReport:
    n = len(df)
    issues: list[Issue] = []
    err = np.zeros(n, dtype=bool)
    warn = np.zeros(n, dtype=bool)

    def add(level: str, code: str, message: str, mask=None, column_level: bool = False):
        rows = [] if mask is None else [int(i) for i in np.flatnonzero(np.asarray(mask, dtype=bool))]
        if not rows and not column_level:
            return
        issues.append(Issue(level, code, message, rows, column_level))
        target = err if level == ERROR else warn
        if column_level:
            target[:] = True
        else:
            target[rows] = True

    # 1. Required columns
    missing_req = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in missing_req:
        add(ERROR, f"missing_column:{c}", f"required column '{c}' is missing", column_level=True)

    # 2. Numeric columns must be numeric
    for c in NUMERIC_COLUMNS:
        if c not in df.columns:
            continue
        raw = df[c]
        coerced = pd.to_numeric(raw, errors="coerce")
        bad = coerced.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
        add(ERROR, f"non_numeric:{c}", f"non-numeric values in '{c}'", bad.to_numpy())

    # 3. Missing required values
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            continue
        if c in NUMERIC_COLUMNS:
            missing = _numeric(df, c).isna() & ~(df[c].notna() & (df[c].astype(str).str.strip() != ""))
        else:
            missing = df[c].isna() | (df[c].astype(str).str.strip() == "")
        add(ERROR, f"missing_value:{c}", f"missing {c}", missing.to_numpy())

    # 4. Zero BE power (undefined percentage error)
    if TARGET in df.columns:
        zero = (_numeric(df, TARGET) == 0)
        add(ERROR, "zero_be", f"{TARGET} is zero (percentage error undefined)", zero.to_numpy())

    # 5. Non-negative constraints
    for c in NON_NEGATIVE_COLUMNS:
        if c not in df.columns:
            continue
        neg = _numeric(df, c) < 0
        add(ERROR, f"negative:{c}", f"negative {c}", neg.to_numpy())

    # 6. Duplicate keys
    keys = [c for c in KEY_COLUMNS if c in df.columns]
    if keys:
        dup = df.duplicated(subset=keys, keep=False)
        add(ERROR, "duplicate_key", "duplicate measurement key (" + ", ".join(keys) + ")", dup.to_numpy())

    # 7. Missing optional features (warning)
    feat_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    feat_absent = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if feat_present:
        any_missing = np.zeros(n, dtype=bool)
        for c in feat_present:
            any_missing |= _numeric(df, c).isna().to_numpy()
        add(WARNING, "missing_feature", "contain missing optional physical features", any_missing)
    if feat_absent:
        issues.append(Issue(WARNING, "absent_feature_columns",
                            "feature columns not present: " + ", ".join(feat_absent), [], column_level=True))

    # 8. Suspicious outliers (warning)
    if TARGET in df.columns and "fe_physical_mw" in df.columns:
        lo, hi = RATIO_BOUNDS
        ratio_mask, logz_mask = outlier_masks(_numeric(df, TARGET), _numeric(df, "fe_physical_mw"))
        add(WARNING, "outlier_ratio", f"fe_physical/be ratio outside [{lo}, {hi}]", ratio_mask)
        add(WARNING, "outlier_be", f"be_mw is an extreme outlier (|z| > {LOG_Z_LIMIT:g} on log scale)", logz_mask)

    return ValidationReport(n_rows=n, issues=issues, error_mask=err, warning_mask=warn)
