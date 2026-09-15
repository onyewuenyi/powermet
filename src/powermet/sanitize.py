"""Correlation-dataset sanitization (V1).

Goes beyond schema validation: uses the long provenance table, lineage table and build
metadata to detect problems that only show up across sources or across builds. Produces
per-row quality flags and a sanitized dataset of usable rows. Nothing is deleted: flagged
rows stay in measurements.parquet and quality_flags.parquet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermet.config import Config, Project
from powermet.extract import SOURCE_METRICS
from powermet.extract.base import CANONICAL_UNITS
from powermet.ingest import write_table
from powermet.lineage import BLOCKING_ISSUES, PHYSICAL_REQUIRED, TIMING_ISSUES
from powermet.schema import KEY_COLUMNS, NON_NEGATIVE_COLUMNS, POWER_GROUP_COLUMNS, REQUIRED_POWER_COLUMNS
from powermet.storage import load_dataset, load_table, register_views, sanitized_path
from powermet.textfmt import fmt_int, fmt_pct
from powermet.validation import outlier_masks

REQUIRED_METRICS = REQUIRED_POWER_COLUMNS
PHYSICAL_METRICS = PHYSICAL_REQUIRED

# check -> (label, blocks usability?)
CHECKS: dict[str, tuple[str, bool]] = {
    "missing_metric": ("Missing power metrics", True),
    "missing_physical": ("Missing physical data", True),
    "duplicate": ("Duplicate records", True),
    "duplicate_source_row": ("Duplicate rows in source reports", False),
    "unit_normalized": ("Unit conversions applied", False),
    "unit_suspect": ("Unit inconsistencies (magnitude)", True),
    "negative": ("Impossible negative values", True),
    "near_zero_be": ("Zero / near-zero BE power", True),
    "stale_build": ("Stale / superseded builds", True),
    "outlier": ("Outliers flagged", False),
    "lineage_mismatch": ("FE/BE lineage mismatches", True),
    "timing_missing": ("Timing missing (partition not in PrimeTime)", False),
    "timing_suspect": ("Timing inconsistencies (WNS > period, negative period)", False),
    "vectorless_power": ("BE power from vectorless activity (lower trust)", False),
    "intent_missing": ("No UPF power domain for FUB", False),
    "intent_mismatch": ("UPF voltage != operating point voltage", False),
    "unmapped_objects": ("Unmapped report objects", False),
    "leakage_suspect": ("Leakage > total or varies with workload (component mismatch)", False),
    "group_sum_mismatch": ("Power groups do not sum to BE total (> 5%)", False),
}
BLOCKING_LINEAGE = BLOCKING_ISSUES
TIMING_LINEAGE = TIMING_ISSUES


@dataclass
class QualityReport:
    n_rows: int
    n_fubs: int
    counts: dict[str, int]                # check -> rows affected
    usable_mask: np.ndarray
    flags: pd.Series                      # per-row ';'-joined checks
    details: dict[str, list[str]] = field(default_factory=dict)
    n_unmapped_objects: int = 0
    n_source_duplicates: int = 0
    n_unit_conversions: int = 0
    metric_quality: pd.DataFrame | None = None

    @property
    def n_usable(self) -> int:
        return int(self.usable_mask.sum())

    @property
    def usable_pct(self) -> float:
        return self.n_usable / self.n_rows * 100 if self.n_rows else float("nan")

    def render(self) -> str:
        w = 34
        lines = ["CORRELATION DATA QUALITY", "-" * 44]
        lines.append(f"{'Measurements analyzed':<{w}}{fmt_int(self.n_rows):>10}")
        lines.append(f"{'FUB x build combinations':<{w}}{fmt_int(self.n_fubs):>10}")
        lines.append(f"{'Usable':<{w}}{fmt_int(self.n_usable):>10}")
        for check, (label, blocks) in CHECKS.items():
            n = self.counts.get(check, 0)
            if check == "unmapped_objects":
                n = self.n_unmapped_objects
            elif check == "duplicate_source_row":
                n = self.n_source_duplicates
            elif check == "unit_normalized":
                n = self.n_unit_conversions
            if n:
                lines.append(f"{label:<{w}}{fmt_int(n):>10}{'' if blocks else '   info'}")
        lines.append("")
        lines.append(f"{'Usable dataset':<{w}}{fmt_pct(self.usable_pct):>10}")
        for check, items in self.details.items():
            if items:
                lines.append("")
                lines.append(CHECKS.get(check, (check, False))[0] + ":")
                for it in items[:8]:
                    lines.append(f"  {it}")
                if len(items) > 8:
                    lines.append(f"  ... {len(items) - 8} more")
        return "\n".join(lines)


def sanitize(df: pd.DataFrame, cfg: Config, long: pd.DataFrame | None = None, lineage: pd.DataFrame | None = None,
             unmapped: pd.DataFrame | None = None, intent: pd.DataFrame | None = None) -> QualityReport:
    n = len(df)
    flags: dict[int, list[str]] = {}
    counts: dict[str, int] = {}
    details: dict[str, list[str]] = {}
    blocking = np.zeros(n, dtype=bool)

    def mark(check: str, mask, detail: list[str] | None = None):
        m = np.asarray(mask, dtype=bool)
        counts[check] = int(m.sum())
        if detail:
            details[check] = detail
        if CHECKS[check][1]:
            blocking[m] = True
        for i in np.flatnonzero(m):
            flags.setdefault(int(i), []).append(check)

    num = {c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index) for c in
           set(REQUIRED_METRICS) | set(PHYSICAL_METRICS) | set(NON_NEGATIVE_COLUMNS)}

    # 1 missing metrics
    miss = np.zeros(n, dtype=bool)
    for c in REQUIRED_METRICS:
        miss |= num[c].isna().to_numpy()
    mark("missing_metric", miss)

    # 2 incomplete physical mapping
    missp = np.zeros(n, dtype=bool)
    for c in PHYSICAL_METRICS:
        missp |= num[c].isna().to_numpy()
    fubs = sorted(df.loc[missp, "fub"].astype(str).unique()) if "fub" in df else []
    mark("missing_physical", missp, [f"{f}" for f in fubs][:20])

    # 3 duplicates
    keys = [c for c in KEY_COLUMNS if c in df.columns]
    mark("duplicate", df.duplicated(subset=keys, keep=False).to_numpy() if keys else np.zeros(n, bool))

    # 4 negatives
    neg = np.zeros(n, dtype=bool)
    neg_cols = []
    for c in NON_NEGATIVE_COLUMNS:
        if c in df.columns:
            m = (num[c] < 0).to_numpy()
            if m.any():
                neg_cols.append(f"{c}: {int(m.sum())} rows")
            neg |= m
    mark("negative", neg, neg_cols)

    # 4b leakage component sanity: leakage above the total, or leakage that changes with the workload at a fixed
    #    operating point, means the components were read from different scenarios or columns (the convergence
    #    metrics Cdyn / leakage power depend on this split). A high leakage share by itself is legitimate at idle.
    if "be_leakage_mw" in df.columns:
        lk = pd.to_numeric(df["be_leakage_mw"], errors="coerce")
        bad = np.array((lk > num["be_mw"] * 1.001).fillna(False).to_numpy(), dtype=bool)   # writable copy
        gk = [c for c in ("design", "build", "fub", "operating_point") if c in df.columns]
        if gk and "workload" in df.columns:
            g = lk.groupby([df[c] for c in gk])
            spread = (g.transform("max") - g.transform("min")) / g.transform("mean").where(lambda s: s > 0)
            bad |= (spread * 100 > cfg.leakage_workload_tol_pct).fillna(False).to_numpy()
        mark("leakage_suspect", bad)
    else:
        mark("leakage_suspect", np.zeros(n, dtype=bool))

    # 4c power groups: when a groups report is present its clock / register / combinational / memory columns
    #    should reconstruct the BE total; a larger gap means the group report came from another run or netlist
    gcols = [c for c in POWER_GROUP_COLUMNS if c in df.columns]
    if gcols:
        gsum = sum(pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in gcols)
        has = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in gcols], axis=1).notna().any(axis=1)
        gap = ((gsum - num["be_mw"]).abs() / num["be_mw"].where(num["be_mw"] > 0)) * 100
        mark("group_sum_mismatch", (has & (gap > cfg.group_sum_tol_pct)).fillna(False).to_numpy())
    else:
        mark("group_sum_mismatch", np.zeros(n, dtype=bool))

    # 5 near-zero denominators
    mark("near_zero_be", (num["be_mw"].abs() < cfg.near_zero_mw).fillna(False).to_numpy())

    # 6 unit magnitude check: be_mw wildly off the design median suggests a unit mismatch
    sus = np.zeros(n, dtype=bool)
    if "design" in df.columns:
        med = num["be_mw"].groupby(df["design"]).transform("median")
        ratio = num["be_mw"] / med.where(med > 0)
        sus = ((ratio > cfg.unit_magnitude_ratio) | (ratio < 1 / cfg.unit_magnitude_ratio)).fillna(False).to_numpy()
        sus = sus & ~(num["be_mw"].abs() < cfg.near_zero_mw).fillna(False).to_numpy()   # near-zero is its own check
    mark("unit_suspect", sus)

    # 7 stale builds
    stale = np.zeros(n, dtype=bool)
    stale_detail = []
    if "build_status" in df.columns:
        stale |= (df["build_status"].astype(str) == "superseded").to_numpy()
    if "build_date" in df.columns and "design" in df.columns:
        dates = pd.to_datetime(df["build_date"], errors="coerce")
        newest = dates.groupby(df["design"]).transform("max")
        old = ((newest - dates).dt.days > cfg.stale_days).fillna(False).to_numpy()
        stale |= old
    if stale.any():
        stale_detail = sorted(set(df.loc[stale, "design"].astype(str) + "/" + df.loc[stale, "build"].astype(str)))
    mark("stale_build", stale, stale_detail)

    # 8 outliers (non-blocking) - same rule as validation
    ratio_mask, logz_mask = outlier_masks(num["be_mw"], num["fe_physical_mw"])
    mark("outlier", ratio_mask | logz_mask)

    # 9 lineage mismatches (from lineage table): FE/BE issues block, timing issues are informational
    lm = np.zeros(n, dtype=bool)
    tm = np.zeros(n, dtype=bool)
    lm_detail: list[str] = []
    tm_detail: list[str] = []
    if lineage is not None and len(lineage) and "lineage_ok" in lineage.columns:
        bad = lineage[~lineage["lineage_ok"].astype(bool)]
        if len(bad):
            key = df["design"].astype(str) + "|" + df["build"].astype(str) + "|" + df["fub"].astype(str)
            issues = bad["lineage_issues"].astype(str)
            is_block = issues.apply(lambda x: any(i in x for i in BLOCKING_LINEAGE))
            is_time = issues.apply(lambda x: any(i in x for i in TIMING_LINEAGE))
            def keys(sub):
                return set(sub["design"].astype(str) + "|" + sub["build"].astype(str) + "|" + sub["fub"].astype(str))
            lm = key.isin(keys(bad[is_block])).to_numpy()
            tm = key.isin(keys(bad[is_time])).to_numpy()
            lm_detail = [f"{r.design}/{r.build} {r.fub}: {r.lineage_issues}" for r in bad[is_block].itertuples()]
            tm_detail = sorted(set(f"{r.design}/{r.build} partition {r.partition}" for r in bad[is_time].itertuples()))
            missing_rows = bad[bad["lineage_issues"].str.contains("be_hier_not_in_reports")]
            if len(missing_rows):
                lm_detail.append(f"{len(missing_rows)} FUB/build entries have no BE row (BE hierarchy absent from reports)")
    mark("lineage_mismatch", lm, lm_detail)
    mark("timing_missing", tm, tm_detail)

    # 10 timing consistency
    ts = np.zeros(n, dtype=bool)
    if "wns_ps" in df.columns and "clock_period_ps" in df.columns:
        per = pd.to_numeric(df["clock_period_ps"], errors="coerce")
        wns = pd.to_numeric(df["wns_ps"], errors="coerce")
        ts = ((wns > per) | (per <= 0)).fillna(False).to_numpy()
    mark("timing_suspect", ts)

    # 11 activity provenance and power intent (informational)
    vl = np.zeros(n, dtype=bool)
    if "be_activity_mode" in df.columns:
        vl = df["be_activity_mode"].astype(str).str.lower().str.startswith("vectorless").to_numpy()
    mark("vectorless_power", vl)
    im = np.zeros(n, dtype=bool)
    mm = np.zeros(n, dtype=bool)
    if intent is not None and len(intent):
        key = df["design"].astype(str) + "|" + df["build"].astype(str) + "|" + df["fub"].astype(str)
        ikey = intent["design"].astype(str) + "|" + intent["build"].astype(str) + "|" + intent["fub"].astype(str)
        issues = intent["intent_issues"].astype(str)
        im = key.isin(set(ikey[issues.str.contains("no_power_domain|multiple_power_domains")])).to_numpy()
        mm = key.isin(set(ikey[issues.str.contains("domain_voltage_mismatch|domain_state_missing")])).to_numpy()
        det = sorted(set(intent.loc[issues.str.contains("mismatch"), "voltage_mismatch"].astype(str)))[:8]
        mark("intent_mismatch", mm, det)
    else:
        mark("intent_mismatch", mm)
    mark("intent_missing", im)

    # long-table derived info
    n_unit = n_dup_src = 0
    if long is not None and len(long):
        n_unit = int((long["unit_original"].astype(str) != long["unit"].astype(str)).sum())
        cols = [c for c in ("design", "build", "object", "fub", "metric", "workload", "operating_point", "source_file") if c in long.columns]
        n_dup_src = int(long.duplicated(subset=cols, keep="first").sum())
        if n_unit:
            conv = long[long["unit_original"].astype(str) != long["unit"].astype(str)]
            details["unit_normalized"] = sorted(set(conv["source"].astype(str) + ": " + conv["unit_original"].astype(str)
                                                    + " -> " + conv["unit"].astype(str)))
    n_unmapped = 0
    if unmapped is not None and len(unmapped):
        um = unmapped[unmapped["object_kind"] != "design"] if "object_kind" in unmapped.columns else unmapped
        n_unmapped = int(len(um))
        if n_unmapped:
            details["unmapped_objects"] = sorted(set(um["design"].astype(str) + "/" + um["build"].astype(str) + " " + um["object"].astype(str)))[:20]

    flag_series = pd.Series({i: ";".join(v) for i, v in flags.items()}, dtype=object).reindex(range(n)).fillna("")
    fub_combos = int(df[[c for c in ("design", "build", "fub") if c in df.columns]].drop_duplicates().shape[0]) if n else 0
    return QualityReport(n, fub_combos, counts, ~blocking, flag_series, details, n_unmapped, n_dup_src, n_unit)


# ----------------------------------------------------------------------------- per-metric trust

TRUST_METRICS = ("fe_logical_mw", "fe_physical_mw", "be_voltus_mw", "wire_cap_pf", "cell_cap_pf", "area", "cell_count", "fanout",
                 "activity", "bits_per_cycle", "cg_efficiency", "wire_length_um", "avg_net_length_um", "frequency_ghz", "voltage_v",
                 "wns_ps", "fmax_ghz")


def metric_quality(df: pd.DataFrame, long: pd.DataFrame | None = None, target: str = "be_mw") -> pd.DataFrame:
    """Per-metric trust table: coverage, association with the target, build-to-build stability of that
    association, unit conversions, outliers, and a verdict. This is the 'sanitize each metric' view."""
    from powermet.correlation import pearson

    rows = []
    y = pd.to_numeric(df[target], errors="coerce") if target in df.columns else None
    conv = {}
    if long is not None and len(long):
        c = long[long["unit_original"].astype(str) != long["unit"].astype(str)]
        conv = c.groupby("metric").size().to_dict()
    for m in TRUST_METRICS:
        if m not in df.columns:
            continue
        x = pd.to_numeric(df[m], errors="coerce")
        cov = float(x.notna().mean() * 100) if len(x) else float("nan")
        r_all, n, _ = pearson(x, y) if y is not None else (float("nan"), 0, float("nan"))
        rs = []
        if "build" in df.columns and y is not None:
            for b, g in df.groupby("build"):
                r, nb, _ = pearson(pd.to_numeric(g[m], errors="coerce"), pd.to_numeric(g[target], errors="coerce"))
                if nb >= 10 and np.isfinite(r):
                    rs.append(r)
        r_min, r_max = (min(rs), max(rs)) if rs else (float("nan"), float("nan"))
        pos = x.dropna()
        n_out = 0
        if len(pos) >= 10 and pos.std(ddof=0) > 0:
            z = (pos - pos.mean()) / pos.std(ddof=0)
            n_out = int((z.abs() > 4).sum())
        verdict = "trusted"
        reasons = []
        if cov < 90:
            verdict, reasons = "partial", reasons + [f"coverage {cov:.0f}%"]
        if np.isfinite(r_min) and np.isfinite(r_max) and (r_max - r_min) > 0.25:
            verdict, reasons = "unstable", reasons + [f"r varies {r_min:.2f}..{r_max:.2f} across builds"]
        if conv.get(m):
            reasons.append(f"{conv[m]} unit conversions")
        if n_out:
            reasons.append(f"{n_out} outliers")
        if cov < 50:
            verdict = "unusable"
        rows.append({"metric": m, "coverage_pct": cov, "r_target": r_all, "r_min_build": r_min, "r_max_build": r_max,
                     "unit_conversions": int(conv.get(m, 0)), "outliers": n_out, "verdict": verdict, "notes": "; ".join(reasons)})
    return pd.DataFrame(rows)


def render_metric_quality(mq: pd.DataFrame) -> str:
    from powermet.schema import label
    from powermet.textfmt import fmt_r, table

    if not len(mq):
        return ""
    rows = [[label(r["metric"]), f"{r['coverage_pct']:.0f}%", fmt_r(r["r_target"]),
             f"{fmt_r(r['r_min_build'])}..{fmt_r(r['r_max_build'])}" if np.isfinite(r["r_min_build"]) else "n/a",
             r["unit_conversions"] or "", r["outliers"] or "", r["verdict"], r["notes"]] for _, r in mq.iterrows()]
    return ("METRIC QUALITY (association with BE power; trust = coverage + stability across builds)\n\n"
            + table(["Metric", "Coverage", "r", "r by build", "Unit conv", "Outliers", "Verdict", "Notes"], rows,
                    ["l", "r", "r", "r", "r", "r", "l", "l"]))


def run_sanitize(project: Project, cfg: Config | None = None, write: bool = True) -> tuple[QualityReport, pd.DataFrame]:
    cfg = cfg or project.load_config()
    df = load_dataset(project, cfg, raw=True)
    long, lineage, unmapped = load_table(project, "measurements_long"), load_table(project, "lineage"), load_table(project, "unmapped")
    intent = load_table(project, "power_intent")
    rep = sanitize(df, cfg, long if len(long) else None, lineage if len(lineage) else None, unmapped if len(unmapped) else None,
                   intent if len(intent) else None)
    rep.metric_quality = metric_quality(df, long if len(long) else None, cfg.target)
    flagged = df.copy()
    flagged["quality_flags"] = rep.flags.to_numpy()
    flagged["usable"] = rep.usable_mask
    if write:
        pdir = project.processed_dir
        write_table(flagged[["design", "build", "fub", "workload", "operating_point", "quality_flags", "usable"]]
                    if all(c in flagged.columns for c in ("workload", "operating_point")) else flagged, pdir / "quality_flags.parquet")
        clean = flagged[flagged["usable"]].drop(columns=["usable"]).reset_index(drop=True)
        write_table(clean, sanitized_path(project))
        register_views(project, {"measurements_sanitized": sanitized_path(project), "quality_flags": pdir / "quality_flags.parquet"})
        if rep.metric_quality is not None:
            write_table(rep.metric_quality, pdir / "metric_quality.parquet")
        from powermet.catalog import record_quality

        record_quality(project, rep, rep.metric_quality)
    return rep, flagged
