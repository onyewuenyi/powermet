"""powermet command-line interface (argparse, stdlib only)."""

from __future__ import annotations

import argparse
import sys

from powermet import __version__


class CliError(Exception):
    """User-facing error; message printed, exit code 1."""


# --------------------------------------------------------------------------- handlers

def cmd_doctor(args: argparse.Namespace) -> int:
    from powermet.deps import doctor_report

    text, ready = doctor_report()
    print(text)
    return 0 if ready else 1



def cmd_demo_generate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from powermet.demo import DemoSpec, generate, make_dirty
    from powermet.ingest import write_table

    spec = DemoSpec(n_designs=args.designs, n_builds=args.builds, n_fubs=args.fubs, seed=args.seed)
    df = generate(spec)
    out_dir = Path(args.output_dir)
    ext = ".csv" if args.format == "csv" else ".parquet"
    path = write_table(df, out_dir / f"power_measurements{ext}")
    if path.suffix != ext:
        print("note: pyarrow not available; wrote CSV instead of Parquet")
    print(f"Wrote {len(df):,} rows ({spec.n_designs} designs x {spec.n_builds} builds x {spec.n_fubs} FUBs) -> {path}")
    if args.dirty:
        dpath = write_table(make_dirty(df, seed=spec.seed), out_dir / "power_measurements_dirty.csv")
        print(f"Wrote dirty sample with injected defects -> {dpath}")
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    from powermet.ingest import read_table
    from powermet.validation import validate

    try:
        df = read_table(args.file)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise CliError(str(exc))
    report = validate(df)
    print(f"File: {args.file}")
    print()
    print(report.render())
    return 0 if report.ok else 1


def _project(args: argparse.Namespace):
    from powermet.config import Project

    return Project(args.project_dir)


def cmd_data_import(args: argparse.Namespace) -> int:
    from powermet.storage import import_file

    project = _project(args)
    try:
        res = import_file(project, args.file, replace=not args.append)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise CliError(str(exc))
    print(f"Source:   {res.source}")
    print(f"SHA-256:  {res.sha256[:16]}...")
    print()
    print(res.report.render())
    print()
    print(f"Imported: {res.n_imported:,} rows -> {res.dataset_path}")
    if res.n_rejected:
        print(f"Rejected: {res.n_rejected:,} rows -> {res.rejected_path} (see reject_reason column)")
    print(f"Project:  {project.root}")
    return 0


def _load(args: argparse.Namespace):
    from powermet.storage import load_dataset, sanitized_path

    project = _project(args)
    try:
        cfg = project.load_config()
        df = load_dataset(project, cfg)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    if cfg.use_sanitized and not sanitized_path(project).exists() and project.processed_dir.exists() \
            and (project.processed_dir / "lineage.parquet").exists():
        print("NOTE: no sanitized dataset (a new ingest invalidates it); analysing RAW rows including flagged ones. "
              "Run `powermet sanitize` first.", file=sys.stderr)
    return project, cfg, df


def cmd_analyze_summary(args: argparse.Namespace) -> int:
    from powermet.errors import analyze_errors
    from powermet.modeling import latest_model_metadata, render_model_comparison
    from powermet.schema import label
    from powermet.summary import compare_stages_text, render_summary, summarize
    from powermet.textfmt import fmt_r

    project, cfg, df = _load(args)
    s = summarize(df)
    print(render_summary(s))
    ea = analyze_errors(df, "physical", top_n=cfg.top_n_errors)
    if ea.assoc_pct:
        top = ea.assoc_pct[0]
        print()
        print("Top FE Physical -> BE error association (percentage error)")
        print(f"  {label(top.x):<22}r={fmt_r(top.r)}")
    meta = latest_model_metadata(project)
    if meta:
        print()
        print(render_model_comparison(meta))
    print()
    print("Key finding")
    print("  " + compare_stages_text(s))
    if not meta:
        print()
        print("No trained model yet. Run `powermet model train` to test whether physical features improve prediction.")
    return 0


def cmd_analyze_correlation(args: argparse.Namespace) -> int:
    from powermet.correlation import ANALYSIS_FEATURES, be_correlations, correlation_matrix, render_associations, strength_word
    from powermet.schema import label
    from powermet.textfmt import fmt_r, heading, table

    from powermet.profiling import Profiler

    project, cfg, df = _load(args)
    print(heading("Correlation Analysis (Pearson, association only)"))
    print()
    prof = Profiler("analyze correlation")
    with prof.stage("correlation", rows=len(df)):
        assocs = be_correlations(df)
    prof.save(project)
    print(render_associations(assocs, "Correlation with BE Power"))
    print()
    cols = ["fe_logical_mw", "fe_physical_mw", "be_mw", *ANALYSIS_FEATURES]
    m = correlation_matrix(df, cols)
    names = [label(c) for c in m.columns]
    short = [n[:10] for n in names]
    rows = [[names[i]] + [fmt_r(m.iat[i, j]) for j in range(len(names))] for i in range(len(names))]
    print("Correlation matrix")
    print()
    print(table([""] + short, rows))
    print()
    print("Interpretation")
    for a in assocs[:3]:
        print(f"  {label(a.x)} is {strength_word(a.r)} associated with BE power (r={fmt_r(a.r)}, n={a.n:,}).")
    print("  Correlation does not establish cause; features that scale with block size correlate with everything.")
    return 0


def cmd_analyze_errors(args: argparse.Namespace) -> int:
    from powermet.errors import STAGES, analyze_errors, interpret, render_assoc, render_by_group, render_top
    from powermet.textfmt import heading
    from powermet.visualization import make_charts, plots_available

    from powermet.profiling import Profiler

    project, cfg, df = _load(args)
    stage = args.stage
    prof = Profiler("analyze errors")
    with prof.stage("error analysis", rows=len(df)):
        ea = analyze_errors(df, stage, top_n=args.top)
    name = STAGES[stage][3]
    print(heading(f"{name} -> BE Error Analysis"))
    print()
    print(render_top(ea.top_pct, ea.fe_col, f"Largest {name} -> BE errors (by percentage)"))
    print()
    print(render_top(ea.top_abs, ea.fe_col, f"Largest {name} -> BE errors (by mW)"))
    print()
    print(render_assoc(ea.assoc_pct, f"Association with {name} -> BE percentage error"))
    print()
    print(render_assoc(ea.assoc_mw, f"Association with {name} -> BE error (mW)"))
    print()
    print(render_by_group(ea.by_build, "build", "Error by build"))
    print()
    print(render_by_group(ea.by_design, "design", "Error by design"))
    print()
    print("Interpretation")
    for line in interpret(ea).splitlines():
        print("  " + line)
    if not args.no_plots:
        print()
        if plots_available():
            from powermet.errors import stage_metrics
            stats = {
                "logical_r2": stage_metrics(df, "logical").get("r2"),
                "physical_r2": stage_metrics(df, "physical").get("r2"),
                "wire_cap_r": next((a.r for a in analyze_errors(df, "physical").assoc_pct if a.x == "wire_cap_pf"), None),
            }
            with prof.stage("charts"):
                paths = make_charts(df, project.reports_dir, stats)
            print("Charts written:")
            for p in paths:
                print(f"  {p}")
        else:
            print("Charts skipped: matplotlib is not installed (text analysis above is complete).")
    prof.save(project)
    return 0


def cmd_model_train(args: argparse.Namespace) -> int:
    from powermet.ingest import file_sha256
    from powermet.modeling import (MODEL_NAMES, best_model_key, improvement_pct, render_importance,
                                   render_model_comparison, save, train)
    from powermet.textfmt import fmt_pct, heading

    from powermet.profiling import Profiler

    project, cfg, df = _load(args)
    if args.test_fraction is not None:
        cfg.test_fraction = args.test_fraction
    prof = Profiler("model train")
    try:
        with prof.stage("model training (holdout)", rows=len(df)):
            res = train(df, cfg, cv=False)
        if args.cv:
            from powermet.modeling import cross_validate_builds
            with prof.stage("leave-one-build-out CV", rows=len(df)):
                res.cv = cross_validate_builds(df, cfg)
    except ValueError as exc:
        raise CliError(str(exc))
    with prof.stage("save artifact"):
        sha = file_sha256(project.dataset_path(cfg))
        mpath, jpath = save(project, res, cfg, dataset_sha=sha)
    prof.save(project)
    import json

    meta = json.loads(jpath.read_text())
    print(heading("Power Model Training"))
    print()
    print(res.split.describe())
    for n in res.notes:
        print(f"NOTE: {n}")
    print()
    print("Features")
    for k, feats in res.features.items():
        print(f"  {MODEL_NAMES[k]:<22}" + ", ".join(feats))
    print()
    print(render_model_comparison(meta))
    if res.cv:
        from powermet.modeling import render_cv
        print()
        print(render_cv(res.cv))
    print()
    print(render_importance(meta))
    print()
    base = res.metrics_test["baseline"]
    best = best_model_key(res.metrics_test)
    print("Result")
    scaled = res.metrics_test.get("scaled")
    if scaled:
        print(f"  A single scale factor on FE physical power (bias correction only) gives MAPE {fmt_pct(scaled.get('mape'))} "
              f"vs baseline {fmt_pct(base.get('mape'))}.")
        best_feat = min((k for k in res.metrics_test if k in ("linear", "physics", "tree")), key=lambda k: res.metrics_test[k]["mape"])
        gain = improvement_pct(scaled, res.metrics_test[best_feat])
        print(f"  Adding physical features ({MODEL_NAMES[best_feat]}) changes MAPE by a further {fmt_pct(gain)} relative to the scaled baseline.")
    if best == "baseline":
        print("  No model beat the FE physical baseline on held-out builds. Physical features do not add")
        print("  predictive value here beyond FE physical power; check feature quality before adding models.")
    else:
        imp = improvement_pct(base, res.metrics_test[best])
        print(f"  {MODEL_NAMES[best]} reduces held-out MAPE from {fmt_pct(base.get('mape'))} to "
              f"{fmt_pct(res.metrics_test[best].get('mape'))} ({fmt_pct(imp)} relative improvement over the baseline).")
    print()
    print(f"Saved: {mpath}")
    print(f"       {jpath}")
    print("Final models were refit on all builds for deployment; the metrics above are held-out.")
    return 0


def cmd_model_validate(args: argparse.Namespace) -> int:
    from powermet.modeling import cross_validate_builds, render_cv
    from powermet.profiling import Profiler
    from powermet.textfmt import heading

    project, cfg, df = _load(args)
    prof = Profiler("model validate")
    with prof.stage("leave-one-build-out CV", rows=len(df)):
        cv = cross_validate_builds(df, cfg)
    prof.save(project)
    print(heading("Build-based Cross-validation"))
    print()
    print(render_cv(cv))
    print()
    print("Each build is predicted by models trained on every other build. The spread across builds is the")
    print("honest uncertainty of predicting a new build; the 90% interval is the empirical relative error band.")
    return 0


def cmd_model_predict(args: argparse.Namespace) -> int:
    from powermet.modeling import load
    from powermet.whatif import parse_override, render_whatif, run_whatif, select_rows

    project, cfg, df = _load(args)
    try:
        payload, meta = load(project, args.model)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    overrides = []
    for text in args.set or []:
        overrides.append(parse_override(text, "set"))
    for text in args.scale or []:
        overrides.append(parse_override(text, "scale"))
    for flag, feat in (("wire_cap", "wire_cap_pf"), ("cell_cap", "cell_cap_pf"), ("area", "area"), ("fanout", "fanout"),
                       ("frequency", "frequency_ghz"), ("voltage", "voltage_v"), ("activity", "activity")):
        v = getattr(args, flag, None)
        if v is not None:
            from powermet.whatif import Override
            overrides.append(Override(feat, "set", v))
    if not overrides:
        raise CliError("give at least one change: --wire-cap 0.8, --set feature=value, or --scale feature=factor")
    try:
        rows = select_rows(df, args.design, args.build, args.fub, args.workload, args.operating_point)
        res = run_whatif(rows, overrides, payload, meta, model_key=args.model_kind or cfg.whatif_model)
    except ValueError as exc:
        raise CliError(str(exc))
    print(render_whatif(res))
    return 0


def cmd_model_evaluate(args: argparse.Namespace) -> int:
    from powermet.modeling import (MODEL_NAMES, best_model_key, evaluate, improvement_pct, load,
                                   render_importance, render_model_comparison)
    from powermet.schema import label
    from powermet.textfmt import fmt_pct, fmt_r, heading, kv

    project, cfg, df = _load(args)
    try:
        payload, meta = load(project, args.model)
        metrics = evaluate(df, payload, meta)
    except (FileNotFoundError, ValueError) as exc:
        raise CliError(str(exc))
    best = best_model_key(metrics)
    m = metrics[best]
    base = metrics["baseline"]
    print(heading("Power Model Evaluation"))
    print()
    print(f"Model file: {meta['model_file']}  (trained {meta['created_at']})")
    print()
    print("Target")
    print(f"  {label(meta['target'])} (BE total power)")
    print()
    print(f"Features ({MODEL_NAMES[best]})")
    for f in meta["features"][best]:
        print(f"  {label(f)}")
    print()
    print("Test split")
    print(f"  {meta['split_strategy']}-based: builds " + ", ".join(meta["test_builds"]) + f"  ({m['n']:,} rows)")
    print()
    print(f"Results ({MODEL_NAMES[best]})")
    print(kv([("MAPE", fmt_pct(m.get("mape"))), ("P50 error", fmt_pct(m.get("p50_ape"))),
              ("P95 error", fmt_pct(m.get("p95_ape"))), ("R^2", fmt_r(m.get("r2")))]))
    print()
    print("Baseline (BE = FE physical)")
    print(kv([("MAPE", fmt_pct(base.get("mape"))), ("P95 error", fmt_pct(base.get("p95_ape"))),
              ("R^2", fmt_r(base.get("r2")))]))
    print()
    imp = improvement_pct(base, m)
    print("Improvement over baseline")
    print(f"  {fmt_pct(imp)} relative MAPE reduction" if best != "baseline" else "  none (baseline is best)")
    print()
    print(render_model_comparison(meta, metrics))
    print()
    print(render_importance(meta))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from powermet.reporting import write_report

    project, cfg, df = _load(args)
    path = write_report(project, cfg, df, make_plots=not args.no_plots)
    print(f"Report written: {path}")
    charts = sorted(project.reports_dir.glob("*.png"))
    if charts:
        print("Charts: " + ", ".join(c.name for c in charts))
    return 0


# ----------------------------------------------------------------------------- V1: extraction / lineage / quality

def cmd_mock_generate(args: argparse.Namespace) -> int:
    from powermet.demo import DemoSpec
    from powermet.mockdata import write_mock_runs

    spec = DemoSpec(n_designs=args.designs, n_builds=args.builds, n_fubs=args.fubs, seed=args.seed,
                    workloads=tuple(args.workloads.split(",")), operating_points=tuple(args.operating_points.split(",")))
    root, data, log = write_mock_runs(args.output_dir, spec, defects=not args.no_defects)
    n_runs = spec.n_designs * spec.n_builds
    print(f"Wrote {n_runs} mock run directories under {root}")
    print(f"  {spec.n_designs} designs x {spec.n_builds} builds x {spec.n_fubs} FUBs x "
          f"{len(spec.workloads)} workloads x {len(spec.operating_points)} operating points "
          f"= {len(data.measurements):,} expected measurements")
    print("  sources per run: metadata.json, mapping/fub_map.csv, pprtl/, primepower/, starrc/, implementation/, activity/, perf/")
    if log:
        print(f"  injected defects ({len(log)}):")
        for line in log:
            print(f"    - {line}")
    print()
    print(f"Next: powermet ingest scan {root}")
    return 0


def _print_ingest_summary(summary, verbose: bool) -> None:
    from powermet.textfmt import table

    rows = []
    for r in summary.runs:
        rows.append([r.design, r.build, r.status, len(r.reports), f"{len(r.long):,}", f"{len(r.wide):,}",
                     len(r.flags), len(r.errors)])
    print(table(["Design", "Build", "Status", "Reports", "Records", "FUB rows", "Lineage flags", "Errors"], rows,
                ["l", "l", "l", "r", "r", "r", "r", "r"]))
    for r in summary.runs:
        if r.errors or (verbose and r.flags):
            print()
            print(f"{r.design}/{r.build}:")
            for e in r.errors:
                print(f"  ERROR  {e}")
            if verbose:
                for f in r.flags:
                    print(f"  FLAG   {f}")
    print()
    print(f"Dataset: {summary.n_rows:,} rows -> {summary.dataset_path}")
    if summary.n_rejected:
        print(f"Rejected at validation: {summary.n_rejected:,} rows (see rejected.parquet)")
    print()
    print(f"Ingest wall time: {summary.profiler.total_wall_s:.2f} s   (use --profile for stage timings)")
    print("Next: powermet sanitize")


def cmd_ingest_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from powermet.pipeline import ingest_runs

    project = _project(args)
    try:
        summary = ingest_runs(project, [Path(args.run_dir)], replace=not args.append)
    except (FileNotFoundError, ValueError) as exc:
        raise CliError(str(exc))
    _print_ingest_summary(summary, args.verbose)
    return 0


def cmd_ingest_scan(args: argparse.Namespace) -> int:
    from pathlib import Path

    from powermet.pipeline import find_runs, ingest_runs

    project = _project(args)
    runs = find_runs(Path(args.root))
    if not runs:
        raise CliError(f"no run directories (*/*/metadata.json) found under {args.root}")
    print(f"Found {len(runs)} run directories under {args.root}")
    try:
        summary = ingest_runs(project, runs, replace=not args.append)
    except (FileNotFoundError, ValueError) as exc:
        raise CliError(str(exc))
    _print_ingest_summary(summary, args.verbose)
    return 0


def cmd_sanitize(args: argparse.Namespace) -> int:
    from powermet.profiling import Profiler
    from powermet.sanitize import run_sanitize
    from powermet.storage import sanitized_path

    project = _project(args)
    cfg = project.load_config()
    prof = Profiler("sanitize")
    try:
        with prof.stage("sanitization") as st:
            rep, flagged = run_sanitize(project, cfg, write=not args.report_only)
            st.rows = rep.n_rows
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    prof.save(project)
    print(rep.render())
    print()
    if rep.metric_quality is not None and len(rep.metric_quality):
        from powermet.sanitize import render_metric_quality
        print(render_metric_quality(rep.metric_quality))
        print()
    if args.report_only:
        print("Report only; sanitized dataset not written.")
    else:
        print(f"Sanitized dataset: {sanitized_path(project)}  ({rep.n_usable:,} usable rows)")
        print(f"Row flags:         {project.processed_dir / 'quality_flags.parquet'}")
        print("Analysis and modeling commands now use the sanitized dataset (config: use_sanitized).")
    return 0


def cmd_lineage_show(args: argparse.Namespace) -> int:
    from powermet.lineage import render_chain
    from powermet.storage import load_dataset, load_table

    project = _project(args)
    lin = load_table(project, "lineage")
    if not len(lin):
        raise CliError("no lineage table; run `powermet ingest scan <root>` first")
    sel = lin[lin["fub"].astype(str) == args.fub]
    if args.design:
        sel = sel[sel["design"] == args.design]
    if args.build:
        sel = sel[sel["build"] == args.build]
    if not len(sel):
        raise CliError(f"FUB '{args.fub}' not found in lineage table")
    if not args.build:
        sel = sel.sort_values(["design", "build"]).groupby("design").tail(1)
    try:
        df = load_dataset(project, raw=True)
    except FileNotFoundError:
        df = None
    for _, row in sel.iterrows():
        meas = None
        if df is not None:
            meas = df[(df["design"] == row["design"]) & (df["build"] == row["build"]) & (df["fub"].astype(str) == args.fub)]
        print(render_chain(row, meas))
        print()
    return 0


def cmd_db_tables(args: argparse.Namespace) -> int:
    from powermet.catalog import db_path, query, tables
    from powermet.textfmt import table

    project = _project(args)
    names = tables(project)
    if not names:
        raise CliError("no metadata catalog yet; run an ingest first")
    rows = [[t, int(query(project, f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0])] for t in names]
    print(f"Catalog: {db_path(project)}")
    print()
    print(table(["Table", "Rows"], rows))
    print()
    print("Parquet tables (DuckDB views): measurements, measurements_sanitized, measurements_long, lineage, unmapped, performance, quality_flags")
    return 0


def cmd_db_query(args: argparse.Namespace) -> int:
    import pandas as pd

    from powermet.catalog import query
    from powermet.storage import query as duck_query

    project = _project(args)
    try:
        df = duck_query(project, args.sql) if args.engine == "duckdb" else query(project, args.sql)
    except Exception as exc:
        raise CliError(f"query failed: {exc}")
    with pd.option_context("display.max_rows", args.limit, "display.width", 200, "display.max_columns", 40):
        print(df.head(args.limit).to_string(index=False) if len(df) else "(no rows)")
    return 0


def cmd_measure_get(args: argparse.Namespace) -> int:
    from powermet.measurements import MeasurementStore

    project = _project(args)
    try:
        store = MeasurementStore.from_project(project)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    filters = {k: v for k, v in (("fub", args.fub), ("model_root", args.model_root), ("design", args.design), ("build", args.build),
                                 ("stage", args.stage), ("metric", args.metric), ("workload", args.workload),
                                 ("operating_point", args.operating_point)) if v is not None}
    hits = store.find(**filters)
    if not hits:
        raise CliError(f"no measurement matches {filters}")
    from powermet.textfmt import table
    rows = [[m.model_root, m.build, m.stage, m.metric, f"{m.value:.4g}", m.unit, m.workload or "", m.operating_point or "",
             m.source or "", m.tool_version or "", m.run_id or ""] for m in hits[: args.limit]]
    print(table(["Model root", "Build", "Stage", "Metric", "Value", "Unit", "Workload", "OP", "Source", "Tool ver", "Run id"], rows,
                ["l", "l", "l", "l", "r", "l", "l", "l", "l", "l", "l"]))
    if len(hits) > args.limit:
        print(f"... {len(hits) - args.limit} more (use --limit)")
    if args.show_file and hits:
        print()
        print(f"source file: {hits[0].source_file}   hierarchy: {hits[0].hierarchy}")
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    from powermet.profiling import load_profiles, render_profile

    project = _project(args)
    entries = load_profiles(project, last=args.last)
    if not entries:
        raise CliError("no profiles recorded yet; run an ingest, sanitize, or model command first")
    for e in entries:
        print(render_profile(e))
        print()
    return 0


# ----------------------------------------------------------------------------- V3: workload / exploration

def cmd_workload_summary(args: argparse.Namespace) -> int:
    from powermet.storage import load_table
    from powermet.textfmt import heading
    from powermet.workload import render_workload_summary, summarize_workloads

    project, cfg, df = _load(args)
    perf = load_table(project, "performance")
    ws = summarize_workloads(df, perf if len(perf) else None)
    print(heading("Workload and Performance Summary"))
    print()
    if not len(perf):
        print("No performance table found (perf/ source not ingested); energy per op unavailable.")
        print()
    print(render_workload_summary(ws))
    return 0


def _explorer(args, project, cfg, df):
    from powermet.explore import Explorer
    from powermet.modeling import load
    from powermet.storage import load_table

    try:
        payload, meta = load(project, args.model)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    perf = load_table(project, "performance")
    return Explorer(df, perf if len(perf) else None, payload, meta, model_key=args.model_kind or cfg.whatif_model)


def cmd_explore_sweep(args: argparse.Namespace) -> int:
    from powermet.explore import render_rows

    project, cfg, df = _load(args)
    ex = _explorer(args, project, cfg, df)
    values = [float(v) for v in args.values.split(",")]
    mode = "scale" if args.scale else "set"
    try:
        rows = ex.sweep(args.design, args.workload, args.param, values, mode, args.operating_point, args.fixed_voltage)
    except ValueError as exc:
        raise CliError(str(exc))
    print(render_rows(rows, ex.model_key, f"Sweep {args.param} ({mode})  design={args.design} workload={args.workload}"))
    return 0


def cmd_explore_opmap(args: argparse.Namespace) -> int:
    from powermet.explore import render_rows

    project, cfg, df = _load(args)
    ex = _explorer(args, project, cfg, df)
    cands = []
    for c in args.add or []:
        kv = dict(part.split("=") for part in c.split(","))
        cands.append((float(kv["v"]), float(kv["f"])))
    try:
        rows = ex.opmap(args.design, args.workload, cands)
    except ValueError as exc:
        raise CliError(str(exc))
    print(render_rows(rows, ex.model_key, f"Operating-point map  design={args.design} workload={args.workload}"))
    return 0


def cmd_explore_scenario(args: argparse.Namespace) -> int:
    from powermet.explore import load_scenarios, render_rows

    project, cfg, df = _load(args)
    ex = _explorer(args, project, cfg, df)
    try:
        scs = load_scenarios(args.file)
        rows = ex.scenarios(scs)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise CliError(f"scenario file: {exc}")
    print(render_rows(rows, ex.model_key, f"Scenarios from {args.file}"))
    return 0


# ----------------------------------------------------------------------------- timing / energy / integration

def cmd_analyze_energy(args: argparse.Namespace) -> int:
    from powermet.decomposition import decompose, render_decomposition
    from powermet.modeling import load
    from powermet.selection import DatasetSlice
    from powermet.textfmt import heading

    project, cfg, df = _load(args)
    try:
        payload, meta = load(project, args.model)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    if "datamove" not in payload["models"]:
        raise CliError("no data-movement model in the artifact (needs bits_per_cycle and avg_net_length_um); retrain with `model train`")
    try:
        sel = DatasetSlice(design=args.design, workload=args.workload, operating_point=args.operating_point,
                           default_workload=False, default_operating_point=False).apply(df)
    except ValueError as exc:
        raise CliError(str(exc))
    dec = decompose(sel, payload["models"]["datamove"])
    print(heading("Energy Decomposition (latest build per design)"))
    print()
    print(render_decomposition(dec, payload["models"]["datamove"], top=args.top))
    if args.csv:
        dec.to_csv(args.csv, index=False)
        print(f"\nPer-FUB decomposition written to {args.csv}")
    return 0


def cmd_analyze_deltas(args: argparse.Namespace) -> int:
    from powermet.deltas import build_deltas, render_deltas
    from powermet.textfmt import heading

    project, cfg, df = _load(args)
    if "wns_ps" not in df.columns:
        print("NOTE: no timing columns in the dataset; deltas will cover power and physical metrics only.")
    pairs = None
    if args.builds:
        a, _, b = args.builds.partition(":")
        pairs = [(a, b)]
    designs = [args.design] if args.design else sorted(df["design"].astype(str).unique())
    print(heading("Build-to-build power / timing / physical deltas"))
    print()
    for d in designs:
        deltas = build_deltas(df, d, args.workload, args.operating_point, pairs)
        if not deltas:
            print(f"{d}: fewer than two builds")
            continue
        if not args.all and pairs is None:
            deltas = deltas[-1:]
        print(render_deltas(deltas, top=args.top))
        print()
    return 0


def cmd_analyze_frontier(args: argparse.Namespace) -> int:
    from powermet.frontier import frontier, render_frontier
    from powermet.storage import load_table
    from powermet.textfmt import heading

    project, cfg, df = _load(args)
    if "fmax_ghz" not in df.columns or df["fmax_ghz"].isna().all():
        raise CliError("no timing data (fmax_ghz) in the dataset; ingest PrimeTime reports first")
    perf = load_table(project, "performance")
    designs = [args.design] if args.design else sorted(df["design"].astype(str).unique())
    print(heading("Power x Timing frontier across builds"))
    print()
    all_pts = {}
    for d in designs:
        pts = frontier(df, d, args.workload, args.operating_point, perf if len(perf) else None)
        all_pts[d] = pts
        print(render_frontier(pts, args.workload, args.operating_point))
        print()
    if not args.no_plots:
        from powermet.visualization import frontier_chart, plots_available
        if plots_available():
            p = frontier_chart(all_pts, project.reports_dir / "power_timing_frontier.png")
            print(f"Chart written: {p}")
        else:
            print("Chart skipped: matplotlib not installed.")
    return 0


def cmd_model_export(args: argparse.Namespace) -> int:
    import json

    from powermet.integrate import export_compact
    from powermet.modeling import load
    from powermet.storage import load_table

    project, cfg, df = _load(args)
    try:
        payload, meta = load(project, args.model)
    except FileNotFoundError as exc:
        raise CliError(str(exc))
    perf = load_table(project, "performance")
    doc = export_compact(df, perf if len(perf) else None, payload, meta, model_key=args.model_kind)
    out = args.output or str(project.models_dir / "compact_power_model.json")
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=2))
    n_fubs = sum(len(d["fubs"]) for d in doc["designs"].values())
    print(f"Compact power model written: {out}")
    print(f"  kind: {doc['model_kind']}   designs: {len(doc['designs'])}   FUBs: {n_fubs}   terms: {', '.join(doc['terms']['features'])}")
    print("  contents per design: latest-build physical features per FUB, per-workload activity/traffic, operating points,")
    print("  DVFS curve, per-workload throughput scaling, per-partition timing model.")
    print("  Load without powermet: json + numpy/pandas via powermet.integrate.CompactPowerModel, or reimplement the 10-line evaluator.")
    return 0


def cmd_integrate_trace(args: argparse.Namespace) -> int:
    import pandas as pd

    from powermet.integrate import CompactPowerModel, render_trace, run_trace

    project = _project(args)
    path = args.compact or str(project.models_dir / "compact_power_model.json")
    try:
        model = CompactPowerModel.load(path)
    except FileNotFoundError:
        raise CliError(f"compact model not found at {path}; run `powermet model export` first")
    try:
        trace = pd.read_csv(args.trace)
        res = run_trace(model, trace)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise CliError(f"trace: {exc}")
    print(render_trace(res, model.doc["model_kind"]))
    if args.csv:
        res.timeline.to_csv(args.csv, index=False)
        print(f"\nTimeline written to {args.csv}")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    from powermet.modeling import MODEL_REGISTRY, WHATIF_CHOICES

    LINEAR_CHOICES = tuple(m.key for m in MODEL_REGISTRY if m.family == "linear")

    p = argparse.ArgumentParser(
        prog="powermet",
        description="Power Metrology & Modeling V0: FE <-> BE power correlation (local CLI).",
    )
    p.add_argument("--version", action="version", version=f"powermet {__version__}")
    p.add_argument("--profile", action="store_true", help="Print stage runtime/memory after the command.")
    p.add_argument(
        "--project-dir",
        default=None,
        help="Location of the .powermet/ project directory (default: ./.powermet or $POWERMET_HOME).",
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sp = sub.add_parser("doctor", help="Check the local Python environment.")
    sp.set_defaults(func=cmd_doctor)


    sp = sub.add_parser("demo", help="Synthetic demo data.")
    demo_sub = sp.add_subparsers(dest="demo_command", metavar="<subcommand>")
    demo_sub.required = True
    g = demo_sub.add_parser("generate", help="Generate a deterministic synthetic dataset.")
    g.add_argument("--output-dir", default="data/demo", help="Output directory (default: data/demo).")
    g.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    g.add_argument("--designs", type=int, default=5)
    g.add_argument("--builds", type=int, default=10)
    g.add_argument("--fubs", type=int, default=50)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dirty", action="store_true", help="Also write a small CSV with injected defects.")
    g.set_defaults(func=cmd_demo_generate)

    sp = sub.add_parser("data", help="Validate and import measurement files.")
    data_sub = sp.add_subparsers(dest="data_command", metavar="<subcommand>")
    data_sub.required = True
    v = data_sub.add_parser("validate", help="Validate a CSV/Parquet file against the schema.")
    v.add_argument("file")
    v.set_defaults(func=cmd_data_validate)
    i = data_sub.add_parser("import", help="Validate, normalize and store a file in the project.")
    i.add_argument("file")
    i.add_argument("--append", action="store_true", help="Append to the existing dataset instead of replacing it.")
    i.set_defaults(func=cmd_data_import)

    sp = sub.add_parser("analyze", help="Summary, correlation and error analysis.")
    an_sub = sp.add_subparsers(dest="analyze_command", metavar="<subcommand>")
    an_sub.required = True
    a = an_sub.add_parser("summary", help="Engineering summary of the dataset and FE -> BE accuracy.")
    a.set_defaults(func=cmd_analyze_summary)
    a = an_sub.add_parser("correlation", help="Pearson correlations between power and physical features.")
    a.set_defaults(func=cmd_analyze_correlation)
    a = an_sub.add_parser("errors", help="Rank FE -> BE errors and find associated features.")
    a.add_argument("--stage", choices=["physical", "logical"], default="physical")
    a.add_argument("--top", type=int, default=10, help="Rows to show in ranked tables.")
    a.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation.")
    a.set_defaults(func=cmd_analyze_errors)

    a = an_sub.add_parser("energy", help="Compute / wire / data-movement / leakage decomposition (compact energy model).")
    a.add_argument("--design", default=None)
    a.add_argument("--workload", default=None)
    a.add_argument("--operating-point", default=None)
    a.add_argument("--top", type=int, default=8)
    a.add_argument("--csv", default=None, help="Write the per-FUB decomposition to CSV.")
    a.add_argument("--model", default=None)
    a.set_defaults(func=cmd_analyze_energy)
    a = an_sub.add_parser("deltas", help="Build-to-build deltas: power, timing (WNS/Fmax), wire cap, area; classify the trade.")
    a.add_argument("--design", default=None)
    a.add_argument("--builds", default=None, help="Specific pair, e.g. B021:B022 (default: latest consecutive pair).")
    a.add_argument("--all", action="store_true", help="Every consecutive pair.")
    a.add_argument("--workload", default=None)
    a.add_argument("--operating-point", default=None)
    a.add_argument("--top", type=int, default=5)
    a.set_defaults(func=cmd_analyze_deltas)
    a = an_sub.add_parser("frontier", help="Power x timing (Fmax) frontier across builds with Pareto classification.")
    a.add_argument("--design", default=None)
    a.add_argument("--workload", default=None)
    a.add_argument("--operating-point", default=None)
    a.add_argument("--no-plots", action="store_true")
    a.set_defaults(func=cmd_analyze_frontier)

    sp = sub.add_parser("model", help="Train and evaluate simple BE-power models.")
    mo_sub = sp.add_subparsers(dest="model_command", metavar="<subcommand>")
    mo_sub.required = True
    t = mo_sub.add_parser("train", help="Train baseline, linear and tree models with a build-based split.")
    t.add_argument("--test-fraction", type=float, default=None, help="Fraction of builds to hold out (default from config).")
    t.add_argument("--cv", action="store_true", help="Also run leave-one-build-out cross-validation (slower).")
    t.set_defaults(func=cmd_model_train)
    v2 = mo_sub.add_parser("validate", help="Leave-one-build-out cross-validation of all models.")
    v2.set_defaults(func=cmd_model_validate)
    pr = mo_sub.add_parser("predict", help="What-if: change a parameter and predict BE power.")
    pr.add_argument("--fub", default=None)
    pr.add_argument("--design", default=None)
    pr.add_argument("--build", default=None, help="Default: latest build.")
    pr.add_argument("--workload", default=None)
    pr.add_argument("--operating-point", default=None)
    pr.add_argument("--wire-cap", type=float, default=None, dest="wire_cap", help="New wire cap (pF).")
    pr.add_argument("--cell-cap", type=float, default=None, dest="cell_cap")
    pr.add_argument("--area", type=float, default=None)
    pr.add_argument("--fanout", type=float, default=None)
    pr.add_argument("--frequency", type=float, default=None, help="GHz")
    pr.add_argument("--voltage", type=float, default=None, help="V")
    pr.add_argument("--activity", type=float, default=None)
    pr.add_argument("--set", action="append", help="feature=value (repeatable)")
    pr.add_argument("--scale", action="append", help="feature=factor, e.g. wire_cap_pf=0.8 (repeatable)")
    pr.add_argument("--model-kind", choices=WHATIF_CHOICES, default=None)
    pr.add_argument("--model", default=None, help="Path to model_<ts>.json (default: latest).")
    pr.set_defaults(func=cmd_model_predict)
    ex_ = mo_sub.add_parser("export", help="Export a compact JSON power model for performance-tool integration.")
    ex_.add_argument("--output", default=None, help="Default: .powermet/models/compact_power_model.json")
    ex_.add_argument("--model-kind", choices=LINEAR_CHOICES, default="datamove")
    ex_.add_argument("--model", default=None)
    ex_.set_defaults(func=cmd_model_export)
    e = mo_sub.add_parser("evaluate", help="Re-score a saved model on its held-out builds.")
    e.add_argument("--model", default=None, help="Path to a model_<ts>.json (default: latest).")
    e.set_defaults(func=cmd_model_evaluate)

    r = sub.add_parser("report", help="Write the Markdown methodology report to .powermet/reports/.")
    r.add_argument("--no-plots", action="store_true")
    r.set_defaults(func=cmd_report)

    sp = sub.add_parser("mock", help="Mock raw EDA run directories (V1 demo input).")
    mk = sp.add_subparsers(dest="mock_command", metavar="<subcommand>")
    mk.required = True
    g = mk.add_parser("generate", help="Write mock PPRTL/PrimePower/StarRC/implementation/activity/perf reports.")
    g.add_argument("--output-dir", default="mock_runs")
    g.add_argument("--designs", type=int, default=5)
    g.add_argument("--builds", type=int, default=10)
    g.add_argument("--fubs", type=int, default=50)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--workloads", default="idle,typical,compute,memory")
    g.add_argument("--operating-points", default="eco,nom,turbo")
    g.add_argument("--no-defects", action="store_true", help="Do not inject data-quality defects.")
    g.set_defaults(func=cmd_mock_generate)

    sp = sub.add_parser("ingest", help="Automated extraction from EDA run directories.")
    ig = sp.add_subparsers(dest="ingest_command", metavar="<subcommand>")
    ig.required = True
    r1 = ig.add_parser("run", help="Ingest one run directory (contains metadata.json).")
    r1.add_argument("run_dir")
    r1.add_argument("--append", action="store_true", help="Merge into the existing dataset instead of replacing it.")
    r1.add_argument("--verbose", action="store_true", help="Print lineage flags per run.")
    r1.set_defaults(func=cmd_ingest_run)
    r2 = ig.add_parser("scan", help="Ingest every <design>/<build>/metadata.json under a root directory.")
    r2.add_argument("root")
    r2.add_argument("--append", action="store_true")
    r2.add_argument("--verbose", action="store_true")
    r2.set_defaults(func=cmd_ingest_scan)

    sp = sub.add_parser("sanitize", help="Cross-source data-quality checks; writes the sanitized dataset.")
    sp.add_argument("--report-only", action="store_true", help="Print the quality report without writing files.")
    sp.set_defaults(func=cmd_sanitize)

    sp = sub.add_parser("lineage", help="Show where a FUB's numbers came from.")
    ln = sp.add_subparsers(dest="lineage_command", metavar="<subcommand>")
    ln.required = True
    l1 = ln.add_parser("show", help="Print the FUB -> FE -> synth -> BE -> instances -> measurement chain.")
    l1.add_argument("fub")
    l1.add_argument("--design", default=None)
    l1.add_argument("--build", default=None)
    l1.set_defaults(func=cmd_lineage_show)

    sp = sub.add_parser("measure", help="Measurement-level access with provenance (model root / FUB / build / stage / metric).")
    ms = sp.add_subparsers(dest="measure_command", metavar="<subcommand>")
    ms.required = True
    m1 = ms.add_parser("get", help="Look up measurements, e.g. --fub Scheduler --build B003 --stage FE --metric fe_physical_mw")
    for flag in ("fub", "model-root", "design", "build", "stage", "metric", "workload", "operating-point"):
        m1.add_argument(f"--{flag}", default=None)
    m1.add_argument("--limit", type=int, default=40)
    m1.add_argument("--show-file", action="store_true")
    m1.set_defaults(func=cmd_measure_get)

    sp = sub.add_parser("db", help="Metadata catalog (SQLite) and analytical (DuckDB) queries.")
    dbs = sp.add_subparsers(dest="db_command", metavar="<subcommand>")
    dbs.required = True
    d1 = dbs.add_parser("tables", help="List catalog tables and row counts.")
    d1.set_defaults(func=cmd_db_tables)
    d2 = dbs.add_parser("query", help="Run SQL against the catalog (sqlite) or the Parquet views (duckdb).")
    d2.add_argument("sql")
    d2.add_argument("--engine", choices=["sqlite", "duckdb"], default="sqlite")
    d2.add_argument("--limit", type=int, default=50)
    d2.set_defaults(func=cmd_db_query)

    sp = sub.add_parser("profile", help="Runtime / memory profiles of previous commands.")
    pf = sp.add_subparsers(dest="profile_command", metavar="<subcommand>")
    pf.required = True
    p1 = pf.add_parser("show", help="Show the most recent stage timings and peak memory.")
    p1.add_argument("--last", type=int, default=3)
    p1.set_defaults(func=cmd_profile_show)

    sp = sub.add_parser("integrate", help="Performance-tool integration: evaluate workload traces with the compact model.")
    it = sp.add_subparsers(dest="integrate_command", metavar="<subcommand>")
    it.required = True
    i1 = it.add_parser("trace", help="Power/throughput/energy timeline for a phase trace CSV.")
    i1.add_argument("trace", help="CSV: design, interval, duration_s, workload, operating_point | frequency_ghz[,voltage_v], activity_scale")
    i1.add_argument("--compact", default=None, help="Compact model JSON (default: latest export).")
    i1.add_argument("--csv", default=None, help="Write the timeline to CSV.")
    i1.set_defaults(func=cmd_integrate_trace)

    sp = sub.add_parser("workload", help="Workload / performance / energy-per-op analysis (V3).")
    wl = sp.add_subparsers(dest="workload_command", metavar="<subcommand>")
    wl.required = True
    w1 = wl.add_parser("summary", help="Power, throughput and energy per op by workload and operating point.")
    w1.set_defaults(func=cmd_workload_summary)

    sp = sub.add_parser("explore", help="Design-space what-if exploration (V3).")
    ex = sp.add_subparsers(dest="explore_command", metavar="<subcommand>")
    ex.required = True
    e1 = ex.add_parser("sweep", help="Sweep one parameter and predict power, throughput, energy/op.")
    e1.add_argument("--design", required=True)
    e1.add_argument("--workload", default="typical")
    e1.add_argument("--operating-point", default=None)
    e1.add_argument("--param", required=True, help="e.g. frequency_ghz, wire_cap_pf, voltage_v, activity")
    e1.add_argument("--values", required=True, help="comma-separated values (or factors with --scale)")
    e1.add_argument("--scale", action="store_true", help="Treat values as multiplicative factors.")
    e1.add_argument("--fixed-voltage", action="store_true", help="Do not follow the DVFS curve when sweeping frequency.")
    e1.add_argument("--model-kind", choices=WHATIF_CHOICES, default=None)
    e1.add_argument("--model", default=None)
    e1.set_defaults(func=cmd_explore_sweep)
    e2 = ex.add_parser("opmap", help="Compare measured operating points and candidate (V, f) points.")
    e2.add_argument("--design", required=True)
    e2.add_argument("--workload", default="typical")
    e2.add_argument("--add", action="append", help='candidate point "v=0.78,f=2.3" (repeatable)')
    e2.add_argument("--model-kind", choices=WHATIF_CHOICES, default=None)
    e2.add_argument("--model", default=None)
    e2.set_defaults(func=cmd_explore_opmap)
    e3 = ex.add_parser("scenario", help="Evaluate named scenarios from a TOML file.")
    e3.add_argument("file")
    e3.add_argument("--model-kind", choices=WHATIF_CHOICES, default=None)
    e3.add_argument("--model", default=None)
    e3.set_defaults(func=cmd_explore_scenario)

    return p


def _print_new_profiles(project_dir, before: int) -> None:
    from powermet.config import Project
    from powermet.profiling import load_profiles, render_profile

    entries = load_profiles(Project(project_dir), last=1000)
    for e in entries[before:]:
        print()
        print(render_profile(e))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    n_before = 0
    if getattr(args, "profile", False):
        from powermet.config import Project
        from powermet.profiling import load_profiles

        n_before = len(load_profiles(Project(args.project_dir), last=1000))
    try:
        rc = int(args.func(args) or 0)
        if getattr(args, "profile", False):
            _print_new_profiles(args.project_dir, n_before)
        return rc
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
