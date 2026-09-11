# Reusable components: what to lift, where it lives, how it is tested

Each row is one abstraction that can be copied into another pipeline on its own. "Depends on"
lists the powermet modules it imports; anything not listed is standalone apart from pandas/numpy.

| Component | Module | What it gives you | Depends on | Tests |
|---|---|---|---|---|
| **Source adapter contract** | `extract/base.py` | `SourceInputs`, `Located`, `ParsedReport`, `record()`, `convert_unit()` with the canonical-unit table, `locate()` with `{workload}`/`{operating_point}` patterns and per-source overrides, `parse_header()` / `find_table_start()` / `to_float()` text helpers, `OBJECT_KINDS`, `tool_name()` policy | – | `test_extract_units.py` (base helpers, locate overrides, record/kind validation) |
| Adapters | `extract/{pprtl,primepower,primetime,starrc,implementation,saif,perf,metadata}.py` | One `get_files()` + `parse()` per tool; representative formats, a real SAIF 2.0 parser (per-instance activity, bits switched/cycle), unit normalisation, header-driven run id / build / scenario | `extract/base.py` | `test_extract_units.py` (literal report strings per adapter, unit variants, error paths); `test_extract_pipeline.py` (mock round-trip) |
| Metric registry | `extract/__init__.py` | `SOURCES`, `SOURCE_METRICS`, `METRIC_SCOPE` (which keys a metric varies by), `metrics_with_scope()` | adapters | `test_extract_units.py::test_registry_consistency` |
| **Design identity** | `identity.py` | `ModelRoot` / `FubSpec`: FUB list, `model_root`, FUB → partition, FE/BE hierarchy, `resolve(obj, kind)`; the only place object names are matched | – | `test_identity_selection.py::TestModelRoot` |
| Lineage | `lineage.py` | `resolve_objects()` (records → FUB incl. partition fan-out), `lineage_table()` with issue codes, `render_chain()` | `identity`, `extract/base` | `test_pipeline_units.py`, `test_timing_energy.py::test_primetime_parser_and_partition_fanout` |
| **Measurement accessor** | `measurements.py` | `Measurement` (atomic record with provenance), `MeasurementStore.get(fub=, build=, stage=, metric=)`, `find()`, `pivot()`, `get_measurement(project, ...)` | `storage` (only for `from_project`) | `test_identity_selection.py::TestMeasurementStore`; CLI `measure get` in `test_cli.py` |
| Dataset selection | `selection.py` | `DatasetSlice` (latest build / default workload & operating point / design / FUB), `build_order()`, `latest_build[_per_design]()`, `default_value()` | – | `test_identity_selection.py::TestSelection` |
| Ingest pipeline | `pipeline.py` | `extract_run()` (extract → normalise → lineage → pivot), `verify_run_consistency()`, `join_metric()` / `pivot_fub_records()` / `attach_identity()` / `stamp_provenance()` / `pivot_perf_records()`, `ingest_runs()` with idempotent append | most of the above | `test_pipeline_units.py` (each building block, append regressions, consistency), `test_extract_pipeline.py` |
| Canonical schema | `schema.py` | `ColumnSpec` table by role; derived column lists (`FUB_METRICS`, `POWER_COLUMNS`, `TIMING_COLUMNS`, ...), `identity_key()`, labels | – | exercised by every pipeline/validation test |
| Validation | `validation.py` | `validate()` → `ValidationReport` (errors vs warnings, per-row masks, reject reasons); `outlier_masks()` | `schema` | `test_demo_validation.py` |
| **Sanitization** | `sanitize.py` | `sanitize()` → `QualityReport` with one mask per `CHECKS` entry (blocking vs info), `metric_quality()` per-metric trust table, `run_sanitize()` | `validation`, `lineage`, `storage` | `test_units_analysis.py` (one check at a time, verdict rules), `test_timing_energy.py` |
| Metadata catalog | `catalog.py` | SQLite tables `build`, `source_file`, `import_run`, `quality_run`, `model`, `profile_run`; idempotent writers, `query()` | – | `test_timing_energy.py::test_catalog_records`, `test_units_analysis.py::test_catalog_empty_project`, append test |
| Profiling | `profiling.py` | `Profiler.stage()` context (wall, CPU, peak RSS, optional heap), `aggregate_stages()`, `render_profile()` | `catalog` (save) | `test_extract_pipeline.py::test_profiler_records_and_aggregates`, `test_units_analysis.py::test_profiler_heap_and_linux_rss` |
| Engineered features | `features.py` | `add_engineered_features()` (CV²f, leakage, cell/wire/data-movement terms), `rescale_fe_physical()` | – | `test_units_analysis.py` |
| **Model registry + models** | `modeling.py` | `ModelSpec` / `MODEL_REGISTRY` (add a kind = add one spec), `build_models()`, `LinearModel` (OLS, `contributions()`), `TreeModel` (monotone GBDT), `split_by_build()`, `cross_validate_builds()`, `permutation_importance()`, `pick_model_key()`, artifact save/load | `config`, `metrics`, `catalog` | `test_modeling.py`, `test_v2_v3.py`, `test_units_analysis.py` |
| Accuracy metrics | `metrics.py` | `prediction_metrics()` (MAE/RMSE/MAPE/R²/P50/P95), `add_derived_metrics()`, `fmax_from_timing()` | `schema` | `test_demo_validation.py`, `test_pipeline_units.py` |
| Energy decomposition | `decomposition.py` | `decompose()` per-FUB compute / wire / movement / leakage shares, pJ per bit-mm | `features`, `modeling` | `test_units_analysis.py::test_pj_per_bit_mm_math`, `test_timing_energy.py` |
| Build deltas | `deltas.py` | `build_deltas()` per design / partition / FUB, `classify()` trade classification | `selection`, `schema` | `test_units_analysis.py::test_classify_boundaries`, `test_timing_energy.py` |
| Power × timing frontier | `frontier.py` | `frontier()` with Pareto marking, `ascii_scatter()` | `deltas`, `selection` | `test_units_analysis.py::test_frontier_ties_and_dominance` |
| Curve fits | `curves.py` | `PerfModel` (throughput ~ f^b), `DvfsCurve` (V(f)), `TimingModel` (delay ~ V^k per partition, Fmax(V)) | `selection` | `test_units_analysis.py`, `test_v2_v3.py` |
| What-if | `whatif.py` | `Override` / `parse_override()` aliases, `run_whatif()` with CV interval | `features`, `modeling`, `selection` | `test_units_analysis.py`, `test_v2_v3.py` |
| Exploration | `explore.py` | `Explorer.sweep/opmap/scenarios`, timing feasibility, `mark_pareto()`, TOML scenarios | `curves`, `features`, `selection`, `whatif` | `test_units_analysis.py::test_mark_pareto_with_infeasible_and_ties`, `test_v2_v3.py`, `test_timing_energy.py` |
| **Compact model / integration** | `integrate.py` | `export_compact()` JSON, `CompactPowerModel` standalone evaluator (matches `LinearModel.predict` exactly), `run_trace()` | `curves`, `features`, `modeling` | `test_units_analysis.py` (equivalence, f/V traces, fmax guards), `test_timing_energy.py` |
| Text output | `textfmt.py` | fixed-width `table()`, `fmt_pct/fmt_mw/fmt_r` with sensible rounding | – | indirectly everywhere |

## Templates for a new environment

`templates/` holds the files you fill in at a target company: `metadata.template.json` (run identity,
tool versions, operating points, `design_type`, activity-flow provenance), `fub_map.template.csv`
(the model-root export), `config.template.toml` (source patterns, thresholds, validation) and
`run_directory.README.md` (layout). See `docs/portability.md` for the adaptation checklist.

## Patterns worth copying even without the code

- **Long-then-wide.** Every adapter emits `(object, object_kind, metric, value, unit, unit_original)`;
  provenance is attached once in `_normalize`; the wide table is a pivot governed by `METRIC_SCOPE`.
  New sources never touch the pivot.
- **Identity from the model root, not from name heuristics.** `ModelRoot.resolve(obj, kind)` is the single
  matching function; partition-level data fans out to member FUBs there.
- **Checks as a registry.** `sanitize.CHECKS` maps a code to (label, blocks-usability). Adding a check =
  one `mark()` call and one dict entry; the report, the flags column and the usable mask follow.
- **Models as a registry.** `MODEL_REGISTRY` drives what is trained, what the CLI accepts and what is
  allowed in what-if; a new model kind is one `ModelSpec`.
- **Selection as a value object.** `DatasetSlice(design=..., workload=...)` replaces ad-hoc filtering and
  makes "latest build, typical workload, nominal corner" an explicit, testable default.
- **Never split rows at random.** `split_by_build` and `cross_validate_builds` hold out whole builds; the
  empirical residual quantiles from CV become the interval on every prediction.
- **Consistency before correlation.** `verify_run_consistency` refuses to pair FE and BE reports from
  different signoff runs; `sanitize` refuses stale builds and unmapped hierarchies.
- **Everything profiled, everything catalogued.** `Profiler.stage()` around each pipeline step and
  SQLite rows for each build, file, import, quality run and model make the methodology auditable.

## Known limitations to keep in mind when lifting

- Adapter formats are representative, not vendor-exact, except SAIF which follows the SAIF 2.0 grammar; `parse()` is the function to rewrite.
- `PARTITION_OBJECT_PATTERNS` in `identity.py` encodes how partition objects are named in reports.
- `stage` in the wide schema is always `FE_BE`; it is reserved for other pairings.
- `TimingModel` fits per partition from the latest build and only borrows earlier builds for the
  voltage exponent when the latest build has a single corner.
