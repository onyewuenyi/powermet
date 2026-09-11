# powermet methodology (V0)

## Question

How well does FE (front-end) power predict BE (back-end, signoff) power at FUB level, which
physical characteristics are associated with the difference, and does a simple model that uses
those characteristics predict BE power better than FE physical power alone?

## Data model

One row = one FUB measurement identified by `(design, build, fub, stage, workload, operating_point)`.
FUB is an *analytical* dimension; V0 assumes the input has already been mapped to FUBs and does
not reconcile FE and BE hierarchies. See `src/powermet/schema.py` for the column table.

Provenance columns (`source_file`, `tool`, `tool_version`, `imported_at`) are carried on every row.
Raw files are never modified; import writes a processed copy under `.powermet/data/processed/`
and logs the source path and SHA-256 in `.powermet/cache/imports.jsonl` (and the DuckDB `imports`
table when DuckDB is present).

## Derived metrics

| Column | Definition |
|---|---|
| `logical_error_mw` | `be_mw - fe_logical_mw` |
| `logical_error_pct` | `(be_mw - fe_logical_mw) / be_mw * 100` |
| `physical_error_mw` | `be_mw - fe_physical_mw` |
| `physical_error_pct` | `(be_mw - fe_physical_mw) / be_mw * 100` |
| `logical_to_be_ratio` | `fe_logical_mw / be_mw` |
| `physical_to_be_ratio` | `fe_physical_mw / be_mw` |
| `wire_cap_fraction` | `wire_cap_pf / (wire_cap_pf + cell_cap_pf)` (analysis feature; size-independent) |

Positive error means FE *underestimates* BE. Division by zero or NaN yields NaN; rows with
`be_mw == 0` or missing are rejected at import because percentage error is undefined.

## Accuracy metrics

For a predictor `p` of BE power `y` over `n` rows:

- MAE = mean |p - y|; RMSE = sqrt(mean (p - y)^2)
- MAPE = mean |p - y| / y * 100; P50 / P95 are percentiles of |p - y| / y * 100
- R^2 = 1 - sum (p - y)^2 / sum (y - mean y)^2. For the FE stages this is the R^2 of the
  identity line `BE = FE` (no fitting), so it penalizes bias; the Pearson r reported alongside
  does not.

## Association analysis

`analyze correlation` and `analyze errors` report Pearson r (with p-value when scipy is present)
between features and BE power, and between features and FE -> BE error. Two views are shown:

- **percentage error**: size-independent; the preferred view for "what explains the gap".
- **mW error**: confounded by block size (bigger blocks have bigger absolute error), shown for
  completeness.

All of these are associations in the dataset at hand. None establish physical cause; features
that scale with block size are strongly collinear.

## Models and split

Rows are never split at random. Builds are sorted naturally (`B2 < B10`) and the latest
`ceil(test_fraction * n_builds)` builds are held out, so metrics approximate "predict the next
build". Fewer than 5 builds triggers a warning; fewer than 2 refuses to train.

| Model | Form | Purpose |
|---|---|---|
| Baseline | `BE = FE_physical` | the number to beat |
| Scaled | `BE = a * FE_physical + b` | bias correction only: separates "fix the scale" from "features add information" |
| Linear | OLS on `fe_physical_mw, wire_cap_pf, cell_cap_pf` (numpy) | explainable feature contribution |
| Tree | `HistGradientBoostingRegressor` on all available physical features (scikit-learn, optional) | non-linear check |

Feature importance is **permutation importance on the held-out builds** (increase in MAPE when
a feature is shuffled, normalized to shares). It is evidence about this model on this data, not
proof of physical cause. Linear coefficients are reported as mW change per one standard deviation
of the feature.

Artifacts: `.powermet/models/model_<timestamp>.joblib` (or `.pkl`) plus a JSON sidecar with model
types, features, target, train/test builds, metrics, importances, the config used and the dataset
SHA-256, so any result is reproducible from input + config + command.

## Reporting conventions

Percentages to one decimal, mW to one decimal, r and R^2 to two decimals. The wording is
"associated with", never "caused by".

## Not in V0

RTL/PrimePower/StarRC parsing, FE <-> BE hierarchy mapping, servers/APIs/UI, cloud storage,
neural networks, workload modeling beyond the basic columns.

---

# V1 — methodology infrastructure

## Extraction and provenance

`powermet ingest` walks run directories, calls each source adapter (`docs/extractors.md`), and
builds a *long* provenance table before anything is pivoted: one record per (object, metric)
with `source`, `source_file`, `tool`, `tool_version`, `run_id`, `report_date`, canonical `unit`
and `unit_original`. The wide FUB dataset is derived from it, so any number in any table can be
traced back to the report line it came from.

## Lineage

The FUB map (`fub, fe_hier, synth_object, be_hier`) is the analytical mapping input. Objects in
FE-side reports are resolved through `fe_hier`, BE-side reports through `be_hier`, design-level
records apply to all FUBs. The lineage table records, per FUB and build:

```
FUB -> FE hierarchy -> synthesis object -> BE hierarchy -> physical instances (cell count) -> measurement sources
```

and flags `fe_hier_not_in_reports`, `be_hier_not_in_reports` and `incomplete_physical_mapping`.
Report objects no FUB claims are kept in `unmapped.parquet` (e.g. instances the BE flow renamed
after the map was made).

## Sanitization

`powermet sanitize` runs cross-source checks that schema validation cannot:

| Check | Blocks use | Source of evidence |
|---|---|---|
| missing power metrics | yes | wide dataset |
| missing physical data (incomplete mapping) | yes | wide dataset + lineage |
| duplicate measurement keys | yes | wide dataset |
| duplicate rows inside source reports | no | long table |
| unit conversions applied | no | long table (`unit_original != unit`) |
| unit inconsistency by magnitude (>50x design median) | yes | wide dataset |
| impossible negatives | yes | wide dataset |
| zero / near-zero BE power | yes | wide dataset |
| stale / superseded builds (metadata status or > `stale_days` behind newest) | yes | metadata |
| suspicious outliers (ratio / log-z) | no | wide dataset |
| FE/BE lineage mismatches | yes | lineage table |
| unmapped report objects | no | unmapped table |

Nothing is deleted. `quality_flags.parquet` holds every row's flags; `measurements_sanitized.parquet`
holds usable rows and is what analysis/model commands load (`use_sanitized = true`).

## Runtime and memory

`Profiler` wraps each stage with wall time, CPU time and peak RSS (`resource.getrusage`), saved to
`cache/profiles.jsonl`. Stages: extraction, normalization, lineage mapping, pivot, validation,
store (Parquet + DuckDB), sanitization, correlation, error analysis, model training, CV, save.

# V2 — predictive model

## Models

| Model | Form | Use |
|---|---|---|
| baseline | BE = FE_physical | number to beat |
| scaled | BE = a·FE_physical + b | separates bias correction from feature value |
| linear | OLS on FE_physical, wire cap, cell cap | explainable |
| physics | OLS on `dyn_term` = activity·(wire+cell cap)·V²·f, `leak_term` = area·V³, FE_physical | physically structured; extrapolates sensibly in V, f, cap, activity; default for what-if |
| tree | HistGradientBoosting with monotonic constraints (power non-decreasing in FE physical, caps, area, activity, V, f) | non-linear check; cannot extrapolate |

Engineered features live in `features.py` and are recomputed after any what-if override.

## Validation

- Holdout: latest `ceil(test_fraction · n_builds)` builds.
- Leave-one-build-out CV: each build predicted from all others; per-fold MAPE, mean/sd/worst,
  and the empirical 5th/50th/95th percentiles of relative error, used as the **90% interval**
  on every what-if prediction.
- Deployed models are refit on all builds after the held-out metrics are recorded.

## What-if

`model predict` selects rows (design/FUB/build/workload/operating point; latest build by default),
applies `--set`/`--scale` overrides, recomputes engineered features, rescales FE physical power by
the change in the CV²f term (so the model is not fed a stale FE estimate), predicts with an
extrapolating model, and reports current vs proposed power with the CV interval and the other
models' predictions for comparison.

# V3 — workload, performance and exploration

- Energy per op: Σ BE power (mW) / throughput (Gops/s) = pJ/op, per design × build × workload × operating point.
- Perf model: `log(throughput) = a + b·log(f)` per (design, workload); `b < 1` indicates memory-bound saturation.
- DVFS curve: `V = c0 + c1·f` per design from the measured operating points; frequency sweeps follow it unless `--fixed-voltage`.
- `explore sweep` / `opmap` / `scenario` evaluate scenarios with the power model + perf model, report
  ΔP, ΔPerf, ΔE/op and mark Pareto-optimal points (no other point has lower power and ≥ throughput).

All V3 numbers are model predictions layered on model predictions; the tables carry the power
model's CV interval and the perf model's fit quality is reported by `workload summary`.

# Activity from SAIF

Switching activity originates in the RTL simulation FSDB per workload. The FSDB → SAIF flow
(Verdi; inputs: workload FSDB, core, FE/BE mapping data, partition list) writes a SAIF in the
back-end physical hierarchy, the same file that drives SAIF-based power optimization in early
Fusion Compiler. powermet reads that SAIF and aggregates per instance including descendants:
`activity` = mean over nets of TC / cycles (the activity factor), `bits_per_cycle` = Σ TC / cycles
(the data-movement traffic proxy), `net_count`. Cycles come from DURATION × TIMESCALE and the
simulation clock period declared in `metadata.json` (`activity_flow.sim_clock_period_ps`, else
the nominal operating point). The flow's inputs are recorded as provenance on every activity record.

# Timing, identity and the data-movement model

## Identity: model_root

`model_root` (design.partition.fub) is the canonical FUB identity carried through FE, BE, timing,
lineage, quality flags and the compact model. FE and BE hierarchy paths are preserved next to it so
a poorly correlating FUB can be drilled into. Duplicate detection still uses the measurement key.

## Timing at partition level

PrimeTime reports per partition (physical block), not per FUB. The FUB map declares each FUB's
partition; lineage fans every partition record (period, WNS, TNS, violating endpoints) out to its
FUBs, and `fmax_ghz = 1000 / (period - WNS)`. A partition missing from the timing report is a
non-blocking `timing_missing` flag (power correlation does not need timing); `wns > period` or a
non-positive period is `timing_suspect`.

## Build-to-build deltas and the power x timing frontier

`analyze deltas` aggregates consecutive builds per design (and per partition, per FUB): BE power,
FE physical power, wire cap, cell cap, area, cell count, wire length, worst-partition WNS and Fmax.
Each move is classified:

| power | timing (WNS) | class |
|---|---|---|
| down (or flat) | up (or flat) | pareto improvement |
| up (or flat) | down (or flat) | regression |
| up | up | power-for-performance trade |
| down | down | performance-for-power trade |

`analyze frontier` places every build at (worst-partition Fmax, total BE power) and marks builds
no other build dominates. The performance axis is Fmax from PrimeTime; throughput from the perf
model is shown for reference only.

## Timing feasibility in exploration

Per (design, partition), `log(delay) = a + k log(V)` is fitted from the measured operating points
of the latest build; `Fmax(V) = 1000 / delay(V)`. Sweeps, operating-point maps and scenarios mark a
point VIOLATES when `f > min_partition Fmax(V)`, and infeasible points are excluded from Pareto marking.

## Data-movement energy model

Post-layout features `wire_length_um`, `avg_net_length_um` (implementation) and `bits_per_cycle`
(SAIF: bits switched per cycle, the sum of net toggle counts over cycles) feed engineered terms:

| term | definition | attributed to |
|---|---|---|
| `cell_dyn_term` | activity · cell_cap · V² · f | compute (cell switching) |
| `wire_dyn_term` | activity · wire_cap · V² · f | wire switching |
| `move_term` | bits_per_cycle · avg_net_length · V² · f | data movement (bits x distance) |
| `leak_term` | area · V³ | leakage |

The `datamove` model is OLS on these four terms without any FE power estimate: a compact analytical
energy model whose fitted coefficients decompose each FUB's predicted power into the four shares
(`analyze energy`), and whose `move_term` coefficient x V² is the pJ per bit-mm. It is validated
like every other model (latest-build holdout and leave-one-build-out CV) and is the default what-if
and exploration model because it depends only on physical and activity inputs.

## Per-metric trust

`sanitize` also scores every input metric: coverage, Pearson r with BE power, the min/max of that
r across builds (stability), unit conversions applied, outliers, and a verdict (trusted / partial /
unstable / unusable). Sparse or unstable metrics are candidates for calibration or exclusion before
they reach a model.

## Compact model and performance-tool integration

`model export` writes a JSON document a performance simulator can evaluate without powermet: the
fitted terms, per design the latest-build physical features per FUB, per-workload activity and
bits/cycle, operating points, the DVFS curve, per-workload throughput scaling (`throughput ~ f^b`)
and the per-partition timing model. `integrate trace` walks a phase trace (workload, operating point
or (f, V), activity scale, duration) and reports power, throughput, energy, pJ/op and timing
feasibility per phase and in total. `powermet.integrate.CompactPowerModel` is the reference evaluator.

## Metadata catalog

`metrology.db` (SQLite) records builds, every parsed source file with SHA-256, tool and version,
import runs, quality runs (including the metric trust table), models and runtime profiles. It is the
laptop stand-in for a shared catalog; the schema is plain SQL.

## Not yet built (V4 and beyond)

Vendor-exact parsers, automatic FE ↔ BE hierarchy mapping, endpoint-level timing, multi-corner
signoff, pre/post-silicon calibration feedback (`model calibrate`), a joint power/performance
optimizer, servers/APIs/UI, cloud storage, neural networks.
