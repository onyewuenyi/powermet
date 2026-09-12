# powermet — Power Metrology & Modeling (V0 → V3, with timing)

Local command-line tool that bridges physical implementation and architectural decisions:
post-layout data → trustworthy per-metric features → compact power/energy model → performance and
timing integration → what-if. Power, timing (PrimeTime, partition level) and physical data share one
canonical FUB identity (`model_root`).

No server, no REST API, no UI, no cloud. Everything lives under a local `.powermet/`
directory you can delete to start over. Python 3.11+.

## Purpose

A battle-tested set of abstractions and a general workflow, built from public knowledge and current
tools, that can be applied to a company's power methodology in its own compute and execution
environment for accelerator, GPU, ASIC or SoC projects. It was developed on a CPU program; the unit
of analysis (`fub`), the workloads and the data-movement terms are deliberately generic so the same
pipeline transfers. Environment limitations are expected: they are absorbed by adapters, config and
the files under `templates/`, not by the core abstractions. `docs/portability.md` lists what is
expected to change, where it lands, and how the workflow stays current as signoff engines and
activity flows change; `docs/tool-landscape.md` is the reference of current Synopsys, Cadence, Keysight
and data/AI tools per methodology stage, each to be confirmed at the target company. `powermet sources` shows every adapter with the tool family and versions it
was written against.

| Version | Question it answers | Commands |
|---|---|---|
| **V0** | Can I correlate FE to BE? | `demo`, `data`, `analyze`, `model train/evaluate`, `report` |
| **V1** | Can I build a trustworthy, automated methodology for that correlation? | `mock`, `ingest`, `sanitize`, `lineage`, `profile`, `--profile` |
| **V2** | Can I predict BE power before BE is complete? | `model train --cv`, `model validate`, `model predict` |
| **V3** | Can I answer workload / design what-if questions? | `workload summary`, `explore sweep/opmap/scenario`, `model export`, `integrate trace` |
| **Timing** | What physical changes improve timing but hurt power? | `analyze deltas`, `analyze frontier`, timing-feasible `explore` |
| **Closure** | Are we within budget at this milestone, is the power intent consistent, can I trust this engine? | `budget check`, `intent show`, `qualify`, `analyze hotspots` |
| **Convergence** | Will the design hit its CdynTot and LkgPwr targets, and which techniques close the gap? | `converge --plan`, `techniques assess` |

## Install

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[all,dev]"    # dev laptop
pip install -e ".[all]"                                          # environment with packages present
powermet doctor
```

Required: pandas, numpy, pyarrow, duckdb, scipy, scikit-learn, joblib. Optional: matplotlib (charts), psutil.

## The full workflow on mock data

```bash
powermet mock generate                        # mock_runs/<design>/<build>/ with tool-style reports
powermet ingest scan mock_runs                # extract -> normalize -> lineage -> FUB dataset (profiled)
powermet sanitize                             # CORRELATION DATA QUALITY report + sanitized dataset
powermet lineage show Scheduler --design GPU_A
powermet analyze summary
powermet analyze correlation
powermet analyze errors
powermet model train --cv                     # holdout on latest builds + leave-one-build-out CV
powermet model evaluate
powermet model predict --design GPU_A --fub Scheduler --workload typical --operating-point nom --scale wire_cap_pf=0.8
powermet analyze energy --design GPU_A        # compute / wire / data-movement / leakage decomposition, pJ per bit-mm
powermet analyze deltas --design GPU_A        # build N -> N+1: power, WNS/Fmax, wire cap, area; classify the trade
powermet analyze frontier --design GPU_A      # power x Fmax frontier across builds (Pareto / regression / trade)
powermet workload summary                     # power, throughput, pJ/op by workload x operating point
powermet explore sweep --design GPU_A --workload compute --param frequency_ghz --values 1.6,1.8,2.0,2.2,2.4
powermet explore opmap --design GPU_A --add "v=0.78,f=2.3"
powermet explore scenario examples/scenarios/gpu_a.toml
powermet budget check --history               # budgets per design/partition vs milestone tolerance; ON TRACK / AT RISK / OVER
powermet converge --plan                      # CdynTot / LkgPwr / total targets: gap, trend, builds-to-target, ranked closure plan
powermet analyze hotspots --design GPU_A      # share, power density, growth vs previous build, clock-gating efficiency
powermet qualify --a be_mw --b be_voltus_mw   # engine-to-engine qualification with a tolerance and PASS/FAIL
powermet intent show                          # UPF domains per FUB, missing domains, voltage mismatches
powermet model export                         # compact JSON power model for a performance simulator
powermet integrate trace mock_runs/traces/GPU_A_phases.csv   # power / throughput / energy timeline of a phase trace
powermet measure get --fub Scheduler --design GPU_A --stage FE --metric fe_physical_mw   # one number with provenance
powermet profiles list                        # methodology profiles: same hierarchy, separate + map, replicated units, flattened BE, RTL-only activity
powermet init --design NPU_X --design-type ai_accelerator --profile replicated_units   # config + templates for that setup
powermet techniques list                      # power-optimisation techniques: problem, mechanism, trade-off, considerations
powermet techniques assess --design GPU_A     # rank candidate FUBs per technique with stated assumptions
powermet sources                              # adapters: stage, tool family, versions written against, patterns, metrics
powermet ingest plan mock_runs | sh           # scheduler fan-out: one worker job per run writes a Parquet partition
powermet ingest merge                         # single-writer merge of partitions into the dataset and catalog
powermet db tables                            # SQLite metadata catalog; `db query "<sql>" [--engine duckdb]`
powermet profile show
powermet report                               # .powermet/reports/power_metrology_report.md (24 sections)
```

The V0 flat-file path still works: `powermet demo generate`, `powermet data validate <file>`,
`powermet data import <file>`.

## Data sources (V1)

`powermet ingest` reads one run directory per design × build:

```
<root>/<design>/<build>/
  metadata.json                           design, build, date, run_id, status, tool versions, operating points (V, f)
  mapping/fub_map.csv                     fub, model_root, partition, fe_hier, synth_object, be_hier
  pprtl/<wl>_<op>/{rtl,physical}_power.rpt   FE logical / FE physical power per FE hierarchy
  primepower/<wl>_<op>/power_hier.rpt     BE power per BE hierarchy (indented report_power -hierarchy style)
  primetime/<op>/timing_summary.rpt       period, WNS, TNS, violating endpoints per PARTITION (inherited by its FUBs)
  starrc/parasitics_summary.rpt           wire cap / pin cap per BE hierarchy
  implementation/qor_summary.rpt          area, cell count, fanout, wire length, avg net length per BE hierarchy
  activity/<wl>.saif                      SAIF in the physical hierarchy from the FSDB -> SAIF flow (activity, bits switched/cycle)
  perf/<wl>_<op>.csv                      design-level ipc / throughput (optional)
  voltus/<wl>_<op>/power_hier.rpt         second signoff engine (optional), for `qualify`
  intent/<design>.upf                     power intent: domains, supplies, port states
<root>/budgets.toml                       power budgets with per-milestone tolerance
<root>/traces/<design>_phases.csv         workload phase trace for `integrate trace`
```

**Activity.** The source of truth for switching activity is the RTL simulation FSDB per workload
(logical hierarchy). The existing FSDB → SAIF flow (Verdi) takes the workload FSDB, the core, the FE/BE
mapping data and the partition list, and writes a SAIF in the back-end physical hierarchy; that same SAIF
drives SAIF-based power optimization in early Fusion Compiler. powermet reads that SAIF directly: per
instance, `activity` = mean toggles per cycle per net, `bits_per_cycle` = total toggles per cycle, with the
simulation clock period and the flow's inputs (FSDB path, core, mapping, partition list) recorded in
`metadata.json` under `activity_flow` and carried as provenance.

**Methodology variants.** Companies differ in how FE and BE hierarchies relate: same hierarchy, separate
hierarchies with a map, renamed or uniquified instances, replicated units (cores, SMs, PEs), merged blocks
after synthesis ungrouping, FUBs split across partitions. `identity.IdentityStrategy` (kind, replica policy,
merge basis, name rules) handles each; the FUB map may carry several rows per FUB, a shared BE path with
`be_share`, or a glob BE path; extensive metrics add and intensive ones average across many-to-one objects;
`lineage.parquet` records the relationship per FUB. `docs/methodology-variants.md` catalogues the setups,
the problem each creates and the switch that handles it; profiles bundle the switches. The mock generator
produces every layout (`--methodology`).

**Identity.** The model root is the source of truth: `ModelRoot` (`identity.py`) loads the FUB list with
`model_root` (e.g. `GPU_A.PCORE0.Scheduler`), partition and FE/BE hierarchy from `mapping/fub_map.csv`,
and is the only place report object names are resolved to FUBs. `fub` is the short name. Reports whose run
id or build disagree with `metadata.json` are dropped (`strict_consistency`) so FE and BE are never paired
across signoff runs. FE and BE hierarchy paths are kept so a
poorly correlating FUB can be drilled into. **Timing** is a partition attribute: PrimeTime reports per
partition, the FUB map says which partition implements each FUB, and every FUB inherits its partition's
WNS/TNS/Fmax.

Each source is one module in `src/powermet/extract/` with `get_files(inputs)` (where the data is;
**this is the function to change when the real location or naming is known**) and `parse(path)`
(what the file looks like). Patterns can be overridden per source in `.powermet/config.toml`
(`source_patterns`). Units are normalized (W→mW, fF→pF, MHz→GHz) and the original unit is kept in the
long provenance table. See `docs/extractors.md`.

## What each layer produces

- **Provenance**: `measurements_long.parquet` holds every (object, metric, value) with source, file, tool,
  version, run_id and original unit. The wide FUB dataset carries run_id, build_date, source run dir.
- **Lineage**: `lineage.parquet` has one row per FUB/build with the chain
  FUB → FE hierarchy → synthesis object → BE hierarchy → physical instances → measurement sources, plus
  mismatch flags (`fe_hier_not_in_reports`, `be_hier_not_in_reports`, `incomplete_physical_mapping`);
  `unmapped.parquet` lists report objects that no FUB claims.
- **Sanitization**: per-row `quality_flags.parquet` and `measurements_sanitized.parquet` (usable rows).
  Checks: missing metrics, missing physical data, duplicates (dataset and source-report level), unit
  conversions and magnitude-based unit suspicion, negatives, near-zero BE, stale/superseded builds,
  outliers, lineage mismatches, missing/inconsistent timing, unmapped objects. Plus a **per-metric trust
  table** (`metric_quality.parquet`): coverage, association with BE power, stability of that association
  across builds, unit conversions, outliers, verdict.
- **Fan-out**: `ingest plan` prints one worker command per run directory (wrapped in the scheduler
  template from config); `ingest run --partition-dir` extracts one run and writes only a Parquet partition
  (no dataset, no catalog, no locks); `ingest merge` is the single writer that validates, merges and
  catalogs. Serial and fan-out produce identical datasets (tested).
- **Catalog**: `metrology.db` (SQLite) holds builds, source files with SHA-256 and tool versions, import
  runs, quality runs, models and profiles; Parquet holds the measurements; DuckDB queries both.
- **Profiling**: every ingest/sanitize/analyze/train command appends stage wall/CPU time and peak RSS to
  `cache/profiles.jsonl`; `--profile` prints it, `profile show` lists recent runs.
- **Models**: baseline (BE = FE physical), scaled baseline, linear OLS, physics-structured OLS
  (activity·C·V²·f + bits·distance·V²·f + area·V³ + FE physical), a **data-movement decomposition** model
  (cell switching + wire switching + bits × distance + leakage, no FE power: the compact analytical energy
  model), and monotone-constrained gradient boosting.
  Latest-builds holdout plus leave-one-build-out CV with empirical 90% error intervals. Artifacts in
  `models/model_<ts>.joblib` + JSON metadata (features, builds, metrics, CV, coefficients, config, dataset SHA).
- **What-if**: overrides on raw parameters propagate through the engineered features; FE physical power is
  rescaled with the CV²f term; extrapolating models are used by default; the CV interval is attached.
- **Workload / performance**: energy per op = Σ BE power / throughput (pJ/op); per-workload throughput ~ f^b
  fit; per-design DVFS curve V(f) from measured operating points; FUB workload sensitivity.
- **Exploration**: parameter sweeps (frequency follows the DVFS curve unless `--fixed-voltage`), operating-point
  maps with candidate (V, f) points, TOML scenario files; Pareto marking on (power, throughput); every point
  is checked against the per-partition timing model (delay ~ V^k) and marked VIOLATES when f exceeds Fmax.
- **Timing analysis**: `analyze deltas` compares consecutive builds (power, WNS, Fmax, wire cap, cell cap,
  area, per partition and per FUB) and classifies each move as Pareto improvement, regression, or a
  power-for-performance trade; `analyze frontier` plots power vs worst-partition Fmax across builds.
- **Energy decomposition**: `analyze energy` splits predicted power into compute, wire, data-movement and
  leakage shares and reports the fitted pJ per bit-mm.
- **Techniques**: `techniques list` documents clock gating, power gating, DVFS, wire-cap reduction, multi-Vt,
  operand isolation, glitch reduction and memory sleep (problem, mechanism, trade-off, considerations, data
  needed); `techniques assess` ranks candidate FUBs and estimates savings under stated assumptions, and says
  which source would enable the techniques it cannot assess (`docs/power-techniques.md`).
- **Power convergence**: the thing the whole flow drives. Targets are set per milestone on the design-owned
  quantities, not just total power: **CdynTot** (`cdyn_pf`, effective switched capacitance = dynamic power with
  V²f divided out, so it compares FE to BE, build to build and corner to corner) and **LkgPwr**
  (`be_leakage_mw`, tracked separately because its levers and corner sensitivity differ). PrimePower and PPRTL
  leakage columns are kept as `be_leakage_mw` / `fe_leakage_mw`; `sanitize` flags leakage that exceeds the total or
  varies with the workload. `converge` reports gap, cut needed, trend and builds-to-target with a
  CONVERGED / CONVERGING / FLAT / DIVERGING verdict; `converge --plan` ranks the techniques that move the target's
  component (dynamic levers for a CdynTot gap, leakage levers for LkgPwr, corner changes never counted against a
  fixed-corner target) by the share of the gap each covers (`docs/methodology.md`, "Power convergence").
- **Power closure**: `budget check` tracks budgets per design / partition / FUB against a per-milestone
  tolerance with trend, FUB coverage and the model's error band; `intent show` verifies UPF domains and
  supply states against the operating points; `qualify` compares two estimates of the same quantity (engine
  vs engine, version vs version, vectorless vs SAIF) with a PASS / FAIL against a tolerance; `analyze
  hotspots` ranks power share, density, growth and clock-gating efficiency. Vectorless BE numbers carry
  `be_activity_mode` and are flagged lower-trust.
- **Integration**: `model export` writes a compact JSON model (per-FUB physical features, per-workload
  activity/traffic, operating points, DVFS, throughput scaling, partition timing); `integrate trace`
  evaluates a workload phase trace into a power / throughput / energy timeline with timing feasibility.

## Project layout on disk

```
.powermet/
  config.toml                 features, split, sanitization thresholds, source patterns
  metrology.db                SQLite catalog: build, source_file, import_run, quality_run, model, profile_run
  data/processed/             measurements.parquet, measurements_long.parquet, lineage.parquet, unmapped.parquet,
                              performance.parquet, quality_flags.parquet, metric_quality.parquet,
                              measurements_sanitized.parquet, rejected.parquet
  models/                     model_<ts>.joblib + model_<ts>.json, compact_power_model.json
  reports/                    power_metrology_report.md, *.png (incl. power_timing_frontier.png)
  cache/                      powermet.duckdb (views over the Parquet files)
```

## Lifting pieces into another pipeline

`docs/reusable-components.md` maps every reusable abstraction (adapter contract, `ModelRoot` identity,
`MeasurementStore`, `DatasetSlice`, model and check registries, catalog, profiler, compact model) to its
module, its dependencies and the tests that pin its behaviour.

## Development

```bash
.venv/bin/pytest -q
```

`src/powermet/`: `cli.py`, `extract/` (base, metadata, pprtl, primepower, primetime, starrc, implementation,
activity, perf), `identity.py`, `lineage.py`, `measurements.py`, `selection.py`, `pipeline.py`, `sanitize.py`,
`catalog.py`, `profiling.py`, `mockdata.py`, `demo.py`, `schema.py`, `validation.py`, `ingest.py`, `storage.py`,
`metrics.py`, `features.py`, `correlation.py`, `errors.py`, `summary.py`, `budgets.py`, `intent.py`, `qualify.py`,
`hotspots.py`, `techniques.py`, `profiles.py`, `modeling.py`, `decomposition.py`,
`deltas.py`, `frontier.py`, `curves.py`, `whatif.py`, `workload.py`, `explore.py`, `integrate.py`,
`visualization.py`, `reporting.py` (`energy.py` is a compatibility facade).

## Principles

Raw data is immutable; every number keeps its provenance; FUB is an analytical dimension, not the physical
hierarchy; no fake precision; every analysis is reproducible from input + config + command; explainable
models before sophisticated ones; correlation is reported as association, never cause; what-if numbers are
predictions to be validated against a real build.
