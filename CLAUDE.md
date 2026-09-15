# powermet

Power Metrology & Modeling: a local Python CLI that bridges physical implementation data and
architectural power decisions. It ingests EDA reports (PPRTL, PrimePower, PrimeTime, StarRC, Fusion
Compiler, SAIF activity from the FSDB → SAIF flow, performance) per design/build run directory, keys everything on the model root
(canonical FUB identity, FUB → partition), sanitizes each metric, correlates FE to BE power, fits
build-validated power/energy models (including a compact data-movement model), and answers what-if
questions with timing feasibility and workload traces.

## Purpose

Battle-tested, portable abstractions plus a general workflow (public knowledge, current tools) intended
to be applied to a company's power methodology for accelerator / GPU / ASIC / SoC projects; the reference
setup is CPU-style (separate FE/BE hierarchies, partition-level timing). Environment limitations are absorbed by adapters, config and `templates/`, never the core
abstractions. `docs/portability.md` is the adaptation checklist; keep it and `powermet sources` current.

## Layout

- `src/powermet/extract/` — one adapter per EDA source (`get_files()` locates, `parse()` reads). Contract in `base.py`.
- `identity.py` (ModelRoot), `lineage.py`, `measurements.py` (MeasurementStore), `selection.py` (DatasetSlice), `pipeline.py` (ingest).
- `sanitize.py` (quality checks registry + per-metric trust), `validation.py`, `catalog.py` (SQLite), `profiling.py`.
- Power closure: `budgets.py` (milestone budgets on any convergence metric), `intent.py` (UPF), `qualify.py` (engine/estimator qualification), `hotspots.py`.
- Power analysis (the analysis half of the role): `anomalies.py` (comparative rules registry `RULES` -> power bugs with owner, new/persisting/cleared; catalog `anomaly`), `timeprofile.py` (time-based profile: peak window, peak/avg, max step, energy, reconciliation), `extract/power_groups.py` (clock/register/comb/memory per FUB -> `clock_fraction`), `extract/power_profile.py` (design power per window -> `power_profile.parquet`), `owner` column in the FUB map. Docs: `docs/power-analysis.md` (rules table and the EDA reports each needs).
- Power convergence (what the flow drives): `schema.CONVERGENCE_METRICS` (Cdyn `cdyn_pf`, leakage power `be_leakage_mw`, total), `metrics.add_convergence_metrics`, `convergence.py` (gap, trend, projection, verdict; `plan()` ranks techniques by the component they move via `Technique.reduces` / `TechniqueResult.saving_for`). Leakage comes from the PrimePower / PPRTL leakage columns (`be_leakage_mw`, `fe_leakage_mw`).
- Methodology variants: `identity.IdentityStrategy` + `NameRules` (same hierarchy, explicit map, replicated, merged, split), `profiles.py` + `profiles/*.toml`, `extract.METRIC_AGG`; `techniques.py` registry. Docs: `docs/methodology-variants.md`, `docs/power-techniques.md`.
- `modeling.py` (MODEL_REGISTRY, build-based split/CV), `features.py`, `decomposition.py`, `deltas.py`, `frontier.py`, `curves.py`, `whatif.py`, `explore.py`, `integrate.py`.
- `cli.py` is argparse only; business logic lives in modules. `reporting.py` builds the 30-section Markdown report.
- `docs/reusable-components.md` maps every liftable abstraction to module, deps and tests. `docs/methodology.md`, `docs/extractors.md`, `docs/portability.md`, `docs/tool-landscape.md` (vendor/AI tool options per stage; confirm per company). `docs/ppa-convergence-playbook.md` (milestone loop, gap triage, P/P/A trade table, correlation gates).
- Fan-out: `ingest plan` -> worker `ingest run --partition-dir` (Parquet only) -> `ingest merge` (single writer). Keep workers free of catalog writes.

## Working rules

- Python 3.11+. Deps: pandas, numpy, pyarrow, duckdb, scipy, scikit-learn, joblib (required); matplotlib, psutil (optional). No new deps without a reason; no server/UI/cloud.
- Dev env: `uv venv --python 3.12 .venv && uv pip install -e ".[all,dev]"`. Tests: `.venv/bin/pytest -q` (~40 s).
- Keep abstractions single-sourced: object names resolve only in `ModelRoot` (strategy + map shape, never pipeline code); dataset slicing only via `DatasetSlice`; metric lists derive from `schema.py` / `extract/__init__.py`; new model kinds are a `ModelSpec`; new quality checks are a `CHECKS` entry plus one `mark()`.
- Rows are never split at random; hold out whole builds. Report associations, never causes. Round output sensibly.
- A new anomaly rule is one `Rule` entry in `anomalies.RULES` plus a function; thresholds are `anomaly_*` config fields; a rule that lacks its inputs must say so in `skipped`, never return an empty list silently. Findings name the owner from the FUB map; nothing else assigns ownership.
- Convergence targets are extensive sums over a scope in the metric's own unit; a new target metric is a `CONVERGENCE_METRICS` entry (name, unit, component) and its derivation in `metrics.py`. Technique savings must state which component they move; never count a corner change against a Cdyn target.
- Mock data (`powermet mock generate`) is a fixture with injected defects; when real report samples arrive, change `parse()`/`get_files()` in the matching adapter and extend `mockdata.py` to match.
- Adapter formats are representative, not vendor-exact (SAIF follows the real SAIF 2.0 grammar). `stage` is always `FE_BE`. Timing is a partition attribute inherited by FUBs. BE hierarchy paths go through the partition (`top/part_<p>/u_<fub>`); activity comes from the physical-hierarchy SAIF written by the Verdi FSDB → SAIF flow, with the flow inputs recorded in `metadata.json` → `activity_flow`.

## Quick demo

```bash
powermet mock generate && powermet ingest scan mock_runs && powermet sanitize
powermet model train --cv && powermet analyze deltas --design GPU_A && powermet report
powermet analyze anomalies --design GPU_A && powermet analyze profile --design GPU_A
```
