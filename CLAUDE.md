# powermet

Power Metrology & Modeling: a local Python CLI that bridges physical implementation data and
architectural power decisions. It ingests EDA reports (PPRTL, PrimePower, PrimeTime, StarRC, Fusion
Compiler, SAIF activity from the FSDB → SAIF flow, performance) per design/build run directory, keys everything on the model root
(canonical FUB identity, FUB → partition), sanitizes each metric, correlates FE to BE power, fits
build-validated power/energy models (including a compact data-movement model), and answers what-if
questions with timing feasibility and workload traces.

## Layout

- `src/powermet/extract/` — one adapter per EDA source (`get_files()` locates, `parse()` reads). Contract in `base.py`.
- `identity.py` (ModelRoot), `lineage.py`, `measurements.py` (MeasurementStore), `selection.py` (DatasetSlice), `pipeline.py` (ingest).
- `sanitize.py` (quality checks registry + per-metric trust), `validation.py`, `catalog.py` (SQLite), `profiling.py`.
- `modeling.py` (MODEL_REGISTRY, build-based split/CV), `features.py`, `decomposition.py`, `deltas.py`, `frontier.py`, `curves.py`, `whatif.py`, `explore.py`, `integrate.py`.
- `cli.py` is argparse only; business logic lives in modules. `reporting.py` builds the 24-section Markdown report.
- `docs/reusable-components.md` maps every liftable abstraction to module, deps and tests. `docs/methodology.md`, `docs/extractors.md`.

## Working rules

- Python 3.11+. Deps: pandas, numpy, pyarrow, duckdb, scipy, scikit-learn, joblib (required); matplotlib, psutil (optional). No new deps without a reason; no server/UI/cloud.
- Dev env: `uv venv --python 3.12 .venv && uv pip install -e ".[all,dev]"`. Tests: `.venv/bin/pytest -q` (~40 s).
- Keep abstractions single-sourced: object names resolve only in `ModelRoot`; dataset slicing only via `DatasetSlice`; metric lists derive from `schema.py` / `extract/__init__.py`; new model kinds are a `ModelSpec`; new quality checks are a `CHECKS` entry plus one `mark()`.
- Rows are never split at random; hold out whole builds. Report associations, never causes. Round output sensibly.
- Mock data (`powermet mock generate`) is a fixture with injected defects; when real report samples arrive, change `parse()`/`get_files()` in the matching adapter and extend `mockdata.py` to match.
- Adapter formats are representative, not vendor-exact (SAIF follows the real SAIF 2.0 grammar). `stage` is always `FE_BE`. Timing is a partition attribute inherited by FUBs. BE hierarchy paths go through the partition (`top/part_<p>/u_<fub>`); activity comes from the physical-hierarchy SAIF written by the Verdi FSDB → SAIF flow, with the flow inputs recorded in `metadata.json` → `activity_flow`.

## Quick demo

```bash
powermet mock generate && powermet ingest scan mock_runs && powermet sanitize
powermet model train --cv && powermet analyze deltas --design GPU_A && powermet report
```
