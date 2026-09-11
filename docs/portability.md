# Purpose and portability

## Why this project exists

powermet is a battle-tested set of abstractions and a general workflow for power metrology and
modeling, built from public knowledge and current tools and patterns. It was developed on a CPU
program (the author's current team) but the target is any accelerator, GPU, ASIC or SoC power
methodology: at a new company the pieces get applied where a methodology already exists, or the
whole workflow gets implemented where it does not. It is also a proof of concept for roles such as
SoC power analysis and optimization or power methodology and modeling engineering, where the
questions are the same: how trustworthy is the correlation data, how early can power be predicted,
and what design change buys what.

The expectation is that every target environment has limitations (compute and execution
environment, where reports live, which engines are licensed, how the model root is exported) that
force changes. Those changes should land in adapters and templates, never in the core abstractions.

## What is expected to change, and where

| Dimension | Likely difference at a target company | Where it lands |
|---|---|---|
| Report locations and naming | NFS trees, artifact stores, per-team conventions | `source_patterns` in config; `get_files()` per adapter |
| Report formats and engine versions | vendor-exact output, new PrimePower / PrimeTime / StarRC releases, Cadence instead of Synopsys | `parse()` per adapter; `SUPPORTED_VERSIONS`; `tool_version` in provenance flags the change first |
| Execution environment | batch schedulers, no local write access, restricted Python | CLI stays argparse and pure Python; `.powermet/` root via `--project-dir` / `POWERMET_HOME` |
| Storage | shared catalog, object storage | `catalog.py` schema is plain SQL (SQLite → PostgreSQL); Parquet stays; DuckDB optional |
| Identity export | model root as a database, a YAML, a tool query | `ModelRoot.from_frame()` from any frame; `identity.py` is the only matching code |
| Unit of analysis | FUB (CPU) → unit, SM, cluster, PE, tile (GPU / accelerator) | the schema column is still `fub`; `model_root` carries the real name; labels are one dict |
| Workloads | vectors / traces (CPU) → kernels, layers, graphs (accelerator) | `workload` is a free string; traces drive `integrate trace` |
| Data-movement emphasis | far larger on accelerators (NoC, SRAM, HBM traffic) | `move_term` and the datamove model are the extension point; add terms as `ModelSpec` features |
| Timing granularity | partition-level here; endpoint or path-group elsewhere | PrimeTime adapter emits partition objects; `ModelRoot.resolve_partition` fans out |
| Power convergence engines | new signoff engines, different activity flows | one adapter each; the long-record contract does not change |

## Adaptation checklist for a new environment

1. Run `powermet sources` and compare each adapter's tool family and versions with what the
   environment produces. Rewrite `parse()` against one real sample per source; keep the records long.
2. Fill `templates/metadata.template.json` from the flow that owns run identity; set `design_type`.
3. Export the model root to `templates/fub_map.template.csv` shape (FUB list, canonical id,
   partition, FE and BE paths). Confirm BE paths match the physical hierarchy the reports print.
4. Copy `templates/config.template.toml`, set `source_patterns` and `disabled_sources`.
5. Ingest one build; read the lineage flags and the `CORRELATION DATA QUALITY` block before
   anything else. Unmapped objects and consistency errors are the environment telling you where
   its conventions differ.
6. Only then correlate, model and explore. Calibrate against the next signoff build.

## Keeping it current as the technology changes

- Engine upgrades appear as a new `tool_version` in provenance and the catalog; add the version to
  the adapter's `SUPPORTED_VERSIONS` once the parser is verified against it.
- New sources (a different activity flow, a thermal or IR-drop report, silicon measurements) are
  one adapter plus a `SOURCE_METRICS` / `METRIC_SCOPE` entry; the pivot and downstream do not change.
- New physics (a different data-movement decomposition, a leakage model with temperature) is one
  engineered feature plus one `ModelSpec`.
- New quality rules are one `CHECKS` entry plus one `mark()`.
- The mock generator is the regression harness: extend it when a real format is added so the
  pipeline is always runnable without proprietary data.
