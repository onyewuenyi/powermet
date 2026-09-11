# Extractors: where the data comes from and how to repoint them

Every EDA source is a module in `src/powermet/extract/` that follows one contract
(`extract/base.py`):

| Symbol | Meaning |
|---|---|
| `SOURCE` | short name recorded in provenance (`primepower`, `starrc`, ...) |
| `DEFAULT_PATTERN` | path pattern under the run directory; may use `{workload}`, `{operating_point}` and glob wildcards |
| `OBJECT_KIND` | one of `base.OBJECT_KINDS` (`FE_HIER`, `BE_HIER`, `PARTITION`, `DESIGN`); resolution lives in `identity.ModelRoot.resolve()` |
| `get_files(inputs) -> list[Located]` | **where** the files are for a run. Change this when the real location is known. |
| `parse(path, **context) -> ParsedReport` | **what** the file looks like. Returns long records `(object, object_kind, metric, value, unit, unit_original)` plus tool, version, run_id, date. |

`SourceInputs` is the common input every locator receives: `design`, `build`, `run_dir`,
`workloads`, `operating_points`, and `patterns` (per-source overrides read from
`.powermet/config.toml` → `source_patterns`).

Adapter conventions (enforced by `tests/test_extract_units.py`): build records with `base.record()`,
convert with `convert_unit()` and keep `unit_original`, fill `tool` via `tool_name(hdr, default)`,
surface `run_id` / `build` from the header so `pipeline.verify_run_consistency()` can reject stale
or misplaced reports, and put `workload` / `operating_point` on the `ParsedReport` unless one file
spans several (then a per-record column, as `metadata.py` does).

## Changing where a source lives

Three levels, from cheapest to most flexible:

1. **Different file names / layout, same run tree**: set `source_patterns` in `config.toml`, e.g.
   `source_patterns = { primepower = "pt/{workload}/{operating_point}/report_power_hier.txt" }`.
2. **Different tree or a fetch step (NFS, artifact store, database export)**: edit `get_files()` in that
   source's module. It only has to return `Located(path, context)` entries; download or copy into a temp dir
   if needed and return those paths. `locate()` in `base.py` is the shared glob helper you can keep using.
3. **Different file format**: edit `parse()`; keep returning the same long records. Unit handling goes through
   `convert_unit()` so a report in W or fF is normalized and the original unit is preserved.

## Report formats the mock writer produces

These are representative, not vendor-exact. Each parser is written the way a real one would be
(header key/value parsing, table detection, indentation-based hierarchy reconstruction for PrimePower,
unit normalization), so replacing them with vendor-exact parsing is a local change.

- `metadata.json`: `design`, `build`, `build_date`, `run_id`, `status` (`current`/`superseded`), `tools{}`,
  `workloads[]`, `operating_points{name: {voltage_v, frequency_ghz}}`.
- `mapping/fub_map.csv`: `fub, model_root, partition, fe_hier, synth_object, be_hier` (`model_root` and
  `partition` optional; without them identity falls back to `fub` and timing cannot be attached).
- PrimeTime: `Time units: ps|ns`; `Scenario: <op>`; `Partition Clock Period WNS TNS Violating Endpoints` rows keyed
  by `<top>/part_<partition>`; `object_kind = partition`, fanned out to every FUB of that partition by lineage.
- PPRTL: `Mode: rtl | physical-aware` selects `fe_logical_mw` / `fe_physical_mw`; table columns
  `Hierarchy Internal Switching Leakage Total`.
- PrimePower: `Power Units = 1W|1mW`; `Scenario: <workload>@<op>`; indented `name (ref) Int Switch Leak Total %`
  rows; full path rebuilt from indentation, top-level row skipped.
- StarRC: `Capacitance units: pF|fF`; `Instance Nets TotalCap WireCap PinCap` → `wire_cap_pf`, `cell_cap_pf`.
- Implementation: `Area units: um^2`; `Hierarchy CellArea CellCount AvgFanout Utilization [WireLength AvgNetLen]` →
  `area`, `cell_count`, `fanout`, optional `wire_length_um`, `avg_net_length_um` (data-movement distance proxy).
- Activity: `Hierarchy AvgToggleRate NetCount [BitsPerCycle]` → `activity`, optional `bits_per_cycle` (per workload).
- Perf: CSV `metric,value,unit` with `ipc`, `throughput_gops` (per workload × operating point).

## Metric scope

`extract/__init__.py::METRIC_SCOPE` states which keys a metric varies by. Power varies by
workload and operating point; activity and bits/cycle by workload only; frequency/voltage and timing
by operating point; capacitance/area/count/length by build only. The pipeline uses this to broadcast build-level metrics
across every workload × operating point row, so a new source only needs to declare the scope of
its metrics.

## Adding a source

1. Copy `activity.py`, set `SOURCE`, `DEFAULT_PATTERN`, `OBJECT_KIND`, write `parse()`.
2. Register it in `extract/__init__.py` (`SOURCES`, `SOURCE_METRICS`, `METRIC_SCOPE`).
3. Add the metric to `schema.py` if it should be a column of the wide dataset, and to
   `pipeline.FUB_METRICS` (per-FUB) or `PERF_METRICS` (design-level).
4. Have `mockdata.py` emit a representative file so the pipeline test covers it.
