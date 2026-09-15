# Methodology variants: how FE and BE relate at different companies, and what powermet does about it

Every company organises RTL, synthesis and physical design differently, and every difference changes what
"the power of a FUB" means and where a number has to come from. This page catalogues the setups seen in
practice, the concrete problem each creates, the implication for the metrology, and the switch in
powermet that handles it. Profiles bundle the switches: `powermet profiles list`, `powermet init --profile`.

## Identity: how a FUB appears in the back end

| Setup | Seen where | Problem it creates | Implication | powermet |
|---|---|---|---|---|
| **Same hierarchy** (BE keeps the RTL hierarchy) | small ASICs, IP blocks, flows without cross-module ungrouping | none for identity; risk that one optimisation pass ungroups a block silently | FE-to-BE is one-to-one; gate-level SAIF needs no name mapping | `identity.kind = "same_hierarchy"`; map needs only `fe_hier`; profile `same_hierarchy` |
| **Separate hierarchies with a map** (the reference setup) | CPU cores, large SoCs with a physical partitioning step | nothing in the reports says which BE instance is which FUB; the map is the first deliverable | every unmapped object is missing power; report coverage with every total | default `explicit_map`; six-column FUB map exported from the model root; `unmapped.parquet`; budgets marked INCOMPLETE |
| **Renamed / uniquified instances** | any flow: synthesis uniquify (`_0`, `_1`), ECO / physical renames (`_phys`, `_r2`), escaped Verilog names, `.` vs `/` dividers | exact-name matching fails after the map is exported | names must be normalised on both sides before matching | `identity.name_rules` (regex rewrites), `divider`, `strip_top`, `lowercase`, `unescape` |
| **Replicated units** (cores, SMs, PEs, slices) | GPUs, AI accelerators, multi-core CPUs | one RTL module, N BE instances that differ by placement and activity | choose between the unit total (FE × N comparable) and per-instance rows (outlier hunting) | glob BE path in the map (`.../u_sm_*`); `replica_policy = "sum"` or `"per_instance"` (rows become `SM@3`); extensive metrics add, intensive average, timing takes the worst |
| **Merged / flattened blocks** (synthesis ungrouping) | ASIC flows optimising QoR, small FUBs | BE power exists only for the merged block; per-FUB BE power is an apportionment | block totals are exact, per-FUB numbers carry the apportion assumption and should be labelled | several map rows share one `be_hier` with `be_share` (cell-count or area ratio, or a group power report); `merge_basis = "share"` or `"equal"`; lineage `relationship = merged` |
| **Split FUBs** (one FUB across two partitions) | floorplan-driven splits, memory placed apart from its controller | the FUB's power is the sum of two BE objects in two partitions; timing comes from two partitions | sum extensive metrics, take the worse timing, list both partitions | several map rows per FUB with different `be_hier` and `partition`; `relationship = split` |
| **Aggregate rows** (partition and top totals in BE reports) | every hierarchical report | they look like unmapped objects | must not be counted as missing | `ModelRoot.is_ancestor()` classifies them as aggregates |
| **Mixed** | most real designs | all of the above in one design | per-FUB relationship must be visible | `lineage.parquet` carries `relationship` per FUB; mock `--methodology mixed` |

## Activity: where switching data comes from

| Setup | Problem | Implication | powermet |
|---|---|---|---|
| **RTL FSDB → physical-hierarchy SAIF** (a flow with core, mapping, partition list; drives Fusion Compiler SAIF-based optimisation) | none for identity, but the flow's inputs must be recorded | activity is already in BE names | `activity_flow.hierarchy = "be"`, flow inputs as provenance |
| **Gate-level FSDB / VCD** | large files | `read_saif` / `vcd2saif` converts directly; names are BE names when the hierarchy is preserved | same as above; nothing to map |
| **RTL-hierarchy SAIF only** | generate blocks, escaped identifiers, retimed and synthesis-created registers have no BE counterpart; tools use `read_saif -map_names` | activity coverage drops; provenance must say which nets carried data | `activity_flow.hierarchy = "fe"` + `name_rules`; `metric_quality` coverage column; profile `rtl_activity_only` |
| **Vectorless** | fast, biased, noisier | never mix silently into a correlation | `Activity:` header → `be_activity_mode`; sanitize flags `vectorless_power`; `qualify` shows the gap |
| **Multiple windows per workload** (peak vs average intervals) | one SAIF per window | each window is a workload variant | name windows as workloads (`gemm_peak`, `gemm_avg`); traces replay them |

## Timing and physical hierarchy

| Setup | Implication | powermet |
|---|---|---|
| Partition-level timing (physical hierarchy is partition-based) | FUBs inherit their partition's slack; a split FUB inherits the worse | `primetime` adapter, `wns_ps` aggregated with `min` |
| Block-level or endpoint-level timing | finer than partition; needs an endpoint → FUB grouping | a second timing adapter with `object_kind = be_hier`; same pivot |
| Hierarchical signoff (block-level runs with boundary models) | block totals exclude top-level interconnect | an explicit "top" FUB for the glue; watch `is_ancestor` rows |

## How the strategies plug in

- `identity.IdentityStrategy` (kind, replica policy, merge basis, name rules) is read from `config.toml`
  `[identity]` and applied inside `ModelRoot`; nothing else matches names.
- `extract.METRIC_AGG` says how each metric combines across many-to-one objects; `pipeline.aggregate_metric`
  applies weights first (merged blocks) then sums, averages or takes the worst value.
- `lineage.parquet` records `relationship` per FUB so every downstream table can say whether a number is
  measured, summed or apportioned.
- Profiles in `src/powermet/profiles/*.toml` are documented presets; add one per company as its setup is learnt.

The mock generator produces each layout (`powermet mock generate --methodology same_hierarchy|replicated|merged|split|mixed`)
and the tests assert that totals are conserved in every layout and per-FUB values are exact in all but the merged case.
