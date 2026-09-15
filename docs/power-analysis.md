# Power analysis: comparative analysis, power bugs, ownership, and what the engines must produce

A power *methodology and modeling* role asks how early power can be predicted and how much a change buys.
A power *analysis* role (the fullchip / unit-level analysis half of the same team) asks a different set of
questions every build: which numbers are wrong for what the block is doing, what the bug probably is, who
owns it, and did the fix land. This document covers the pieces that answer those questions, and the EDA
engine outputs they need beyond the hierarchical power report.

## Comparative analysis: `powermet analyze anomalies`

A hotspot is where power is large. An anomaly is where power is wrong relative to something that should
agree with it. `anomalies.py` is a registry of such comparisons (`RULES`); each rule names what it
compares, the class of power bug it points at, the technique that usually addresses it, the columns it
needs, and a threshold in `config.toml` (`anomaly_*`).

| Rule | Compares | Points at | Sev | Technique | Needs |
|---|---|---|---|---|---|
| `idle_dynamic` | dynamic power at the idle workload vs the reference workload, same FUB | clocks or datapaths not gated when the block is idle | high | clock_gating | `idle` workload (`idle_workload` in config) |
| `activity_power_mismatch` | dynamic power per unit activity x capacitance x V^2 f vs design peers (robust z on the log ratio) | glitching, an ungated clock tree, or activity not reaching the power run | high | glitch_reduction | activity, StarRC caps, corner |
| `unexplained_regression` | this build vs the FUB's last good build, net of what activity x capacitance explains | gating lost, a new always-on path, a tool setting changed | high | clock_gating | two builds |
| `creeping_growth` | the last N builds of the same FUB, all rising | incremental ECOs adding power nobody signed up for | medium | - | N+1 builds |
| `clock_dominant` | clock-network share of the FUB's own dynamic power | clock tree feeding idle registers; enable coverage | medium | clock_gating | power-groups report |
| `replica_divergence` | replica instances (FUB@i) of one module against each other | placement, activity mapping, or a per-instance constraint | medium | wire_cap_reduction | `replica_policy = "per_instance"` |
| `leakage_share` | leakage fraction vs the design median | low-Vt mix or missing power gating on a mostly-idle block | medium | vt_swap | leakage split |

Each finding carries the model root, partition, **owner** (from the FUB map), the value, the threshold,
the evidence sentence, and a status: `new` (absent in the previous build) or `persisting`. Findings that
were present in the previous build and are absent now are listed as *cleared*, which is how a fix is
seen to land without anyone updating a tracker. The CLI prints a per-owner summary (who to drive), the
catalog keeps every run in the `anomaly` table, and the report has a section. `--strict` returns a
non-zero exit when a high-severity finding is open, for a gate in a nightly flow.

```
$ powermet analyze anomalies --design GPU_A
Model root              Owner       Rule                      Sev     Value  Status      Evidence
GPU_A.PCORE0.L1I        rtl-pcore0  idle_dynamic              high    0.75   persisting  idle dynamic 19.4 mW is 75% of typical (25.8 mW); activity ratio 0.09
GPU_A.PCORE1.TLB        rtl-pcore1  unexplained_regression    high    31.9%  new         dynamic 25.4 -> 34.3 mW (+34.9%) vs B003; activity x capacitance explains +3.0%
GPU_A.PCORE0.Scheduler  rtl-pcore0  activity_power_mismatch   high    2.03   persisting  dynamic power per activity*C*V^2f is 2.0x the design median (robust z 5.5)
GPU_A.PCORE0.Dispatch   rtl-pcore0  clock_dominant            medium  0.65   persisting  clock network is 65% of dynamic power; clock-gating efficiency 0.34

By owner (who to drive)
Owner       Findings  High  FUBs
rtl-pcore0         5     3     4
rtl-pcore1         1     1     1
```

Rules are associations. The evidence says what disagrees with what; the designer confirms the bug in
RTL (a missing enable, a free-running counter, a debug bus left on) or the flow (SAIF not annotated on
a hierarchy, a netlist swap between runs). Thresholds are tuned once against real magnitudes, and every
rule says why it was skipped when the data it needs is absent, so a silent "no anomalies" cannot mean
"no inputs".

The mock generator plants three bugs so the command has something to find without proprietary data: a
FUB whose clocks keep toggling at idle (75% of its busy dynamic power), a FUB whose dynamic power jumps
25% at a later build with no physical or activity change, and a FUB that burns 2.4x per unit of
activity and capacitance. They sit on the smallest FUBs of each design so totals and models barely move.

### Ownership

`mapping/fub_map.csv` accepts an optional `owner` column (team alias or engineer). It travels with the
model root into every dataset row (`owner`), so anomalies, hotspots and budgets can be cut by owner
without a second table to keep in sync. Where the FUB map is generated from the design database, the
owner comes from the same export that names the RTL hierarchy.

## Power by cell group: `power_groups` source

Turning a hotspot into a bug needs to know *where inside the block* the dynamic power sits. The
optional `power_groups` adapter reads a hierarchical report split by cell group and yields four metrics
per FUB: `be_clock_mw`, `be_register_mw`, `be_comb_mw`, `be_memory_mw`; `clock_fraction` (clock /
dynamic) is derived. Sanitize checks that the four groups reconstruct the BE total within
`group_sum_tol_pct` and flags `group_sum_mismatch` otherwise (the mock's vectorless idle run is such a
case: its groups came from the SAIF run).

**EDA requirement (representative; confirm the exact switches against the installed release):**

| Engine | Report | Notes |
|---|---|---|
| Synopsys PrimePower | `report_power -hierarchy -groups {clock_network register combinational memory io black_box}` with full hierarchy names, same scenario and run as `power_hier.rpt` | one file per workload x operating point; groups are read by column name so extra groups or a different order parse |
| Cadence Voltus | `report_power -hierarchy -format detailed` (category columns) | same shape after column renaming in `parse()` |
| RTL tools (PowerArtist, PrimePower RTL, Joules) | clock / register / logic / memory category columns in the hierarchical report | would map to FE group columns; not yet in the schema |

## Time-based power profile: `power_profile` source and `analyze profile`

The averaged number hides what matters on GPU and AI workloads: the peak window that sets the thermal
and IR vector, the largest step between windows (a di/dt proxy), and the energy of the run. The optional
`power_profile` adapter reads design power per time window for one workload x operating point and the
pipeline keeps it in `power_profile.parquet` (the object is the design over time, not a FUB, so it stays
out of the FUB dataset). `powermet analyze profile` then reports per workload: average, peak window,
peak-to-average, max step, energy, pJ/op when throughput is known, and the reconciliation of the profile
average with the averaged hierarchical report (`anomaly_profile_tol_pct`); a `MISMATCH` means the two
runs used different netlists, parasitics or activity windows.

```
$ powermet analyze profile --design GPU_A
Workload  Op   Average   Peak      Peak/avg  Peak window     Max step mW/ns  Energy      pJ/op  vs report
compute   nom  1,124 mW  1,461 mW  1.30      1,100-1,200 ns  4.0             4,496 uJ    8.7    -3.1%
typical   nom  1,038 mW  1,182 mW  1.14      2,900-3,000 ns  1.5             4,153 uJ    ...
idle      nom    409 mW    421 mW  1.03      1,000-1,100 ns  0.2             1,636 uJ    ...

Peak-power vector: compute/nom at 1,100-1,200 ns (1,461 mW, 1.30x its average); use that window for IR / thermal signoff, the averages for energy.
```

**EDA requirement (representative):**

| Source | What to produce | Notes |
|---|---|---|
| PrimePower time-based | `set_power_analysis_mode time_based`, waveform interval set so the output is a table (start, end, total, dynamic, leakage per window), same netlist and parasitics as the averaged run, FSDB/VCD of the workload | the reconciliation check exists to catch a profile run on an older netlist |
| Emulation (Palladium DPA, ZeBu, Veloce) | per-window design power exported as the same CSV | for workloads too long for gate-level simulation; record the tool in the header so provenance says which |
| Per-partition columns | not read yet; add columns and a partition dimension in `parse()` | partition-level peaks would feed the IR review directly |

## Where this sits in the per-build loop

```
powermet sanitize                         # trust the data
powermet analyze anomalies --strict       # new / persisting / cleared power bugs, by owner
powermet analyze hotspots                 # where power is large
powermet analyze profile                  # peak window and energy per workload
powermet converge --plan                  # is the target still reachable, and by which lever
```

Anomalies come before hotspots because a wrong number should not be optimized; it should be fixed.
