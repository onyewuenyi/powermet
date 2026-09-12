# Power-optimisation techniques: what they solve, why, the trade-off, and how powermet assesses them

`powermet techniques list` prints the catalogue; `powermet techniques assess` ranks candidate FUBs in the
dataset and gives an order-of-magnitude saving under stated assumptions. Nothing here is a measurement:
the point is to decide what to try first and to say what data would make the estimate better.

| Technique | Stage | Moves | Problem it solves | Mechanism | Trade-off | Considerations | Assessed from |
|---|---|---|---|---|---|---|---|
| Clock gating | RTL | dynamic → CdynTot | clock network and idle registers toggle every cycle | ICG stops the clock when the enable is false | ICG area, enable-path timing, clock-tree balance; nothing gained where enables are always true | high dynamic power with low `cg_efficiency`; verify with a workload that idles the block | `be_mw`, `cg_efficiency` (PPRTL) |
| Power gating | physical | leakage → LkgPwr | leakage and idle clocking in blocks that are off for long stretches | switched supply per UPF domain; retention or restore on wake | wake latency, rush current, isolation on every boundary, always-on logic, UPF verification | long idle intervals, clean domain boundary, an owner for the on/off policy | idle-workload `be_mw`, duty cycle |
| DVFS / AVS | runtime | corner (V, f); CdynTot unchanged | turbo corner used when the workload does not need it | P ∝ V²f while performance ∝ f | regulators, closure at every corner, transition latency, control stability | energy per op, not raw power; memory-bound workloads benefit most; corner must be timing-feasible | `be_mw`, `voltage_v`, `frequency_ghz`, `explore opmap` |
| Wire-cap reduction | physical | dynamic → CdynTot | wire-dominated blocks charge interconnect, which FE estimates miss | tighter placement, shorter nets, fewer buffers, better layers | density, congestion, timing on long nets | rank by wire-cap fraction; validate with a what-if and the next build | `wire_cap_pf`, `cell_cap_pf` |
| Multi-Vt / downsizing | physical | low-Vt high-drive cells leak on paths with slack | swap non-critical cells to higher Vt or smaller drive | consumes slack, may move the critical path; bounded by leakage share | only in partitions with positive slack; needs a leakage split | `wns_ps`, leakage from the decomposition |
| Operand isolation | RTL | dynamic → CdynTot | datapaths switch while their results are discarded | gate operands when the output is unused | gating logic on wide buses, risk of gating a live path | needs per-net toggles with a result-used signal | not assessable from block-level SAIF |
| Glitch reduction | synthesis | dynamic → CdynTot | spurious transitions from unequal arrival times, 5-20% of dynamic power in some datapaths | balance delays, restructure, selective gating | area and effort; needs a glitch-aware run to see it | add a glitch-mode PrimePower report as a source | not assessable without it |
| Memory sleep / banking | architecture | leakage → LkgPwr | SRAM leakage and periphery clocking with idle arrays | light/deep sleep, shutdown, finer banks | wake latency per mode, control logic, bandwidth loss | needs memory instances per FUB and macro mode power | not assessable without it |

## Convergence: which technique answers which target

Each technique declares the convergence component it moves (`Technique.reduces`): **dynamic** techniques lower
CdynTot, **leakage** techniques lower LkgPwr, and a **corner** change (DVFS) lowers power at a different V/f
without touching Cdyn at all. `powermet converge --plan` uses this to rank techniques against an open gap in
the target's own unit (dynamic mW convert to pF by dividing out V²f at the target's corner) and to refuse to
count a corner change against a target stated at a fixed corner. Power gating removes idle dynamic power as
well, but only its leakage share counts toward convergence: gating a block during idle phases does not lower
CdynTot at the target workload. The leakage each assessment uses comes from the PrimePower leakage column
when present, else the data-movement decomposition, else an assumed 15%; the assumptions list says which.

## Reading an assessment

- **Candidates** are ranked by estimated saving; the columns shown are the evidence (dynamic power, gating
  efficiency, idle power, slack, wire-cap fraction).
- **Assumptions** are printed with every technique: register share of dynamic power, achievable gating
  ceiling, idle duty cycle, assumed wire-cap cut, Vt leakage reduction. Change them with the CLI flags
  where they exist; the estimate moves linearly.
- **Dynamic / leakage split** comes from the data-movement decomposition when a model has been trained,
  else a fixed 85/15 split is used and stated.
- A technique that reports "needs: ..." names the source that would enable it; adding that source is one
  adapter plus a metric-scope entry.

## Why this shape

Every technique above appears in most flows regardless of vendor. What differs per company is which ones
are already applied, which have an owner, and what data exists to judge them. Keeping the catalogue as a
registry (`techniques.TECHNIQUES`) means a new technique, or a company-specific variant of one, is one entry
with its own assessment function, and the documentation is generated from the same object the CLI runs.
