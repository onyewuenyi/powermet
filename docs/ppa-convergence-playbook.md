# PPA convergence playbook, power first

How a program drives a design from its first RTL power estimate to a signed-off number that meets
its target, with performance (timing) and area as the constraints power is traded against. The
playbook is tool-agnostic and built from public methodology; the commands are powermet's, and every
step names the check that says it is done. Read `docs/methodology.md` for the definitions and
`docs/power-techniques.md` for the levers.

## Principles

1. **Two targets, one corner.** Sign up to Cdyn (`cdyn_pf`, dynamic power with V²f divided out) and
   leakage (`be_leakage_mw`) at a stated workload and operating point. Total power is the product
   number and is reported too, but it is the sum of the two at one corner, and a total lets one
   component hide the other.
2. **Design moves, corner does not.** A corner change (DVFS, a lower Vmin) changes the product
   operating point; it never counts toward a fixed-corner Cdyn or leakage target. `converge --plan`
   refuses it by construction.
3. **Signoff-consistent numbers only.** FE estimates are used to steer; BE signoff is the reference.
   An FE number enters a status report only with its correlation to BE from the last build attached.
4. **Whole builds, never rows.** Every model and every error band is validated by holding out entire
   builds; a random-row split flatters everything.
5. **Coverage before status.** A budget scope missing FUBs is INCOMPLETE, not on track. Unmapped
   objects, stale builds and vectorless numbers are excluded by name, never silently.
6. **Association, not cause.** The pipeline says which physical feature moves with the error; a
   what-if plus the next build says whether a change worked.

## Milestones: what exists, what is checked, what "done" means

| Milestone | Tolerance | What is available | Power questions to close | Exit check |
|---|---|---|---|---|
| **rtl** | 25% | RTL power (logical), vectorless or early SAIF, area from synthesis trials | Is the architecture inside the budget with headroom? Which units dominate Cdyn? Is the workload set right? | `budget check` ON TRACK on FE physical for every design scope; hotspot list agreed with unit owners |
| **synthesis** | 15% | RTL physical-aware power, clock-gating efficiency, first FUB map, gate-level SAIF for early blocks | Clock-gating and operand-isolation coverage; leakage split by Vt mix; FE-to-FE (logical vs physical) drift | `techniques assess` clock gating candidates assigned; `qualify --a fe_logical_mw --b fe_physical_mw` reviewed |
| **placement** | 10% | First BE power (vectorless and SAIF), StarRC wire cap, PrimeTime per partition, UPF | First FE→BE correlation; wire-cap fraction as the error driver; partitions over budget; intent vs corners | `sanitize` usable ≥ 90%; `analyze correlation` FE physical MAPE inside the milestone tolerance; `intent show` no voltage mismatch |
| **route** | 5% | Routed BE power per build, full timing, glitch-aware reports if available | Build-to-build deltas; power-for-performance trades; leakage recovery on slack; convergence trend per target | `converge` CONVERGING or CONVERGED on every target; `analyze frontier` latest build on the Pareto front |
| **signoff** | 0% | Final PrimePower / Voltus at all corners, activity from the release workloads | Does the signed-off number meet the target at the stated corner? Are the two engines within tolerance? | `converge` CONVERGED; `qualify --a be_mw --b be_voltus_mw` PASS; report archived with provenance |

Tolerances are `budgets.toml [defaults].tolerance_pct`; set them per program, not per build.

## The per-build loop

Run once per build drop. Each step feeds the next; the whole loop is one afternoon on a fresh build.

```
powermet ingest scan <runs>                # or: ingest plan | scheduler ; ingest merge
powermet sanitize                          # usable %, per-metric trust, what was excluded and why
powermet budget check --history            # status per scope with tolerance, trend, coverage
powermet converge --plan --top 5           # gap per target, trend, projection, ranked closure plan
powermet analyze deltas --design <d>       # what moved since the last build, classified
powermet analyze frontier --design <d>     # is this build on the power x Fmax front
powermet analyze anomalies --strict        # comparative analysis: power bugs by owner; new / persisting / cleared
powermet analyze hotspots                  # where to spend effort: share, density, growth, CG efficiency
powermet analyze profile                   # peak window (thermal / IR vector), peak/avg, energy per workload
powermet techniques assess --design <d>    # candidates per lever with stated assumptions
powermet model predict ... --scale ...     # what-if the top action, with the CV error band
powermet report                            # one Markdown report with every table above
```

**1. Trust the data before reading it.** `sanitize` first. A usable fraction that dropped, a metric
whose correlation flipped sign, or a scope that went INCOMPLETE explains more status changes than the
design does. Fix the map or the report path before anyone reacts to a number.

**1b. Fix wrong numbers before optimizing large ones.** `analyze anomalies` compares each FUB with what should
agree with it (idle vs busy, peers at the same activity and capacitance, its own previous builds, its replicas,
its clock-network share) and names the owner. A block whose idle power is 75% of its busy power is a gating
bug to drive to its RTL owner, not a hotspot to place better; a regression that activity and capacitance do
not explain is a flow question before it is a design one. A finding that clears in the next build is the
fix landing; one that persists goes on the milestone review with the owner's name on it.

**2. Read status against the milestone, verdict against the trend.** `converge` prints both. Inside
tolerance but DIVERGING is an action item now; over tolerance but CONVERGING at a rate that lands before
the next milestone is a watch item. The projection column says which.

**3. Triage a gap in this order.**

| Question | Evidence | If yes |
|---|---|---|
| Is it a data artefact? | `sanitize`: coverage, stale builds, vectorless flag, run-id mismatches | fix the input; no design action |
| Which component? | `converge`: Cdyn gap vs leakage gap | dynamic levers or leakage levers, never mixed |
| Which scope? | `budget check` per partition / FUB; `analyze hotspots` | the owner of the top three FUBs by share × growth |
| Which lever? | `converge --plan`: techniques ranked by gap coverage in the target's unit | assign the first one that covers the gap; the rest are backup |
| Is it timing-feasible? | `analyze frontier`; `explore` marks VIOLATES beyond Fmax | if not, the lever moves to the trade table below |
| Does any lever reach it? | the plan's remainder line | escalate: architecture change, renegotiated target, or missing data |

**4. Predict before you commit.** Every assigned action gets a `model predict` or `explore scenario`
line with the 90% interval. A predicted saving inside the interval is noise; say so and pick a bigger
lever or wait for the build.

**5. Confirm on the next build.** `analyze deltas` classifies what actually happened: Pareto
improvement, regression, or a trade. A predicted saving that did not appear is a correlation problem
first (`analyze errors`) and a design problem second.

## Trading power against performance and area

Every lever has a cost on the other two axes. Decide with the frontier, not with a single number.

| Lever | Power component | Performance cost | Area cost | Confirm with |
|---|---|---|---|---|
| Clock gating (ICG insertion, enable coverage) | Cdyn | enable-path timing, clock-tree balance | ICG cells | `techniques assess` → next build's `cg_efficiency` |
| Operand isolation / data gating | Cdyn | gating logic on the datapath | small | per-net toggles (needs SAIF at gate level) |
| Glitch reduction (balancing, retiming) | Cdyn | may lengthen paths | none to small | glitch-aware power report |
| Wire-cap reduction (placement, routing, buffering) | Cdyn | timing on long nets, congestion | density | `wire_cap_fraction` in `analyze errors`; `model predict --scale wire_cap_pf` |
| Multi-Vt / downsizing on slack | leakage | consumes slack, may move the critical path | none | PrimeTime WNS per partition after the ECO |
| Power gating / memory sleep | leakage (and idle Cdyn) | wake latency, isolation and retention logic | switches, always-on | `intent show` domains; UPF verification tool |
| Fmax reduction at fixed V | total, not Cdyn | direct | none | not a convergence lever; a product decision |
| DVFS / AVS / lower Vmin | total, not Cdyn | throughput per `explore sweep` | regulators | energy per op; timing feasibility |

Rules of thumb the frontier enforces: a build that raises both power and Fmax is a *trade*, judged by
whether its exchange rate beats earlier builds; a build that raises power without Fmax is a regression
regardless of what else improved; a lever that closes power but breaks timing is not closed.

## Correlation gates: when an FE number may be quoted

| Gate | Threshold (set per program) | Command |
|---|---|---|
| FE physical → BE MAPE inside the current milestone tolerance | e.g. ≤ 15% at synthesis, ≤ 10% at placement | `analyze correlation` |
| Leave-one-build-out interval narrower than the gap being discussed | interval width < gap | `model train --cv` |
| Metric trust `trusted` for every feature in the model | no `unstable` / `unusable` | `sanitize` (metric_quality) |
| Second engine or new tool version within tolerance of the reference | MAPE ≤ 5%, P95 ≤ 10% | `qualify` |
| Activity is vector-based for every quoted BE number | no `vectorless_power` flags in scope | `sanitize` |

A version bump of any engine reopens the qualify gate; the catalog records the version per file, so
the diff is a query, not a memory.

## Deliverables by cadence

- **Per build:** the convergence table (target, actual, gap, trend, verdict), the closure plan, the
  hotspot list with owners, the deltas classification. All from `powermet report`.
- **Per milestone:** budget status per scope with coverage; the correlation report (FE→BE MAPE, R²,
  error drivers, drift across builds); the model table with CV intervals; qualification results;
  intent coverage.
- **On demand:** what-if answers with error bands, sweeps and scenarios with timing feasibility, the
  compact model export for the performance team, trace energy per phase.

## Anti-patterns the pipeline refuses, and why

- Comparing power across corners without dividing out V²f: a lower Vmin looks like a design win.
- Mixing vectorless and SAIF-based numbers in one correlation: the bias lands in the model.
- Pairing FE and BE reports from different signoff runs: run-id mismatches are rejected at ingest.
- Random-row cross-validation: leaks the build into both sides; the interval becomes meaningless.
- A total-power budget with no component split: leakage growth hides behind a clock-gating win.
- A budget with no coverage column: two renamed instances read as a 3% saving.
- Counting the same registers for clock gating and operand isolation: the plan reports cumulative
  as an upper bound and says so.

## Where the tools fit

The flow steers with RTL power (PrimePower RTL / Joules / PowerArtist), converges on implementation
QoR (Fusion Compiler / Innovus, including their in-design power optimization), and signs off on
PrimePower / Voltus with vector-based activity. AI flow optimizers (DSO.ai, Fusion Compiler adaptive
flows, Cerebrus) are one more build each, with their own run id, judged by `deltas` and `frontier`
like any other. `docs/tool-landscape.md` has the current options per stage.
