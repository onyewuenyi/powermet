# Tool landscape for the methodology (reference, to be confirmed per company)

Status as of September 2026, from vendor material and public announcements. Everything here must be
confirmed at the target company on three axes before it enters the plan: **licensed and installed**,
**version in production use**, and **ease of getting output into this pipeline** (a report an adapter
can read beats an API that needs a new integration). Ratings are the author's reading, not vendor claims.

## Where each tool plugs in

| Methodology stage | powermet seam | Candidate tools (confirm at company) | Production ease |
|---|---|---|---|
| RTL power (FE logical / physical-aware) | `extract/pprtl.py` | Synopsys PrimePower RTL; Siemens PowerPro; Keysight PowerArtist (acquired from Ansys, Oct 2025); Cadence Joules RTL Power / Joules RTL Design Studio | high: all print hierarchical power reports |
| BE signoff power | `extract/primepower.py`, `extract/voltus.py` | Synopsys PrimePower; Cadence Voltus (and Voltus-XFi) | high; keep both engines as separate columns and `qualify` them |
| Timing | `extract/primetime.py` | Synopsys PrimeTime; Cadence Tempus | high (partition summaries); endpoint-level is a larger adapter |
| Parasitics | `extract/starrc.py` | Synopsys StarRC; Cadence Quantus | high |
| Implementation QoR, wire length | `extract/implementation.py` | Synopsys Fusion Compiler; Cadence Innovus / Genus | high |
| Activity | `extract/saif.py` | Synopsys Verdi (FSDB → SAIF); Cadence Xcelium / Simvision SAIF dump; VCS | high: SAIF is a standard |
| Power intent | `intent.py` | UPF 3.x from the design team; Synopsys VC LP and Cadence Conformal Low Power for the full static checks (powermet only checks structure and voltages) | medium: use the signoff tool for isolation / retention rules |
| Power by cell group (clock / register / comb / memory) | `extract/power_groups.py` | PrimePower `report_power -hierarchy -groups`; Voltus detailed hierarchical report; RTL tools print the same categories | high: one more report from the same run; enables the clock-dominant rule |
| Time-based power / emulation power | `extract/power_profile.py` | PrimePower time-based mode (waveform interval); Cadence Palladium DPA; Synopsys ZeBu power; Siemens Veloce power | medium: the time-based run must use the same netlist and activity window as the averaged run (the reconciliation check catches it); emulation is the only route for long AI / graphics workloads |
| IR drop / power integrity | none yet (candidate new source; the profile's peak window and max step are the inputs it needs) | Ansys RedHawk-SC (now Synopsys); Cadence Voltus-XFi / Celsius | medium: per-instance IR data is large; start with partition summaries |
| Design-space / AI optimization | `explore.py` produces the scenarios; results would be a new source | Synopsys DSO.ai and the Synopsys.ai Copilot assistants (Knowledge, Workflow for PrimeTime and Fusion Compiler, 2026); Cadence Cerebrus Intelligent Chip Explorer; Cadence JedAI (Joint Enterprise Data and AI) for cross-run data | low to medium: enterprise deployment, licensing and data governance decide this, not the tool |
| Analytics / data | storage, catalog, modeling | DuckDB + Parquet (in use), Polars (faster frames at scale), PostgreSQL (shared catalog), MLflow or DVC (model and dataset versioning), Dask or Ray (fan-out beyond a scheduler) | high for DuckDB/Parquet/PostgreSQL; medium for MLflow/DVC (needs a server or shared store) |
| Open-source flow for practice | mock generator today | OpenROAD / OpenSTA / Yosys, OpenROAD-flow-scripts: real reports without licenses | high for learning; not a substitute for signoff engines |

## Notes that matter for planning

- **PowerArtist changed hands.** Ansys sold PowerArtist to Keysight (closed October 2025) as a condition of
  the Synopsys–Ansys merger; RedHawk-SC stayed with Ansys and is now under Synopsys. At a company that
  standardised on PowerArtist the RTL power adapter targets a Keysight product going forward.
- **Cadence Joules RTL Design Studio** integrates with Cerebrus for design-space exploration and with JedAI
  for cross-version trend analysis, which overlaps with what `deltas` / `frontier` / `explore` do here. If a
  company already has JedAI, powermet's catalog is the lightweight substitute, not a competitor.
- **Synopsys.ai Copilots** (2026) are assistants inside PrimeTime and Fusion Compiler. They do not replace a
  methodology pipeline; they speed up using the tools that feed it.
- **Vectorless vs vector-based.** Every engine above can run vectorless; the `Activity:` header convention and
  the `vectorless_power` flag exist so that mode never gets mixed into a correlation silently.
- **Version discipline.** Each adapter declares `SUPPORTED_VERSIONS`; the catalog stores the version of every
  parsed file. A new engine release is a version diff first and a parser change second.

## The three tools to build hands-on depth in (public state, September 2026)

What each vendor has said publicly in the last 18 months, and what it means for this pipeline. Talk
decks from SNUG and CadenceLIVE sit behind SolvNetPlus / Cadence Support logins; the public record is
press releases, vendor blogs, datasheets and trade-press coverage.

**Synopsys Fusion Compiler** (RTL-to-GDSII; the `implementation` adapter's source)
- Positioning: one RTL-to-GDSII tool with shared engines and data model, signoff timing / extraction /
  power built in; 500+ customer tapeouts cited on the product page.
- Named features to know: PPA(V) (performance-per-watt optimization with variable operating voltage),
  Adaptive Scenario Compression, Parametric Scenario Extension, ML-based macro placement, backside
  routing support, native DSO.ai integration.
- 2026 "adaptive flows": the flow re-sequences itself (alternative heuristics, selective steps,
  step re-ordering, recovery) with a factory-pretrained model; vendor claims up to 7% power and 2%
  area improvement, 2-3x faster time-to-PPA-target, 5-10x less compute than general DSO.ai search.
- Synopsys.ai Copilot Workflow Assistant is available inside Fusion Compiler and PrimeTime
  (script generation; vendor claims up to 60% faster, 10-20x on PrimeTime scripting tasks).
  AgentEngineer is the L1-L5 autonomy framework; an L4 multi-agent design-and-verification
  workflow was demonstrated at Synopsys Converge (March 2026).
- June 2026 (SNUG India): first "Multiphysics Fusion" releases post-Ansys — PrimeTime linked with
  RedHawk-SC / RedHawk-SC Electrothermal / StarRC / HFSS-IC for power-, thermal- and stress-aware
  timing signoff; PrimeClosure linked with RedHawk-SC for power-integrity-aware closure.
- For powermet: the QoR report stays the seam; the pipeline should treat an adaptive-flow or DSO.ai
  run as one more build with its own `run_id`, so `deltas` / `frontier` measure what the flow
  bought. Any FC-side "power" number is an in-design estimate, not the `be_mw` reference, and
  belongs in `qualify` against PrimePower.

**Synopsys PrimePower / PrimePower RTL** (the `primepower` and `pprtl` adapters' sources)
- PrimePower RTL uses the RTL Architect predictive engine plus PrimeTime STA and PrimePower gate
  engines, so RTL estimates are physically aware and "signoff-consistent" by construction. Reports:
  average, peak, glitch, clock, dynamic, leakage and multi-voltage power; vector-based (simulation
  or emulation FSDB/SAIF) and vectorless.
- Glitch power is the current emphasis for AI/ML datapaths: PrimePower RTL points to the RTL line
  generating the most glitch; the gate-level engine closes the loop.
- "PPRTL" is industry shorthand (it appears in public job postings) for PrimePower RTL; Synopsys
  marketing says PrimePower RTL.
- For powermet: FE-vs-BE correlation on `fe_physical_mw` is exactly the claim PrimePower RTL makes;
  the sanitize trust table and `qualify` are how a team checks that claim per build rather than
  assuming it. Glitch power is a candidate new column (`fe_glitch_mw`) if reports carry it.

**Cadence Innovus (Innovus+) / Cerebrus / Voltus / Joules** (the `voltus` adapter; Innovus is the
`implementation` alternative)
- Innovus+ bundles Genus synthesis with Innovus and integrates Tempus (timing), Quantus
  (extraction), Voltus (power integrity) and Pegasus (physical verification) for signoff-accurate
  in-design analysis; power-integrity-aware placement, optimization, CTS and routing address IR/EM
  during implementation. Certified for TSMC N2 and A16 (backside "Super Power Rail"), with NanoFlex
  Pro standard-cell DTCO tuning in the April 2026 TSMC announcement.
- Cerebrus Intelligent Chip Explorer (RL flow tuning) and, since 2025, Cerebrus AI Studio: agentic,
  multi-block, multi-user hierarchical SoC optimization (vendor claims 5-10x faster delivery).
  ChipStack (acquired Nov 2025) is the "AI Super Agent" for design and verification; at
  CadenceLIVE Silicon Valley (April 2026) it was paired with Google Gemini on Google Cloud.
- Joules RTL Design Studio (2023-) shares Genus/Innovus/Joules engines for RTL-stage PPA and
  congestion visibility, integrates with Cerebrus for design-space exploration and JedAI for
  cross-run trends. Voltus covers RTL power, glitch estimation and gate-level power signoff.
- For powermet: a Cadence shop swaps three adapters (Innovus QoR, Voltus hierarchy power, Tempus
  partition timing); the identity, sanitize, modeling and convergence layers do not change. Keep
  Voltus and PrimePower as separate columns and let `qualify` say which is the reference.

**Open-source practice ground.** OpenROAD / OpenROAD-flow-scripts (Yosys + OpenSTA + OpenROAD) give
a full RTL-to-GDS flow with timing and power reports and no licence; OpenSTA power is reported to
correlate reasonably with commercial tools. It is the fastest way to generate real (non-mock) reports
to write adapters against, and to practise the vocabulary (SAIF, UPF, SPEF, QoR) the commercial
tools share.

Sources (public): Synopsys Fusion Compiler product page and "AI-Driven Chip Design: Dynamic, Adaptive
Flows" blog; Synopsys.ai Copilot blog and datasheet; Synopsys AgentEngineer blog (May 2025) and
"Vision for Engineering the Future" press release (March 2026); EE Times, "SNUG India 2026: Synopsys
unveils first Multiphysics Fusion tools since Ansys deal" (June 2026); Synopsys PrimePower product
page and datasheet; Cadence "Collaborates with TSMC" press release (April 2026); Cadence Cerebrus AI
Studio and Innovus product pages; Cadence Joules RTL Design Studio announcement; The Next Platform on
CadenceLIVE 2026; The OpenROAD Project.

## Confirmation checklist for a new company

1. Which of the tools above are licensed, which version is in production, and who owns the flow.
2. Where each tool's reports land (path convention, retention, permissions) and whether they are
   already hierarchical per partition / block.
3. Whether the activity flow produces SAIF in the physical hierarchy or only RTL FSDB.
4. Whether UPF is authoritative for voltages or the operating points live elsewhere.
5. Whether a shared data platform (JedAI, an internal run database, PostgreSQL) already exists to
   replace the SQLite catalog.
6. Whether any AI optimization tool is in production; if so, its runs are one more source to ingest
   and qualify, not a reason to change the methodology.

Sources consulted (September 2026): Keysight and Ansys press releases on the PowerArtist sale; Synopsys
PrimePower product pages; Cadence Joules RTL Design Studio datasheet and announcements; Synopsys blog on
Synopsys.ai Copilot assistants.
