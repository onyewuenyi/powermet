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
| IR drop / power integrity | none yet (candidate new source) | Ansys RedHawk-SC (now Synopsys); Cadence Voltus-XFi / Celsius | medium: per-instance IR data is large; start with partition summaries |
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
