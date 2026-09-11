"""SAIF adapter: switching activity per hierarchy instance, per workload.

Source of truth for activity is the RTL simulation FSDB (logical hierarchy) per workload. An
FSDB -> SAIF flow (Verdi) takes the workload FSDB, the core, the FE/BE mapping data and the
partition list (back-end physical hierarchy) and writes a SAIF in the physical hierarchy; that
same SAIF drives SAIF-based power optimization in early Fusion Compiler (or the equivalent
Cadence flow). This adapter reads that SAIF, so activity lands on BE hierarchy objects by
default (`activity_hierarchy = "fe"` in metadata.json switches to the RTL-hierarchy SAIF).

Format (SAIF 2.0, backward direction):

    (SAIFILE
    (SAIFVERSION "2.0") (DIRECTION "backward") (DESIGN "gpu_a_top") (DATE "...")
    (VENDOR "Synopsys") (PROGRAM_NAME "Verdi") (VERSION "V-2024.09")
    (DIVIDER / ) (TIMESCALE 1 ps) (DURATION 2000000)
    (INSTANCE gpu_a_top
      (INSTANCE u_scheduler
        (NET
          (data_in\\[0\\] (T0 1200000) (T1 800000) (TX 0) (TC 1234) (IG 0))
          ...
        )
        (INSTANCE u_sub ...)
      )
    ))

Per instance (own nets plus descendants):
    activity        = mean over nets of TC / cycles         toggles per cycle per net (activity factor)
    bits_per_cycle  = sum over nets of TC / cycles          bits switched per cycle (data-movement traffic proxy)
    net_count       = number of nets
with cycles = DURATION * TIMESCALE / sim_clock_period_ps. The simulation clock period comes from
metadata.json (`activity_flow.sim_clock_period_ps`), else the nominal operating point's frequency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from powermet.extract.base import (BE_HIER, FE_HIER, Located, ParsedReport, ParseError, SourceInputs, locate, make_records,
                                   record)

SOURCE = "saif"
DEFAULT_PATTERN = "activity/{workload}.saif"
OBJECT_KIND = BE_HIER          # default; per-file kind decided from context (activity_hierarchy)
TIMESCALE_TO_PS = {"fs": 1e-3, "ps": 1.0, "ns": 1e3, "us": 1e6, "ms": 1e9, "s": 1e12}

_TOKEN = re.compile(r"\(|\)|\"[^\"]*\"|[^\s()]+")
_HEADER_KEYS = ("SAIFVERSION", "DIRECTION", "DESIGN", "DATE", "VENDOR", "PROGRAM_NAME", "VERSION", "DIVIDER", "TIMESCALE", "DURATION")


def get_files(inputs: SourceInputs) -> list[Located]:
    return locate(inputs, SOURCE, DEFAULT_PATTERN)


@dataclass
class _Inst:
    name: str
    path: str
    tc_sum: float = 0.0
    nets: int = 0
    children: list["_Inst"] = field(default_factory=list)

    def aggregate(self) -> tuple[float, int]:
        tc, n = self.tc_sum, self.nets
        for c in self.children:
            ctc, cn = c.aggregate()
            tc += ctc
            n += cn
        return tc, n


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def parse_saif(text: str) -> tuple[dict, list[_Inst]]:
    """Parse SAIF text into (header, top-level instances). Only TC per net is retained."""
    toks = _tokens(text)
    pos = 0
    header: dict = {}
    roots: list[_Inst] = []

    def expect(tok: str):
        nonlocal pos
        if pos >= len(toks) or toks[pos] != tok:
            raise ParseError(f"SAIF: expected '{tok}' at token {pos}, got {toks[pos] if pos < len(toks) else 'EOF'}")
        pos += 1

    def skip_group():
        """Skip the remainder of a parenthesised group (current pos is just after its opening name)."""
        nonlocal pos
        depth = 1
        while pos < len(toks) and depth:
            if toks[pos] == "(":
                depth += 1
            elif toks[pos] == ")":
                depth -= 1
            pos += 1

    def parse_net_block(inst: _Inst):
        nonlocal pos
        # at '(' NET already consumed; entries: '(' name attrs... ')'
        while pos < len(toks) and toks[pos] != ")":
            expect("(")
            pos += 1                       # net name
            tc = 0.0
            while pos < len(toks) and toks[pos] != ")":
                expect("(")
                key = toks[pos]
                val = toks[pos + 1] if pos + 1 < len(toks) else "0"
                if key == "TC":
                    tc = float(val)
                pos += 2
                expect(")")
            expect(")")
            inst.tc_sum += tc
            inst.nets += 1
        expect(")")

    def parse_instance(parent_path: str, divider: str) -> _Inst:
        nonlocal pos
        name = toks[pos]
        pos += 1
        path = f"{parent_path}{divider}{name}" if parent_path else name
        inst = _Inst(name, path)
        while pos < len(toks) and toks[pos] != ")":
            expect("(")
            kw = toks[pos]
            pos += 1
            if kw == "INSTANCE":
                inst.children.append(parse_instance(path, divider))
            elif kw == "NET":
                parse_net_block(inst)
            else:
                skip_group()
        expect(")")
        return inst

    expect("(")
    expect("SAIFILE")
    while pos < len(toks) and toks[pos] != ")":
        expect("(")
        kw = toks[pos]
        pos += 1
        if kw in _HEADER_KEYS:
            vals = []
            while pos < len(toks) and toks[pos] != ")":
                vals.append(toks[pos].strip('"'))
                pos += 1
            header[kw] = " ".join(vals)
            expect(")")
        elif kw == "INSTANCE":
            roots.append(parse_instance("", header.get("DIVIDER", "/").strip() or "/"))
        else:
            skip_group()
    return header, roots


def _cycles(header: dict, sim_clock_period_ps: float | None) -> float:
    ts = header.get("TIMESCALE", "1 ps").split()
    scale = float(ts[0]) * TIMESCALE_TO_PS.get(ts[1] if len(ts) > 1 else "ps", 1.0)
    duration_ps = float(header.get("DURATION", "0")) * scale
    if duration_ps <= 0:
        raise ParseError("SAIF: missing or zero DURATION")
    if not sim_clock_period_ps or sim_clock_period_ps <= 0:
        raise ParseError("SAIF: simulation clock period unknown; set activity_flow.sim_clock_period_ps in metadata.json")
    return duration_ps / sim_clock_period_ps


def parse(path: Path, **context) -> ParsedReport:
    text = Path(path).read_text(errors="replace")
    header, roots = parse_saif(text)
    kind = FE_HIER if str(context.get("activity_hierarchy", "be")).lower().startswith("fe") else BE_HIER
    cycles = _cycles(header, context.get("sim_clock_period_ps"))
    rows = []

    def emit(inst: _Inst, depth: int):
        tc, n = inst.aggregate()
        if n and depth > 0:              # skip the top-level design instance (not a FUB object)
            rows.append(record(inst.path, kind, "activity", tc / n / cycles, "ratio"))
            rows.append(record(inst.path, kind, "bits_per_cycle", tc / cycles, "bits"))
            rows.append(record(inst.path, kind, "net_count", float(n), "count"))
        for c in inst.children:
            emit(c, depth + 1)

    for r in roots:
        emit(r, 0)
    notes = [f"hierarchy={'FE' if kind == FE_HIER else 'BE'}", f"cycles={cycles:.0f}"]
    flow = context.get("activity_flow") or {}
    if flow:
        notes.append("flow inputs: " + ", ".join(f"{k}={v}" for k, v in flow.items() if k not in ("sim_clock_period_ps",)))
    return ParsedReport(
        source=SOURCE, path=Path(path), tool=header.get("PROGRAM_NAME", "saif"), tool_version=header.get("VERSION", "?"),
        records=make_records(rows), report_date=header.get("DATE"), workload=context.get("workload"), notes=notes,
    )
