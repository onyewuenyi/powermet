"""Power intent (UPF): which power domain and supply each FUB lives in, and whether that agrees
with the operating points the analysis uses.

Reads the subset of UPF that carries structure:
    create_power_domain PD_CORE -elements {top/part_pcore0 top/part_pcore1}
    create_supply_port VDD_CORE
    create_supply_net  VDD_CORE
    set_domain_supply_net PD_CORE -primary_power_net VDD_CORE -primary_ground_net VSS
    add_port_state VDD_CORE -state {nom 0.80} -state {turbo 0.88} -state {off off}

Elements are hierarchy prefixes: every FUB whose BE (or FE) path starts with an element belongs to
that domain. Consistency checks:
    no_power_domain            FUB not covered by any domain
    multiple_power_domains     FUB covered by more than one domain (ambiguous intent)
    domain_voltage_mismatch    domain's state voltage differs from the operating point voltage in metadata
    domain_state_missing       an operating point has no matching UPF state
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from powermet.identity import ModelRoot

DEFAULT_PATTERN = "intent/*.upf"
_CMD = re.compile(r"^\s*(create_power_domain|set_domain_supply_net|add_port_state|create_supply_net|create_supply_port)\s+(\S+)(.*)$")
_ELEMENTS = re.compile(r"-elements\s+\{([^}]*)\}")
_PRIMARY = re.compile(r"-primary_power_net\s+(\S+)")
_STATE = re.compile(r"-state\s+\{\s*(\S+)\s+(\S+)\s*\}")


@dataclass
class PowerDomain:
    name: str
    elements: list[str] = field(default_factory=list)
    supply_net: str | None = None
    states: dict[str, float] = field(default_factory=dict)     # state name -> voltage (V)


@dataclass
class PowerIntent:
    domains: dict[str, PowerDomain]
    port_states: dict[str, dict[str, float]]
    source: str | None = None

    def domain_of(self, path: str) -> list[str]:
        hits = []
        for d in self.domains.values():
            for el in d.elements:
                if path == el or path.startswith(el.rstrip("/") + "/"):
                    hits.append(d.name)
                    break
        return hits


def _join_continuations(text: str) -> list[str]:
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        if cur.strip():
            out.append(cur)
        cur = ""
    if cur.strip():
        out.append(cur)
    return out


def parse_upf(text: str, source: str | None = None) -> PowerIntent:
    domains: dict[str, PowerDomain] = {}
    port_states: dict[str, dict[str, float]] = {}
    for line in _join_continuations(text):
        m = _CMD.match(line)
        if not m:
            continue
        cmd, name, rest = m.groups()
        if cmd == "create_power_domain":
            d = domains.setdefault(name, PowerDomain(name))
            em = _ELEMENTS.search(rest)
            if em:
                d.elements.extend(tok for tok in em.group(1).split() if tok)
        elif cmd == "set_domain_supply_net":
            d = domains.setdefault(name, PowerDomain(name))
            pm = _PRIMARY.search(rest)
            if pm:
                d.supply_net = pm.group(1)
        elif cmd == "add_port_state":
            st = port_states.setdefault(name, {})
            for sname, sval in _STATE.findall(rest):
                try:
                    st[sname] = float(sval)
                except ValueError:
                    pass                          # 'off' and friends
    for d in domains.values():
        if d.supply_net and d.supply_net in port_states:
            d.states = dict(port_states[d.supply_net])
    return PowerIntent(domains, port_states, source)


def load_upf(path: str | Path) -> PowerIntent:
    return parse_upf(Path(path).read_text(errors="replace"), source=str(path))


def intent_table(intent: PowerIntent, model: ModelRoot, design: str, build: str,
                 operating_points: dict | None = None) -> pd.DataFrame:
    """One row per FUB: domain, supply, state voltages and consistency issues."""
    ops = operating_points or {}
    rows = []
    for spec in model:
        hits = intent.domain_of(spec.be_hier) or intent.domain_of(spec.fe_hier)
        issues = []
        if not hits:
            issues.append("no_power_domain")
        elif len(hits) > 1:
            issues.append("multiple_power_domains")
        dom = intent.domains.get(hits[0]) if hits else None
        mism = []
        if dom is not None and ops:
            for op, vals in ops.items():
                v_op = vals.get("voltage_v")
                if op not in dom.states:
                    if "domain_state_missing" not in issues:
                        issues.append("domain_state_missing")
                elif v_op is not None and abs(float(dom.states[op]) - float(v_op)) > 0.005:
                    mism.append(f"{op}:{dom.states[op]:.3f}V!={float(v_op):.3f}V")
            if mism:
                issues.append("domain_voltage_mismatch")
        rows.append({
            "design": design, "build": build, "fub": spec.fub, "model_root": spec.model_root, "partition": spec.partition,
            "power_domain": dom.name if dom else None, "supply_net": dom.supply_net if dom else None,
            "domain_states": ";".join(f"{k}={v:.3f}" for k, v in (dom.states.items() if dom else [])),
            "intent_ok": not issues, "intent_issues": ";".join(issues), "voltage_mismatch": ";".join(mism),
        })
    return pd.DataFrame(rows)


def render_intent_summary(table: pd.DataFrame) -> str:
    from powermet.textfmt import table as _t

    if not len(table):
        return "No power intent (UPF) ingested."
    g = table.groupby(["design", "build"])
    rows = []
    for (d, b), grp in g:
        n = len(grp)
        rows.append([d, b, grp["power_domain"].nunique(), n, int(grp["intent_ok"].sum()),
                     int(grp["intent_issues"].str.contains("no_power_domain").sum()),
                     int(grp["intent_issues"].str.contains("domain_voltage_mismatch").sum())])
    return "Power intent (UPF) coverage\n\n" + _t(["Design", "Build", "Domains", "FUBs", "Consistent", "No domain", "V mismatch"], rows)
