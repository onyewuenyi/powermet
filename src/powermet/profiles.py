"""Methodology profiles: named, documented presets of strategy configuration for a company / program setup.

A profile is a TOML file under powermet/profiles (or a user path) with a description of the setup, the
problem it creates, the implication for the methodology, and the config values that handle it
(identity strategy, activity hierarchy, notes). `powermet profiles list|show` documents them and
`powermet init --profile <name>` applies one.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from powermet.config import Config

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


@dataclass
class Profile:
    name: str
    title: str
    setup: str
    problem: str
    implication: str
    identity: dict = field(default_factory=dict)
    activity: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    source: str | None = None

    def apply(self, cfg: Config) -> Config:
        cfg.identity = {**cfg.identity, **self.identity}
        cfg.profile = self.name
        return cfg


def load_profile(name_or_path: str) -> Profile:
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILE_DIR / f"{name_or_path}.toml"
    if not p.exists():
        raise FileNotFoundError(f"profile '{name_or_path}' not found (built-ins: {', '.join(list_profiles())})")
    with open(p, "rb") as fh:
        d = tomllib.load(fh)
    return Profile(d["name"], d.get("title", d["name"]), d.get("setup", "").strip(), d.get("problem", ""), d.get("implication", ""),
                   dict(d.get("identity", {})), dict(d.get("activity", {})), dict(d.get("notes", {})), str(p))


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.toml"))


def render_profile(pr: Profile) -> str:
    lines = [f"{pr.name}: {pr.title}", "", "Setup", "  " + pr.setup.replace("\n", "\n  "), "",
             f"Problem      {pr.problem}", f"Implication  {pr.implication}", "", "Config"]
    for k, v in pr.identity.items():
        lines.append(f"  identity.{k} = {v!r}")
    for k, v in pr.activity.items():
        lines.append(f"  activity_flow.{k} = {v!r}")
    for k, v in pr.notes.items():
        lines.append(f"  note: {k}: {v}")
    return "\n".join(lines)
