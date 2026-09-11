"""Engineered features shared by storage, modeling and what-if.

Kept in one place so a what-if override of a raw feature (e.g. wire cap) is propagated to
every derived feature before prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED = ("wire_cap_fraction", "total_cap_pf", "dyn_term", "leak_term", "cell_dyn_term", "wire_dyn_term", "move_term")

DESCRIPTIONS = {
    "wire_cap_fraction": "wire_cap / (wire_cap + cell_cap)  (size-independent wire dominance)",
    "total_cap_pf": "wire_cap + cell_cap",
    "dyn_term": "activity * total_cap * V^2 * f  (CV^2f dynamic-power proxy; activity/V/f default to 1 when absent)",
    "leak_term": "area * V^3  (leakage proxy; V defaults to 1 when absent)",
    "cell_dyn_term": "activity * cell_cap * V^2 * f  (compute / cell-switching energy proxy)",
    "wire_dyn_term": "activity * wire_cap * V^2 * f  (wire-switching energy proxy)",
    "move_term": "bits_per_cycle * avg_net_length_um * V^2 * f  (data-movement energy proxy: bits x distance x V^2 x rate)",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num = lambda c, default=np.nan: pd.to_numeric(out[c], errors="coerce").astype(float) if c in out.columns else pd.Series(default, index=out.index, dtype=float)
    wire, cell = num("wire_cap_pf"), num("cell_cap_pf")
    total = wire + cell
    out["total_cap_pf"] = total
    out["wire_cap_fraction"] = (wire / total.where(total > 0)).astype(float)
    act = num("activity").fillna(1.0)
    v = num("voltage_v").fillna(1.0)
    f = num("frequency_ghz").fillna(1.0)
    area = num("area")
    out["dyn_term"] = act * total * v ** 2 * f
    out["leak_term"] = area * v ** 3
    out["cell_dyn_term"] = act * cell * v ** 2 * f
    out["wire_dyn_term"] = act * wire * v ** 2 * f
    bits = num("bits_per_cycle")
    dist = num("avg_net_length_um")
    out["move_term"] = (bits * dist * v ** 2 * f).where(bits.notna() & dist.notna(), np.nan)
    return out


# parameters whose change alters what an FE physical-power estimate would have reported
FE_RESCALE_TRIGGERS = ("wire_cap_pf", "cell_cap_pf", "frequency_ghz", "voltage_v", "activity")


def rescale_fe_physical(current: pd.DataFrame, proposed: pd.DataFrame, changed: set[str] | None = None) -> tuple[pd.DataFrame, bool]:
    """Scale fe_physical_mw in `proposed` by the change in the CV^2f term relative to `current`.

    Both frames must already carry engineered features. The rescale is applied only when a trigger
    parameter changed and fe_physical_mw itself was not overridden (changed=None means "always").
    Returns (frame, applied).
    """
    if "fe_physical_mw" not in proposed.columns or "dyn_term" not in proposed.columns:
        return proposed, False
    if changed is not None and ((not changed & set(FE_RESCALE_TRIGGERS)) or "fe_physical_mw" in changed):
        return proposed, False
    out = proposed.copy()
    ratio = (out["dyn_term"] / current["dyn_term"].where(current["dyn_term"] > 0)).fillna(1.0)
    out["fe_physical_mw"] = current["fe_physical_mw"] * ratio
    return out, True
