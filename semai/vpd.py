"""Vapour-pressure-deficit maths shared by every SEMAI part.

Formulas are copied from the rig firmware (docs/SPEC.md §4) so the laptop and
the Pico agree on what "dry" means.

    SVP(T)   = 0.6108 * exp(17.27*T / (T + 237.3))        [kPa], T in °C
    air VPD  = SVP(T_air) * (1 - RH/100)                   [kPa]
    leaf VPD = SVP(T_leaf) - SVP(T_air) * RH/100            [kPa]

Rig thresholds (leaf VPD, kPa):
    DRY  (mist)        when VPD > 1.2
    OPTIMAL            0.8 – 1.1
    SATURATED (purge)  when VPD < 0.6
    gaps keep the previous state (hysteresis)

All functions accept floats or numpy arrays / pandas Series and pass NaN through.
"""
from __future__ import annotations

import numpy as np

# Rig thresholds (kPa)
VPD_DRY = 1.2
VPD_OPT_LOW = 0.8
VPD_OPT_HIGH = 1.1
VPD_SAT = 0.6

# Rig status strings (docs/SPEC.md §4)
STATUSES = (
    "OPTIMAL",
    "HIGH_VPD_MISTING",
    "LOW_VPD_PURGE",
    "ERR_LOW_WATER",
    "ERR_SENSOR",
    "WAITING_FOR_PICO",
    "PICO_OFFLINE",
)


def svp(t_c):
    """Saturation vapour pressure in kPa for temperature in °C (Tetens, as on the rig)."""
    t = np.asarray(t_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def air_vpd(t_air_c, rh_pct):
    """Air VPD in kPa: SVP(T_air) * (1 - RH/100)."""
    rh = np.asarray(rh_pct, dtype=float)
    return svp(t_air_c) * (1.0 - rh / 100.0)


def leaf_vpd(t_leaf_c, t_air_c, rh_pct):
    """Leaf VPD in kPa: SVP(T_leaf) - SVP(T_air) * RH/100."""
    rh = np.asarray(rh_pct, dtype=float)
    return svp(t_leaf_c) - svp(t_air_c) * rh / 100.0


def is_dry(vpd_kpa):
    """True where VPD exceeds the rig's misting threshold (1.2 kPa). NaN -> False."""
    v = np.asarray(vpd_kpa, dtype=float)
    return np.where(np.isnan(v), False, v > VPD_DRY)


def rig_state_sequence(vpd_kpa):
    """Replica of the rig's state machine with hysteresis.

    Walks a sequence of VPD readings and returns a list of states, one per
    reading: "DRY" (VPD > 1.2, misting), "SATURATED" (VPD < 0.6, fan purge),
    "OPTIMAL" (0.8–1.1). Readings in the gaps (0.6–0.8 and 1.1–1.2) keep the
    previous state. The first reading, if it lands in a gap, is treated as
    OPTIMAL. NaN readings also keep the previous state.
    """
    v = np.asarray(vpd_kpa, dtype=float)
    states = []
    prev = "OPTIMAL"
    for x in v:
        if np.isnan(x):
            s = prev
        elif x > VPD_DRY:
            s = "DRY"
        elif x < VPD_SAT:
            s = "SATURATED"
        elif VPD_OPT_LOW <= x <= VPD_OPT_HIGH:
            s = "OPTIMAL"
        else:
            s = prev  # hysteresis gap
        states.append(s)
        prev = s
    return states


if __name__ == "__main__":  # tiny self-check
    assert abs(float(svp(25.0)) - 3.1687) < 0.01, svp(25.0)
    assert abs(float(air_vpd(30.0, 60.0)) - 1.697) < 0.01, air_vpd(30.0, 60.0)
    assert rig_state_sequence([0.9, 1.15, 1.3, 1.15, 0.7, 0.5, 0.7]) == [
        "OPTIMAL", "OPTIMAL", "DRY", "DRY", "DRY", "SATURATED", "SATURATED"]
    print("vpd.py self-check OK")
