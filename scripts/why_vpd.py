"""SEMAI - "Why VPD, not humidity" (docs/SPEC.md section 6, Step 3). No ML.

Uses the cached Penang hourly weather (2023-2025, Open-Meteo / ERA5) and the
rig's own VPD thresholds (semai.vpd) to answer three questions:

  1. How much of the time would the rig be DRY / OPTIMAL / SATURATED in Penang
     outdoor air?  (hysteresis replica of the rig state machine, plus the plain
     threshold bands without hysteresis)
  2. If we misted on the simple rule "RH < 60 %" instead of "VPD > 1.2 kPa",
     how many dry-stress hours would we miss, and what do they look like?
  3. Which hours of the day are dry?

Outputs (all in results/):
  why_vpd.json, W1_rig_states_penang.png, W2_rh_rule_misses.png, W3_dry_by_hour.png

Run:  python scripts/why_vpd.py     (idempotent; seed 0 for the scatter sample)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from semai import penang_data  # noqa: E402
from semai.vpd import (  # noqa: E402
    VPD_DRY, VPD_OPT_HIGH, VPD_OPT_LOW, VPD_SAT, rig_state_sequence, svp,
)

RESULTS_DIR = os.path.join(ROOT, "results")
JSON_PATH = os.path.join(RESULTS_DIR, "why_vpd.json")
W1_PATH = os.path.join(RESULTS_DIR, "W1_rig_states_penang.png")
W2_PATH = os.path.join(RESULTS_DIR, "W2_rh_rule_misses.png")
W3_PATH = os.path.join(RESULTS_DIR, "W3_dry_by_hour.png")

RH_RULE = 60.0                       # "mist when RH < 60 %"
RH_TABLE = [55, 60, 65, 70, 75]      # thresholds for the sensitivity table
SEED = 0
SCATTER_MAX = 4000
STATES = ("DRY", "OPTIMAL", "SATURATED")
YEARS = (2023, 2024, 2025)
DPI = 200

# Colours (validated categorical palette, fixed order; gray = recessive)
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_RED = "#e34948"
C_GRAY = "#b8b7b2"
C_INK = "#52514e"
STATE_COLOURS = {"DRY": C_ORANGE, "OPTIMAL": C_AQUA, "SATURATED": C_BLUE}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 3) if d else float("nan")


def state_shares(states: np.ndarray) -> dict:
    n = int(len(states))
    out = {"n_hours": n}
    for s in STATES:
        k = int(np.sum(states == s))
        out[s] = {"hours": k, "share_pct": pct(k, n)}
    return out


def band_shares(vpd: np.ndarray) -> dict:
    """Plain threshold bands, no hysteresis. Boundaries follow semai.vpd."""
    n = int(len(vpd))
    bands = {
        "dry_gt_1.2": vpd > VPD_DRY,
        "gap_1.1_to_1.2": (vpd > VPD_OPT_HIGH) & (vpd <= VPD_DRY),
        "optimal_0.8_to_1.1": (vpd >= VPD_OPT_LOW) & (vpd <= VPD_OPT_HIGH),
        "gap_0.6_to_0.8": (vpd >= VPD_SAT) & (vpd < VPD_OPT_LOW),
        "saturated_lt_0.6": vpd < VPD_SAT,
    }
    out = {"n_hours": n}
    total = 0
    for k, m in bands.items():
        c = int(m.sum())
        total += c
        out[k] = {"hours": c, "share_pct": pct(c, n)}
    assert total == n, f"bands do not partition the series: {total} != {n}"
    return out


def rh_rule_stats(t, rh, vpd, rh_thr: float) -> dict:
    dry = vpd > VPD_DRY
    rule = rh < rh_thr
    missed = dry & ~rule
    false_alarm = rule & ~dry
    both = dry & rule
    n_dry = int(dry.sum())
    out = {
        "rh_threshold_pct": rh_thr,
        "n_hours": int(len(vpd)),
        "n_dry_stress": n_dry,
        "n_rh_rule": int(rule.sum()),
        "n_both": int(both.sum()),
        "n_missed": int(missed.sum()),
        "missed_share_of_dry_pct": pct(int(missed.sum()), n_dry),
        "n_false_alarm": int(false_alarm.sum()),
    }
    if missed.any():
        q = {}
        for name, arr in (("t_c", t), ("rh_pct", rh), ("vpd_kpa", vpd)):
            v = arr[missed]
            q[name] = {
                "p25": round(float(np.percentile(v, 25)), 3),
                "median": round(float(np.median(v)), 3),
                "p75": round(float(np.percentile(v, 75)), 3),
            }
        out["typical_missed_hour"] = q
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def draw_w1(rig: dict, path: str) -> None:
    groups = ["all 2023-2025"] + [str(y) for y in YEARS]
    keys = ["overall"] + [str(y) for y in YEARS]
    x = np.arange(len(groups))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(STATES):
        vals = [rig[k][s]["share_pct"] for k in keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=s,
                      color=STATE_COLOURS[s], edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                        textcoords="offset points", color=C_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("period (hourly Penang outdoor air, Open-Meteo / ERA5)")
    ax.set_ylabel("share of hours (%)")
    ax.set_ylim(0, max(rig[k][s]["share_pct"] for k in keys for s in STATES) * 1.15)
    ax.set_title("W1 - Rig state replica (with hysteresis) on Penang outdoor air\n"
                 f"DRY: VPD > {VPD_DRY} kPa, OPTIMAL: {VPD_OPT_LOW}-{VPD_OPT_HIGH} kPa, "
                 f"SATURATED: VPD < {VPD_SAT} kPa; gaps keep the previous state")
    ax.legend(title="rig state", frameon=False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def draw_w2(t, rh, vpd, rule: dict, path: str) -> dict:
    n = len(vpd)
    rng = np.random.default_rng(SEED)
    k = min(SCATTER_MAX, n)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    dry = vpd > VPD_DRY
    caught = dry & (rh < RH_RULE)
    missed = dry & ~(rh < RH_RULE)
    not_dry = ~dry

    fig, ax = plt.subplots(figsize=(9, 6))
    classes = [
        (not_dry, C_GRAY, f"not dry-stress (VPD <= {VPD_DRY} kPa)", 8, 0.55, "o"),
        (caught, C_BLUE, f"dry-stress caught by RH < {RH_RULE:.0f} % rule", 14, 0.85, "o"),
        (missed, C_ORANGE, f"dry-stress MISSED by RH < {RH_RULE:.0f} % rule", 14, 0.85, "o"),
    ]
    sample_counts = {}
    for mask, colour, label, size, alpha, marker in classes:
        m = mask[idx]
        sample_counts[label] = int(m.sum())
        ax.scatter(t[idx][m], rh[idx][m], s=size, c=colour, alpha=alpha,
                   marker=marker, linewidths=0, label=f"{label} (n={int(m.sum())} in sample)")

    # the two rules
    tt = np.linspace(float(t.min()) - 0.5, float(t.max()) + 0.5, 300)
    rh_curve = 100.0 * (1.0 - VPD_DRY / svp(tt))
    ax.plot(tt, rh_curve, color="#0b0b0b", linewidth=2,
            label=f"VPD = {VPD_DRY} kPa curve: RH = 100(1 - {VPD_DRY}/SVP(T))")
    ax.axhline(RH_RULE, color=C_RED, linewidth=2, linestyle="--",
               label=f"RH = {RH_RULE:.0f} % rule")
    ax.fill_between(tt, RH_RULE, rh_curve, where=rh_curve > RH_RULE,
                    color=C_ORANGE, alpha=0.10, linewidth=0,
                    label="miss zone: dry by VPD (below curve) but RH >= 60 % (above line)")
    ax.set_xlabel("air temperature T (deg C)")
    ax.set_ylabel("relative humidity RH (%)")
    ax.set_xlim(tt.min(), tt.max())
    ax.set_ylim(max(0.0, float(rh.min()) - 3), 101)
    ax.set_title(
        f"W2 - The RH < {RH_RULE:.0f} % rule misses {rule['n_missed']:,} of "
        f"{rule['n_dry_stress']:,} dry-stress hours ({rule['missed_share_of_dry_pct']:.1f} %)\n"
        f"Penang outdoor air 2023-2025: random sample of {k:,} of {n:,} hours (seed {SEED}), "
        "weather numbers only", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return {"sample_size": int(k), "seed": SEED, "sample_class_counts": sample_counts}


def draw_w3(byhour: dict, path: str) -> None:
    hours = np.arange(24)
    share = np.array(byhour["dry_share_pct_by_hour"])
    mean_vpd = np.array(byhour["mean_vpd_kpa_by_hour"])
    peak_h = byhour["peak_hour"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
    colours = [C_ORANGE if h == peak_h else C_BLUE for h in hours]
    bars = ax1.bar(hours, share, color=colours, edgecolor="white", linewidth=1,
                   label=f"share of hours with VPD > {VPD_DRY} kPa")
    ax1.annotate(f"peak {share[peak_h]:.1f}% at {peak_h:02d}:00",
                 (peak_h, share[peak_h]), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=9, color="#0b0b0b", fontweight="bold")
    ax1.set_ylabel("dry share (%)")
    ax1.set_ylim(0, max(share.max() * 1.2, 1))
    ax1.set_title("W3 - Dry hours by hour of day, Penang outdoor air 2023-2025 (local time)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2.plot(hours, mean_vpd, color=C_BLUE, linewidth=2, marker="o", markersize=4,
             label="mean air VPD by hour")
    ax2.axhline(VPD_DRY, color=C_RED, linewidth=1.5, linestyle="--",
                label=f"rig DRY threshold ({VPD_DRY} kPa)")
    ax2.set_ylabel("mean air VPD (kPa)")
    ax2.set_xlabel("hour of day (local time, Asia/Kuala_Lumpur)")
    ax2.set_xticks(hours)
    ax2.set_ylim(0, max(float(mean_vpd.max()), VPD_DRY) * 1.2)
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax2.set_axisbelow(True)
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = penang_data.load()
    df = df.sort_values("time").reset_index(drop=True)   # chronological order
    t = df["t"].to_numpy(dtype=float)
    rh = df["rh"].to_numpy(dtype=float)
    vpd = df["vpd"].to_numpy(dtype=float)
    years = df["time"].dt.year.to_numpy()
    hours = df["time"].dt.hour.to_numpy()
    n = len(df)
    assert n == penang_data.EXPECTED_ROWS, f"expected {penang_data.EXPECTED_ROWS} rows, got {n}"
    assert not np.isnan(vpd).any(), "NaN in VPD series"
    assert np.all(np.diff(df["time"].to_numpy()).astype("timedelta64[h]") == np.timedelta64(1, "h")), \
        "time series is not continuous hourly"

    # 1. rig state replica (hysteresis) over the full continuous series
    states = np.array(rig_state_sequence(vpd))
    rig = {"overall": state_shares(states)}
    for y in YEARS:
        rig[str(y)] = state_shares(states[years == y])
    bands = {"overall": band_shares(vpd)}
    for y in YEARS:
        bands[str(y)] = band_shares(vpd[years == y])

    # 2. RH rule vs VPD rule
    rule = rh_rule_stats(t, rh, vpd, RH_RULE)
    rh_table = [rh_rule_stats(t, rh, vpd, float(thr)) for thr in RH_TABLE]
    for row in rh_table:
        row.pop("typical_missed_hour", None)

    # 3. dry share by hour of day
    dry = vpd > VPD_DRY
    share_by_hour = [pct(int(dry[hours == h].sum()), int((hours == h).sum())) for h in range(24)]
    mean_vpd_by_hour = [round(float(vpd[hours == h].mean()), 4) for h in range(24)]
    peak_hour = int(np.argmax(share_by_hour))
    byhour = {
        "dry_share_pct_by_hour": share_by_hour,
        "mean_vpd_kpa_by_hour": mean_vpd_by_hour,
        "peak_hour": peak_hour,
        "peak_dry_share_pct": share_by_hour[peak_hour],
        "hours_per_slot": int((hours == 0).sum()),
    }

    # 4. figures
    draw_w1(rig, W1_PATH)
    w2_info = draw_w2(t, rh, vpd, rule, W2_PATH)
    draw_w3(byhour, W3_PATH)

    # 5. JSON
    meta = penang_data.meta()
    out = {
        "computed_at_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": {
            "url": meta.get("url"),
            "fetched_at_utc": meta.get("fetched_at_utc"),
            "rows": meta.get("rows", n),
            "source": meta.get("source"),
            "csv": os.path.relpath(penang_data.CSV_PATH, ROOT).replace(os.sep, "/"),
            "period_local": [str(df["time"].iloc[0]), str(df["time"].iloc[-1])],
            "timezone": meta.get("timezone", "Asia/Kuala_Lumpur"),
            "n_hours": n,
        },
        "thresholds": {
            "vpd_dry_kpa": VPD_DRY, "vpd_opt_low_kpa": VPD_OPT_LOW,
            "vpd_opt_high_kpa": VPD_OPT_HIGH, "vpd_sat_kpa": VPD_SAT,
            "rh_rule_pct": RH_RULE, "rh_table_pct": RH_TABLE,
            "vpd_formula": "air VPD = 0.6108*exp(17.27*T/(T+237.3)) * (1 - RH/100)  [kPa]",
            "hysteresis": "gaps 0.6-0.8 and 1.1-1.2 kPa keep the previous state (semai.vpd.rig_state_sequence)",
        },
        "rig_states_hysteresis": rig,
        "vpd_bands_no_hysteresis": bands,
        "rh_rule_vs_vpd_rule": rule,
        "rh_threshold_table": rh_table,
        "dry_by_hour_of_day": byhour,
        "figures": {
            "W1_rig_states_penang.png": "share of hours per rig state (hysteresis replica), overall and per year",
            "W2_rh_rule_misses.png": {"description": "T vs RH scatter, 3 classes, RH=60 line and VPD=1.2 curve", **w2_info},
            "W3_dry_by_hour.png": "dry share by hour of day (top) and mean VPD by hour with the 1.2 kPa line (bottom)",
        },
        "seed": SEED,
        "script": "scripts/why_vpd.py",
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # console summary
    o = rig["overall"]
    b = bands["overall"]
    tm = rule.get("typical_missed_hour") or {
        k: {"p25": float("nan"), "median": float("nan"), "p75": float("nan")} for k in ("t_c", "rh_pct", "vpd_kpa")}
    print(f"hours: {n}  (fetched {meta.get('fetched_at_utc')})")
    print("rig states (hysteresis): " + ", ".join(f"{s} {o[s]['share_pct']:.1f}%" for s in STATES))
    print(f"bands (no hysteresis): >1.2 {b['dry_gt_1.2']['share_pct']:.1f}%, "
          f"0.8-1.1 {b['optimal_0.8_to_1.1']['share_pct']:.1f}%, <0.6 {b['saturated_lt_0.6']['share_pct']:.1f}%")
    print(f"RH<{RH_RULE:.0f} rule misses {rule['n_missed']} of {rule['n_dry_stress']} dry-stress hours "
          f"({rule['missed_share_of_dry_pct']:.1f}%), false alarms {rule['n_false_alarm']}")
    print(f"typical missed hour: {tm['t_c']['median']:.1f} degC, {tm['rh_pct']['median']:.0f}% RH, "
          f"{tm['vpd_kpa']['median']:.2f} kPa")
    print(f"dry share peaks at {byhour['peak_dry_share_pct']:.1f}% at {peak_hour:02d}:00")
    print("wrote", JSON_PATH, W1_PATH, W2_PATH, W3_PATH, sep="\n  ")
    return out


if __name__ == "__main__":
    main()
