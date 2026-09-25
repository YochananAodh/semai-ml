"""Train and evaluate SEMAI Forecast end to end (docs/SPEC.md §6 step 2, items 2-6).

    python scripts/train_forecast.py

Reads data/penang_hourly_2023_2025.csv (via semai.penang_data), builds the
features and label with semai.forecast.build_features, trains on 2023-2024,
tests on 2025 against three baselines, and writes:
    models/forecast.joblib
    results/forecast_metrics.json
    results/F1_model_vs_baselines.png
    results/F2_confusion_2025.png
    results/F3_feature_importance.png
    results/F4_replay_3days.png
Idempotent: re-running overwrites the same files with the same numbers (seed 0).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import joblib  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

from semai import forecast, penang_data  # noqa: E402
from semai.forecast import FEATURES, THRESHOLD, TEST_YEAR  # noqa: E402
from semai.vpd import VPD_DRY  # noqa: E402

SEED = 0
MODEL_PARAMS = {"max_iter": 300, "learning_rate": 0.05, "random_state": SEED}
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_PATH = os.path.join(MODELS_DIR, "forecast.joblib")
METRICS_PATH = os.path.join(RESULTS_DIR, "forecast_metrics.json")

# Colours (categorical slots, fixed order: baselines first, model last)
C_ALWAYS_NO = "#eda100"    # yellow
C_PERSIST = "#eb6834"      # orange
C_CLIM = "#1baf7a"         # aqua
C_MODEL = "#2a78d6"        # blue
C_VPD = "#4a3aa7"          # violet (actual VPD line in F4)
C_SHADE = "#e34948"        # red (label shading in F4)
C_TEXT2 = "#52514e"
FIG_DPI = 200
TEST_TITLE = "2025 test (model never saw this year)"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def metrics_for(y_true, y_pred, score=None, auc_kind=None) -> dict:
    """accuracy / f1 / precision / recall / auc / confusion for one predictor."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": None,
        "auc_kind": None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y_true)),
        "n_predicted_positive": int(y_pred.sum()),
    }
    if score is not None:
        out["auc"] = float(roc_auc_score(y_true, np.asarray(score, dtype=float)))
        out["auc_kind"] = auc_kind
    return out


def to_native(o):
    """Make a nested structure JSON-serialisable (numpy scalars -> python)."""
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_f1(metrics: dict, n_test: int, path: str) -> None:
    names = ["always_no", "persistence", "climatology", "model"]
    labels = ['Always "no"', "Persistence (dry now?)", "Hour-of-day climatology", "SEMAI Forecast (model)"]
    colors = [C_ALWAYS_NO, C_PERSIST, C_CLIM, C_MODEL]
    keys = ["accuracy", "f1", "precision", "recall"]
    klabels = ["Accuracy", "F1", "Precision", "Recall"]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (nm, lb, col) in enumerate(zip(names, labels, colors)):
        vals = [metrics[nm][k] for k in keys]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=col, label=lb)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xticklabels(klabels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (fraction, 0-1)")
    ax.set_xlabel("Metric (positive class = misting needed within the next 2 h)")
    ax.set_title(f"SEMAI Forecast vs baselines: {TEST_TITLE}, n={n_test:,} hours")
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def _draw_cm(ax, cm: dict, title: str, color: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    pct = np.where(row_sum > 0, mat / np.maximum(row_sum, 1) * 100, 0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("c", ["#ffffff", color])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            txt = f"{int(mat[i, j]):,} h\n({pct[i, j]:.1f}% of row)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if pct[i, j] > 60 else "#0b0b0b")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["predicted: no mist", "predicted: mist"])
    ax.set_yticklabels(["actual: no mist", "actual: mist"], rotation=90, va="center")
    ax.set_xlabel("Prediction (hours)")
    ax.set_ylabel("Actual label (hours)")
    ax.set_title(title, fontsize=10)


def fig_f2(metrics: dict, n_test: int, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    p = metrics["persistence"]
    m = metrics["model"]
    _draw_cm(axes[0], p["confusion"],
             f"Persistence baseline (mist if dry now)\nacc {p['accuracy']:.3f}, F1 {p['f1']:.3f}", C_PERSIST)
    _draw_cm(axes[1], m["confusion"],
             f"SEMAI Forecast model (prob >= {THRESHOLD})\nacc {m['accuracy']:.3f}, F1 {m['f1']:.3f}", C_MODEL)
    fig.suptitle(f"Confusion matrices: {TEST_TITLE}, n={n_test:,} hours; counts and row-percentages",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_f3(pi: list, model_f1: float, persist_f1: float, path: str) -> None:
    feats = [d["feature"] for d in pi][::-1]
    means = np.array([d["mean"] for d in pi])[::-1]
    stds = np.array([d["std"] for d in pi])[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(feats, means, xerr=stds, color=C_MODEL, ecolor=C_TEXT2, capsize=2.5, height=0.65,
            label="mean +/- std over 10 shuffles")
    for i, (mval, sval) in enumerate(zip(means, stds)):
        ax.text(max(mval + sval, 0) + 0.002, i, f"{mval:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel("F1 drop when feature is shuffled (2025 test)")
    ax.set_ylabel("Feature (t in degC, rh in %, vpd in kPa, lagN = N hours earlier)")
    ax.set_title(f"Permutation importance: {TEST_TITLE}\n"
                 f"model F1 {model_f1:.3f} vs persistence baseline F1 {persist_f1:.3f}", fontsize=10)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def fig_f4(days: list, persist_f1: float, model_f1: float, path: str) -> None:
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.6), sharex=True)
    for ax, d in zip(axes, days):
        rows = d["hours"]
        h = np.array([r["hour"] for r in rows])
        vpd = np.array([r["vpd"] for r in rows])
        prob = np.array([np.nan if r["prob"] is None else r["prob"] for r in rows])
        lab = np.array([False if r["actual_dry"] is None else r["actual_dry"] for r in rows])
        # shading: hours where the label says misting is needed within the next 2 h
        for hh, lv in zip(h, lab):
            if lv:
                ax.axvspan(hh - 0.5, hh + 0.5, color=C_SHADE, alpha=0.13, lw=0)
        ax.plot(h, vpd, color=C_VPD, lw=2, marker="o", ms=3.5)
        ax.axhline(VPD_DRY, color=C_VPD, ls="--", lw=1.2)
        ax.set_ylabel("Air VPD (kPa)")
        ax.set_ylim(0, max(2.0, float(np.nanmax(vpd)) * 1.15))
        ax2 = ax.twinx()
        ax2.plot(h, prob, color=C_MODEL, lw=2, marker="s", ms=3)
        ax2.axhline(THRESHOLD, color=C_MODEL, ls=":", lw=1.2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Model probability (0-1)", color=C_MODEL)
        ax2.tick_params(axis="y", colors=C_MODEL)
        n_lab = int(lab.sum())
        ax.set_title(f"{d['date']}: {d['category']}; max {d['daily_max_vpd']:.2f} kPa, "
                     f"{n_lab}/24 h labelled 'mist within 2 h'", fontsize=9.5, loc="left")
        ax.grid(color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Hour of day (local time, Asia/Kuala_Lumpur)")
    axes[-1].set_xticks(range(0, 24, 2))
    handles = [
        Line2D([], [], color=C_VPD, lw=2, marker="o", ms=3.5, label="actual air VPD (kPa, left axis)"),
        Line2D([], [], color=C_VPD, ls="--", lw=1.2, label="rig mist threshold 1.2 kPa"),
        Line2D([], [], color=C_MODEL, lw=2, marker="s", ms=3, label="model probability (0-1, right axis)"),
        Line2D([], [], color=C_MODEL, ls=":", lw=1.2, label=f"decision threshold {THRESHOLD}"),
        Patch(facecolor=C_SHADE, alpha=0.13, label="label = 1: VPD > 1.2 within the next 2 h (misting followed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("SEMAI Forecast: outdoor Penang air, replay of 2025 (model never saw this year)\n"
                 f"model F1 {model_f1:.3f} vs persistence baseline F1 {persist_f1:.3f} on all of 2025",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checks = []

    def check(name, cond, detail=""):
        checks.append({"check": name, "passed": bool(cond), "detail": detail})
        log(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        assert cond, f"{name}: {detail}"

    # 1. data ----------------------------------------------------------------
    df = penang_data.load()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    data_meta = penang_data.meta()
    log(f"data rows={len(df)}  {df['time'].iloc[0]} .. {df['time'].iloc[-1]}")
    check("hourly series is continuous",
          bool((df["time"].diff().dropna() == pd.Timedelta(hours=1)).all()), "every step is 1 h")

    # 2-3. label + features ----------------------------------------------------
    X_all, y_all, meta_all = forecast.build_features(df)
    check("feature names/order", list(X_all.columns) == FEATURES, ", ".join(FEATURES))
    keep = ~(X_all.isna().any(axis=1) | y_all.isna())
    n_dropped_nan = int((~keep).sum())
    X = X_all.loc[keep]
    y = y_all.loc[keep].astype(int)
    m = meta_all.loc[keep]
    log(f"rows after dropping NaN features/labels: {len(X)} (dropped {n_dropped_nan})")

    # 4. split ----------------------------------------------------------------
    year = m["time"].dt.year.to_numpy()
    is_test = year == TEST_YEAR
    is_train_raw = year <= TEST_YEAR - 1
    # leakage guard: a train row's label window (t+1, t+2) must not reach 2025
    window_end = m["time"] + pd.Timedelta(hours=max(forecast.HORIZON_HOURS))
    crosses = (window_end.dt.year.to_numpy() >= TEST_YEAR) & is_train_raw
    is_train = is_train_raw & ~crosses
    n_train_dropped_for_leakage = int(crosses.sum())
    X_train, y_train = X.loc[is_train], y.loc[is_train]
    X_test, y_test = X.loc[is_test], y.loc[is_test]
    t_train = m.loc[is_train, "time"]
    t_test = m.loc[is_test, "time"]
    log(f"train rows={len(X_train)} ({t_train.min()} .. {t_train.max()}), "
        f"dropped for leakage={n_train_dropped_for_leakage}")
    log(f"TEST SIZE = {len(X_test)} rows ({t_test.min()} .. {t_test.max()})")
    check("every test row is in 2025", bool((t_test.dt.year == TEST_YEAR).all()), f"n_test={len(X_test)}")
    check("no train row is in 2025", bool((t_train.dt.year < TEST_YEAR).all()), f"n_train={len(X_train)}")
    check("no train label uses a 2025 hour",
          bool(((t_train + pd.Timedelta(hours=max(forecast.HORIZON_HOURS))).dt.year < TEST_YEAR).all()),
          f"last train hour {t_train.max()}, its label window ends {t_train.max() + pd.Timedelta(hours=2)}; "
          f"{n_train_dropped_for_leakage} train rows dropped")
    check("train and test are disjoint", not bool((is_train & is_test).any()), "")
    n_2025_rows = int((meta_all["time"].dt.year == TEST_YEAR).sum())
    check("test set = all 2025 rows with a defined label", len(X_test) == int(is_test.sum()),
          f"{len(X_test)} of {n_2025_rows} 2025 hours; the {n_2025_rows - len(X_test)} last hours of "
          f"2025-12-31 have no label (t+2 is outside the data)")
    train_pos = float(y_train.mean())
    test_pos = float(y_test.mean())
    log(f"positive rate train={train_pos:.4f} test={test_pos:.4f}")

    # 4. model ---------------------------------------------------------------
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0
    trained_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log(f"model trained in {train_seconds:.1f} s, iterations used={model.n_iter_}")
    p_test = model.predict_proba(X_test)[:, 1]
    pred_model = (p_test >= THRESHOLD).astype(int)
    metrics = {"model": metrics_for(y_test, pred_model, p_test, "probability")}

    # 4. baselines -----------------------------------------------------------
    y_test_np = y_test.to_numpy()
    metrics["always_no"] = metrics_for(y_test_np, np.zeros(len(y_test_np), dtype=int))
    pers = (X_test["vpd"].to_numpy() > VPD_DRY).astype(int)
    metrics["persistence"] = metrics_for(y_test_np, pers, pers, "binary")

    hour_train = t_train.dt.hour.to_numpy()
    hour_test = t_test.dt.hour.to_numpy()
    rates = pd.Series(y_train.to_numpy()).groupby(hour_train).mean()
    rates = rates.reindex(range(24), fill_value=0.0)
    rate_arr = rates.to_numpy()
    grid = np.unique(rate_arr)                       # the 24 distinct rates
    best_thr, best_f1 = None, -1.0
    y_train_np = y_train.to_numpy()
    for thr in grid:
        pr = (rate_arr[hour_train] >= thr).astype(int)
        f = f1_score(y_train_np, pr, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    clim_score = rate_arr[hour_test]
    clim_pred = (clim_score >= best_thr).astype(int)
    metrics["climatology"] = metrics_for(y_test_np, clim_pred, clim_score, "hourly rate")
    metrics["climatology"]["threshold"] = best_thr
    metrics["climatology"]["threshold_train_f1"] = float(best_f1)
    metrics["climatology"]["threshold_selection"] = (
        "grid over the 24 distinct training hourly rates; the smallest rate with the "
        "highest training F1 wins")
    metrics["climatology"]["hourly_rates"] = {str(h): float(rate_arr[h]) for h in range(24)}
    metrics["climatology"]["hours_predicted_positive"] = [int(h) for h in range(24) if rate_arr[h] >= best_thr]

    # 4. permutation importance ------------------------------------------------
    t0 = time.perf_counter()
    pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=SEED,
                                scoring="f1", n_jobs=4)
    pi_seconds = time.perf_counter() - t0
    order = np.argsort(-pi.importances_mean)
    pi_list = [{"feature": FEATURES[i], "mean": float(pi.importances_mean[i]),
                "std": float(pi.importances_std[i])} for i in order]
    log(f"permutation importance done in {pi_seconds:.1f} s; top: "
        + ", ".join(f"{d['feature']}={d['mean']:.3f}" for d in pi_list[:5]))

    # 5. save model ---------------------------------------------------------------
    bundle = {"model": model, "features": list(FEATURES), "threshold": THRESHOLD,
              "trained_on": "2023-2024", "test_year": TEST_YEAR, "seed": SEED,
              "sklearn_version": sklearn.__version__, "model_params": MODEL_PARAMS,
              "trained_at_utc": trained_at}
    joblib.dump(bundle, MODEL_PATH)
    log(f"saved {MODEL_PATH}")
    forecast.reset_cache()
    check("saved model reloads through semai.forecast", forecast.available(), MODEL_PATH)
    p_check = forecast.load_model()["model"].predict_proba(X_test.iloc[:50])[:, 1]
    check("reloaded model reproduces probabilities", bool(np.allclose(p_check, p_test[:50])),
          "first 50 test rows")

    # 6. replay days for F4 ---------------------------------------------------------
    test_meta = meta_all.loc[meta_all["time"].dt.year == TEST_YEAR]
    daily_max = test_meta.groupby(test_meta["time"].dt.strftime("%Y-%m-%d"))["vpd"].max()
    med = float(daily_max.median())
    driest = str(daily_max.idxmax())
    wettest = str(daily_max.idxmin())
    mixed = str((daily_max - med).abs().idxmin())
    replay_days = []
    for dt, cat in ((driest, "driest day (highest daily max VPD)"),
                    (wettest, "wettest day (lowest daily max VPD)"),
                    (mixed, "mixed day (daily max VPD closest to the 2025 median)")):
        rows = forecast.replay(dt)
        replay_days.append({"date": dt, "category": cat, "daily_max_vpd": float(daily_max[dt]),
                            "hours": rows})
    check("F4 picks three distinct 2025 days", len({d["date"] for d in replay_days}) == 3,
          ", ".join(f"{d['date']} ({d['daily_max_vpd']:.2f} kPa)" for d in replay_days))
    r = forecast.replay("2025-03-15")
    check("replay returns 24 rows, each with a probability",
          len(r) == 24 and all(x["prob"] is not None for x in r), "2025-03-15")
    # the replay path (semai.forecast) must agree with the training path on the same hours
    idx = pd.Index(t_test)
    rep_probs = np.array([x["prob"] for x in r])
    pos = idx.get_indexer(pd.date_range("2025-03-15", periods=24, freq="h"))
    check("replay probabilities match the training-path probabilities",
          bool(np.allclose(rep_probs, p_test[pos])), "2025-03-15, 24 hours")

    # 5. metrics JSON --------------------------------------------------------------
    out = {
        "data": {k: data_meta.get(k) for k in ("url", "fetched_at_utc", "rows", "source")},
        "label_definition": forecast.LABEL_DEFINITION,
        "split_definition": forecast.SPLIT_DEFINITION,
        "features": list(FEATURES),
        "n_rows_total": int(len(df)),
        "n_rows_dropped_nan": n_dropped_nan,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_dropped_for_leakage": n_train_dropped_for_leakage,
        "train_range": [str(t_train.min()), str(t_train.max())],
        "test_range": [str(t_test.min()), str(t_test.max())],
        "train_positive_rate": train_pos,
        "test_positive_rate": test_pos,
        "model": "sklearn.ensemble.HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "model_iterations_used": int(model.n_iter_),
        "threshold": THRESHOLD,
        "metrics": metrics,
        "permutation_importance": pi_list,
        "permutation_importance_params": {"n_repeats": 10, "random_state": SEED, "scoring": "f1",
                                          "n_jobs": 4, "seconds": pi_seconds},
        "replay_days": replay_days,
        "replay_daily_max_vpd_median_2025": med,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "seed": SEED,
        "trained_at_utc": trained_at,
        "train_seconds": train_seconds,
        "self_checks": checks,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(out), f, indent=2)
    log(f"saved {METRICS_PATH}")

    # 6. figures -------------------------------------------------------------------
    fig_f1(metrics, len(X_test), os.path.join(RESULTS_DIR, "F1_model_vs_baselines.png"))
    fig_f2(metrics, len(X_test), os.path.join(RESULTS_DIR, "F2_confusion_2025.png"))
    fig_f3(pi_list, metrics["model"]["f1"], metrics["persistence"]["f1"],
           os.path.join(RESULTS_DIR, "F3_feature_importance.png"))
    fig_f4(replay_days, metrics["persistence"]["f1"], metrics["model"]["f1"],
           os.path.join(RESULTS_DIR, "F4_replay_3days.png"))
    log("saved F1..F4 figures in results/")

    # table ------------------------------------------------------------------------
    log()
    log(f"SEMAI Forecast: {TEST_TITLE}, n={len(X_test):,} hours, positives={test_pos:.1%}")
    hdr = (f"{'method':<13}{'accuracy':>10}{'f1':>8}{'precision':>11}{'recall':>8}{'auc':>9}"
           f"{'tn':>7}{'fp':>6}{'fn':>6}{'tp':>6}")
    log(hdr)
    for nm in ("always_no", "persistence", "climatology", "model"):
        mm = metrics[nm]
        auc = "null" if mm["auc"] is None else f"{mm['auc']:.3f}"
        if mm["auc_kind"] == "binary":
            auc += "b"
        c = mm["confusion"]
        log(f"{nm:<13}{mm['accuracy']:>10.4f}{mm['f1']:>8.3f}{mm['precision']:>11.3f}{mm['recall']:>8.3f}"
            f"{auc:>9}{c['tn']:>7}{c['fp']:>6}{c['fn']:>6}{c['tp']:>6}")
    log("(auc 'b' = AUC of the binary prediction; climatology threshold "
        f"{best_thr:.3f} chosen on training data, train F1 {best_f1:.3f})")
    log(f"train {train_seconds:.1f} s; permutation importance {pi_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
