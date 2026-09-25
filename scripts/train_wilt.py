"""SEMAI Wilt Watch training (docs/SPEC.md §6 step 4).

Pipeline
  2. frozen MobileNetV3-Large (IMAGENET1K_V2, classifier=Identity, 960-d) features
     for the 904 Kaggle houseplant photos, cached in data/wilt_features.npz
     (§3 no-torch fallback: HSV hist 16x3 + HOG 128x128 grey, same head)
  3. near-duplicate grouping: cosine similarity > 0.97 on L2-normalised features,
     union-find -> group ids
  4. StandardScaler -> LogisticRegression(C=0.1, max_iter=3000)
     headline: StratifiedGroupKFold(5, shuffle, seed 0) with duplicate groups
     also:     naive StratifiedKFold(5, shuffle, seed 0)  (why grouping matters)
     baseline: 50 % (balanced set, always-one-class)
  5. final fit on all 904 -> models/wilt.joblib
  6. timing of semai.wilt.predict (target < 300 ms / frame on CPU)
  8. figures R1_wilt_cv_grouped_vs_naive.png, R2_wilt_confusion.png (no photos)
     metrics results/wilt_metrics.json

Run:  python scripts/train_wilt.py
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

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 0
DATA_DIR = os.path.join(ROOT, "data", "wilt", "houseplant_images")
FEATURE_CACHE = os.path.join(ROOT, "data", "wilt_features.npz")
MODEL_PATH = os.path.join(ROOT, "models", "wilt.joblib")
RESULTS = os.path.join(ROOT, "results")
METRICS_JSON = os.path.join(RESULTS, "wilt_metrics.json")
FIG_R1 = os.path.join(RESULTS, "R1_wilt_cv_grouped_vs_naive.png")
FIG_R2 = os.path.join(RESULTS, "R2_wilt_confusion.png")
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/russellchan/healthy-and-wilted-houseplant-images"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/russellchan/healthy-and-wilted-houseplant-images"
DUP_THRESHOLD = 0.97
BASELINE_ACC = 0.5  # balanced 452/452: always-one-class

CLASS_DIRS = (("healthy", 0), ("wilted", 1))
IMG_EXT = (".jpg", ".jpeg", ".png")

decisions: list[str] = []
fallbacks: list[str] = []
self_checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    self_checks.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {detail}")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. file list
# ---------------------------------------------------------------------------
def list_images() -> tuple[list[str], np.ndarray]:
    files, labels = [], []
    for sub, lab in CLASS_DIRS:
        d = os.path.join(DATA_DIR, sub)
        names = sorted(n for n in os.listdir(d) if n.lower().endswith(IMG_EXT))
        files += [f"{sub}/{n}" for n in names]
        labels += [lab] * len(names)
    return files, np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. features (cached)
# ---------------------------------------------------------------------------
def extract_features(files: list[str], labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (X, info). Uses data/wilt_features.npz when it matches the file list."""
    from PIL import Image

    import semai.wilt as W

    # choose extractor: torch first, §3 fallback after 3 failed tries
    extractor = None
    try:
        extractor = W._load_extractor(W.TORCH_EXTRACTOR)
    except Exception as e:
        msg = (f"torch/torchvision backbone unavailable ({type(e).__name__}: {e}); "
               f"using SPEC §3 no-torch fallback (HSV hist 16x3 + HOG 128x128)")
        print("FALLBACK:", msg)
        fallbacks.append(msg)
        extractor = W._load_extractor(W.FALLBACK_EXTRACTOR)
    name = extractor["name"]

    if os.path.isfile(FEATURE_CACHE):
        try:
            z = np.load(FEATURE_CACHE, allow_pickle=False)
            if (str(z["extractor"]) == name and list(z["files"]) == files
                    and np.array_equal(z["labels"], labels)):
                X = z["X"].astype(np.float32)
                info = json.loads(str(z["info"]))
                info["cache_hit"] = True
                print(f"feature cache hit: {FEATURE_CACHE} {X.shape}")
                return X, info
            print("feature cache present but stale; re-extracting")
        except Exception as e:  # corrupt cache -> re-extract
            print("feature cache unreadable, re-extracting:", e)

    print(f"extracting {len(files)} images with {name} ...")
    t0 = time.perf_counter()
    X_parts = []
    bs = 32
    for i in range(0, len(files), bs):
        ims = [Image.open(os.path.join(DATA_DIR, f)).convert("RGB") for f in files[i:i + bs]]
        X_parts.append(extractor["fn"](ims, bs))
        for im in ims:
            im.close()
        if (i // bs) % 5 == 0:
            print(f"  {min(i + bs, len(files))}/{len(files)}  {time.perf_counter() - t0:.1f}s")
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    secs = time.perf_counter() - t0
    info = {
        "extractor": name,
        "feature_dim": int(X.shape[1]),
        "extraction_seconds": round(secs, 2),
        "torch_num_threads": extractor.get("threads"),
        "batch_size": bs,
        "weights_cache_file": extractor.get("cache_file"),
        "weights_url": extractor.get("weights_url"),
        "preprocess": extractor.get("transform"),
        "extracted_at_utc": now_utc(),
        "cache_hit": False,
    }
    np.savez_compressed(FEATURE_CACHE, X=X, labels=labels, files=np.asarray(files),
                        extractor=np.asarray(name), info=np.asarray(json.dumps(info)))
    print(f"features {X.shape} in {secs:.1f}s -> {FEATURE_CACHE}")
    return X, info


# ---------------------------------------------------------------------------
# 3. near-duplicate grouping
# ---------------------------------------------------------------------------
def group_duplicates(X: np.ndarray, files: list[str], labels: np.ndarray) -> tuple[np.ndarray, dict]:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    n = len(files)
    iu, ju = np.triu_indices(n, k=1)
    mask = S[iu, ju] > DUP_THRESHOLD
    pi, pj = iu[mask], ju[mask]

    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in zip(pi, pj):
        union(int(a), int(b))
    roots = np.asarray([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    sizes = np.bincount(groups)

    pairs = [{"a": files[int(a)], "b": files[int(b)], "similarity": round(float(S[a, b]), 4)}
             for a, b in zip(pi, pj)]
    mism = [p for p, a, b in zip(pairs, pi, pj) if labels[a] != labels[b]]
    involved = np.unique(np.concatenate([pi, pj])) if len(pi) else np.asarray([], dtype=int)
    info = {
        "threshold": DUP_THRESHOLD,
        "n_pairs": int(len(pairs)),
        "n_images_involved": int(len(involved)),
        "n_groups": int(len(sizes)),
        "n_groups_2plus": int((sizes >= 2).sum()),
        "largest_group_size": int(sizes.max()),
        "n_label_mismatch_pairs": int(len(mism)),
        "label_mismatch_pairs": mism,
        "pairs": pairs,
    }
    return groups, info


# ---------------------------------------------------------------------------
# 4. cross-validation
# ---------------------------------------------------------------------------
def make_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(C=0.1, max_iter=3000))])


def run_cv(X, y, splits, name: str) -> dict:
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    n = len(y)
    oof_p = np.full(n, np.nan)
    oof_pred = np.full(n, -1)
    fold_acc, fold_sizes = [], []
    for k, (tr, te) in enumerate(splits):
        pipe = make_pipeline()
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        oof_p[te] = p
        oof_pred[te] = pred
        acc = float((pred == y[te]).mean())
        fold_acc.append(acc)
        fold_sizes.append(int(len(te)))
        print(f"  {name} fold {k}: n_test={len(te)} acc={acc:.4f}")
    assert not np.isnan(oof_p).any() and (oof_pred >= 0).all(), "every image must be OOF-predicted once"
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "fold_accuracy": [round(a, 6) for a in fold_acc],
        "fold_test_sizes": fold_sizes,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "pooled_accuracy": float((oof_pred == y).mean()),
        "precision": float(precision_score(y, oof_pred, pos_label=1)),
        "recall": float(recall_score(y, oof_pred, pos_label=1)),
        "f1": float(f1_score(y, oof_pred, pos_label=1)),
        "auc": float(roc_auc_score(y, oof_p)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_class": "wilted",
        "baseline_accuracy": BASELINE_ACC,
    }


# ---------------------------------------------------------------------------
# 8. figures
# ---------------------------------------------------------------------------
def fig_r1(grouped: dict, naive: dict) -> None:
    g = np.asarray(grouped["fold_accuracy"]) * 100
    nv = np.asarray(naive["fold_accuracy"]) * 100
    k = len(g)
    x = np.arange(k + 1)
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x[:k] - w / 2, g, w, color="#2a7f62", label="grouped 5-fold CV (near-duplicates kept together) - headline")
    ax.bar(x[:k] + w / 2, nv, w, color="#c96f2b", label="naive 5-fold CV (duplicates may leak across folds)")
    ax.bar(x[k] - w / 2, g.mean(), w, color="#2a7f62", hatch="//", edgecolor="black",
           yerr=g.std(), capsize=5, label=f"grouped mean {g.mean():.1f} % +/- {g.std():.1f} (std)")
    ax.bar(x[k] + w / 2, nv.mean(), w, color="#c96f2b", hatch="//", edgecolor="black",
           yerr=nv.std(), capsize=5, label=f"naive mean {nv.mean():.1f} % +/- {nv.std():.1f} (std)")
    ax.axhline(BASELINE_ACC * 100, ls="--", color="black", lw=1.2,
               label=f"baseline {BASELINE_ACC * 100:.0f} % (always one class, 452/452 balanced)")
    for xi, v in zip(x[:k] - w / 2, g):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)
    for xi, v in zip(x[:k] + w / 2, nv):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i + 1}" for i in range(k)] + ["mean"])
    ax.set_xlabel("cross-validation fold")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SEMAI Wilt Watch: 5-fold CV accuracy, grouped vs naive, on 904 Kaggle photos")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_R1, dpi=200)
    plt.close(fig)


def fig_r2(grouped: dict) -> None:
    c = grouped["confusion"]
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    row_pct = m / m.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(row_pct, cmap="Greens", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            col = "white" if row_pct[i, j] > 60 else "black"
            ax.text(j, i, f"{int(m[i, j])}\n({row_pct[i, j]:.1f} % of row)",
                    ha="center", va="center", fontsize=12, color=col)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["predicted HEALTHY", "predicted WILTED"])
    ax.set_yticklabels(["actual HEALTHY", "actual WILTED"])
    ax.set_xlabel("model prediction (class)")
    ax.set_ylabel("true label (class)")
    acc = grouped["pooled_accuracy"] * 100
    ax.set_title(f"Wilt Watch pooled confusion matrix, grouped 5-fold CV (904 photos)\n"
                 f"accuracy {acc:.1f} % vs {BASELINE_ACC * 100:.0f} % baseline (always one class)\n"
                 f"F1 {grouped['f1']:.3f}, recall {grouped['recall']:.3f}, precision {grouped['precision']:.3f}",
                 fontsize=10, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label("share of row (%)")
    fig.tight_layout(pad=1.2)
    fig.savefig(FIG_R2, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. timing / smoke test of semai.wilt.predict
# ---------------------------------------------------------------------------
def time_predict(files: list[str], labels: np.ndarray) -> dict:
    import cv2

    import semai.wilt as W

    # fresh process-level state so we exercise lazy loading of the saved joblib
    W._bundle = None
    ok = W.available()
    check("semai.wilt.available() after training", ok, str(W._unavailable_reason()))
    rng = np.random.default_rng(SEED)
    warm = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    r0 = W.predict(warm)  # warm-up (lazy load + first forward pass)
    times = []
    for _ in range(20):
        fr = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        W.predict(fr)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.asarray(times)
    # smoke test on 6 training images (NOT an accuracy claim: these are training data)
    hidx = [i for i, l in enumerate(labels) if l == 0][:3]
    widx = [i for i, l in enumerate(labels) if l == 1][:3]
    smoke = []
    for i in hidx + widx:
        fr = cv2.imread(os.path.join(DATA_DIR, files[i]), cv2.IMREAD_COLOR)
        r = W.predict(fr)
        # plumbing check only (training images): print the label, keep just the timing in the JSON
        print(f"  smoke {files[i]}: true={'wilted' if labels[i] else 'healthy'} -> {r['label']} p={r['p_wilted']:.3f}")
        smoke.append({"file": files[i], "ms": round(r["ms"], 1)})
    return {
        "frame_size": "800x600 BGR synthetic (uniform noise, seed 0), roi 0.7",
        "warmup_ms": round(r0["ms"], 1),
        "n_frames": int(len(times)),
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "max_ms": float(times.max()),
        "min_ms": float(times.min()),
        "target_ms": 300,
        "meets_target": bool(times.max() < 300),
        "torch_num_threads_at_inference": W._extractor.get("threads") if W._extractor else None,
        "real_photo_timing_note": ("6 dataset images (3 healthy, 3 wilted) loaded with cv2.imread; "
                                   "these are TRAINING images, so only the per-frame time is kept here "
                                   "(labels are printed to stdout and are not an accuracy measurement)"),
        "real_photo_timing_ms": smoke,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    import joblib
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    np.random.seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "rig_frames", "healthy"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "rig_frames", "wilted"), exist_ok=True)

    t_start = time.perf_counter()
    files, y = list_images()
    n_h, n_w = int((y == 0).sum()), int((y == 1).sum())
    check("dataset has 904 images (452/452)", len(files) == 904 and n_h == 452 and n_w == 452,
          f"n={len(files)} healthy={n_h} wilted={n_w}")

    X, feat_info = extract_features(files, y)
    check("feature matrix finite", bool(np.isfinite(X).all()), f"shape={X.shape}")
    if feat_info["extractor"].startswith("mobilenet"):
        check("feature dim 960", X.shape[1] == 960, f"dim={X.shape[1]}")
        cf = feat_info.get("weights_cache_file")
        check("MobileNet weights cached in torch hub", bool(cf and os.path.isfile(cf)), str(cf))

    groups, dup = group_duplicates(X, files, y)
    print(f"duplicates: {dup['n_pairs']} pairs over {dup['n_images_involved']} images, "
          f"{dup['n_groups']} groups ({dup['n_groups_2plus']} with 2+), largest {dup['largest_group_size']}, "
          f"{dup['n_label_mismatch_pairs']} label mismatches")
    check("no label-mismatched duplicate pairs", dup["n_label_mismatch_pairs"] == 0,
          f"{dup['n_label_mismatch_pairs']} mismatches")

    # --- grouped CV (headline) with the leak assertion
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    g_splits = list(sgkf.split(X, y, groups))
    leaks = 0
    for tr, te in g_splits:
        overlap = set(groups[tr].tolist()) & set(groups[te].tolist())
        leaks += len(overlap)
        assert not overlap, f"duplicate group(s) {sorted(overlap)[:5]} appear in both train and test"
    check("grouped CV: no duplicate group in both train and test of any fold", leaks == 0,
          f"{len(g_splits)} folds checked")
    print("grouped CV:")
    grouped = run_cv(X, y, g_splits, "grouped")

    # --- naive CV (for contrast)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_splits = list(skf.split(X, y))
    naive_leaks = 0
    for tr, te in n_splits:
        naive_leaks += len(set(groups[tr].tolist()) & set(groups[te].tolist()))
    print("naive CV:")
    naive = run_cv(X, y, n_splits, "naive")
    naive["n_duplicate_groups_split_across_train_test"] = int(naive_leaks)
    grouped["n_duplicate_groups_split_across_train_test"] = 0
    print(f"grouped mean acc {grouped['mean_accuracy']:.4f}  naive mean acc {naive['mean_accuracy']:.4f}  "
          f"(naive leaks {naive_leaks} groups across folds)")
    check("grouped CV beats 50 % baseline", grouped["mean_accuracy"] > BASELINE_ACC,
          f"{grouped['mean_accuracy']:.4f}")

    # --- final model on all 904
    final = make_pipeline()
    final.fit(X, y)
    train_acc_note = "not reported: SPEC §2 rule 2 forbids accuracy on training data"
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        pass
    bundle = {
        "pipeline": final,
        "feature_extractor": feat_info["extractor"],
        "input_size": 224,
        "classes": ["healthy", "wilted"],
        "positive": "wilted",
        "unsure_band": [0.35, 0.65],
        "seed": SEED,
        "roi_default": 0.7,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch_version,
        "feature_dim": int(X.shape[1]),
        "trained_at_utc": now_utc(),
        "n_train_images": int(len(files)),
    }
    joblib.dump(bundle, MODEL_PATH)
    check("models/wilt.joblib written", os.path.isfile(MODEL_PATH),
          f"{os.path.getsize(MODEL_PATH)} bytes")

    # --- timing + smoke test through semai.wilt.predict (loads the joblib we just saved)
    timing = time_predict(files, y)
    print(f"predict timing: mean {timing['mean_ms']:.1f} ms, median {timing['median_ms']:.1f} ms, "
          f"max {timing['max_ms']:.1f} ms (target < 300 ms)")
    check("predict < 300 ms per frame (max of 20)", timing["meets_target"],
          f"max {timing['max_ms']:.1f} ms, mean {timing['mean_ms']:.1f} ms")
    print("real-photo timing (ms):", [s["ms"] for s in timing["real_photo_timing_ms"]])

    # --- figures
    fig_r1(grouped, naive)
    fig_r2(grouped)
    check("R1/R2 figures written", os.path.isfile(FIG_R1) and os.path.isfile(FIG_R2))
    bad = [f for f in os.listdir(RESULTS) if f.lower().endswith((".jpg", ".jpeg"))]
    assert not bad, f"results/ must not contain dataset photos: {bad}"
    check("results/ holds no .jpg/.jpeg", not bad, f"{len(os.listdir(RESULTS))} files scanned")

    # --- metrics JSON
    try:
        import torchvision
        tv_version = torchvision.__version__
    except Exception:
        tv_version = None
    metrics = {
        "part": "SEMAI Wilt Watch",
        "trained_at_utc": now_utc(),
        "seed": SEED,
        "dataset": {
            "name": "Healthy and Wilted Houseplant Images (Kaggle, russellchan)",
            "download_url": KAGGLE_URL,
            "page_url": KAGGLE_PAGE,
            "n_images": int(len(files)),
            "n_healthy": n_h,
            "n_wilted": n_w,
            "licence": "see the Kaggle dataset page",
            "usage": "cite-only: no dataset image is copied into results/ (SPEC §2 rule 7)",
            "local_path": os.path.relpath(DATA_DIR, ROOT).replace("\\", "/"),
        },
        "features": feat_info,
        "duplicates": dup,
        "model": {
            "pipeline": "StandardScaler() -> LogisticRegression(C=0.1, max_iter=3000)",
            "threshold": 0.5,
            "unsure_band": [0.35, 0.65],
            "roi_default": 0.7,
            "input_size": 224,
            "path": "models/wilt.joblib",
            "training_accuracy": train_acc_note,
        },
        "baseline": {"name": "always one class (balanced 452/452)", "accuracy": BASELINE_ACC},
        "cv_grouped": {"scheme": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0), "
                                 "groups = near-duplicate group ids", "headline": True, **grouped},
        "cv_naive": {"scheme": "StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
                     "headline": False, **naive},
        "predict_timing_ms": timing,
        "rig_eval": "see results/wilt_rig_eval.json (scripts/eval_rig_frames.py)",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch_version,
            "torchvision": tv_version,
            "platform": platform.platform(),
        },
        "extraction_seconds": feat_info.get("extraction_seconds"),
        "figures": [os.path.basename(FIG_R1), os.path.basename(FIG_R2)],
        "self_checks": self_checks,
        "decisions": decisions or ["see DECISIONS.md, Step 4 (recorded by the orchestrator from the builder report)"],
        "fallbacks": fallbacks,
        "total_script_seconds": round(time.perf_counter() - t_start, 1),
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {METRICS_JSON}")
    print(f"HEADLINE grouped CV acc {grouped['mean_accuracy'] * 100:.1f} % "
          f"(naive {naive['mean_accuracy'] * 100:.1f} %, baseline 50 %), "
          f"F1 {grouped['f1']:.3f}, AUC {grouped['auc']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
