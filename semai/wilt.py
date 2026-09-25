"""SEMAI Wilt Watch inference (docs/SPEC.md §6 step 4.6).

Public API (exactly what the console uses):

    available() -> bool
        models/wilt.joblib exists and the feature backbone loads. Never raises.
    roi_box(h, w, roi=0.7) -> (x0, y0, x1, y1)
        Centred box covering `roi` of the width and `roi` of the height.
    predict(frame_bgr, roi=0.7) -> {"label": "HEALTHY"|"WILTED"|"UNSURE",
                                    "p_wilted": float, "ms": float}
        Crop to roi_box, BGR->RGB, the *same* preprocessing as training,
        backbone features, pipeline.predict_proba. UNSURE when 0.35 < p < 0.65.
        Raises RuntimeError("wilt model unavailable") when not available().

The backbone (frozen ImageNet MobileNetV3-Large, torchvision IMAGENET1K_V2,
classifier = Identity, 960-d) and the joblib bundle are lazy-loaded once and
cached in module globals. Torch is pinned to min(4, cpu_count) threads.

The underscore helpers (_pil_features, _load_extractor, _unavailable_reason, ...) are the single
implementation of preprocessing + feature extraction; scripts/train_wilt.py
imports them so training and inference are identical by construction.
The §3 no-torch fallback (HSV histogram 16 bins x 3 + HOG on 128x128 grey)
lives here too and is selected by the "feature_extractor" field of the joblib.
"""
from __future__ import annotations

import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "wilt.joblib")

# Names stored in wilt.joblib["feature_extractor"]
TORCH_EXTRACTOR = "mobilenet_v3_large IMAGENET1K_V2 classifier=Identity 960-d"
FALLBACK_EXTRACTOR = "fallback HSV-hist(16x3)+HOG(128x128 grey) no-torch"

INPUT_SIZE = 224
UNSURE_LOW, UNSURE_HIGH = 0.35, 0.65
ROI_DEFAULT = 0.7
CLASSES = ["healthy", "wilted"]

# ---- module-level caches --------------------------------------------------
_bundle: dict | None = None          # loaded joblib
_extractor: dict | None = None       # {"name", "fn": callable(list[PIL]) -> np.ndarray, ...}
_load_error: str | None = None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def roi_box(h: int, w: int, roi: float = ROI_DEFAULT) -> tuple[int, int, int, int]:
    """Centred box covering `roi` of the width and `roi` of the height (ints)."""
    roi = float(min(max(roi, 0.05), 1.0))
    bw = max(1, int(round(w * roi)))
    bh = max(1, int(round(h * roi)))
    x0 = (w - bw) // 2
    y0 = (h - bh) // 2
    return int(x0), int(y0), int(x0 + bw), int(y0 + bh)


# ---------------------------------------------------------------------------
# Feature extractors (shared with scripts/train_wilt.py)
# ---------------------------------------------------------------------------
def _torch_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _load_torch_extractor(tries: int = 3, sleep_s: float = 3.0) -> dict:
    """Load frozen MobileNetV3-Large (IMAGENET1K_V2) with classifier=Identity.

    Downloads the weights into the torch hub cache on first use (about 22 MB),
    retrying `tries` times. Raises on failure so the caller can fall back.
    """
    import torch
    import torchvision
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    torch.set_num_threads(_torch_threads())
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    last: Exception | None = None
    model = None
    for i in range(tries):
        try:
            model = mobilenet_v3_large(weights=weights)
            break
        except Exception as e:  # download / IO failure
            last = e
            if i < tries - 1:
                time.sleep(sleep_s * (i + 1))
    if model is None:
        raise RuntimeError(f"MobileNetV3 weights failed to load after {tries} tries: {last!r}")
    model.classifier = torch.nn.Identity()
    model.eval()
    # The weights' own preprocessing: resize 232 -> centre-crop 224 -> ImageNet norm.
    transform = weights.transforms()
    cache_file = os.path.join(torch.hub.get_dir(), "checkpoints",
                              os.path.basename(weights.url))

    def fn(pil_images: list, batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = torch.stack([transform(im) for im in pil_images[i:i + batch_size]])
                out.append(model(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 960), np.float32)

    return {
        "name": TORCH_EXTRACTOR,
        "fn": fn,
        "dim": 960,
        "cache_file": cache_file,
        "weights_url": weights.url,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "threads": torch.get_num_threads(),
        "transform": repr(transform),
    }


def _fallback_features_one(pil_img) -> np.ndarray:
    """§3 no-torch fallback: HSV histogram (16 bins x 3 channels) + HOG on 128x128 grey."""
    from skimage.feature import hog

    im = pil_img.convert("RGB").resize((128, 128))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    hist = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=16, range=(0, 256))
        hist.append(h.astype(np.float32) / max(1, hsv[..., c].size))
    grey = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    hg = hog(grey, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
             feature_vector=True).astype(np.float32)
    return np.concatenate([np.concatenate(hist), hg]).astype(np.float32)


def _load_fallback_extractor() -> dict:
    from PIL import Image  # noqa: F401  (import check)
    import skimage

    def fn(pil_images: list, batch_size: int = 32) -> np.ndarray:
        if not pil_images:
            return np.zeros((0, 0), np.float32)
        return np.stack([_fallback_features_one(im) for im in pil_images])

    return {"name": FALLBACK_EXTRACTOR, "fn": fn, "dim": None,
            "skimage_version": skimage.__version__, "threads": None}


def _load_extractor(name: str = TORCH_EXTRACTOR) -> dict:
    """Return the extractor dict for `name` (cached)."""
    global _extractor
    if _extractor is not None and _extractor["name"] == name:
        return _extractor
    if name == TORCH_EXTRACTOR:
        _extractor = _load_torch_extractor()
    elif name == FALLBACK_EXTRACTOR:
        _extractor = _load_fallback_extractor()
    else:
        raise ValueError(f"unknown feature extractor {name!r}")
    return _extractor


def _pil_features(pil_images: list, name: str = TORCH_EXTRACTOR, batch_size: int = 32) -> np.ndarray:
    """Features for a list of PIL RGB images using extractor `name`."""
    return _load_extractor(name)["fn"](pil_images, batch_size)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        import joblib
        b = joblib.load(MODEL_PATH)
        if "pipeline" not in b or "feature_extractor" not in b:
            raise ValueError("wilt.joblib is missing 'pipeline' or 'feature_extractor'")
        _bundle = b
    return _bundle


def _ensure_loaded() -> tuple[dict, dict]:
    """Load joblib + matching extractor. Raises on any failure."""
    b = _load_bundle()
    ex = _load_extractor(b["feature_extractor"])
    return b, ex


def available() -> bool:
    """True when models/wilt.joblib exists and the backbone loads. Never raises."""
    global _load_error
    try:
        if not os.path.isfile(MODEL_PATH):
            _load_error = "models/wilt.joblib not found"
            return False
        _ensure_loaded()
        _load_error = None
        return True
    except Exception as e:  # pragma: no cover - defensive by contract
        _load_error = f"{type(e).__name__}: {e}"
        return False


def _unavailable_reason() -> str | None:
    """Why available() last returned False (None when available)."""
    return _load_error


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(frame_bgr: np.ndarray, roi: float = ROI_DEFAULT) -> dict:
    """Classify one BGR frame (H, W, 3 uint8) -> label / p_wilted / ms."""
    t0 = time.perf_counter()
    if not available():
        raise RuntimeError("wilt model unavailable")
    bundle, ex = _ensure_loaded()

    from PIL import Image

    frame = np.asarray(frame_bgr)
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame_bgr must be a uint8 ndarray (H, W, 3)")
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame_bgr must be an HxWx3 BGR array")
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi_box(h, w, roi)
    crop = frame[y0:y1, x0:x1, :3]
    rgb = np.ascontiguousarray(crop[..., ::-1])       # BGR -> RGB
    pil = Image.fromarray(rgb, mode="RGB")            # same as training: PIL RGB
    feats = ex["fn"]([pil], 1)
    p = float(bundle["pipeline"].predict_proba(feats)[0, 1])
    lo, hi = bundle.get("unsure_band", [UNSURE_LOW, UNSURE_HIGH])
    if lo < p < hi:
        label = "UNSURE"
    elif p >= hi:
        label = "WILTED"
    else:
        label = "HEALTHY"
    ms = (time.perf_counter() - t0) * 1000.0
    return {"label": label, "p_wilted": p, "ms": float(ms)}


if __name__ == "__main__":  # tiny self-check
    assert roi_box(600, 800, 0.7) == (120, 90, 680, 510), roi_box(600, 800, 0.7)
    assert roi_box(10, 10, 1.0) == (0, 0, 10, 10)
    print("available:", available(), _unavailable_reason())
    if available():
        rng = np.random.default_rng(0)
        fr = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
        print(predict(fr))
    print("wilt.py self-check OK")
