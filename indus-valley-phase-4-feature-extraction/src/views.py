# src/views.py
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .config import SEALS_PHASE1_DIR, PREPROC_SUFFIXES, PREPROC_DIRS


def _safe_read_image(path: Path, flags=cv2.IMREAD_GRAYSCALE) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return cv2.imread(str(path), flags)


def seal_image_path(seal_id: str, view_name: str) -> Path:
    """
    Expect files like:
      data/seals_phase1/<view_name>/<seal_id><suffix>

    Example:
      data/seals_phase1/binary/page001_M-1_S_c0_binary.png
    """
    subdir = PREPROC_DIRS[view_name]
    suffix = PREPROC_SUFFIXES[view_name]
    filename = f"{seal_id}{suffix}"
    return SEALS_PHASE1_DIR / subdir / filename


def crop_bbox(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop image using [x1, y1, x2, y2] in (col, row) coordinates, clamped to bounds.
    """
    h, w = img.shape[:2]
    x1c = max(0, min(w - 1, x1))
    x2c = max(0, min(w, x2))
    y1c = max(0, min(h - 1, y1))
    y2c = max(0, min(h, y2))

    if x2c <= x1c or y2c <= y1c:
        # degenerate bbox, return a 1x1 black patch
        return np.zeros((1, 1), dtype=img.dtype)

    return img[y1c:y2c, x1c:x2c]


def resize_to_square(img: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Resize crop to a square canvas (size x size) while preserving aspect ratio
    and padding with zeros (black).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img.dtype)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=resized.dtype)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2

    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def make_symbol_views_from_seal(
    seal_id: str,
    bbox_orig: Dict[str, int],
    target_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    For a given symbol (identified by seal_id + bbox_orig),
    load all Phase-1 preprocessed seal images and crop the same bounding box.

    bbox_orig should be a dict with keys: x1, y1, x2, y2 in original seal coordinates.

    Returns a dict:
    {
      "gray": gray_view,
      "clahe": clahe_view,
      "binary": binary_view,
      "clean": clean_view,
      "skeleton": skeleton_view,
      "edge": edge_view,
    }
    with each view as a size x size np.ndarray (uint8).
    """
    x1 = int(bbox_orig["x1"])
    y1 = int(bbox_orig["y1"])
    x2 = int(bbox_orig["x2"])
    y2 = int(bbox_orig["y2"])

    views: Dict[str, np.ndarray] = {}

    for view_name in PREPROC_SUFFIXES.keys():
        seal_path = seal_image_path(seal_id, view_name)
        img = _safe_read_image(seal_path, flags=cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Missing file: return a black patch for this view
            views[view_name] = np.zeros((target_size, target_size), dtype=np.uint8)
            continue

        crop = crop_bbox(img, x1, y1, x2, y2)
        crop_sq = resize_to_square(crop, size=target_size)
        views[view_name] = crop_sq

    return views
