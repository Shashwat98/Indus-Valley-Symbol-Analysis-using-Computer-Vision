import os
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ------------------ config ------------------ #

INPUT_DIR = Path("data/tablets_raw")
OUT_BASE  = Path("data/preprocessed")

MAX_SIZE  = 768        # max dimension after resizing
BORDER_THRESH = 240    # for cropping white margin
MIN_SYMBOL_AREA = 80   # tweak depending on resolution

# -------------------------------------------- #

def ensure_dirs():
    for sub in ["gray", "clahe", "binary", "clean", "skeleton", "symbols"]:
        (OUT_BASE / sub).mkdir(parents=True, exist_ok=True)


def crop_border(gray, thresh=BORDER_THRESH, margin=5):
    """
    Crop away the light border (PDF background).
    Works well for your example images where the margin is very bright.
    """
    mask = gray < thresh  # True where tablet is present
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        # nothing darker than thresh, just return original
        return gray

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, gray.shape[0])
    x1 = min(x1 + margin, gray.shape[1])

    return gray[y0:y1, x0:x1]


def resize_max(img, max_size=MAX_SIZE):
    h, w = img.shape[:2]
    scale = max(h, w) / max_size
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_adaptive(gray_eq):
    # Adaptive threshold, inverted so carvings = white
    bin_img = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,   # blockSize (odd)
        5     # C (tweakable)
    )
    return bin_img


def clean_binary(bin_img, min_area=MIN_SYMBOL_AREA):
    """
    Morphological closing + remove small components (noise).
    """
    # small closing to close gaps in strokes
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # remove tiny components via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)

    for i in range(1, num_labels):  # 0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def make_skeleton(clean_bin):
    """
    Convert binary (255/0) to skeleton (255/0).
    """
    bool_img = clean_bin > 0
    skel_bool = skeletonize(bool_img)
    skel = (skel_bool.astype(np.uint8) * 255)
    return skel


def extract_symbol_crops(clean_bin, base_name):
    """
    Use connected components to get candidate symbol crops.
    Saves to OUT_BASE / "symbols".
    """
    out_dir = OUT_BASE / "symbols"
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)

    symbol_idx = 0
    for i in range(1, num_labels):  # skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_SYMBOL_AREA:
            continue

        crop = clean_bin[y:y+h, x:x+w]

        # optional: ignore components that are almost the whole image (= the tablet outline)
        if w * h > 0.8 * clean_bin.shape[0] * clean_bin.shape[1]:
            continue

        symbol_name = f"{base_name}_sym{symbol_idx:03d}.png"
        cv2.imwrite(str(out_dir / symbol_name), crop)
        symbol_idx += 1


def preprocess_one(path: Path):
    base_name = path.stem
    print(f"Processing {base_name}...")

    # 1. load & grayscale
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] Could not read {path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. crop white border
    gray_cropped = crop_border(gray)

    # 3. resize
    gray_resized = resize_max(gray_cropped)

    # 4. CLAHE
    gray_clahe = apply_clahe(gray_resized)

    # 5. binarize
    bin_img = binarize_adaptive(gray_clahe)

    # 6. clean binary
    clean_bin = clean_binary(bin_img)

    # 7. skeleton
    skel = make_skeleton(clean_bin)

    # 8. save all intermediates
    cv2.imwrite(str(OUT_BASE / "gray" / f"{base_name}_gray.png"), gray_resized)
    cv2.imwrite(str(OUT_BASE / "clahe" / f"{base_name}_clahe.png"), gray_clahe)
    cv2.imwrite(str(OUT_BASE / "binary" / f"{base_name}_binary.png"), bin_img)
    cv2.imwrite(str(OUT_BASE / "clean" / f"{base_name}_clean.png"), clean_bin)
    cv2.imwrite(str(OUT_BASE / "skeleton" / f"{base_name}_skeleton.png"), skel)

    # 9. extract symbol crops
    extract_symbol_crops(clean_bin, base_name)


def main():
    ensure_dirs()
    images = sorted(list(INPUT_DIR.glob("*.png"))) + \
             sorted(list(INPUT_DIR.glob("*.jpg"))) + \
             sorted(list(INPUT_DIR.glob("*.jpeg")))

    print(f"Found {len(images)} images.")
    for p in images:
        preprocess_one(p)


if __name__ == "__main__":
    main()
