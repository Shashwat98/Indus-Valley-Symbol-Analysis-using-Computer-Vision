import os
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

# ------------------ config ------------------ #

INPUT_DIR = Path("data/tablets_raw")
OUT_BASE  = Path("data/preprocessed")

MAX_SIZE        = 768        # max dimension after resizing
BORDER_THRESH   = 240        # for cropping white margin

# FIX 4: separate thresholds for "cleaning" vs "symbol crops"
MIN_CLEAN_AREA  = 150        # remove small specks when cleaning edge-map
MIN_SYMBOL_AREA = 80         # keep this smaller for symbol crops

USE_CLAHE       = True

# -------------------------------------------- #

def ensure_dirs():
    """Create all output subfolders."""
    for sub in ["gray", "clahe", "edge", "binary", "clean",
                "skeleton", "symbols"]:
        (OUT_BASE / sub).mkdir(parents=True, exist_ok=True)


def crop_border(gray, thresh=BORDER_THRESH, margin=5):
    """
    Crop away light border (page background) around the tablet.
    Assumes background is much lighter than seal.
    """
    mask = gray < thresh
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        # nothing darker than thresh; just return original
        return gray

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, gray.shape[0])
    x1 = min(x1 + margin, gray.shape[1])

    return gray[y0:y1, x0:x1]


def resize_max(img, max_size=MAX_SIZE):
    """Resize so that max(width, height) == max_size (or smaller)."""
    h, w = img.shape[:2]
    scale = max(h, w) / max_size
    if scale <= 1.0:
        return img
    new_w = int(w / scale)
    new_h = int(h / scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe(gray):
    """Local contrast enhancement (mild)."""
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    return clahe.apply(gray)


def edge_based_binary(gray_proc):
    """
    Main edge-based binarization step.

    1. Strong denoising (fast non-local means)
    2. Edge detection (Scharr x + y)
    3. Normalize edge magnitude
    4. Otsu threshold on edge image
    """

    # FIX 1: slightly stronger denoising to kill more stone texture
    denoised = cv2.fastNlMeansDenoising(
        gray_proc, None,
        h=22,                 # was 15
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. Scharr edges (more sensitive than Sobel)
    edge_x = cv2.Scharr(denoised, cv2.CV_64F, 1, 0)
    edge_y = cv2.Scharr(denoised, cv2.CV_64F, 0, 1)

    mag = np.abs(edge_x) + np.abs(edge_y)

    # 3. Normalize to [0, 255]
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)

    # FIX 2: light blur on edge magnitude before Otsu
    mag_blur = cv2.GaussianBlur(mag, (3, 3), 0)

    # 4. Otsu threshold on edges
    _, edge_bin = cv2.threshold(
        mag_blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mag, edge_bin


def clean_binary(bin_img, min_area=MIN_CLEAN_AREA):
    """
    Clean binary edge image.
    - Opening removes isolated speckles
    - Closing reconnects strokes
    - Remove small connected components
    """
    kernel = np.ones((3, 3), np.uint8)

    # FIX 3: slightly more aggressive morphology
    # Remove tiny noise
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Close gaps in strokes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extra light opening to kill small dots introduced by closing
    clean_morph = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small CCs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean_morph, connectivity=8
    )
    cleaned = np.zeros_like(clean_morph)

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def make_skeleton(clean_bin):
    """Convert cleaned binary image to 1-pixel-wide skeleton."""
    bool_img = clean_bin > 0
    skel_bool = skeletonize(bool_img)
    return (skel_bool.astype(np.uint8) * 255)


def extract_symbol_crops(clean_bin, base_name):
    """
    Extract candidate symbol crops using connected components.
    Saves them in OUT_BASE / 'symbols'.
    This works on the edge-map; you’ll get glyph fragments, animal parts, etc.
    """
    out_dir = OUT_BASE / "symbols"
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean_bin, connectivity=8
    )

    symbol_idx = 0
    H, W = clean_bin.shape[:2]

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # filter out tiny bits again – use MIN_SYMBOL_AREA here
        if area < MIN_SYMBOL_AREA:
            continue

        # skip extremely large CC that is basically the frame
        if w * h > 0.85 * H * W:
            continue

        crop = clean_bin[y:y + h, x:x + w]
        symbol_name = f"{base_name}_sym{symbol_idx:03d}.png"
        cv2.imwrite(str(out_dir / symbol_name), crop)
        symbol_idx += 1


def preprocess_one(path: Path):
    """Run full edge-based pipeline on one tablet image."""
    base_name = path.stem
    print(f"Processing {base_name}...")

    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARN] Could not read {path}")
        return

    # 1. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. crop white border
    gray_cropped = crop_border(gray)

    # 3. resize
    gray_resized = resize_max(gray_cropped)

    # 4. optional CLAHE or just use resized
    if USE_CLAHE:
        gray_proc = apply_clahe(gray_resized)
    else:
        gray_proc = gray_resized

    # 5–8. edge-based binary
    edge_mag, edge_bin = edge_based_binary(gray_proc)

    # 9. clean
    clean_bin = clean_binary(edge_bin)

    # 10. skeleton
    skel = make_skeleton(clean_bin)

    # 11. save intermediates
    cv2.imwrite(str(OUT_BASE / "gray"   / f"{base_name}_gray.png"), gray_resized)
    cv2.imwrite(str(OUT_BASE / "clahe"  / f"{base_name}_clahe.png"), gray_proc)
    cv2.imwrite(str(OUT_BASE / "edge"   / f"{base_name}_edge.png"), edge_mag)
    cv2.imwrite(str(OUT_BASE / "binary" / f"{base_name}_binary.png"), edge_bin)
    cv2.imwrite(str(OUT_BASE / "clean"  / f"{base_name}_clean.png"), clean_bin)
    cv2.imwrite(str(OUT_BASE / "skeleton" / f"{base_name}_skeleton.png"), skel)

    # 12. symbol crops
    extract_symbol_crops(clean_bin, base_name)


def run_all():
    """Find all images in INPUT_DIR and preprocess them."""
    print("Current working directory:", os.getcwd())
    print("INPUT_DIR:", INPUT_DIR)
    print("Exists?", INPUT_DIR.exists())

    ensure_dirs()

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    images = []
    for pat in patterns:
        images.extend(list(INPUT_DIR.glob(pat)))

    print(f"Found {len(images)} images in {INPUT_DIR}")

    for p in tqdm(sorted(images)):
        preprocess_one(p)


if __name__ == "__main__":
    run_all()
