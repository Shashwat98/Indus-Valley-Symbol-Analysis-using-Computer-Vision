import cv2
import json
import numpy as np
from pathlib import Path

# Band detection hyperparameters
ROW_SUM_THRESHOLD_FRAC = 0.15  # fraction of max row stroke count to call a row "active"
TEXT_MAX_CENTER_Y_FRAC = 0.6   # text band center should be in top 60% of image
TEXT_MAX_HEIGHT_FRAC = 0.4     # text band height <= 40% of image
ANIMAL_MARGIN_FRAC = 0.05      # small gap between text band and animal region

# Symbol grouping hyperparameters
GAP_FRAC = 0.6                # max horizontal gap as fraction of avg component width
MIN_VERTICAL_OVERLAP_FRAC = 0.3  # min vertical overlap between components to merge

# Normalization of symbol crops
TARGET_SIZE = 64              # final symbol image size (64x64)
PADDING_FRAC = 0.1            # 10% padding inside the square canvas


# =========================
# CONFIG
# =========================

# Root of this segmentation project
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories (change if your data lives elsewhere)
DATA_DIR = PROJECT_ROOT / "data"
CROPPED_DIR = DATA_DIR / "cropped"   # original cropped seal images
CLEAN_DIR = DATA_DIR / "clean"       # clean binary images

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
SYMBOLS_DIR = OUTPUT_DIR / "symbols"     # will use later
ANIMALS_DIR = OUTPUT_DIR / "animals"     # will use later
METADATA_DIR = OUTPUT_DIR / "metadata"   # will use later

# Make sure output dirs exist
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)
ANIMALS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Connected component hyperparameters
MIN_AREA = 50            # ignore tiny specks
MAX_AREA_FRACTION = 0.8  # ignore blobs bigger than 80% of image area (likely background/border)


# =========================
# UTILITIES
# =========================

def list_clean_images():
    """
    Return sorted list of all clean images (.png).
    """
    return sorted(CLEAN_DIR.glob("*.png"))


def seal_id_from_path(clean_path: Path) -> str:
    """
    Derive a seal ID from the clean image filename.
    Example: 'A-123_1.png' -> 'A-123_1'
    """
    stem = clean_path.stem
    if stem.endswith("_clean"):
        stem = stem[:-6]  # remove 6 characters: len("_clean")
    return stem


# =========================
# CONNECTED COMPONENTS
# =========================

def find_components(clean_bin: np.ndarray):
    """
    Run connected components on a binary clean image (foreground = 255).
    Returns a list of component dicts:
      {
        'label': int,
        'bbox': [x_min, y_min, x_max, y_max],
        'area': int,
        'centroid': [cx, cy]
      }
    Filter by area thresholds.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_bin, connectivity=8
    )

    h, w = clean_bin.shape[:2]
    img_area = h * w

    components = []

    # Label 0 is background
    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]

        # Area-based filtering
        if area < MIN_AREA:
            continue
        if area > MAX_AREA_FRACTION * img_area:
            continue

        cx, cy = centroids[label]

        comp = {
            "label": int(label),
            "bbox": [int(x), int(y), int(x + w_box), int(y + h_box)],
            "area": int(area),
            "centroid": [float(cx), float(cy)],
        }
        components.append(comp)

    # Sort by area (largest first) – useful later for animal detection
    components.sort(key=lambda c: c["area"], reverse=True)
    return components

def find_active_bands(clean_bin: np.ndarray):
    """
    From a binary image, find contiguous vertical bands (in Y) where strokes are dense.
    Returns a list of (start_row, end_row) inclusive.
    """
    h, w = clean_bin.shape[:2]
    # stroke count per row
    row_counts = np.sum(clean_bin > 0, axis=1)
    max_row = row_counts.max()

    if max_row == 0:
        return []

    # rows with enough strokes
    thresh = ROW_SUM_THRESHOLD_FRAC * max_row
    active = row_counts > thresh

    bands = []
    i = 0
    while i < h:
        if active[i]:
            start = i
            while i + 1 < h and active[i + 1]:
                i += 1
            end = i
            bands.append((start, end))
        i += 1

    return bands

def separate_text_and_animal(clean_bin: np.ndarray, components):
    """
    Given the clean binary image + list of components, estimate:
      - text_band: (y1, y2) or None
      - animal_bbox: [x1, y1, x2, y2] or None

    Uses vertical stroke density and component centroids.
    """
    h, w = clean_bin.shape[:2]
    bands = find_active_bands(clean_bin)

    text_band = None
    animal_bbox = None

    if not bands:
        return None, None

    # Choose the text band:
    #  - prefer bands whose center is in the top part of the image
    #  - prefer smaller height (narrow band)
    def band_center_frac(b):
        y1, y2 = b
        return (y1 + y2) / 2.0 / h

    def band_height(b):
        y1, y2 = b
        return (y2 - y1 + 1)

    # candidates in top region
    candidates = [
        b for b in bands
        if band_center_frac(b) <= TEXT_MAX_CENTER_Y_FRAC
        and band_height(b) <= TEXT_MAX_HEIGHT_FRAC * h
    ]

    if not candidates:
        # fallback: just pick the narrowest band overall
        candidates = bands

    if candidates:
        text_band = min(candidates, key=band_height)

    # Now estimate animal bbox from components below the text band
    if text_band is not None:
        ty1, ty2 = text_band
        margin = int(ANIMAL_MARGIN_FRAC * h)
        cutoff_y = ty2 + margin

        animal_components = [
            c for c in components
            if c["centroid"][1] > cutoff_y
        ]

        if animal_components:
            x1 = min(c["bbox"][0] for c in animal_components)
            y1 = min(c["bbox"][1] for c in animal_components)
            x2 = max(c["bbox"][2] for c in animal_components)
            y2 = max(c["bbox"][3] for c in animal_components)
            animal_bbox = [x1, y1, x2, y2]

    return text_band, animal_bbox


# =========================
# VISUALIZATION
# =========================

def make_overlay(seal_id: str, components):
    orig_path = CROPPED_DIR / f"{seal_id}.png"
    clean_path = CLEAN_DIR / f"{seal_id}_clean.png"

    orig_img = cv2.imread(str(orig_path))
    clean_img = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)

    if orig_img is None or clean_img is None:
        print(f"[WARN] Could not read image for seal_id={seal_id}")
        return

    oh, ow = orig_img.shape[:2]
    ch, cw = clean_img.shape[:2]

    # Compute scaling factors
    scale_x = ow / cw
    scale_y = oh / ch

    overlay = orig_img.copy()

    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]

        # Scale bbox from clean-space to original-image-space
        x1s = int(x1 * scale_x)
        x2s = int(x2 * scale_x)
        y1s = int(y1 * scale_y)
        y2s = int(y2 * scale_y)

        # Draw
        cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), (0, 0, 255), 1)

    out_path = OVERLAY_DIR / f"{seal_id}_components.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"[INFO] Saved scaled overlay: {out_path}")

def make_overlay_with_roles(seal_id: str,
                            components,
                            clean_shape,
                            text_band=None,
                            animal_bbox=None):
    """
    Draw:
      - all components (red)
      - text band (green horizontal band)
      - animal bbox (blue)
    on the original cropped image.
    """
    orig_path = CROPPED_DIR / f"{seal_id}.png"
    orig_img = cv2.imread(str(orig_path))

    if orig_img is None:
        print(f"[WARN] Could not read original cropped image: {orig_path}")
        return

    oh, ow = orig_img.shape[:2]
    ch, cw = clean_shape[:2]

    # scaling from clean coords -> original coords
    scale_x = ow / cw
    scale_y = oh / ch

    overlay = orig_img.copy()

    # 1) draw all components in red
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        x1s = int(x1 * scale_x)
        x2s = int(x2 * scale_x)
        y1s = int(y1 * scale_y)
        y2s = int(y2 * scale_y)
        cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), (0, 0, 255), 1)

    # 2) draw text band in green
    if text_band is not None:
        ty1, ty2 = text_band
        ty1s = int(ty1 * scale_y)
        ty2s = int(ty2 * scale_y)

        ty1s = max(0, min(ty1s, oh - 1))
        ty2s = max(0, min(ty2s, oh - 1))

        cv2.rectangle(overlay, (0, ty1s), (ow - 1, ty2s), (0, 255, 0), 2)

    # 3) draw animal bbox in blue
    if animal_bbox is not None:
        ax1, ay1, ax2, ay2 = animal_bbox
        ax1s = int(ax1 * scale_x)
        ax2s = int(ax2 * scale_x)
        ay1s = int(ay1 * scale_y)
        ay2s = int(ay2 * scale_y)

        ax1s, ay1s, ax2s, ay2s = clamp_bbox_to_image(
            [ax1s, ay1s, ax2s, ay2s], ow, oh
        )

        cv2.rectangle(overlay, (ax1s, ay1s), (ax2s, ay2s), (255, 0, 0), 2)


    out_path = OVERLAY_DIR / f"{seal_id}_roles.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"[INFO] Saved overlay with roles: {out_path}")

def make_full_overlay(seal_id: str,
                      components,
                      clean_shape,
                      text_band=None,
                      animal_bbox=None,
                      symbol_boxes=None):
    """
    Draw:
      - components (red, thin)
      - text band (green)
      - animal bbox (blue)
      - symbol boxes (yellow, thick)
    on the original cropped image.
    """
    orig_path = CROPPED_DIR / f"{seal_id}.png"
    orig_img = cv2.imread(str(orig_path))
    if orig_img is None:
        print(f"[WARN] Could not read original cropped image: {orig_path}")
        return

    oh, ow = orig_img.shape[:2]
    ch, cw = clean_shape[:2]
    scale_x = ow / cw
    scale_y = oh / ch

    overlay = orig_img.copy()

    # all components: red
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        x1s = int(x1 * scale_x); x2s = int(x2 * scale_x)
        y1s = int(y1 * scale_y); y2s = int(y2 * scale_y)
        cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), (0, 0, 255), 1)

    # text band: green
    if text_band is not None:
        ty1, ty2 = text_band
        ty1s = max(0, min(int(ty1 * scale_y), oh - 1))
        ty2s = max(0, min(int(ty2 * scale_y), oh - 1))
        cv2.rectangle(overlay, (0, ty1s), (ow - 1, ty2s), (0, 255, 0), 2)

    # animal bbox: blue
    if animal_bbox is not None:
        ax1, ay1, ax2, ay2 = animal_bbox
        ax1s = int(ax1 * scale_x); ax2s = int(ax2 * scale_x)
        ay1s = int(ay1 * scale_y); ay2s = int(ay2 * scale_y)
        ax1s, ay1s, ax2s, ay2s = clamp_bbox_to_image([ax1s, ay1s, ax2s, ay2s], ow, oh)
        cv2.rectangle(overlay, (ax1s, ay1s), (ax2s, ay2s), (255, 0, 0), 2)

    # symbol boxes: yellow
    if symbol_boxes is not None:
        for b in symbol_boxes:
            x1, y1, x2, y2 = b
            x1s = int(x1 * scale_x); x2s = int(x2 * scale_x)
            y1s = int(y1 * scale_y); y2s = int(y2 * scale_y)
            x1s, y1s, x2s, y2s = clamp_bbox_to_image([x1s, y1s, x2s, y2s], ow, oh)
            cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), (0, 255, 255), 2)

    out_path = OVERLAY_DIR / f"{seal_id}_symbols.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"[INFO] Saved symbol overlay: {out_path}")


def clamp_bbox_to_image(bbox, width, height):
    """
    Clamp [x1, y1, x2, y2] to be inside [0, width-1] x [0, height-1].
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # ensure non-empty box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]

def get_text_components(components, text_band):
    """
    Filter components whose centroid lies inside the text_band (y1, y2).
    """
    if text_band is None:
        return []

    y1, y2 = text_band
    text_comps = [
        c for c in components
        if y1 <= c["centroid"][1] <= y2
    ]
    return text_comps

def group_components_into_symbols(text_components, img_height):
    """
    Group text-row components into symbol bounding boxes.
    Returns a list of symbol bboxes in CLEAN coordinates: [x1, y1, x2, y2].
    """
    if not text_components:
        return []

    # sort left-to-right
    comps = sorted(text_components, key=lambda c: c["bbox"][0])

    # average width of text components
    widths = [c["bbox"][2] - c["bbox"][0] for c in comps]
    avg_w = np.mean(widths) if widths else 1.0
    max_gap = GAP_FRAC * avg_w

    symbol_boxes = []

    # start with first component as current symbol box
    cur_x1, cur_y1, cur_x2, cur_y2 = comps[0]["bbox"]

    for c in comps[1:]:
        x1, y1, x2, y2 = c["bbox"]

        # horizontal gap between current box and this component
        gap = x1 - cur_x2

        # vertical overlap between current box and this one
        overlap_y = min(cur_y2, y2) - max(cur_y1, y1)
        min_h = min(cur_y2 - cur_y1, y2 - y1)
        overlap_frac = overlap_y / min_h if min_h > 0 else 0.0

        if gap <= max_gap and overlap_frac >= MIN_VERTICAL_OVERLAP_FRAC:
            # merge into current symbol box
            cur_x1 = min(cur_x1, x1)
            cur_y1 = min(cur_y1, y1)
            cur_x2 = max(cur_x2, x2)
            cur_y2 = max(cur_y2, y2)
        else:
            # close current symbol and start new
            symbol_boxes.append([cur_x1, cur_y1, cur_x2, cur_y2])
            cur_x1, cur_y1, cur_x2, cur_y2 = x1, y1, x2, y2

    # add last one
    symbol_boxes.append([cur_x1, cur_y1, cur_x2, cur_y2])

    # clamp to image height (just in case)
    clamped = [
        clamp_bbox_to_image(b, width=10**9, height=img_height)  # width unused here
        for b in symbol_boxes
    ]
    return clamped

def scale_bbox_from_clean_to_orig(bbox, clean_shape, orig_shape):
    """
    Scale a bbox from clean-image coordinates to original-cropped-image coordinates.
    """
    ch, cw = clean_shape[:2]
    oh, ow = orig_shape[:2]

    x1, y1, x2, y2 = bbox
    scale_x = ow / cw
    scale_y = oh / ch

    x1s = int(x1 * scale_x)
    x2s = int(x2 * scale_x)
    y1s = int(y1 * scale_y)
    y2s = int(y2 * scale_y)

    x1s, y1s, x2s, y2s = clamp_bbox_to_image([x1s, y1s, x2s, y2s], ow, oh)
    return [x1s, y1s, x2s, y2s]


def crop_and_normalize_symbol(orig_img, bbox_orig):
    """
    Crop bbox from original image, convert to grayscale, normalize to TARGET_SIZE x TARGET_SIZE.
    """
    x1, y1, x2, y2 = bbox_orig
    # +1 because slices are end-exclusive
    crop = orig_img[y1:y2+1, x1:x2+1]

    if crop.size == 0:
        return None

    # convert to grayscale
    if len(crop.shape) == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop

    h, w = crop_gray.shape[:2]
    if h == 0 or w == 0:
        return None

    # inner size after padding
    inner = int(TARGET_SIZE * (1.0 - 2 * PADDING_FRAC))
    inner = max(1, inner)

    # preserve aspect ratio
    if h >= w:
        new_h = inner
        new_w = max(1, int(w * inner / h))
    else:
        new_w = inner
        new_h = max(1, int(h * inner / w))

    resized = cv2.resize(crop_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    y0 = (TARGET_SIZE - new_h) // 2
    x0 = (TARGET_SIZE - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


# =========================
# MAIN PER-IMAGE PIPELINE
# =========================

def process_one_clean_image(clean_path: Path):
    seal_id = seal_id_from_path(clean_path)
    print(f"\n[INFO] Processing seal_id = {seal_id}")

    clean_gray = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    if clean_gray is None:
        print(f"[WARN] Could not read clean image: {clean_path}")
        return

    _, clean_bin = cv2.threshold(
        clean_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    h, w = clean_bin.shape[:2]

    components = find_components(clean_bin)
    print(f"[INFO] Components after filtering: {len(components)}")

    text_band, animal_bbox = separate_text_and_animal(clean_bin, components)

    text_components = get_text_components(components, text_band)
    print(f"[INFO] Text-row components: {len(text_components)}")

    symbol_boxes_clean = group_components_into_symbols(text_components, img_height=h)
    print(f"[INFO] Symbols detected: {len(symbol_boxes_clean)}")

    # Save symbol crops
    orig_path = CROPPED_DIR / f"{seal_id}.png"
    orig_img = cv2.imread(str(orig_path))
    if orig_img is None:
        print(f"[WARN] Could not read original cropped image: {orig_path}")
        return

    symbol_entries = []
    for idx, bbox_clean in enumerate(symbol_boxes_clean, start=1):
        bbox_orig = scale_bbox_from_clean_to_orig(bbox_clean, clean_gray.shape, orig_img.shape)
        symbol_img = crop_and_normalize_symbol(orig_img, bbox_orig)
        if symbol_img is None:
            continue

        symbol_id = f"{seal_id}_sym_{idx:03d}"
        out_path = SYMBOLS_DIR / f"{symbol_id}.png"
        cv2.imwrite(str(out_path), symbol_img)

        symbol_entries.append({
            "symbol_id": symbol_id,
            "index": idx,
            "bbox_clean": bbox_clean,
            "bbox_orig": bbox_orig,
            "path": str(out_path.relative_to(PROJECT_ROOT))
        })

    # Write per-seal metadata JSON
    meta = {
        "seal_id": seal_id,
        "source_image": str(orig_path.relative_to(PROJECT_ROOT)),
        "clean_image": str(clean_path.relative_to(PROJECT_ROOT)),
        "image_size_orig": list(orig_img.shape[:2][::-1]),   # [w, h]
        "image_size_clean": list(clean_gray.shape[:2][::-1]),
        "text_band": list(text_band) if text_band is not None else None,
        "animal_bbox_clean": animal_bbox,
        "symbols": symbol_entries,
    }

    meta_path = METADATA_DIR / f"{seal_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved metadata: {meta_path}")

    # Debug overlay with symbols
    make_full_overlay(
        seal_id,
        components,
        clean_gray.shape,
        text_band=text_band,
        animal_bbox=animal_bbox,
        symbol_boxes=symbol_boxes_clean,
    )




def main():
    clean_images = list_clean_images()
    print(f"[INFO] Found {len(clean_images)} clean images in {CLEAN_DIR}")

    # For first run, just do a small subset
    for clean_path in clean_images:
        process_one_clean_image(clean_path)


if __name__ == "__main__":
    main()
