import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd


def sanitize_code(code: str) -> str:
    """Turn 'M-13 A' into a safe filename like 'M-13_A'."""
    if not isinstance(code, str):
        code = str(code)
    code = code.strip().replace(" ", "_")
    # remove anything weird
    code = re.sub(r"[^\w\-]+", "_", code)
    return code


def find_big_contours(mask, min_area_ratio=0.005):
    """
    Find 'big' contours on a page mask (binary image).

    Returns a list of (x, y, w, h, area).
    """
    h_img, w_img = mask.shape[:2]
    page_area = float(h_img * w_img)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area_ratio * page_area:
            # too small: likely glyphs/text, not a whole tablet
            continue
        boxes.append((x, y, w, h, area))

    return boxes


def crop_tablets(captions_csv, pages_dir, out_dir, debug_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    df = pd.read_csv(captions_csv)

    # If we still have leftover rows with no caption_index, drop them
    if "caption_index" in df.columns:
        df = df[df["caption_index"].notna()].reset_index(drop=True)

    df["page"] = df["page"].astype(int)

    # Group captions by page
    for page_idx, group in df.groupby("page"):
        page_path = os.path.join(pages_dir, f"page_{page_idx:03d}.png")
        if not os.path.exists(page_path):
            print(f"[WARN] Page image not found for page {page_idx}: {page_path}")
            continue

        img = cv2.imread(page_path)
        if img is None:
            print(f"[WARN] Failed to read image for page {page_idx}: {page_path}")
            continue

        h_img, w_img = img.shape[:2]

        # --- Build a mask of dark regions (tablets + text) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find big contours (candidate tablet boxes)
        big_boxes = find_big_contours(mask)

        if not big_boxes:
            print(f"[WARN] No big contours found on page {page_idx}")
            continue

        if debug_dir is not None:
            debug_img = img.copy()

        print(f"[INFO] Page {page_idx}: {len(group)} captions, {len(big_boxes)} big contours")

        for _, row in group.iterrows():
            cx = int(row["x_center"])
            cap_y = int(row["y"])

            # Find candidate boxes: must cover x_center and be above the caption
            candidates = []
            for (bx, by, bw, bh, barea) in big_boxes:
                x0, x1 = bx, bx + bw
                bottom = by + bh

                # horizontally: caption center should lie within the box
                if not (cx >= x0 and cx <= x1):
                    continue

                # vertically: box must be above caption
                if bottom >= cap_y:
                    continue

                # and not *too* far above (avoid picking header images, etc.)
                if cap_y - bottom > 0.6 * h_img:
                    continue

                candidates.append((barea, bx, by, bw, bh))

            if not candidates:
                print(
                    f"[WARN] No matching contour for caption "
                    f"'{row.get('normalized_code')}' on page {page_idx}"
                )
                continue

            # Take the largest area candidate
            candidates.sort(reverse=True)  # largest area first
            _, bx, by, bw, bh = candidates[0]

            # Add some padding around the box
            pad_x = int(0.02 * w_img)
            pad_y = int(0.02 * h_img)
            x0 = max(0, bx - pad_x)
            y0 = max(0, by - pad_y)
            x1 = min(w_img, bx + bw)
            y1 = min(h_img, by + bh)

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                print(
                    f"[WARN] Empty crop for caption "
                    f"'{row.get('normalized_code')}' on page {page_idx}"
                )
                continue

            code = sanitize_code(row.get("normalized_code", "unknown"))
            cap_idx = int(row.get("caption_index", 0))
            out_name = f"page{page_idx:03d}_{code}_c{cap_idx}.png"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, crop)

            if debug_dir is not None:
                # draw rectangle used for cropping
                cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 0, 255), 4)

        if debug_dir is not None:
            dbg_path = os.path.join(debug_dir, f"page_{page_idx:03d}_debug.png")
            cv2.imwrite(dbg_path, debug_img)


def main():
    parser = argparse.ArgumentParser(
        description="Crop Indus tablet images above each caption using contours."
    )
    parser.add_argument(
        "--captions",
        required=True,
        help="Path to captions.csv produced by build_captions_table.py",
    )
    parser.add_argument(
        "--pages_dir",
        required=True,
        help="Directory with page_XXX.png images (pages_as_images).",
    )
    parser.add_argument(
        "--out_dir",
        default="cropped_tablets",
        help="Output directory for cropped tablet images.",
    )
    parser.add_argument(
        "--debug_dir",
        default=None,
        help="Optional directory to save debug page overlays.",
    )

    args = parser.parse_args()
    crop_tablets(
        captions_csv=args.captions,
        pages_dir=args.pages_dir,
        out_dir=args.out_dir,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
