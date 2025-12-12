import os
import re
import argparse

import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pytesseract import Output


# --------- CAPTION PATTERN(S) ---------
#
# This is designed to handle:
#   "A-123"
#   "a-123 (1)"
#   "A-55 bis"
#   "M-67 A"
#   "M-67 a (2) bis"
#
# Groups:
#   site      -> first letter before "-"
#   number    -> digits after "-"
#   side      -> trailing letter (A/B/a/b etc., often obverse/reverse)
#   part      -> number in parentheses, e.g. (1)
#   variant   -> bis / ter / quater

import re

CAPTION_REGEX = re.compile(
    r"""
    (?P<site>[A-Za-z])            # site code (e.g. M)
    \s*-\s*
    (?P<number>\d+)               # number after '-'
    \s*
    (?P<side>[A-Za-z])?           # optional side letter (A, a, B, b...)
    \s*
    (?:\((?P<part>\d+)\))?        # optional (1), (2)...
    \s*
    (?P<variant>bis|ter|quater)?  # optional variant
    """,
    re.VERBOSE,
)


def normalize_ocr_text(text: str) -> str:
    """
    Clean up some common OCR quirks so the regex works better.
    Examples:
      'M 13-A' -> 'M-13 A'
    """
    # unify weird dashes
    text = text.replace("—", "-")

    # Fix patterns like 'M 13-A' --> 'M-13 A'
    text = re.sub(r"\b([A-Za-z])\s*(\d+)-([A-Za-z])\b", r"\1-\2 \3", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_match(m: re.Match) -> dict:
    d = m.groupdict()
    if d["site"]:
        d["site"] = d["site"].upper()
    if d["number"]:
        d["number"] = int(d["number"])
    if d["part"]:
        d["part"] = int(d["part"])
    return d


def find_captions_in_line(raw_text: str):
    """
    Return *all* caption matches in a line of OCR text.

    Each item: {
        "parsed": {site, number, side, part, variant},
        "span": (start_idx, end_idx)  # in normalized text
        "norm": normalized_text
    }
    """
    norm = normalize_ocr_text(raw_text)
    results = []
    for m in CAPTION_REGEX.finditer(norm):
        results.append(
            {
                "parsed": _parse_match(m),
                "span": m.span(),
                "norm": norm,
            }
        )
    return results



def ocr_page_to_lines(image: Image.Image, page_index: int):
    """
    Run Tesseract on a PIL image and aggregate word boxes into line boxes.
    Returns a list of dicts, each representing one line:
      {
        "page": page_index (1-based),
        "line_id": int,
        "text": str,
        "x": int, "y": int, "w": int, "h": int
      }
    """
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DATAFRAME,
        config="--psm 6",  # assume a single uniform block of text
    )

    # Filter out empty / NaN text and negative conf rows
    data = data[(data.conf != -1) & (data.text.notna())]
    data["text"] = data["text"].astype(str)

    lines = []
    line_counter = 0

    group_cols = ["block_num", "par_num", "line_num"]
    for _, group in data.groupby(group_cols):
        words = [w for w in group["text"].tolist() if w.strip()]
        if not words:
            continue

        text = " ".join(words)
        x_min = int(group["left"].min())
        y_min = int(group["top"].min())
        x_max = int((group["left"] + group["width"]).max())
        y_max = int((group["top"] + group["height"]).max())

        line_counter += 1
        lines.append(
            {
                "page": page_index,
                "line_id": line_counter,
                "text": text,
                "x": x_min,
                "y": y_min,
                "w": x_max - x_min,
                "h": y_max - y_min,
            }
        )

    return lines


def process_pdf(pdf_path: str, out_dir: str, dpi: int = 300):
    """
    Main pipeline for Steps 1–5:
      1. Convert each PDF page to an image.
      2. OCR each page.
      3. Group words into lines.
      4. Detect caption-like lines with regex.
      5. Save caption table as captions.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "pages_as_images")
    os.makedirs(images_dir, exist_ok=True)

    captions = []

    # Step 1: convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)

    for idx, page_img in enumerate(pages, start=1):
        # Save the full page image (useful for debugging and for later cropping)
        page_filename = os.path.join(images_dir, f"page_{idx:03d}.png")
        page_img.save(page_filename)

        # Step 2 & 3: OCR and aggregate into line boxes
        lines = ocr_page_to_lines(page_img, page_index=idx)

        # Step 4: detect caption lines using regex
                # Step 4: detect ALL caption codes in each line
        for line in lines:
            caps_in_line = find_captions_in_line(line["text"])
            if not caps_in_line:
                continue

            for idx, cap in enumerate(caps_in_line):
                parsed = cap["parsed"]
                start, end = cap["span"]
                norm_text = cap["norm"]

                # approximate horizontal center of this caption in the line
                center_char = (start + end) / 2.0
                center_ratio = center_char / max(len(norm_text), 1)
                x_center = int(line["x"] + center_ratio * line["w"])

                entry = {
                    "page": line["page"],
                    "line_id": line["line_id"],
                    "caption_index": idx,      # 0 = left caption, 1 = right caption, etc.
                    "text_raw": line["text"],
                    "text_norm": norm_text,
                    "x": line["x"],
                    "y": line["y"],
                    "w": line["w"],
                    "h": line["h"],
                    "x_center": x_center,
                    "site": parsed.get("site"),
                    "number": parsed.get("number"),
                    "side": parsed.get("side"),
                    "part": parsed.get("part"),
                    "variant": parsed.get("variant"),
                }

                # Build a normalized code like 'M-13 A', 'A-55 bis', etc.
                code_parts = [f"{entry['site']}-{entry['number']}"]
                if entry["side"]:
                    code_parts.append(entry["side"])
                if entry["part"] is not None:
                    code_parts.append(f"({entry['part']})")
                if entry["variant"]:
                    code_parts.append(entry["variant"])
                entry["normalized_code"] = " ".join(code_parts)

                captions.append(entry)

    # Save as CSV
    captions_df = pd.DataFrame(captions)
    out_csv = os.path.join(out_dir, "captions.csv")
    captions_df.to_csv(out_csv, index=False)
    print(f"Saved {len(captions_df)} captions to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract caption lines (A-123, M-67 A, etc.) from a scanned plates PDF."
    )
    parser.add_argument("pdf_path", help="Path to the scanned plates PDF")
    parser.add_argument(
        "--out_dir",
        default="output_captions",
        help="Output directory for captions.csv and page images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF-to-image conversion (300–400 recommended).",
    )

    args = parser.parse_args()
    process_pdf(args.pdf_path, args.out_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
