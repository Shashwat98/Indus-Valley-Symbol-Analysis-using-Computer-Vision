# src/index_symbols.py
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from .config import METADATA_DIR, FEATURES_DIR


def load_metadata_files(metadata_dir: Path) -> List[Path]:
    """
    Return a list of all metadata JSON files in the given directory.
    """
    return sorted(metadata_dir.glob("*.json"))


def parse_metadata_file(path: Path) -> List[Dict[str, Any]]:
    """
    Parse a single metadata JSON file and return a list of symbol records.

    Each record will contain:
    - seal_id
    - source_image
    - clean_image
    - image_size_orig_w, image_size_orig_h
    - image_size_clean_w, image_size_clean_h
    - symbol_id
    - symbol_index
    - bbox_clean_x1, bbox_clean_y1, bbox_clean_x2, bbox_clean_y2
    - bbox_orig_x1, bbox_orig_y1, bbox_orig_x2, bbox_orig_y2
    - symbol_path
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seal_id = data.get("seal_id")
    source_image = data.get("source_image")
    clean_image = data.get("clean_image")

    # image sizes may be stored as [w, h] strings or ints; normalize to ints
    def _to_int_pair(arr):
        if arr is None:
            return None, None
        return int(arr[0]), int(arr[1])

    orig_w, orig_h = _to_int_pair(data.get("image_size_orig"))
    clean_w, clean_h = _to_int_pair(data.get("image_size_clean"))

    symbols = data.get("symbols", [])
    records = []

    for sym in symbols:
        symbol_id = sym.get("symbol_id")
        idx = sym.get("index")

        bbox_clean = sym.get("bbox_clean", [0, 0, 0, 0])
        bbox_orig = sym.get("bbox_orig", [0, 0, 0, 0])
        sym_path = sym.get("path")

        record = {
            "seal_id": seal_id,
            "source_image": source_image,
            "clean_image": clean_image,
            "image_size_orig_w": orig_w,
            "image_size_orig_h": orig_h,
            "image_size_clean_w": clean_w,
            "image_size_clean_h": clean_h,
            "symbol_id": symbol_id,
            "symbol_index": idx,
            "bbox_clean_x1": bbox_clean[0],
            "bbox_clean_y1": bbox_clean[1],
            "bbox_clean_x2": bbox_clean[2],
            "bbox_clean_y2": bbox_clean[3],
            "bbox_orig_x1": bbox_orig[0],
            "bbox_orig_y1": bbox_orig[1],
            "bbox_orig_x2": bbox_orig[2],
            "bbox_orig_y2": bbox_orig[3],
            "symbol_path": sym_path,
        }
        records.append(record)

    return records


def build_symbol_index(metadata_dir: Path, output_csv: Path) -> None:
    """
    Walk through all metadata JSON files and write a global symbol index CSV.
    """
    meta_files = load_metadata_files(metadata_dir)
    all_records: List[Dict[str, Any]] = []

    for path in meta_files:
        recs = parse_metadata_file(path)
        all_records.extend(recs)

    if not all_records:
        print("No symbols found in metadata. Check METADATA_DIR.")
        return

    fieldnames = list(all_records[0].keys())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"Wrote symbol index with {len(all_records)} symbols to {output_csv}")


if __name__ == "__main__":
    output_csv_path = FEATURES_DIR / "symbol_index.csv"
    build_symbol_index(METADATA_DIR, output_csv_path)
