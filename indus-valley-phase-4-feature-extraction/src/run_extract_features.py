# src/run_extract_features.py
import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .config import FEATURES_DIR
from .views import make_symbol_views_from_seal
from .features import extract_features_simple


def _bbox_from_row(row: pd.Series) -> Dict[str, int]:
    """
    Build a bbox dict from a symbol_index row using bbox_orig_* columns.
    """
    return {
        "x1": int(row["bbox_orig_x1"]),
        "y1": int(row["bbox_orig_y1"]),
        "x2": int(row["bbox_orig_x2"]),
        "y2": int(row["bbox_orig_y2"]),
    }


def run_feature_extraction(symbol_index_csv: Path) -> None:
    """
    Main entry point for simple feature extraction (Feature Set A).
    """
    df = pd.read_csv(symbol_index_csv)
    print(f"Loaded symbol index with {len(df)} symbols from {symbol_index_csv}")

    feature_rows: List[Dict[str, Any]] = []
    feature_matrix: List[np.ndarray] = []

    for i, row in df.iterrows():
        seal_id = row["seal_id"]
        symbol_id = row["symbol_id"]
        symbol_index = row["symbol_index"]

        bbox_orig = _bbox_from_row(row)

        # Build all preprocessed views for this symbol
        views = make_symbol_views_from_seal(seal_id, bbox_orig, target_size=64)

        # Compute simple features
        feat_vec, feat_dict = extract_features_simple(views)

        # Build row dict for CSV
        csv_row: Dict[str, Any] = {
            "symbol_id": symbol_id,
            "seal_id": seal_id,
            "symbol_index": symbol_index,
        }
        csv_row.update(feat_dict)

        feature_rows.append(csv_row)
        feature_matrix.append(feat_vec)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(df)} symbols...")

    if not feature_rows:
        print("No features extracted. Check symbol_index.csv and view paths.")
        return

    # Save CSV
    out_csv = FEATURES_DIR / "features_simple.csv"
    fieldnames = list(feature_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in feature_rows:
            writer.writerow(r)

    print(f"Wrote feature table to {out_csv}")

    # Save numeric matrix (.npy)
    X = np.stack(feature_matrix, axis=0)
    out_npy = FEATURES_DIR / "features_simple.npy"
    np.save(out_npy, X)
    print(f"Wrote feature matrix {X.shape} to {out_npy}")


if __name__ == "__main__":
    symbol_index_path = FEATURES_DIR / "symbol_index.csv"
    run_feature_extraction(symbol_index_path)
