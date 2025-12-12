# src/cluster_viz.py
import math
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, SYMBOL_DIR, FEATURES_DIR, VIZ_DIR


def resolve_symbol_image_path(row: pd.Series) -> Optional[Path]:
    """
    Try to find the actual image file for this symbol.

    Priority:
      1) If 'symbol_path' exists in the index and the file exists, use it.
      2) Otherwise, try SYMBOL_DIR / "<symbol_id>.png".
      3) If nothing exists, return None.
    """
    # 1) symbol_path from index (if present)
    if "symbol_path" in row and isinstance(row["symbol_path"], str):
        p = Path(row["symbol_path"])
        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            # relative to project root
            candidates.append(PROJECT_ROOT / p)
            # relative to symbols directory (using only filename)
            candidates.append(SYMBOL_DIR / p.name)

        for c in candidates:
            if c.exists():
                return c

    # 2) Fallback: SYMBOL_DIR / "<symbol_id>.png"
    symbol_id = row["symbol_id"]
    cand = SYMBOL_DIR / f"{symbol_id}.png"
    if cand.exists():
        return cand

    # Nothing found
    return None


def create_cluster_grid(
    df_cluster: pd.DataFrame,
    out_path: Path,
    max_per_cluster: int = 25,
    tile_size: int = 64,
) -> None:
    """
    Given a subset of rows for a single cluster, sample up to max_per_cluster
    symbols and create a grid image of size ~sqrt(N) x sqrt(N).

    Saves the grid to out_path as PNG.
    """
    n = len(df_cluster)
    if n == 0:
        return

    # Sample if cluster is too big
    if n > max_per_cluster:
        df_cluster = df_cluster.sample(max_per_cluster, random_state=42)
        n = max_per_cluster

    # Determine grid size
    grid_cols = int(math.ceil(math.sqrt(n)))
    grid_rows = int(math.ceil(n / grid_cols))

    # Prepare blank canvas (grayscale)
    canvas_h = grid_rows * tile_size
    canvas_w = grid_cols * tile_size
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255  # white background

    idx = 0
    for _, row in df_cluster.iterrows():
        img_path = resolve_symbol_image_path(row)
        if img_path is None:
            # skip if image missing
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize to tile_size x tile_size if needed
        if img.shape[0] != tile_size or img.shape[1] != tile_size:
            img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        r = idx // grid_cols
        c = idx % grid_cols
        y1 = r * tile_size
        y2 = y1 + tile_size
        x1 = c * tile_size
        x2 = x1 + tile_size

        if y2 <= canvas_h and x2 <= canvas_w:
            canvas[y1:y2, x1:x2] = img

        idx += 1
        if idx >= n:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_cluster_viz(
    assignments_csv: Path,
    symbol_index_csv: Path,
    max_per_cluster: int = 25,
    tile_size: int = 64,
) -> None:
    """
    Build small grid images per cluster.

    assignments_csv: e.g. outputs/features/cluster_assignments_simple_k40.csv
    symbol_index_csv: outputs/features/symbol_index.csv
    """
    df_assign = pd.read_csv(assignments_csv)
    df_index = pd.read_csv(symbol_index_csv)

    # Merge to get symbol_path etc.
    df = df_assign.merge(df_index, on=["symbol_id", "seal_id"], how="left")

    print(f"Merged assignments + index: shape={df.shape}")

    # Determine unique clusters
    cluster_ids = sorted(df["cluster_id"].unique().tolist())
    print(f"Found {len(cluster_ids)} clusters.")

    out_dir = VIZ_DIR / "clusters_simple"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in cluster_ids:
        df_c = df[df["cluster_id"] == cid]
        print(f"Cluster {cid}: {len(df_c)} symbols")

        out_path = out_dir / f"cluster_{cid}.png"
        create_cluster_grid(
            df_cluster=df_c,
            out_path=out_path,
            max_per_cluster=max_per_cluster,
            tile_size=tile_size,
        )

    print(f"Cluster grids saved in {out_dir}")


if __name__ == "__main__":
    # default paths for the simple KMeans run
    assignments = FEATURES_DIR / "cluster_assignments_rich_k40.csv"
    symbol_index = FEATURES_DIR / "symbol_index.csv"
    run_cluster_viz(assignments, symbol_index, max_per_cluster=25, tile_size=64)
