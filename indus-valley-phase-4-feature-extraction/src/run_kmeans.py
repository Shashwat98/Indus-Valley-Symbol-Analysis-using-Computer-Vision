# src/run_kmeans.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import FEATURES_DIR


def run_kmeans(
    scaled_features_path: Path,
    k: int = 40,
) -> None:
    """
    Run KMeans on scaled features and save cluster assignments and PCA+cluster CSVs.
    """
    # Load scaled matrix
    X_scaled = np.load(scaled_features_path)
    print(f"Loaded scaled features from {scaled_features_path}, shape={X_scaled.shape}")

    # Load original feature table to get ids
    features_csv = FEATURES_DIR / "features_simple.csv"
    df_feat = pd.read_csv(features_csv)

    # KMeans clustering
    print(f"Running KMeans with k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print("KMeans done.")

    # Save basic assignments
    assign_df = pd.DataFrame({
        "symbol_id": df_feat["symbol_id"],
        "seal_id": df_feat["seal_id"],
        "symbol_index": df_feat["symbol_index"],
        "cluster_id": labels,
    })

    out_assign = FEATURES_DIR / f"cluster_assignments_simple_k{k}.csv"
    assign_df.to_csv(out_assign, index=False)
    print(f"Wrote cluster assignments to {out_assign}")

    # Also join with PCA 2D for quick plotting later
    pca_csv = FEATURES_DIR / "pca_2d_simple.csv"
    if pca_csv.exists():
        pca_df = pd.read_csv(pca_csv)
        merged = pca_df.merge(assign_df, on=["symbol_id", "seal_id"], how="left")
        out_pca_clusters = FEATURES_DIR / f"pca_2d_clusters_simple_k{k}.csv"
        merged.to_csv(out_pca_clusters, index=False)
        print(f"Wrote PCA+cluster table to {out_pca_clusters}")
    else:
        print(f"Warning: {pca_csv} not found, skipping PCA+cluster export.")


if __name__ == "__main__":
    scaled_path = FEATURES_DIR / "features_simple_scaled.npy"
    # You can tweak k here
    run_kmeans(scaled_path, k=40)
