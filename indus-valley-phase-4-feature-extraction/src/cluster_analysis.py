# src/cluster_analysis.py
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from .config import FEATURES_DIR


def load_rich_features_and_clusters(
    k: int = 40,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Load:
      - scaled rich feature matrix X (N x D)
      - rich feature table (ids + raw features)
      - rich cluster assignments (symbol_id, seal_id, cluster_id)
    """
    # Features
    X_scaled_path = FEATURES_DIR / "features_rich_scaled.npy"
    X = np.load(X_scaled_path)
    print(f"Loaded scaled rich features: {X_scaled_path}, shape={X.shape}")

    # Feature table for IDs
    feat_csv = FEATURES_DIR / "features_rich.csv"
    df_feat = pd.read_csv(feat_csv)
    print(f"Loaded rich feature table: {feat_csv}, shape={df_feat.shape}")

    # Cluster assignments
    assign_csv = FEATURES_DIR / f"cluster_assignments_rich_k{k}.csv"
    df_assign = pd.read_csv(assign_csv)
    print(f"Loaded rich cluster assignments: {assign_csv}, shape={df_assign.shape}")

    # Sanity: ensure same ordering / alignment by symbol_id & seal_id
    merged = df_feat.merge(df_assign, on=["symbol_id", "seal_id", "symbol_index"], how="inner")
    if len(merged) != len(df_feat):
        print("Warning: some rich features have no cluster assignment or vice versa.")
        print(f"features_rich rows: {len(df_feat)}, merged rows: {len(merged)}")

    # Reorder X to match merged order
    # Assume original df_feat order is the same as X row order
    # So we just need index mapping from feat to merged
    df_feat["__row_idx"] = np.arange(len(df_feat))
    merged = merged.merge(df_feat[["symbol_id", "seal_id", "symbol_index", "__row_idx"]],
                          on=["symbol_id", "seal_id", "symbol_index"],
                          how="left")
    merged = merged.sort_values("__row_idx")
    row_indices = merged["__row_idx"].values.astype(int)

    X_aligned = X[row_indices]
    print(f"Aligned feature matrix shape: {X_aligned.shape}")

    return X_aligned, merged, df_assign


def compute_cluster_stats(
    X: np.ndarray,
    df_merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-cluster statistics:
      - size (number of symbols)
      - num_seals (number of unique seals)
      - mean_intra_dist (mean distance to cluster centroid)
      - std_intra_dist (std of distances)
    """
    labels = df_merged["cluster_id"].values
    symbol_ids = df_merged["symbol_id"].values
    seal_ids = df_merged["seal_id"].values

    unique_clusters = sorted(np.unique(labels).tolist())
    print(f"Computing stats for {len(unique_clusters)} clusters...")

    rows: List[Dict[str, Any]] = []

    for cid in unique_clusters:
        mask = labels == cid
        X_c = X[mask]
        sym_c = symbol_ids[mask]
        seal_c = seal_ids[mask]

        n = X_c.shape[0]
        if n == 0:
            continue

        # Cluster centroid
        centroid = X_c.mean(axis=0, keepdims=True)
        # Euclidean distances to centroid
        dists = np.linalg.norm(X_c - centroid, axis=1)

        row = {
            "cluster_id": cid,
            "size": int(n),
            "num_seals": int(len(set(seal_c))),
            "mean_intra_dist": float(dists.mean()),
            "std_intra_dist": float(dists.std()),
        }
        rows.append(row)

    df_stats = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return df_stats


def compute_overall_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the overall silhouette score for the clustering.
    """
    # Need at least 2 clusters and more samples than clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Silhouette score not defined for less than 2 clusters.")
        return float("nan")

    if len(labels) <= len(unique_labels):
        print("Not enough samples to compute silhouette score reliably.")
        return float("nan")

    score = silhouette_score(X, labels, metric="euclidean")
    return float(score)


def compare_simple_vs_rich(k_simple: int = 40, k_rich: int = 40) -> Dict[str, float]:
    """
    Compare simple-feature clustering vs rich-feature clustering on the same symbols,
    using ARI and NMI.

    Expects:
      - cluster_assignments_simple_k{k_simple}.csv
      - cluster_assignments_rich_k{k_rich}.csv
    """
    simple_csv = FEATURES_DIR / f"cluster_assignments_simple_k{k_simple}.csv"
    rich_csv = FEATURES_DIR / f"cluster_assignments_rich_k{k_rich}.csv"

    if not simple_csv.exists():
        print(f"Simple assignments file not found: {simple_csv}")
        return {}

    if not rich_csv.exists():
        print(f"Rich assignments file not found: {rich_csv}")
        return {}

    df_simple = pd.read_csv(simple_csv)
    df_rich = pd.read_csv(rich_csv)

    # Merge on symbol_id + seal_id to ensure same rows
    merged = df_simple.merge(
        df_rich,
        on=["symbol_id", "seal_id", "symbol_index"],
        suffixes=("_simple", "_rich"),
        how="inner",
    )

    if len(merged) == 0:
        print("No overlapping symbols between simple and rich assignments.")
        return {}

    labels_simple = merged["cluster_id_simple"].values
    labels_rich = merged["cluster_id_rich"].values

    ari = adjusted_rand_score(labels_simple, labels_rich)
    nmi = normalized_mutual_info_score(labels_simple, labels_rich)

    print(f"Adjusted Rand Index (simple vs rich): {ari:.4f}")
    print(f"Normalized Mutual Information (simple vs rich): {nmi:.4f}")

    return {"ARI": float(ari), "NMI": float(nmi)}


def run_cluster_analysis(k_rich: int = 40, k_simple: int = 40) -> None:
    """
    High-level entry point:
      1) Load rich features + clusters.
      2) Compute per-cluster stats.
      3) Compute overall silhouette score.
      4) Optionally compare simple vs rich clusters (if simple assignments exist).
    """
    # 1) Load
    X_rich, df_merged_rich, _ = load_rich_features_and_clusters(k=k_rich)

    # 2) Per-cluster stats
    df_stats = compute_cluster_stats(X_rich, df_merged_rich)
    stats_csv = FEATURES_DIR / f"cluster_stats_rich_k{k_rich}.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"Wrote rich cluster stats to {stats_csv}")

    # 3) Silhouette
    labels_rich = df_merged_rich["cluster_id"].values
    sil = compute_overall_silhouette(X_rich, labels_rich)
    print(f"Overall silhouette score (rich, k={k_rich}): {sil:.4f}")

    # Save silhouette + global metrics to a small text file
    metrics_txt = FEATURES_DIR / f"cluster_metrics_rich_k{k_rich}.txt"
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Silhouette score (rich, k={k_rich}): {sil:.6f}\n")
        f.write(f"Number of clusters: {len(df_stats)}\n")
        f.write(f"Total symbols: {len(df_merged_rich)}\n")

    print(f"Wrote rich clustering metrics to {metrics_txt}")

    # 4) Compare simple vs rich (if available)
    comp = compare_simple_vs_rich(k_simple=k_simple, k_rich=k_rich)
    if comp:
        with open(metrics_txt, "a", encoding="utf-8") as f:
            f.write(f"ARI (simple vs rich): {comp['ARI']:.6f}\n")
            f.write(f"NMI (simple vs rich): {comp['NMI']:.6f}\n")
        print("Appended simple-vs-rich comparison metrics to metrics file.")


if __name__ == "__main__":
    # Default: k=40 for both
    run_cluster_analysis(k_rich=40, k_simple=40)
