# src/prepare_features.py
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from .config import FEATURES_DIR


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns that are numeric features,
    excluding identifiers.
    """
    exclude = {"symbol_id", "seal_id", "symbol_index"}
    return [c for c in df.columns if c not in exclude]


def run_prepare_features(features_csv: Path) -> None:
    df = pd.read_csv(features_csv)
    print(f"Loaded features from {features_csv} with shape {df.shape}")

    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns: {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)

    # 1) Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_path = FEATURES_DIR / "features_simple_scaled.npy"
    np.save(scaled_path, X_scaled)
    print(f"Saved scaled features to {scaled_path}, shape={X_scaled.shape}")

    scaler_path = FEATURES_DIR / "scaler_simple.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    # 2) PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "symbol_id": df["symbol_id"],
        "seal_id": df["seal_id"],
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
    })

    pca_csv = FEATURES_DIR / "pca_2d_simple.csv"
    pca_df.to_csv(pca_csv, index=False)
    print(f"Saved 2D PCA projection to {pca_csv}")

    pca_model_path = FEATURES_DIR / "pca_simple_2d.joblib"
    joblib.dump(pca, pca_model_path)
    print(f"Saved PCA model to {pca_model_path}")


if __name__ == "__main__":
    features_csv_path = FEATURES_DIR / "features_simple.csv"
    run_prepare_features(features_csv_path)
