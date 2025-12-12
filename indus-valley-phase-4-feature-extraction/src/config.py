# src/config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SYMBOL_DIR = DATA_DIR / "symbols"
SEALS_PHASE1_DIR = DATA_DIR / "seals_phase1"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
VIZ_DIR = OUTPUT_DIR / "viz"

for d in [FEATURES_DIR, CLUSTERS_DIR, VIZ_DIR]:
    os.makedirs(d, exist_ok=True)

# subfolder names inside data/seals_phase1
PREPROC_DIRS = {
    "gray": "gray",
    "clahe": "clahe",
    "binary": "binary",
    "clean": "clean",
    "skeleton": "skeleton",
    "edge": "edge",
}

# filename suffixes (ADJUST EXTENSION IF NEEDED)
PREPROC_SUFFIXES = {
    "gray": "_gray.png",
    "clahe": "_clahe.png",
    "binary": "_binary.png",
    "clean": "_clean.png",
    "skeleton": "_skeleton.png",
    "edge": "_edge.png",
}
