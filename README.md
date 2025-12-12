Powerpoint: https://docs.google.com/presentation/d/13VcAxsmlXFiOYc1bDyohYp6pLs2QslseUz0_fkiM064/edit?usp=sharing
Video: https://drive.google.com/file/d/1VwqPLZ5SS5pcE1xnCGmUw-6AwNcKjlmb/view?usp=sharing

##Repository Structure

indus-valley-cv-pipeline/
│
├── indus-valley/
│   ├── scripts/
│   ├── data/
│   └── README.md
│
├── indus-valley-phase-2-preprocessing/
│   ├── scripts/
│   ├── data/
│   └── README.md
│
├── indus-valley-phase-3-segmentation/
│   ├── scripts/
│   ├── data/
│   └── README.md
│
├── indus-valley-phase-4-feature-extraction/
│   ├── scripts/
│   ├── data/
│   └── README.md
│
├── notebooks/
│   └── ViT_DINOv2_Feature_Extraction.ipynb
│
├── requirements.txt
└── README.md   ← (this file)

##Pipeline Overview

| Phase   | Description        | Input                | Output                       |
| ------- | ------------------ | -------------------- | ---------------------------- |
| Phase 1 | Tablet extraction  | Scanned PDF pages    | Cropped tablet images        |
| Phase 2 | Preprocessing      | Tablet images        | Cleaned tablet images        |
| Phase 3 | Segmentation       | Preprocessed tablets | Individual symbol images     |
| Phase 4 | Classical features | Symbol images        | Feature vectors + clusters   |
| ViT     | Deep embeddings    | Symbol images        | 384-D embeddings + attention |

##System Requirements

Hardware

CPU sufficient for Phases 1–4
GPU recommended for ViT extraction (Colab preferred)

Software

OS: Linux / macOS / Windows
Python: 3.8 – 3.11
Google Colab (for ViT step)

##Dependencies

opencv-python
numpy
pandas
matplotlib
scikit-image
scikit-learn
tqdm
Pillow

##Phase by Phase Execution

**Phase 1 – Tablet Extraction**

Folder: indus-valley/

**Input

Scanned PDF pages from Corpus of Indus Seals and Inscriptions

**Operations

Parse caption metadata
Detect tablet bounding boxes
Crop tablet images
Store metadata as CSV

Run:
cd indus-valley
python scripts/extract_tablets.py

Output:
data/tablets/
├── tablet_001.png
├── tablet_002.png
└── metadata.csv

**Phase 2 – Preprocessing**

Folder: indus-valley-phase-2-preprocessing/

**Input

Tablet images from Phase 1

**Operations

Grayscale conversion
CLAHE contrast enhancement
Adaptive thresholding
Edge detection
Skeletonization 

Run:

cd indus-valley-phase-2-preprocessing
python scripts/preprocess_tablets.py

Output:

data/preprocessed/
├── tablet_001.png
├── tablet_002.png

**Phase 3 – Symbol Segmentation**

Folder: indus-valley-phase-3-segmentation/

Input:

Preprocessed tablet images

Operations:

Contour detection
Noise filtering
Bounding-box based symbol cropping

Run:

cd indus-valley-phase-3-segmentation
python scripts/segment_symbols.py

Output:

data/symbols/
├── symbol_0001.png
├── symbol_0002.png

**Phase 4 – Classical Feature Extraction**

Folder: indus-valley-phase-4-feature-extraction/

**Input

Segmented symbol images

**Features Extracted

Shape descriptors
Geometric statistics
Contour-based features

Run:

cd indus-valley-phase-4-feature-extraction
python scripts/extract_features.py

Output:

data/features/
├── features.npy
├── features.csv
├── clusters.png

**ViT Feature Extraction (Google Collab)

Notebook: notebooks/ViT_DINOv2_Feature_Extraction.ipynb

**Why Colab?

DINOv2 ViT requires GPU & large memory
No local training required

**Steps

Open notebook in Google Colab
Set runtime → GPU
Upload symbol images (data/symbols/)
Run all cells

**Model

DINOv2 ViT-S/14
Self-supervised, no labels
384-dimensional embeddings

**Outputs

embeddings.npy
PCA & UMAP visualizations
Self-attention maps per symbol
Cluster comparison metrics (ARI, NMI)


**License & Data Disclaimer**

Code: Academic use only

Dataset: Rights belong to original publishers

Outputs intended for research and analysis

**Author**
Shashwat Singh
Course: Advanced Perception
