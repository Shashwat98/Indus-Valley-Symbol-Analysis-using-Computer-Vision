## Unsupervised Visual Analysis of Indus Script Symbols using Classical Features and Self Supervised Transformers


Powerpoint: https://docs.google.com/presentation/d/13VcAxsmlXFiOYc1bDyohYp6pLs2QslseUz0_fkiM064/edit?usp=sharing \
Video: https://drive.google.com/file/d/1VwqPLZ5SS5pcE1xnCGmUw-6AwNcKjlmb/view?usp=sharing \
ViT Cluster Outputs: https://drive.google.com/drive/folders/1eKWs-uiHHacwXjqii_ezO9EEHDg2QJvs?usp=sharing


### Project Preview
indus-valley/ \
│ \
├── README.md \
│ \
├── indus-valley/                         # Phase 1 – Caption extraction & tablet cropping \
│   ├── build_captions_table.py \
│   ├── crop_tablets.py \
│   └── requirements.txt \
│ \
├── indus-valley-phase-2-preprocessing/   # Phase 2 – Image preprocessing & symbol candidates \
│   ├── image_preprocessing.py \
│   └── data/ \
│       ├── tablets_raw/                  # Phase 1 output (copied/linked) \
│       └── preprocessed/ \
│           ├── gray/ \
│           ├── clahe/ \
│           ├── edge/ \
│           ├── binary/ \
│           ├── clean/ \
│           ├── skeleton/ \
│           └── symbols/ \
│ \
├── indus-valley-phase-3-segmentation/    # Phase 3 – Symbol segmentation & metadata \
│   ├── segmentation.py \
│   ├── data/ \
│   │   ├── cropped/                      # Original tablet crops \
│   │   └── clean/                        # Clean binary images \
│   └── output/ \
│       ├── symbols/                      # 64×64 normalized symbol crops \
│       ├── overlays/                     # Debug visualizations \
│       ├── metadata/                     # Per-seal JSON metadata \
│       └── animals/ \
│ \
├── indus-valley-phase-4-feature-extraction/  # Phase 4 – Features & clustering \
│   ├── src/ \
│   │   ├── config.py \
│   │   ├── views.py \
│   │   ├── features.py \
│   │   ├── index_symbols.py \
│   │   ├── run_extract_features.py \
│   │   ├── run_extract_features_rich.py \
│   │   ├── prepare_features.py \
│   │   ├── prepare_features_rich.py \
│   │   ├── run_kmeans.py \
│   │   └── run_kmeans_rich.py \
│   │ \
│   ├── data/ \
│   │   ├── metadata/                     # Phase 3 JSON metadata \
│   │   └── seals_phase1/ \
│   │       ├── gray/ \
│   │       ├── clahe/ \
│   │       ├── edge/ \
│   │       ├── binary/ \
│   │       ├── clean/ \
│   │       └── skeleton/ \
│   │ \
│   └── outputs/ \
│       ├── features/ \
│       ├── clusters/ \
│       └── viz/ \
│ \
├── colab/                                # Google Colab experiments \
│   └── vit_dinov2_analysis.ipynb \
│ \
└── docs/ \
    ├── figures/                          # Paper figures \
    ├── tables/                           # CSVs for plots \
    └── report/ \


----------------------------------------------------------------------------------------------------------------------------------------------------

### Dependencies
- pytesseract			- OCR for captions / text
- pdf2image				- Convert scanned PDF pages to images		
- pillow				- Image I/O and manipulation
- pandas				- Metadata tables (CSV)
- numpy					- Numerical processing
- opencv-python			- Image processing & segmentation
- scikit-image			- Filters, skeletons, morphology
- tqdm 					- Progress bars

For tesseract and Poppler, you will have to download the package separately.
https://github.com/tesseract-ocr/tesseract
https://poppler.freedesktop.org/
----------------------------------------------------------------------------------------------------------------------------------------------------

### Phase 1: Caption Extraction & Tablet Cropping

This phase converts a scanned PDF of Indus plates into:

1. Structured caption metadata (captions.csv)
2. Individual cropped tablet images aligned with captions


--------------------------------------------------------------------------


**Step 1**: Build Caption Table from Scanned PDF 

Script: build_captions_table.py 

**Purpose**

1.Converts a scanned PDF into page images \
2.Runs OCR (Tesseract) on each page \
3.Detects caption codes (e.g., A-123, M-67 A (1) bis) \
4.Saves all detected captions with bounding box metadata into a CSV \

**Command to Run** \
```bash
python build_captions_table.py <path_to_pdf> --out_dir <output_dir> --dpi 300 

## Example
python build_captions_table.py data/CorpusVol1.pdf --out_dir phase1_output --dpi 300
```

**Inputs**

path_to_pdf: Scanned PDF of plates (DJVU-converted or scanned book) \
--dpi: DPI for PDF → image conversion (300–400 recommended) 

**Input File Location (Expected)**
data/
 └── CorpusVol1.pdf

**Outputs**
phase1_output/ \
 ├── captions.csv \
 └── pages_as_images/ \
     ├── page_001.png \
     ├── page_002.png \
     ├── ... 

**captions.csv Columns (Key)**
- Page – page number (1-based)
- Line_id – OCR line index
- Caption_index – index within line (0 = left, 1 = right)
- x, y, w, h – bounding box of caption line
- x_center – estimated horizontal center of the caption
- Normalized_code – cleaned caption (e.g., M-13 A, A-55 bis)

**Notes**
- Handles multiple captions per line
- Normalizes OCR quirks like M 13-A → M-13 A
- Page images are reused in the next step (do not delete)

--------------------------------------------------------------------------

**Step 2: Crop Tablet Images Using Captions**

Script: crop_tablets.py 


**Purpose**

- Uses caption locations from captions.csv
- Detects large contours (tablets) above captions
- Crops tablet images with padding

**Command to Run**
```bash
python crop_tablets.py \
  --captions <path_to_captions_csv> \
  --pages_dir <pages_image_dir> \
  --out_dir <output_dir> \
  --debug_dir <optional_debug_dir>

## Example
python crop_tablets.py \
  --captions phase1_output/captions.csv \
  --pages_dir phase1_output/pages_as_images \
  --out_dir phase1_output/cropped_tablets \
  --debug_dir phase1_output/debug_pages
```

**Inputs**

captions.csv:	Output from Step 1
pages_dir:	Directory containing page_XXX.png
--debug_dir:	(Optional) Saves overlay images

Input Directory Structure (Required)
phase1_output/ \
 ├── captions.csv \
 └── pages_as_images/ \
     ├── page_001.png \
     ├── page_002.png \
     ├── ...

Outputs
phase1_output/ \
 ├── cropped_tablets/ \
 │   ├── page003_M-13_A_c0.png \
 │   ├── page003_M-13_A_c1.png \
 │   └── ... \
 └── debug_pages/          (if enabled) \
     ├── page_003_debug.png \
     └── ...

Cropped Image Naming
page{PAGE}_{CAPTION_CODE}_c{INDEX}.png

Example: \
page014_M-67_A_c0.png

**Cropping Logic (Important for README)**

- Tablets are detected using large dark contours
- Selected tablet must:
  - Horizontally align with caption center
	- Appear above the caption
	- Be within reasonable vertical distance

Largest matching contour is selected
Padding ≈ 2% of page size

**Notes**

Small glyphs/text are ignored using area thresholds
Pages with no valid contour matches are skipped (warning logged)
Debug images are strongly recommended for validation

Phase 1 Execution Order (Recommended)
1. build_captions_table.py
2. crop_tablets.py

----------------------------------------------------------------------------------------------------------------------------------------------------


### Phase 2: Image Preprocessing  and  Symbol Crop Extraction

--------------------------------------------------------------------------

Script: image_preprocessing.py 


**Purpose**

Takes the cropped tablet images from Phase 1 and generates:
- grayscale / CLAHE versions
- edge magnitude + edge-binarized images
- cleaned binary edge maps
- skeletonized images
- connected-component symbol crops (glyph fragments, animal parts, etc.)

**Command to Run**
```bash
python image_preprocessing.py
```

**Inputs**
Required Input Folder
The script reads images from:
- data/tablets_raw/

Supported extensions: png, jpg, jpeg, tif, tiff, bmp. 

Copy the Phase 1 cropped tablet images into:

data/tablets_raw/ \
  page003_M-13_A_c0.png \
  page003_M-13_A_c1.png \
  ...

(Your Phase 1 output folder can be different, but for this Phase 2 script to work as-is, the files must be placed in data/tablets_raw/.) 


**Outputs**

All outputs are written under:

data/preprocessed/ \
 ├── gray/ \
 ├── clahe/ \
 ├── edge/ \
 ├── binary/ \
 ├── clean/ \
 ├── skeleton/ \
 └── symbols/ \

Output Files Produced (per input image)

If input is page003_M-13_A_c0.png, outputs include:

data/preprocessed/gray/     page003_M-13_A_c0_gray.png
data/preprocessed/clahe/    page003_M-13_A_c0_clahe.png
data/preprocessed/edge/     page003_M-13_A_c0_edge.png
data/preprocessed/binary/   page003_M-13_A_c0_binary.png
data/preprocessed/clean/    page003_M-13_A_c0_clean.png
data/preprocessed/skeleton/ page003_M-13_A_c0_skeleton.png

Symbol Crops

Connected components are cropped from the cleaned binary edge-map and saved as:

data/preprocessed/symbols/
  page003_M-13_A_c0_sym000.png
  page003_M-13_A_c0_sym001.png
  ...


These are candidate symbol fragments, not guaranteed full glyphs. 

**Processing Details**

- Crops off light borders using threshold BORDER_THRESH=240 
- Resizes so max dimension ≤ MAX_SIZE=768 
- Optional CLAHE contrast enhancement (USE_CLAHE=True) 
- Edge detection uses Scharr on denoised image + Otsu thresholding 
- Cleaning uses morphological open/close + removes small CCs
  - MIN_CLEAN_AREA=150 for cleaning 
  - MIN_SYMBOL_AREA=80 for symbol crops 
- Skeletonization uses skimage.morphology.skeletonize 

Notes:

Phase 2 Run Order
1) Ensure Phase 1 cropped tablet images exist
2) Copy them into data/tablets_raw/
3) Run: python image_preprocessing.py
4) Verify outputs in data/preprocessed/

----------------------------------------------------------------------------------------------------------------------------------------------------

### Phase 3: Segmentation (Text Band, Animal Region, Symbol Crops + Metadata)

**Purpose**

**Given:**

the original cropped seal/tablet images
the corresponding clean binary images (edge-cleaned)

**This script:**

- runs connected components on the clean binary image
- estimates the text band (where symbols are)
- estimates animal region bbox
- groups components into symbol bounding boxes
- writes:
  - normalized 64×64 symbol crops
  - JSON metadata per seal
  - debug overlays (components + text band + animal bbox + symbol boxes)

**Command to Run**
```bash
python segmentation.py
```
**Inputs**

Required Input Folders
<phase-3-segmentation>/ \
 ├── segmentation.py \
 └── data/ \
     ├── cropped/ \
     │   ├── <seal_id>.png \
     │   └── ... \
     └── clean/ \
         ├── <seal_id>_clean.png \
         └── ...

**Outputs**

All outputs go to:

<phase-3-segmentation>/output/

**Created automatically:**

output/ \
 ├── overlays/ \
 ├── symbols/ \
 ├── animals/      (directory created, not populated by this script yet) \
 └── metadata/ 

output/symbols/<seal_id>_sym_###.png

Metadata JSON (per seal)

**Saved to:**
output/metadata/<seal_id>.json

**Contains:**
- seal_id
- paths to source images
- text_band (y1,y2 in clean coords)
- animal_bbox_clean
- list of symbol entries with bboxes (clean + orig) and saved crop path

### Phase 3 Run Order
1) Ensure Phase 1 produced cropped images (original crops)
2) Ensure Phase 2 produced clean images (<seal_id>_clean.png)
3) Copy/link them into Phase 3 folders: 
   - data/cropped/ 
   - data/clean/ 
4) Run: python segmentation.py
5) Verify:
   - output/overlays/*_symbols.png
   - output/symbols/*.png
   - output/metadata/*.json

----------------------------------------------------------------------------------------------------------------------------------------------------

### Phase 4: Feature Extraction

**Run Order**
```bash
python -m src.index_symbols
python -m src.run_extract_features
python -m src.prepare_features
python -m src.run_kmeans
python -m src.run_extract_features_rich
python -m src.prepare_features_rich
python -m src.run_kmeans_rich
```

**At last run the Google Collab notebook cell by cell to extract features using DINOv2. You will need to mount google drive with the folder "symbols_64"
containing the 64x64 symbol crops.**

------------------------------------------------------------------------------------------------------------------------------------

### License & Data Disclaimer

Code: Academic use only
Dataset: Rights belong to original publishers
Outputs intended for research and analysis

Author Shashwat Singh 
Course: Advanced Perception CS-7180

I would like to point out that ChatGPT was of immense help while formulating the methodology, what would be the best practices to overcome a various challenges that I faced throughout the project, and the code structure.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
