# Historical Indic Manuscript Layout Region Detection

An end-to-end, production-grade layout detection pipeline designed for historical Indic manuscripts (Devanagari, Sanskrit, Grantha, etc., across palm-leaf and paper). The system locates and classifies document layout regions into 5 standardized categories: **`header`**, **`footer`**, **`main_text`**, **`side_text`**, and **`filler`**.

Powered by a fine-tuned **YOLOv8** spatial detector combined with automated leaf perimeter estimation and post-processing boundary clipping.

---

## Visual Showcase: Detection in Action

### Complete Layout Segmentation (All 5 Classes on a Single Manuscript Page)
This showcase example demonstrates the detector identifying all five target classes simultaneously on a complex Indic manuscript folio:

![All 5 Classes Layout Detection Showcase](assets/showcase_all_5_classes_stacked.jpg)

### Detected Regions Breakdown

| Region Class | Confidence | Color in Overlay | Bounding Box `[x_min, y_min, x_max, y_max]` | Description & Reasoning |
|---|---|---|---|---|
| **`main_text`** | **91.1%** | **Red** | `[279, 210, 1755, 429]` | Primary manuscript text body envelope (unifying lines around binding hole). |
| **`header`** | **90.8%** | **Blue** | `[353, 106, 926, 182]` | Running title located in the top margin above the main body. |
| **`footer`** | **92.0%** | **Green** | `[857, 439, 1047, 514]` | Bottom margin text line / catchword. |
| **`side_text` (Left)** | **93.9%** | **Cyan** | `[31, 201, 210, 304]` | Lateral margin commentary / scholar gloss outside the main text body. |
| **`side_text` (Right)** | **88.3%** | **Cyan** | `[1785, 333, 1872, 449]` | Right margin gloss. |
| **`filler` (Top)** | **84.6%** | **Gray** | `[370, 5, 1673, 74]` | Modern digitization credit watermark ("Mumukshu Bhawan...") at top edge. |
| **`filler` (Bottom)** | **74.8%** | **Gray** | `[301, 536, 1476, 602]` | Digitization credit watermark ("Panini Kanya...") at bottom edge. |
| **`filler` (Guide)** | **74.7%** | **Gray** | `[247, 193, 267, 398]` | Vertical border ruling line. |
| **`filler` (Hole)** | **69.6%** | **Gray** | `[281, 276, 312, 311]` | Palm-leaf physical cord binding hole. |

<details>
<summary>Click to view JSON Metadata Output for this page</summary>

```json
{
  "image": "manuscript_folio.jpg",
  "width": 1920,
  "height": 607,
  "bbox_format": "[x_min,y_min,x_max,y_max]",
  "method": "yolo",
  "predictions": [
    {
      "label": "side_text",
      "confidence": 0.9387,
      "bbox": [31, 201, 210, 304],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "footer",
      "confidence": 0.9202,
      "bbox": [857, 439, 1047, 514],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "main_text",
      "confidence": 0.9112,
      "bbox": [279, 210, 1755, 429],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "header",
      "confidence": 0.9084,
      "bbox": [353, 106, 926, 182],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "side_text",
      "confidence": 0.8832,
      "bbox": [1785, 333, 1872, 449],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "filler",
      "confidence": 0.8456,
      "bbox": [370, 5, 1673, 74],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "filler",
      "confidence": 0.7483,
      "bbox": [301, 536, 1476, 602],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "filler",
      "confidence": 0.7473,
      "bbox": [247, 193, 267, 398],
      "reason": "yolo_detection",
      "needs_review": false
    },
    {
      "label": "filler",
      "confidence": 0.6956,
      "bbox": [281, 276, 312, 311],
      "reason": "yolo_detection",
      "needs_review": false
    }
  ]
}
```
</details>

---

## 1. What We Did

Historical Indic manuscripts (*pothi* palm leaves and paper folios) differ fundamentally from modern printed documents:
- **Binding Cord Holes**: Palm leaves contain physical string holes flanked by text. While visually resembling columns, semantically they constitute a **single continuous `main_text` body**.
- **Digitization Watermarks & Overlays**: Library stamps, microfilm notices, and modern digital watermarks frequently appear in top/bottom margin bands where headers/footers normally sit.
- **Marginal Glosses & Commentary**: Scribes and scholars added margin notes (`side_text`) with varying stroke thickness, script slants, and arbitrary orientations.
- **Tightly Cropped Pages**: Many digital folios have zero top/bottom margin; inventing absent headers/footers is an error.

### The 5 Standardized Classes
| Class | Semantic Definition | Handling & Domain Logic |
|---|---|---|
| **`main_text`** | Core body of the manuscript | Columns flanking string holes are merged into one comprehensive envelope. |
| **`header`** | Running headers, chapter titles, top folio numbers | Located in the top margin above the primary text body. |
| **`footer`** | Bottom-margin marks, catchwords, scribe signatures | Located in the bottom margin below the primary text body. |
| **`side_text`** | Marginalia, commentary, annotations, side folio numbers | Positioned in lateral margins outside the main body's x-span. |
| **`filler`** | Non-content elements | Modern watermarks/credits, pencil numbers, cord holes, border ruling lines. |

---

## 2. Why We Chose YOLOv8

During development, we implemented and benchmarked two distinct architectures:
1. **Method A: Zero-Shot Classical Vision (Rules Baseline)**:
   - Morphological kernels for text-line grouping, connected-component labeling, CLAHE contrast enhancement, and heuristic geometric rules.
2. **Method B: Deep Spatial Detector (YOLOv8)**:
   - YOLOv8 fine-tuned at 1024px resolution with position-sensitive augmentations.

### Empirical Comparison & Ablation

| Split | Metric | Rules Baseline (A) | YOLOv8 Model (B) | Outcome |
|---|---|---|---|---|
| **Validation** | **Macro mAP@0.5** | **0.253** | **0.992** | **YOLO (+0.739)** |
| Validation | `header` AP50 | 0.091 | 0.989 | YOLO wins |
| Validation | `footer` AP50 | 0.071 | 0.995 | YOLO wins |
| Validation | `main_text` AP50 | 1.000 | 0.995 | Tie within noise |
| Validation | `side_text` AP50 | 0.000 | 0.995 | YOLO wins |
| Validation | `filler` AP50 | 0.103 | 0.984 | YOLO wins |
|---|---|---|---|---|
| **Synthetic Test** | **Macro mAP@0.5** | **0.270** | **0.966** | **YOLO (+0.696)** |
| Synthetic Test | `header` AP50 | 0.223 | 0.986 | YOLO wins |
| Synthetic Test | `footer` AP50 | 0.161 | 0.937 | YOLO wins |
| Synthetic Test | `main_text` AP50 | 0.889 | 0.995 | YOLO wins |
| Synthetic Test | `side_text` AP50 | 0.000 | 0.995 | YOLO wins |
| Synthetic Test | `filler` AP50 | 0.076 | 0.915 | YOLO wins |
|---|---|---|---|---|
| **Real Test (9 Authentic Pages)** | **Macro mAP@0.5** | **0.304** | **0.402** | **YOLO (+0.098)** |
| Real Test | `main_text` AP50 | 1.000 | 1.000 | Tie (perfect recall) |
| Real Test | `filler` AP50 | 0.218 | 0.333 | YOLO wins |

### Why Rules Failed & Why YOLO Was Kept
- **Faded Ink & Script Variations**: Rule-based morphological thresholding shattered on faint ink, pencil annotations, and uneven illumination.
- **Marginalia Collapse**: The rules baseline achieved **0.000 AP on `side_text`** because margin annotations deviate in angle, stroke density, and spacing.
- **Deep Spatial Representation**: YOLOv8 learns script-independent spatial hierarchies, separating typed watermarks (`filler`) from authentic Indic headers with sub-second inference.
- **Decision**: In accordance with the decision criteria in `evaluation/DECISION.md`, the brittle rule-based pipeline was permanently retired to provide a clean, robust, and maintainable production system.

---

## 3. How It Works

```
                     ┌──────────────────────────┐
                     │   Raw Manuscript Image   │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ 1. Non-Destructive Load  │
                     │    & Boundary Estimation │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ 2. YOLOv8 Deep Detection │
                     │    (imgsz=1024, best.pt) │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ 3. Cross-Class NMS       │
                     │    & Area Thresholding   │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ 4. Leaf Boundary Clip    │
                     │    & Coordinate Clamp    │
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ 5. Dual Output Writer    │
                     │    - Annotated PNG copy  │
                     │    - Per-image JSON      │
                     │    - Combined JSON       │
                     └──────────────────────────┘
```

### Pipeline Details

1. **Non-Destructive Image Loading**:
   - Images are loaded strictly as analysis arrays.
   - Original input files are **never modified or overwritten** (verified by SHA-256 byte-immutability tests).
2. **Page & Leaf Boundary Estimation**:
   - Adaptive color and Canny edge segmentation isolate the physical leaf contour from contrasting background mats (e.g. blue or black digitization cloth).
3. **Deep Spatial Detection**:
   - Pretrained YOLOv8 fine-tuned with **position-preserving augmentations**:
     - `mosaic=0.0`: Disabled to prevent quadrant mixing (document headers belong at the top).
     - `fliplr=0.0` / `flipud=0.0`: Disabled to preserve Indic left-to-right text orientation.
     - Affine perturbations: `degrees=2.5`, `scale=0.15`, `shear=1.0`.
4. **Cross-Class Non-Maximum Suppression (NMS)**:
   - Resolves multi-class overlapping candidate boxes, ensuring clean non-conflicting regions.
5. **Geometric Boundary Clipping**:
   - Every detected bounding box `[x_min, y_min, x_max, y_max]` is clipped to both the estimated leaf boundary and image canvas dimensions.
6. **Calibrated Confidence & Review Routing**:
   - Detections with `confidence < 0.40` (configurable in `configs/default.yaml`) automatically set `"needs_review": true` to facilitate downstream human verification.

---

## 4. Repository Structure

```
immverse/
├── inference.py                  # Production CLI (single file and folder batch processing)
├── configs/
│   └── default.yaml              # Consolidated configuration (weights, thresholds, colors)
├── common/
│   ├── __init__.py
│   ├── schema.py                 # Data models, region labels, BGR colors, coordinate validation
│   ├── io_utils.py               # Non-destructive image loading, batch discovery, relative path utils
│   ├── page_boundary.py          # Leaf perimeter estimation & bounding box clipping
│   └── output.py                 # Visual overlay renderer, per-image JSON & predictions.json
├── method_yolo/                  # Production detection engine
│   ├── __init__.py
│   ├── predict.py                # Cached YOLO predictor
│   ├── postprocess.py            # Cross-class NMS, boundary clipping, area filtering
│   ├── kaggle_train.ipynb        # Standalone, self-driving training notebook for Kaggle GPU
│   ├── KAGGLE_STEPS.md           # Reproduction instructions for training on Kaggle
│   └── weights/
│       └── best.pt               # Trained YOLOv8 model weights
├── assets/                       # Visual showcase images for documentation
│   └── showcase_all_5_classes_stacked.jpg
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Precision, recall, AP@0.5, and macro mAP calculation
│   ├── calibration.py            # Expected Calibration Error (ECE) & reliability diagrams
│   ├── robustness.py             # Stress tests: Gaussian blur, ink fade, tilt, stains
│   ├── failure_cases.py          # Worst-5 failure extractor with diagnoses
│   ├── compare.py                # Benchmark comparator documenting the rules vs YOLO ablation
│   ├── DECISION.md               # Decision criteria and model selection record
│   └── reports/                  # Benchmark tables, calibration & robustness charts
├── tests/
│   ├── conftest.py
│   ├── test_schema.py            # Coordinate format & serialization validation
│   ├── test_clipping.py          # Boundary clipping & page isolation checks
│   ├── test_batch.py             # File discovery and non-destructive image loading
│   ├── test_originals.py         # SHA-256 byte immutability verification
│   ├── test_cli.py               # End-to-end CLI execution & missing weights handling
│   └── test_evaluation.py        # IoU and Average Precision numerical checks
├── tools/
│   └── download_data.py          # Automated manuscript scraper (archive.org & Wikimedia Commons)
├── data/
│   └── test_images/              # Sample manuscript images for out-of-the-box CLI execution
├── sources.csv                   # Public domain image origins & license manifest
├── requirements.txt              # Pinned production dependencies
└── README.md                     # Documentation
```

---

## 5. Quickstart & Execution

### 1. Installation
```bash
git clone <repo_url>
cd immverse
pip install -r requirements.txt
```

### 2. Run Inference
The CLI accepts relative paths and handles both a single image file or an entire directory:

```bash
# Process a single image
python inference.py --input ./data/test_images/02_a.jpg --output ./results

# Process an entire folder (batch mode)
python inference.py --input ./data/test_images --output ./results
```

#### Generated Outputs
- `results/annotated/<name>_annotated.png`: Visual overlay displaying color-coded bounding boxes, class labels, and confidence scores.
- `results/json/<name>.json`: Machine-readable metadata per processed image.
- `results/predictions.json`: Consolidated batch results across all images.

### 3. Run Unit Test Suite
```bash
pytest tests/ -v
```
All **19 tests pass**, verifying schema compliance, coordinate bounds, boundary clipping, batch processing, and SHA-256 non-destructive image immutability.

### 4. Run Evaluation Suite
```bash
# AP@0.5 and macro mAP computation across all splits
python evaluation/evaluate.py --method yolo

# Reliability curve and Expected Calibration Error (ECE)
python evaluation/calibration.py --method yolo

# Degradation stress-test (Gaussian blur, fade, tilt, stains)
python evaluation/robustness.py --method yolo

# Worst-5 failure case diagnosis and side-by-side visualization
python evaluation/failure_cases.py --method yolo
```

---

## 6. Data Acquisition & Sourcing

To train and evaluate on authentic historical documents, an automated collection pipeline was developed ([`tools/download_data.py`](tools/download_data.py)) targeting public-domain Indic manuscript collections:
1. **Internet Archive (`archive.org`)**: Advanced queries filtering Sanskrit, Devanagari, Grantha, and palm-leaf manuscripts (eGangotri, Jangamwadi Math, and Royal Asiatic Society collections).
2. **Wikimedia Commons API**: Curated historical Indic folios with verified open licenses across diverse scripts (Devanagari, Sharada, Newar, Grantha).

All source URLs, licenses (Public Domain, CC0, CC BY-SA), and institution titles are cataloged in [`sources.csv`](sources.csv).

To run a test scrape:
```bash
python tools/download_data.py --out ./data --test
```

---

## 7. Training & Reproducibility

Training was performed on a GPU environment using Kaggle. The self-driving notebook [`method_yolo/kaggle_train.ipynb`](method_yolo/kaggle_train.ipynb) contains the end-to-end training pipeline:
1. Upload the dataset to Kaggle.
2. Open `method_yolo/kaggle_train.ipynb`, attach the dataset, and enable GPU (T4 / P100).
3. Click **Run All**. The notebook automatically configures paths, executes training, evaluates checkpoints, and packages `best.pt` with metrics into `outputs.zip`.
4. Place the downloaded `best.pt` in `method_yolo/weights/best.pt`.

Detailed step-by-step instructions are documented in [`method_yolo/KAGGLE_STEPS.md`](method_yolo/KAGGLE_STEPS.md).

---

## 8. Known Edge Cases & Failure Diagnostics

Detailed visual diagnostics are recorded in [`evaluation/reports/failure_cases_yolo.md`](evaluation/reports/failure_cases_yolo.md):
1. **Faint Pencil Annotations**: Extremely low-contrast commentary written in soft pencil can have lower confidence ($< 0.40$), automatically flagged as `needs_review: true`.
2. **Fractured Leaf Perimeters**: Decayed or uneven palm leaf edges can cause tight bounding boxes around marginal text to touch estimated page boundaries.
3. **Sparse Ground Truth in Real Test**: Real-world evaluation pages (9 images) have sparse annotations in draft ground truth; model detections often highlight genuine marginal text not recorded in draft labels.

---

## 9. Licenses & Attribution

- **Ultralytics YOLOv8**: Distributed under the AGPL-3.0 License.
- **Manuscript Scans**: Public domain / CC-licensed historical images sourced from Wikimedia Commons and Internet Archive. Exact source URLs and licenses are cataloged in [`sources.csv`](sources.csv).
