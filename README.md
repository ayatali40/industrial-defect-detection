# Industrial Defect Detection and Localization

Unsupervised anomaly detection and pixel-level localization for industrial inspection,
implemented end-to-end on **MVTec AD** with **PatchCore** and **PaDiM**. Models are trained
only on defect-free images, thresholds are calibrated only on held-out normal images, and
the official test split is used for the final evaluation.

This is the final release of the multi-category prototype. The project scope is **final and
frozen** at exactly six MVTec AD categories: **`transistor`, `cable`, `leather`, `bottle`,
`capsule` and `grid`**. Each supported category has its own independently fitted PatchCore
and PaDiM model, its own normal-only calibration and its own test evaluation; nothing is
shared or silently substituted across categories.

**The remaining nine MVTec AD categories are explicitly out of scope** (`carpet`,
`hazelnut`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `wood`, `zipper`). They were
not downloaded, trained, evaluated or made selectable, and no results are claimed for them.
The scope was frozen to keep the release reproducible and within the local CPU-only/storage
budget. Consequently this is a **six-category prototype, not a complete 15-category MVTec AD
benchmark**, and all reported results and aggregates apply only to these six categories.

This is a research prototype, not a certified production inspection system.

## Results (held-out MVTec AD test splits)

Thresholds are calibrated per category and per model on held-out normal training images
only (`bottle`: 167 fit / 42 calibration; `cable`: 179 / 45; `capsule`: 175 / 44;
`grid`: 211 / 53; `leather`: 196 / 49; `transistor`: 170 / 43).

| Category | Model | Test normal | Test anomalous | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | AUPRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | PatchCore | 20 | 63 | 1.0000 | 1.0000 | 1.0000 | 0.9856 | 0.7541 | 0.4598 | 0.9997 |
| bottle | PaDiM | 20 | 63 | 1.0000 | 1.0000 | 0.9920 | 0.9767 | 0.5750 | 0.4012 | 0.9999 |
| cable | PatchCore | 58 | 92 | 0.9696 | 0.9808 | 0.9348 | 0.9815 | 0.6642 | 0.5348 | 0.9800 |
| cable | PaDiM | 58 | 92 | 0.9798 | 0.9887 | 0.9399 | 0.9789 | 0.5926 | 0.4374 | 0.9881 |
| capsule | PatchCore | 23 | 109 | 0.9517 | 0.9884 | 0.8776 | 0.9874 | 0.4424 | 0.2863 | 0.9918 |
| capsule | PaDiM | 23 | 109 | 0.9541 | 0.9881 | 0.4429 | 0.9811 | 0.3552 | 0.2246 | 0.9936 |
| grid | PatchCore | 21 | 57 | 0.9674 | 0.9893 | 0.9550 | 0.9668 | 0.2656 | 0.1493 | 0.9503 |
| grid | PaDiM | 21 | 57 | 0.9098 | 0.9694 | 0.8679 | 0.9285 | 0.0821 | 0.1227 | 0.8728 |
| leather | PatchCore | 32 | 92 | 1.0000 | 1.0000 | 0.9583 | 0.9890 | 0.3340 | 0.0739 | 1.0000 |
| leather | PaDiM | 32 | 92 | 1.0000 | 1.0000 | 0.9946 | 0.9787 | 0.2138 | 0.0620 | 1.0000 |
| transistor | PatchCore | 60 | 40 | 0.9883 | 0.9867 | 0.9041 | 0.9644 | 0.6334 | 0.6005 | 0.9442 |
| transistor | PaDiM | 60 | 40 | 0.9950 | 0.9929 | 0.7500 | 0.9829 | 0.7443 | 0.6681 | 0.9894 |

At the calibrated image thresholds:

| Category | Model | TP | Anomalies | FP | Normals |
| --- | --- | ---: | ---: | ---: | ---: |
| bottle | PatchCore | 63 | 63 | 0 | 20 |
| bottle | PaDiM | 62 | 63 | 0 | 20 |
| cable | PatchCore | 86 | 92 | 6 | 58 |
| cable | PaDiM | 86 | 92 | 5 | 58 |
| capsule | PatchCore | 86 | 109 | 1 | 23 |
| capsule | PaDiM | 31 | 109 | 0 | 23 |
| grid | PatchCore | 53 | 57 | 1 | 21 |
| grid | PaDiM | 46 | 57 | 3 | 21 |
| leather | PatchCore | 92 | 92 | 8 | 32 |
| leather | PaDiM | 92 | 92 | 1 | 32 |
| transistor | PatchCore | 33 | 40 | 0 | 60 |
| transistor | PaDiM | 24 | 40 | 0 | 60 |

Two operating-point effects appear in the confusion counts:

- **Leather** has perfect image-level ranking, but its ground-truth defects are tiny (median
  0.23-1.68% of the image), so a pixel threshold calibrated at 0.5% normal false positives
  yields much larger predicted regions than the masks (recall 1.0, precision ~0.03). Pixel
  AUROC remains 0.979-0.989.
- **Capsule / PaDiM** ranks well (image AUROC 0.9541, image AP 0.9881) but its calibrated
  image threshold (76.59) is far above most anomalous scores, so only 31/109 anomalies are
  flagged. The threshold was set at the 99th percentile of held-out normal validation scores
  only; it was not tuned on the test split. PatchCore on capsule detects 86/109.

Reports:

- [`reports/final_report.md`](reports/final_report.md) - six-category report, frozen scope,
  aggregation method and limitations
- [`reports/categories/`](reports/categories/) - detailed per-category reports
- [`reports/metrics/cross_category_metrics.csv`](reports/metrics/cross_category_metrics.csv)
  - per-category rows plus explicitly labelled `MACRO_AVERAGE` and `POOLED` rows
- `reports/metrics/<category>/model_comparison.csv` - per-category model comparison
- [`reports/storage_audit.md`](reports/storage_audit.md) - measured storage, unique vs
  duplicate files, and keep/delete classification (nothing was deleted)
- [`reports/final_release_checklist.md`](reports/final_release_checklist.md) - release checks,
  artifact-storage strategy and verdict

### Cross-category aggregation

With six evaluated categories the aggregation code reports both views explicitly:

- **Macro average** gives every evaluated category equal weight (`MACRO_AVERAGE` rows):
  PatchCore image AUROC 0.9795 / pixel AUROC 0.9791 / AUPRO 0.9776; PaDiM 0.9731 / 0.9711 /
  0.9739.
- **Pooled confusions** recompute thresholded P/R/F1 from summed TP/FP/FN, so image rows are
  weighted by image count and pixel rows by pixel count (`POOLED` rows). PatchCore pooled
  image F1 0.9365, pixel F1 0.3456; PaDiM image F1 0.8493, pixel F1 0.3146.
- **AUROC, AP and AUPRO are never pooled** from per-category summaries; only their macro
  average is reported.
- The easiest/hardest ranking is descriptive and reports each metric separately (image AUROC:
  easiest `leather` / PatchCore 1.0000, tied with `bottle` / PatchCore at 1.0000; hardest
  `grid` / PaDiM 0.9098. AUPRO: easiest `leather` / PaDiM 1.0000, hardest `grid` / PaDiM
  0.8728). It is not a single composite score.

## Scope

- **Category-isolated.** Every category has its own fit/calibration split manifest, model
  artifact, image and pixel thresholds, test predictions, metrics and figures. Nothing is
  shared or silently substituted across categories.
- **Exactly six categories are supported** (`transistor`, `cable`, `leather`, `bottle`,
  `capsule`, `grid`). They are declared in `data.supported_categories`; the CLI, Streamlit
  app and reports refuse every other MVTec AD category as out of scope.
- **The scope is permanently frozen.** Additional categories are not part of this release
  and are not enabled by adding data. This keeps the 12 category/model artifacts, metrics
  and cross-category aggregates stable and reproducible.
- **No bounding-box detector.** This is unsupervised anomaly detection plus pixel-level
  localization, not YOLO-style supervised detection.
- **CPU-only.** Developed and evaluated on an AMD64 Windows machine with 12 logical CPUs,
  7.95 GB RAM and no CUDA (`torch 2.10.0+cpu`). Batch commands run categories sequentially
  and release objects between categories.

## Dataset

The [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) is not
included or downloaded by this repository. Obtain it from the official source and follow
[`data/raw/README.md`](data/raw/README.md). The project never downloads from unofficial
mirrors and never generates synthetic dataset samples.

- Supported and evaluated (6): `transistor`, `cable`, `leather`, `bottle`, `capsule`, `grid`
- MVTec AD registry (15 categories) is retained only for name validation; the other nine
  (`carpet`, `hazelnut`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `wood`,
  `zipper`) are explicitly out of scope and are never loaded or evaluated
- `bottle`: 209 defect-free training images, 20 normal and 63 anomalous test images, 63 masks
- `cable`: 224 defect-free training images, 58 normal and 92 anomalous test images, 92 masks
- `capsule`: 219 defect-free training images, 23 normal and 109 anomalous test images,
  109 masks
- `grid`: 264 defect-free training images, 21 normal and 57 anomalous test images, 57 masks
- `leather`: 245 defect-free training images, 32 normal and 92 anomalous test images, 92 masks
- `transistor`: 213 defect-free training images, 60 normal and 40 anomalous test images,
  40 masks; images are 1024x1024 RGB PNG
- The six supported categories live one level next to the project folder and are mapped
  through `data.category_roots` (for example `bottle: ../bottle/bottle`)
- License: **CC BY-NC-SA 4.0** (non-commercial); see the dataset's `license.txt`
- Citation: Bergmann et al., "A Comprehensive Real-World Dataset for Unsupervised Anomaly
  Detection", CVPR 2019

`configs/config.yaml` accepts either:

1. `paths.data_root` pointing at an MVTec root containing `<category>/` folders, or
2. `paths.data_root` pointing at one category folder (the current `transistor/` checkout), or
3. `data.category_roots` mapping individual categories to explicit folders.

These layout options apply to the six supported categories only. The scope is frozen, so a
new category is not enabled merely by supplying data.

## Setup

Python 3.10 is recommended (verified with 3.10.9).

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use .venv/bin/activate on Linux/macOS
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

For a CPU-only PyTorch wheel:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

## Frozen scope

The supported scope is declared in `configs/config.yaml`:

```yaml
data:
  category: transistor
  supported_categories: [transistor, cable, leather, bottle, capsule, grid]
  categories: [transistor, cable, leather, bottle, capsule, grid]
  category_roots:
    cable: ../cable/cable
    leather: ../leather/leather
    bottle: ../bottle/bottle
    capsule: ../capsule/capsule
    grid: ../grid/grid
```

`--categories all` resolves to these six categories, not to the 15-name registry. Passing an
out-of-scope category (for example `--category carpet`) fails with an explicit
`outside the frozen project scope` error. The same check runs before an artifact is loaded,
so no out-of-scope model or threshold can be served, and the app never falls back to another
category. Adding more categories is deliberately not supported in this release.

## Reproduce the reported results

Run from the repository root, in order. `--categories active` processes the configured
scope sequentially and records per-category status:

```bash
python scripts/01_validate_dataset.py --categories active
python scripts/02_explore_dataset.py --categories active
python scripts/04_train.py --categories active
python scripts/05_calibrate.py --categories active
python scripts/06_evaluate.py --categories active
python scripts/08_generate_reports.py
```

Expected here: all six supported categories (`transistor`, `cable`, `leather`, `bottle`,
`capsule`, `grid`) complete. The other nine MVTec AD categories are out of scope and are not
attempted, so they are neither downloaded nor reported as runnable. The commands return a
non-zero exit code only when a requested supported category cannot run.

Measured on the reference machine:

| Category | Audit | Fit PatchCore / PaDiM | Calibration normals | Test evaluation PatchCore / PaDiM |
| --- | --- | --- | ---: | --- |
| bottle | 292 images, 63 masks, 0 critical issues | 22.74 s / 20.52 s | 42 | 17.64 s / 11.65 s for 83 images |
| cable | 374 images, 92 masks, 0 critical issues | 27.88 s / 26.03 s | 45 | 29.76 s / 26.08 s for 150 images |
| capsule | 351 images, 109 masks, 0 critical issues | 27.59 s / 27.26 s | 44 | 25.28 s / 18.84 s for 132 images |
| grid | 342 images, 57 masks, 0 critical issues | 30.26 s / 27.77 s | 53 | 11.05 s / 9.84 s for 78 images |
| leather | 369 images, 92 masks, 0 critical issues | 28.64 s / 24.33 s | 49 | 22.37 s / 21.67 s for 124 images |
| transistor | 313 images, 40 masks, 0 critical issues | 22.89 s / 21.96 s | 43 | 16.58 s / 14.82 s for 100 images |

Generated artifacts:

- `models/<category>/<model>/artifact.pt` - fitted model (git-ignored)
- `models/<category>/<model>/thresholds.json` - calibration record
- `models/<category>/<model>/training.json` - training and reload-validation record
- `reports/metrics/<category>/` - audit, EDA, metrics, model comparison, error analysis
- `reports/metrics/cross_category_metrics.{csv,json}` - explicit cross-category aggregation
- `reports/categories/<category>.md` - detailed per-category report
- `reports/figures/<category>/` - EDA, inspection panels, contact sheets, score distributions
- `reports/predictions/<category>/` - per-image scores and full anomaly maps (git-ignored;
  regenerate with `scripts/06_evaluate.py`)

The existing `transistor` artifacts were migrated to the nested layout without retraining
or recalibration. The migration is verified against the saved test predictions:
`reports/metrics/artifact_migration.json` (maximum score difference ~1e-14).

Notebooks are executed end-to-end and committed with outputs:

- [`notebooks/01_dataset_exploration.ipynb`](notebooks/01_dataset_exploration.ipynb)
- [`notebooks/02_anomaly_detection_experiments.ipynb`](notebooks/02_anomaly_detection_experiments.ipynb)
- [`notebooks/03_results_and_interpretation.ipynb`](notebooks/03_results_and_interpretation.ipynb)

## Inference

List available categories and their status:

```bash
python scripts/07_predict.py --list-categories
```

Single image or folder, for an explicitly selected category:

```bash
python scripts/07_predict.py --category transistor --input path/to/image.png --model both
python scripts/07_predict.py --category transistor --input path/to/folder --model padim \
  --output reports/predictions/demo
```

The selected category must have both an artifact and calibrated thresholds. Otherwise the
CLI fails with an explicit message; it never falls back to another category's model.
Outputs per image: anomaly heatmap, overlay, thresholded defect mask and JSON metadata
(score, thresholds, decision, predicted area, latency). A run-level
`inference_manifest.json` records the environment and all results. Default outputs are
written under `reports/predictions/cli/<category>/`.

Streamlit prototype:

```bash
streamlit run app/streamlit_app.py
```

It provides a selector containing exactly the six supported categories, a model selector
containing only that category's trained models, a status table for the six categories,
PASS/DEFECT status, score/threshold/area metrics, original/heatmap/overlay views, provenance
details and PNG/JSON/ZIP downloads. The nine out-of-scope categories are not offered or
loaded; a missing artifact produces an explicit error with no fallback.

Python API:

```python
from industrial_defect.config import load_config
from industrial_defect.inference import IndustrialInspector

config = load_config("configs/config.yaml")
inspector = IndustrialInspector(config, model_name="patchcore", category="transistor")
result = inspector.inspect("path/to/image.png")
print(result.score, result.decision, result.defect_area_fraction)
result.heatmap      # RGB uint8 (512, 512, 3)
result.overlay      # RGB uint8 (512, 512, 3)
result.predicted_mask
```

## Method

1. **Backbone.** Frozen ImageNet ResNet-50 (`layer2` + `layer3`), 3x3 local average pooling,
   layer3 bilinearly aligned to layer2, concatenated to a 1536-channel 28x28 grid at 224px.
2. **PatchCore.** Random projection to 256 dimensions, candidate patch cap 15000, greedy
   k-center coreset at 1% (1333 patches), k=1 kNN distance map.
3. **PaDiM.** 150 randomly selected feature dimensions, one multivariate Gaussian per
   spatial position, shrinkage epsilon 0.01, Mahalanobis distance, per-position
   standardisation.
4. **Post-processing.** Maps are upsampled to 512x512 and smoothed with a separable
   Gaussian (sigma 4.0) at output resolution. The image score is the maximum map value.
5. **Calibration.** Per category: normal-only split (80% fit / 20% calibration, seed 42,
   recorded manifest). Image threshold = p99 of validation image scores; pixel threshold =
   p99.5 of validation pixel values.
6. **Evaluation.** Official test split per category; pooled pixel metrics at 512x512 (masks
   downsampled with max-pooling); AUPRO up to a pixel false-positive rate of 0.3. Metric
   definitions are unchanged from the single-category implementation.

## Evaluation integrity

- Training never sees anomalous images. `assert_normal_only` enforces this, and each split
  manifest records only `train/good` images for its category.
- Thresholds are computed only from held-out normal training images of the same category.
  The threshold JSON records `test_data_used: false`.
- The test split is used only for the final reported metrics and error-analysis figures;
  no hyperparameter, threshold or model selection is derived from test labels.
- EDA summarises test annotations descriptively (defect area/fragmentation), but no
  modelling decision is derived from it.
- Category isolation is enforced and tested: paths, split manifests, thresholds, model
  state, predictions and metrics are never shared.
- **Disclosed re-evaluation.** During implementation QA, Gaussian smoothing was found to be
  applied on the 28x28 feature grid instead of the output-resolution map, which blurred and
  inflated predicted regions. The bug was fixed in both models, thresholds were recalibrated
  on the same normal-only validation split, and the test evaluation was re-run. Only the
  final (current-artifact) metrics are reported. Test labels were never used for fitting,
  calibration or model selection.

## Tests and lint

```bash
python -m pytest        # 78 tests, ~30 s on the reference machine
python -m ruff check src scripts tests app
```

The suite covers configuration, preprocessing and mask alignment, dataset layout and
normal-only enforcement, the category registry, the frozen six-category scope and
out-of-scope rejection, category-scoped artifact and threshold paths, split isolation,
inference category selection and missing-model handling,
cross-category aggregation, models and coreset determinism, threshold calibration,
image/pixel/PRO metrics, inference validation, artifact-score consistency and visualization
outputs. Integration tests use the real dataset and artifacts and are skipped when they are
absent.

## Project structure

```
app/streamlit_app.py           Streamlit inspection UI (category-aware)
configs/config.yaml            Central configuration and active category scope
data/raw/README.md             Dataset download and license instructions
models/README.md               Artifact notes (binaries git-ignored)
notebooks/                     Executed EDA, experiments and results notebooks
reports/categories/            Detailed per-category reports
reports/figures/<category>/    Committed figures, isolated per category
reports/metrics/<category>/    Committed metrics and metadata, isolated per category
reports/final_report.md        Six-category report, frozen scope and limitations
scripts/01..09_*.py            Pipeline stages (validation, EDA, training, calibration,
                               evaluation, reports, prediction, artifact migration)
src/industrial_defect/         Library code (categories, inventory, cross_category, ...)
tests/                         Pytest suite
```

## Limitations

- The project supports exactly six categories (`transistor`, `cable`, `leather`, `bottle`,
  `capsule`, `grid`). The other nine MVTec AD categories are intentionally out of scope; no
  results are reported or implied for them. Cross-category conclusions are limited to these
  six categories.
- Category test sets are small (78-150 images), so metrics have wide confidence intervals;
  bottle, grid, capsule and leather have only 20-32 normal test images.
- Image thresholds come from 42-53 normal images; the empirical validation false-positive
  rate is ~2%, which is a coarse estimate. A single high-scoring normal validation image can
  move the percentile threshold substantially, as seen for capsule/PaDiM (31/109 detected at
  its calibrated threshold despite image AUROC 0.954).
- Pixel metrics are computed at 512x512, not the native 1024x1024, and are not directly
  comparable to papers reporting 1024x1024 results.
- Pixel precision/F1 depend strongly on defect size and the calibrated false-positive rate;
  leather's sub-1% defects show perfect recall but low precision at a 0.5% pixel FPR.
- Lighting, camera, fixture or material changes are untested; recalibrate per category and
  per deployment site.
- The models have no notion of defect type; they only score deviation from normal.

## License

Code in this repository is released under the MIT license (see [`LICENSE`](LICENSE)). The
MVTec AD dataset is licensed separately under CC BY-NC-SA 4.0, is not included here, and
must not be redistributed.
