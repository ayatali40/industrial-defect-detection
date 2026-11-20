# Industrial Defect Detection and Localization - Multi-Category Technical Report

Generated: 2026-11-18T20:33:28+00:00

Dataset: MVTec AD | Models: PatchCore (primary), PaDiM (secondary) | Normal-only training and calibration

## 1. Executive summary

The project scope is **final and frozen** at exactly six MVTec AD categories: `transistor`, `cable`, `leather`, `bottle`, `capsule`, `grid`. All **6** supported categories completed dataset validation, training, calibration and test evaluation: `transistor`, `cable`, `leather`, `bottle`, `capsule`, `grid`.

- Supported categories: 6 (`transistor`, `cable`, `leather`, `bottle`, `capsule`, `grid`).
- Out of scope: 9 MVTec AD categories (`carpet`, `hazelnut`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `wood`, `zipper`). They were not downloaded, trained, evaluated or made selectable.
- Trained but not evaluated: none.
- Dataset present but not trained: none.
- Supported category with dataset unavailable locally: 0 (none).

The scope was frozen to keep the release reproducible and to stay within the local CPU/storage budget. This is a six-category multi-category prototype; it is **not** a complete 15-category MVTec AD benchmark. A category is reported as evaluated only when its metrics files exist in the category-scoped metrics directory.

## 2. Category scope and status

Only the six supported categories are inventoried. The nine out-of-scope categories are excluded from selection, training, evaluation, reports and the CLI/app; their absence is intentional and not a missing-data failure.

| Category | Dataset | Trained models | Evaluated models |
| --- | --- | --- | --- |
| `transistor` | available | patchcore, padim | patchcore, padim |
| `cable` | available | patchcore, padim | patchcore, padim |
| `leather` | available | patchcore, padim | patchcore, padim |
| `bottle` | available | patchcore, padim | patchcore, padim |
| `capsule` | available | patchcore, padim | patchcore, padim |
| `grid` | available | patchcore, padim | patchcore, padim |

Status is derived from the filesystem: dataset layout, nested artifacts (`models/<category>/<model>/`), calibrated thresholds and evaluated metrics. A supported category can be selected in the CLI/app only when it has both an artifact and calibrated thresholds. Out of scope: `carpet`, `hazelnut`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `wood`, `zipper`.

## 3. Per-category results

### `transistor`

Detailed report: [`reports/categories/transistor.md`](categories/transistor.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 170 | 43 | 60 | 40 | 100 |
| padim | 170 | 43 | 60 | 40 | 100 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9883 | 0.9867 | 1.0000 | 0.8250 | 0.9041 | 0.9644 | 0.6334 | 0.5530 | 0.6569 | 0.6005 | 0.9442 |
| padim | 0.9950 | 0.9929 | 1.0000 | 0.6000 | 0.7500 | 0.9829 | 0.7443 | 0.5281 | 0.9090 | 0.6681 | 0.9894 |

Dataset readiness: **PASS**; train/good=213, test/good=60, test/anomalous=40.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 33 | 0 | 60 | 7 |
| padim | 24 | 0 | 60 | 16 |

### `cable`

Detailed report: [`reports/categories/cable.md`](categories/cable.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 179 | 45 | 58 | 92 | 150 |
| padim | 179 | 45 | 58 | 92 | 150 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9696 | 0.9808 | 0.9348 | 0.9348 | 0.9348 | 0.9815 | 0.6642 | 0.3791 | 0.9076 | 0.5348 | 0.9800 |
| padim | 0.9798 | 0.9887 | 0.9451 | 0.9348 | 0.9399 | 0.9789 | 0.5926 | 0.2839 | 0.9523 | 0.4374 | 0.9881 |

Dataset readiness: **PASS**; train/good=224, test/good=58, test/anomalous=92.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 86 | 6 | 52 | 6 |
| padim | 86 | 5 | 53 | 6 |

### `leather`

Detailed report: [`reports/categories/leather.md`](categories/leather.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 196 | 49 | 32 | 92 | 124 |
| padim | 196 | 49 | 32 | 92 | 124 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 1.0000 | 1.0000 | 0.9200 | 1.0000 | 0.9583 | 0.9890 | 0.3340 | 0.0384 | 1.0000 | 0.0739 | 1.0000 |
| padim | 1.0000 | 1.0000 | 0.9892 | 1.0000 | 0.9946 | 0.9787 | 0.2138 | 0.0320 | 1.0000 | 0.0620 | 1.0000 |

Dataset readiness: **PASS**; train/good=245, test/good=32, test/anomalous=92.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 92 | 8 | 24 | 0 |
| padim | 92 | 1 | 31 | 0 |

### `bottle`

Detailed report: [`reports/categories/bottle.md`](categories/bottle.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 167 | 42 | 20 | 63 | 83 |
| padim | 167 | 42 | 20 | 63 | 83 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9856 | 0.7541 | 0.2987 | 0.9989 | 0.4598 | 0.9997 |
| padim | 1.0000 | 1.0000 | 1.0000 | 0.9841 | 0.9920 | 0.9767 | 0.5750 | 0.2510 | 0.9999 | 0.4012 | 0.9999 |

Dataset readiness: **PASS**; train/good=209, test/good=20, test/anomalous=63.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 63 | 0 | 20 | 0 |
| padim | 62 | 0 | 20 | 1 |

### `capsule`

Detailed report: [`reports/categories/capsule.md`](categories/capsule.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 175 | 44 | 23 | 109 | 132 |
| padim | 175 | 44 | 23 | 109 | 132 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9517 | 0.9884 | 0.9885 | 0.7890 | 0.8776 | 0.9874 | 0.4424 | 0.1688 | 0.9426 | 0.2863 | 0.9918 |
| padim | 0.9541 | 0.9881 | 1.0000 | 0.2844 | 0.4429 | 0.9811 | 0.3552 | 0.1280 | 0.9151 | 0.2246 | 0.9936 |

Dataset readiness: **PASS**; train/good=219, test/good=23, test/anomalous=109.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 86 | 1 | 22 | 23 |
| padim | 31 | 0 | 23 | 78 |

### `grid`

Detailed report: [`reports/categories/grid.md`](categories/grid.md)

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 211 | 53 | 21 | 57 | 78 |
| padim | 211 | 53 | 21 | 57 | 78 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9674 | 0.9893 | 0.9815 | 0.9298 | 0.9550 | 0.9668 | 0.2656 | 0.0815 | 0.8873 | 0.1493 | 0.9503 |
| padim | 0.9098 | 0.9694 | 0.9388 | 0.8070 | 0.8679 | 0.9285 | 0.0821 | 0.0665 | 0.7982 | 0.1227 | 0.8728 |

Dataset readiness: **PASS**; train/good=264, test/good=21, test/anomalous=57.

Decision outcomes:

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 53 | 1 | 20 | 4 |
| padim | 46 | 3 | 18 | 11 |

## 4. Cross-category analysis

### 4.1 Aggregation methods

- **Macro average**: unweighted arithmetic mean of the per-category values. Every evaluated category contributes equally, regardless of test-set size. It is labelled `MACRO_AVERAGE` in `reports/metrics/cross_category_metrics.csv`.
- **Pooled confusions**: thresholded precision/recall/F1 recomputed from TP/FP/FN summed across categories. Image rows are therefore weighted by image count and pixel rows by pixel count. Labelled `POOLED` in the CSV.
- **Ranking metrics (AUROC, AP, AUPRO) are not pooled** from per-category summaries. Only their unweighted macro average is reported; a pooled ranking metric requires the raw scores and is intentionally not computed here.

### 4.2 Per-category metrics

### patchcore

| Category | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transistor` | 0.9883 | 0.9867 | 1.0000 | 0.8250 | 0.9041 | 0.9644 | 0.6334 | 0.5530 | 0.6569 | 0.6005 | 0.9442 |
| `cable` | 0.9696 | 0.9808 | 0.9348 | 0.9348 | 0.9348 | 0.9815 | 0.6642 | 0.3791 | 0.9076 | 0.5348 | 0.9800 |
| `leather` | 1.0000 | 1.0000 | 0.9200 | 1.0000 | 0.9583 | 0.9890 | 0.3340 | 0.0384 | 1.0000 | 0.0739 | 1.0000 |
| `bottle` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9856 | 0.7541 | 0.2987 | 0.9989 | 0.4598 | 0.9997 |
| `capsule` | 0.9517 | 0.9884 | 0.9885 | 0.7890 | 0.8776 | 0.9874 | 0.4424 | 0.1688 | 0.9426 | 0.2863 | 0.9918 |
| `grid` | 0.9674 | 0.9893 | 0.9815 | 0.9298 | 0.9550 | 0.9668 | 0.2656 | 0.0815 | 0.8873 | 0.1493 | 0.9503 |

### padim

| Category | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transistor` | 0.9950 | 0.9929 | 1.0000 | 0.6000 | 0.7500 | 0.9829 | 0.7443 | 0.5281 | 0.9090 | 0.6681 | 0.9894 |
| `cable` | 0.9798 | 0.9887 | 0.9451 | 0.9348 | 0.9399 | 0.9789 | 0.5926 | 0.2839 | 0.9523 | 0.4374 | 0.9881 |
| `leather` | 1.0000 | 1.0000 | 0.9892 | 1.0000 | 0.9946 | 0.9787 | 0.2138 | 0.0320 | 1.0000 | 0.0620 | 1.0000 |
| `bottle` | 1.0000 | 1.0000 | 1.0000 | 0.9841 | 0.9920 | 0.9767 | 0.5750 | 0.2510 | 0.9999 | 0.4012 | 0.9999 |
| `capsule` | 0.9541 | 0.9881 | 1.0000 | 0.2844 | 0.4429 | 0.9811 | 0.3552 | 0.1280 | 0.9151 | 0.2246 | 0.9936 |
| `grid` | 0.9098 | 0.9694 | 0.9388 | 0.8070 | 0.8679 | 0.9285 | 0.0821 | 0.0665 | 0.7982 | 0.1227 | 0.8728 |

### 4.3 Macro average (equal weight per category)

| Model | Categories | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 6 | 0.9795 | 0.9909 | 0.9383 | 0.9791 | 0.5156 | 0.3508 | 0.9776 |
| padim | 6 | 0.9731 | 0.9898 | 0.8312 | 0.9711 | 0.4272 | 0.3193 | 0.9739 |

### 4.4 Pooled thresholded metrics (weighted by sample count)

| Model | Test normal | Test anomalous | Image TP | Image FP | Image FN | Image P | Image R | Image F1 | Pixel P | Pixel R | Pixel F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 214 | 453 | 413 | 16 | 40 | 0.9627 | 0.9117 | 0.9365 | 0.2158 | 0.8676 | 0.3456 |
| padim | 214 | 453 | 341 | 9 | 112 | 0.9743 | 0.7528 | 0.8493 | 0.1886 | 0.9479 | 0.3146 |

### 4.5 Easiest and hardest categories

- **image AUROC**: easiest `leather` / patchcore (1.0000); hardest `grid` / padim (0.9098).
- **AUPRO**: easiest `leather` / padim (1.0000); hardest `grid` / padim (0.8728).

### 4.6 Model comparison

- `transistor`: patchcore image AUROC 0.9883, AUPRO 0.9442; padim image AUROC 0.9950, AUPRO 0.9894.
- `cable`: patchcore image AUROC 0.9696, AUPRO 0.9800; padim image AUROC 0.9798, AUPRO 0.9881.
- `leather`: patchcore image AUROC 1.0000, AUPRO 1.0000; padim image AUROC 1.0000, AUPRO 1.0000.
- `bottle`: patchcore image AUROC 1.0000, AUPRO 0.9997; padim image AUROC 1.0000, AUPRO 0.9999.
- `capsule`: patchcore image AUROC 0.9517, AUPRO 0.9918; padim image AUROC 0.9541, AUPRO 0.9936.
- `grid`: patchcore image AUROC 0.9674, AUPRO 0.9503; padim image AUROC 0.9098, AUPRO 0.8728.

### 4.7 Threshold differences

| Category | Model | Image threshold | Pixel threshold | Validation images |
| --- | --- | ---: | ---: | ---: |
| `transistor` | patchcore | 18.4067 | 15.5496 | 43 |
| `transistor` | padim | 68.1344 | 19.2513 | 43 |
| `cable` | patchcore | 19.4056 | 17.3949 | 45 |
| `cable` | padim | 26.4028 | 13.3443 | 45 |
| `leather` | patchcore | 12.1471 | 10.8563 | 49 |
| `leather` | padim | 19.1718 | 8.7012 | 49 |
| `bottle` | patchcore | 13.9973 | 11.0936 | 42 |
| `bottle` | padim | 50.8292 | 10.7475 | 42 |
| `capsule` | patchcore | 12.6702 | 10.6310 | 44 |
| `capsule` | padim | 76.5921 | 12.6996 | 44 |
| `grid` | patchcore | 14.8415 | 13.0376 | 53 |
| `grid` | padim | 14.2772 | 7.7522 | 53 |

Thresholds are percentile-based per category/model, but raw score scales are not comparable across categories (different normal appearance and distance magnitudes). Only the resulting decisions and metrics are comparable.

### 4.8 False positives and false negatives

- `transistor` / patchcore: 0 false positives, 7 false negatives.
- `transistor` / padim: 0 false positives, 16 false negatives.
- `cable` / patchcore: 6 false positives, 6 false negatives.
- `cable` / padim: 5 false positives, 6 false negatives.
- `leather` / patchcore: 8 false positives, 0 false negatives.
- `leather` / padim: 1 false positives, 0 false negatives.
- `bottle` / patchcore: 0 false positives, 0 false negatives.
- `bottle` / padim: 0 false positives, 1 false negatives.
- `capsule` / patchcore: 1 false positives, 23 false negatives.
- `capsule` / padim: 0 false positives, 78 false negatives.
- `grid` / patchcore: 1 false positives, 4 false negatives.
- `grid` / padim: 3 false positives, 11 false negatives.

### 4.9 Defect appearance

| Category | Defect type | Masks | Median area (% of image) | Max area (%) |
| --- | --- | ---: | ---: | ---: |
| `transistor` | `bent_lead` | 10 | 1.03 | 3.65 |
| `transistor` | `cut_lead` | 10 | 1.48 | 3.58 |
| `transistor` | `damaged_case` | 10 | 2.31 | 5.35 |
| `transistor` | `misplaced` | 10 | 40.45 | 49.32 |
| `cable` | `bent_wire` | 13 | 3.60 | 11.37 |
| `cable` | `cable_swap` | 12 | 5.82 | 6.47 |
| `cable` | `combined` | 11 | 4.77 | 14.17 |
| `cable` | `cut_inner_insulation` | 14 | 3.10 | 9.89 |
| `cable` | `cut_outer_insulation` | 10 | 1.25 | 3.39 |
| `cable` | `missing_cable` | 12 | 7.40 | 22.58 |
| `cable` | `missing_wire` | 10 | 1.80 | 3.88 |
| `cable` | `poke_insulation` | 10 | 2.82 | 4.87 |
| `leather` | `color` | 19 | 0.25 | 1.28 |
| `leather` | `cut` | 19 | 0.38 | 1.70 |
| `leather` | `fold` | 17 | 1.68 | 6.86 |
| `leather` | `glue` | 19 | 0.67 | 1.75 |
| `leather` | `poke` | 18 | 0.23 | 0.40 |
| `bottle` | `broken_large` | 20 | 11.87 | 27.42 |
| `bottle` | `broken_small` | 22 | 2.44 | 8.06 |
| `bottle` | `contamination` | 21 | 7.25 | 16.53 |
| `capsule` | `crack` | 23 | 0.30 | 2.33 |
| `capsule` | `faulty_imprint` | 22 | 0.28 | 4.13 |
| `capsule` | `poke` | 21 | 0.23 | 1.23 |
| `capsule` | `scratch` | 23 | 0.64 | 1.53 |
| `capsule` | `squeeze` | 20 | 2.76 | 8.08 |
| `grid` | `bent` | 12 | 0.99 | 2.16 |
| `grid` | `broken` | 12 | 0.51 | 2.32 |
| `grid` | `glue` | 11 | 0.49 | 0.73 |
| `grid` | `metal_contamination` | 11 | 0.41 | 0.82 |
| `grid` | `thread` | 11 | 1.58 | 3.58 |

### 4.10 Image-level versus pixel-level behaviour

- `transistor` / patchcore: image AUROC 0.9883 vs pixel AUROC 0.9644; image recall 0.8250 vs pixel recall 0.6569 at their calibrated thresholds.
- `transistor` / padim: image AUROC 0.9950 vs pixel AUROC 0.9829; image recall 0.6000 vs pixel recall 0.9090 at their calibrated thresholds.
- `cable` / patchcore: image AUROC 0.9696 vs pixel AUROC 0.9815; image recall 0.9348 vs pixel recall 0.9076 at their calibrated thresholds.
- `cable` / padim: image AUROC 0.9798 vs pixel AUROC 0.9789; image recall 0.9348 vs pixel recall 0.9523 at their calibrated thresholds.
- `leather` / patchcore: image AUROC 1.0000 vs pixel AUROC 0.9890; image recall 1.0000 vs pixel recall 1.0000 at their calibrated thresholds.
- `leather` / padim: image AUROC 1.0000 vs pixel AUROC 0.9787; image recall 1.0000 vs pixel recall 1.0000 at their calibrated thresholds.
- `bottle` / patchcore: image AUROC 1.0000 vs pixel AUROC 0.9856; image recall 1.0000 vs pixel recall 0.9989 at their calibrated thresholds.
- `bottle` / padim: image AUROC 1.0000 vs pixel AUROC 0.9767; image recall 0.9841 vs pixel recall 0.9999 at their calibrated thresholds.
- `capsule` / patchcore: image AUROC 0.9517 vs pixel AUROC 0.9874; image recall 0.7890 vs pixel recall 0.9426 at their calibrated thresholds.
- `capsule` / padim: image AUROC 0.9541 vs pixel AUROC 0.9811; image recall 0.2844 vs pixel recall 0.9151 at their calibrated thresholds.
- `grid` / patchcore: image AUROC 0.9674 vs pixel AUROC 0.9668; image recall 0.9298 vs pixel recall 0.8873 at their calibrated thresholds.
- `grid` / padim: image AUROC 0.9098 vs pixel AUROC 0.9285; image recall 0.8070 vs pixel recall 0.7982 at their calibrated thresholds.

## 5. Evaluation integrity disclosure

- Training used only `train/good` images of the same category.
- Thresholds used only held-out normal training images of the same category (`test_data_used: false`); no test label was used for fitting or calibration.
- Test labels are used only for the final reported metrics and error-analysis figures, after models and thresholds were frozen.
- Artifacts, thresholds, predictions, metrics and figures are category-scoped; no fitted state or threshold is reused across categories.
- Descriptive EDA summarises test annotations but does not influence modelling.

## 6. Limitations

- The report covers only the 6 supported categories. All results, aggregates and rankings apply to these six categories only; this is not a 15-category benchmark, and the nine out-of-scope categories were intentionally excluded.
- Pixel metrics use the configured evaluation resolution with pooled pixels; they are not directly comparable to full-resolution literature numbers.
- AUPRO is capped at the configured false-positive limit and derives thresholds from normal-score quantiles.
- Normal-only calibration cannot guarantee a good defect-type balance; operating points should follow business costs.
- No robustness study for lighting, camera or fixture changes; recalibrate per site.
- Models score deviation from normal; they do not classify defect types.

## 7. Reproducibility

```bash
# Validate the six supported categories (out-of-scope categories are not touched)
python scripts/01_validate_dataset.py --categories active

# Per-category EDA, training, calibration and evaluation (sequential, CPU-friendly)
python scripts/02_explore_dataset.py --categories active
python scripts/04_train.py --categories active
python scripts/05_calibrate.py --categories active
python scripts/06_evaluate.py --categories active

# Cross-category aggregation and reports
python scripts/08_generate_reports.py

# Category-aware inference
python scripts/07_predict.py --list-categories
python scripts/07_predict.py --category transistor --input path/to/image.png --model both
streamlit run app/streamlit_app.py
```

Artifacts live under `models/<category>/<model>/` and outputs under `reports/{metrics,figures,predictions}/<category>/`. The six-category scope is permanently frozen for this release; no additional category is supported. See `README.md` for dataset acquisition and licensing instructions.

## 8. Reference hardware and software

- Platform: Windows-10-10.0.26300-SP0
- Python 3.10.9, torch 2.10.0+cpu, torchvision 0.25.0, numpy 1.26.4, scikit-learn 1.7.2, scipy 1.15.3
- 12 logical CPUs, 7.95 GB RAM, CUDA available: False
- Training device: CPU; torch threads: 8

### 8.1 Pipeline smoke verification

A short smoke fit on a small subset (`--max-images`, `--tag smoke`) exercised the train/save/reload path end to end. Smoke artifacts are never calibrated, evaluated or used by the CLI/app; they exist only to verify the pipeline wiring.

| Artifact | Fit images | Fit time (s) | Reload max diff | Artifact size | Passed |
| --- | ---: | ---: | ---: | ---: | --- |
| `models/transistor/patchcore-smoke` | 8 | 1.17 | 0.0 | 1.6 MB | yes |
