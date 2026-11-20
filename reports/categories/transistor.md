# Category report: `transistor`

Generated: 2026-11-18T20:33:27+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `transistor`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 213 | 0 |
| test/good | 60 | 0 |
| test/anomalous | 40 | 40 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 170 | 22.89 | cpu | projection=256, candidate patches=15000, coreset=1333 | 0 |
| padim | 170 | 21.96 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 43 | 18.4067 | 15.5496 | 99.0 | 99.5 | 0.0233 | 0.005000 |
| padim | 43 | 68.1344 | 19.2513 | 99.0 | 99.5 | 0.0233 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 170 | 43 | 60 | 40 | 100 |
| padim | 170 | 43 | 60 | 40 | 100 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9883 | 0.9867 | 1.0000 | 0.8250 | 0.9041 | 0.9644 | 0.6334 | 0.5530 | 0.6569 | 0.6005 | 0.9442 |
| padim | 0.9950 | 0.9929 | 1.0000 | 0.6000 | 0.7500 | 0.9829 | 0.7443 | 0.5281 | 0.9090 | 0.6681 | 0.9894 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bent_lead` | 10 | 9 | 10 | 6 | 10 |
| `cut_lead` | 10 | 7 | 10 | 4 | 10 |
| `damaged_case` | 10 | 9 | 10 | 4 | 10 |
| `misplaced` | 10 | 8 | 10 | 10 | 10 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 33 | 0 | 60 | 7 |
| padim | 24 | 0 | 60 | 16 |

### Missed anomalies

- **patchcore** misses (7): `test/cut_lead/001.png` (cut_lead, score 18.29), `test/cut_lead/002.png` (cut_lead, score 18.26), `test/cut_lead/007.png` (cut_lead, score 18.17), `test/damaged_case/002.png` (damaged_case, score 17.62), `test/bent_lead/001.png` (bent_lead, score 17.53), `test/misplaced/002.png` (misplaced, score 16.82), `test/misplaced/003.png` (misplaced, score 16.23).
- **padim** misses (16): `test/cut_lead/003.png` (cut_lead, score 67.37), `test/damaged_case/002.png` (damaged_case, score 66.85), `test/damaged_case/005.png` (damaged_case, score 66.14), `test/cut_lead/000.png` (cut_lead, score 65.55), `test/bent_lead/003.png` (bent_lead, score 62.40), `test/bent_lead/004.png` (bent_lead, score 62.36), `test/bent_lead/002.png` (bent_lead, score 59.74), `test/cut_lead/001.png` (cut_lead, score 58.29); 8 more in the per-image CSV.

## Visual evidence

- [model comparison](../../figures/transistor/model_comparison.png)
- [normal samples](../../figures/transistor/eda/normal_training_samples.png)
- [anomalous samples](../../figures/transistor/eda/anomalous_test_samples.png)
- [defect area](../../figures/transistor/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/transistor/patchcore/correct_detections_true_positives.png)
- [patchcore false negatives](../../figures/transistor/patchcore/false_negatives.png)
- [padim true positives](../../figures/transistor/padim/correct_detections_true_positives.png)
- [padim false negatives](../../figures/transistor/padim/false_negatives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `bent_lead` | 10 | 1.03 | 3.65 | 1-2 |
| `cut_lead` | 10 | 1.48 | 3.58 | 1-2 |
| `damaged_case` | 10 | 2.31 | 5.35 | 1-2 |
| `misplaced` | 10 | 40.45 | 49.32 | 1-1 |

Observations:

- Training data contains 213 defect-free images; the test split has 60 normal and 40 anomalous images with 40 masks.
- Ground-truth defects cover 0.22% to 49.32% of an image (median 2.03%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: bent_lead (10), cut_lead (10), damaged_case (10), misplaced (10).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type misplaced covers a median 40.5% of the image, while the other types cover medians of 1.0-2.3%; pooled pixel metrics are therefore dominated by the large misplaced masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
