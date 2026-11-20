# Category report: `bottle`

Generated: 2026-11-18T20:33:28+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `bottle`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 209 | 0 |
| test/good | 20 | 0 |
| test/anomalous | 63 | 63 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 167 | 22.74 | cpu | projection=256, candidate patches=15000, coreset=1309 | 0 |
| padim | 167 | 20.52 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 42 | 13.9973 | 11.0936 | 99.0 | 99.5 | 0.0238 | 0.005000 |
| padim | 42 | 50.8292 | 10.7475 | 99.0 | 99.5 | 0.0238 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 167 | 42 | 20 | 63 | 83 |
| padim | 167 | 42 | 20 | 63 | 83 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9856 | 0.7541 | 0.2987 | 0.9989 | 0.4598 | 0.9997 |
| padim | 1.0000 | 1.0000 | 1.0000 | 0.9841 | 0.9920 | 0.9767 | 0.5750 | 0.2510 | 0.9999 | 0.4012 | 0.9999 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `broken_large` | 20 | 20 | 20 | 20 | 20 |
| `broken_small` | 22 | 22 | 22 | 22 | 22 |
| `contamination` | 21 | 21 | 21 | 20 | 21 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 63 | 0 | 20 | 0 |
| padim | 62 | 0 | 20 | 1 |

### Missed anomalies

- **patchcore**: no false negatives.
- **padim** misses (1): `test/contamination/003.png` (contamination, score 44.42).

## Visual evidence

- [model comparison](../../figures/bottle/model_comparison.png)
- [normal samples](../../figures/bottle/eda/normal_training_samples.png)
- [anomalous samples](../../figures/bottle/eda/anomalous_test_samples.png)
- [defect area](../../figures/bottle/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/bottle/patchcore/correct_detections_true_positives.png)
- [padim true positives](../../figures/bottle/padim/correct_detections_true_positives.png)
- [padim false negatives](../../figures/bottle/padim/false_negatives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `broken_large` | 20 | 11.87 | 27.42 | 1-2 |
| `broken_small` | 22 | 2.44 | 8.06 | 1-3 |
| `contamination` | 21 | 7.25 | 16.53 | 1-2 |

Observations:

- Training data contains 209 defect-free images; the test split has 20 normal and 63 anomalous images with 63 masks.
- Ground-truth defects cover 0.58% to 27.42% of an image (median 5.72%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: broken_large (20), broken_small (22), contamination (21).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type broken_large covers a median 11.9% of the image, while the other types cover medians of 2.4-7.2%; pooled pixel metrics are therefore dominated by the large broken_large masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
