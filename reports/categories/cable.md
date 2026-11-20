# Category report: `cable`

Generated: 2026-11-18T20:33:27+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `cable`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 224 | 0 |
| test/good | 58 | 0 |
| test/anomalous | 92 | 92 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 179 | 27.88 | cpu | projection=256, candidate patches=15000, coreset=1403 | 0 |
| padim | 179 | 26.03 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 45 | 19.4056 | 17.3949 | 99.0 | 99.5 | 0.0222 | 0.005000 |
| padim | 45 | 26.4028 | 13.3443 | 99.0 | 99.5 | 0.0222 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 179 | 45 | 58 | 92 | 150 |
| padim | 179 | 45 | 58 | 92 | 150 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9696 | 0.9808 | 0.9348 | 0.9348 | 0.9348 | 0.9815 | 0.6642 | 0.3791 | 0.9076 | 0.5348 | 0.9800 |
| padim | 0.9798 | 0.9887 | 0.9451 | 0.9348 | 0.9399 | 0.9789 | 0.5926 | 0.2839 | 0.9523 | 0.4374 | 0.9881 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bent_wire` | 13 | 13 | 13 | 13 | 13 |
| `cable_swap` | 12 | 9 | 12 | 12 | 12 |
| `combined` | 11 | 11 | 11 | 11 | 11 |
| `cut_inner_insulation` | 14 | 14 | 14 | 14 | 14 |
| `cut_outer_insulation` | 10 | 10 | 10 | 10 | 10 |
| `missing_cable` | 12 | 12 | 12 | 12 | 12 |
| `missing_wire` | 10 | 10 | 10 | 7 | 9 |
| `poke_insulation` | 10 | 7 | 10 | 7 | 10 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 86 | 6 | 52 | 6 |
| padim | 86 | 5 | 53 | 6 |

### Missed anomalies

- **patchcore** misses (6): `test/poke_insulation/006.png` (poke_insulation, score 19.40), `test/cable_swap/000.png` (cable_swap, score 19.24), `test/poke_insulation/004.png` (poke_insulation, score 18.99), `test/cable_swap/006.png` (cable_swap, score 18.93), `test/cable_swap/003.png` (cable_swap, score 18.85), `test/poke_insulation/002.png` (poke_insulation, score 18.15).
- **padim** misses (6): `test/poke_insulation/004.png` (poke_insulation, score 26.05), `test/poke_insulation/003.png` (poke_insulation, score 25.97), `test/poke_insulation/006.png` (poke_insulation, score 25.48), `test/missing_wire/005.png` (missing_wire, score 21.73), `test/missing_wire/002.png` (missing_wire, score 19.65), `test/missing_wire/000.png` (missing_wire, score 13.20).

## Visual evidence

- [model comparison](../../figures/cable/model_comparison.png)
- [normal samples](../../figures/cable/eda/normal_training_samples.png)
- [anomalous samples](../../figures/cable/eda/anomalous_test_samples.png)
- [defect area](../../figures/cable/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/cable/patchcore/correct_detections_true_positives.png)
- [patchcore false negatives](../../figures/cable/patchcore/false_negatives.png)
- [padim true positives](../../figures/cable/padim/correct_detections_true_positives.png)
- [padim false negatives](../../figures/cable/padim/false_negatives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `bent_wire` | 13 | 3.60 | 11.37 | 1-3 |
| `cable_swap` | 12 | 5.82 | 6.47 | 1-1 |
| `combined` | 11 | 4.77 | 14.17 | 1-11 |
| `cut_inner_insulation` | 14 | 3.10 | 9.89 | 1-3 |
| `cut_outer_insulation` | 10 | 1.25 | 3.39 | 1-4 |
| `missing_cable` | 12 | 7.40 | 22.58 | 1-1 |
| `missing_wire` | 10 | 1.80 | 3.88 | 1-3 |
| `poke_insulation` | 10 | 2.82 | 4.87 | 1-5 |

Observations:

- Training data contains 224 defect-free images; the test split has 58 normal and 92 anomalous images with 92 masks.
- Ground-truth defects cover 0.14% to 22.58% of an image (median 3.68%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: bent_wire (13), cable_swap (12), combined (11), cut_inner_insulation (14), cut_outer_insulation (10), missing_cable (12), missing_wire (10), poke_insulation (10).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type missing_cable covers a median 7.4% of the image, while the other types cover medians of 1.3-5.8%; pooled pixel metrics are therefore dominated by the large missing_cable masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
