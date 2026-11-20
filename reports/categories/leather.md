# Category report: `leather`

Generated: 2026-11-18T20:33:28+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `leather`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 245 | 0 |
| test/good | 32 | 0 |
| test/anomalous | 92 | 92 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 196 | 28.64 | cpu | projection=256, candidate patches=15000, coreset=1537 | 0 |
| padim | 196 | 24.33 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 49 | 12.1471 | 10.8563 | 99.0 | 99.5 | 0.0204 | 0.005000 |
| padim | 49 | 19.1718 | 8.7012 | 99.0 | 99.5 | 0.0204 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 196 | 49 | 32 | 92 | 124 |
| padim | 196 | 49 | 32 | 92 | 124 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 1.0000 | 1.0000 | 0.9200 | 1.0000 | 0.9583 | 0.9890 | 0.3340 | 0.0384 | 1.0000 | 0.0739 | 1.0000 |
| padim | 1.0000 | 1.0000 | 0.9892 | 1.0000 | 0.9946 | 0.9787 | 0.2138 | 0.0320 | 1.0000 | 0.0620 | 1.0000 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `color` | 19 | 19 | 19 | 19 | 19 |
| `cut` | 19 | 19 | 19 | 19 | 19 |
| `fold` | 17 | 17 | 17 | 17 | 17 |
| `glue` | 19 | 19 | 19 | 19 | 19 |
| `poke` | 18 | 18 | 18 | 18 | 18 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 92 | 8 | 24 | 0 |
| padim | 92 | 1 | 31 | 0 |

### Missed anomalies

- **patchcore**: no false negatives.
- **padim**: no false negatives.

## Visual evidence

- [model comparison](../../figures/leather/model_comparison.png)
- [normal samples](../../figures/leather/eda/normal_training_samples.png)
- [anomalous samples](../../figures/leather/eda/anomalous_test_samples.png)
- [defect area](../../figures/leather/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/leather/patchcore/correct_detections_true_positives.png)
- [padim true positives](../../figures/leather/padim/correct_detections_true_positives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `color` | 19 | 0.25 | 1.28 | 1-3 |
| `cut` | 19 | 0.38 | 1.70 | 1-3 |
| `fold` | 17 | 1.68 | 6.86 | 1-1 |
| `glue` | 19 | 0.67 | 1.75 | 1-3 |
| `poke` | 18 | 0.23 | 0.40 | 1-1 |

Observations:

- Training data contains 245 defect-free images; the test split has 32 normal and 92 anomalous images with 92 masks.
- Ground-truth defects cover 0.09% to 6.86% of an image (median 0.46%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: color (19), cut (19), fold (17), glue (19), poke (18).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type fold covers a median 1.7% of the image, while the other types cover medians of 0.2-0.7%; pooled pixel metrics are therefore dominated by the large fold masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
