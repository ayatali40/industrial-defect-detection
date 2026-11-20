# Category report: `grid`

Generated: 2026-11-18T20:33:28+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `grid`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 264 | 0 |
| test/good | 21 | 0 |
| test/anomalous | 57 | 57 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 211 | 30.26 | cpu | projection=256, candidate patches=15000, coreset=1654 | 0 |
| padim | 211 | 27.77 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 53 | 14.8415 | 13.0376 | 99.0 | 99.5 | 0.0189 | 0.005000 |
| padim | 53 | 14.2772 | 7.7522 | 99.0 | 99.5 | 0.0189 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 211 | 53 | 21 | 57 | 78 |
| padim | 211 | 53 | 21 | 57 | 78 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9674 | 0.9893 | 0.9815 | 0.9298 | 0.9550 | 0.9668 | 0.2656 | 0.0815 | 0.8873 | 0.1493 | 0.9503 |
| padim | 0.9098 | 0.9694 | 0.9388 | 0.8070 | 0.8679 | 0.9285 | 0.0821 | 0.0665 | 0.7982 | 0.1227 | 0.8728 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bent` | 12 | 12 | 12 | 12 | 12 |
| `broken` | 12 | 12 | 12 | 12 | 12 |
| `glue` | 11 | 10 | 10 | 6 | 9 |
| `metal_contamination` | 11 | 10 | 11 | 8 | 11 |
| `thread` | 11 | 9 | 11 | 8 | 10 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 53 | 1 | 20 | 4 |
| padim | 46 | 3 | 18 | 11 |

### Missed anomalies

- **patchcore** misses (4): `test/metal_contamination/006.png` (metal_contamination, score 14.82), `test/thread/002.png` (thread, score 14.65), `test/thread/003.png` (thread, score 13.82), `test/glue/008.png` (glue, score 12.71).
- **padim** misses (11): `test/thread/008.png` (thread, score 13.78), `test/glue/000.png` (glue, score 13.62), `test/metal_contamination/007.png` (metal_contamination, score 13.37), `test/metal_contamination/009.png` (metal_contamination, score 13.28), `test/glue/001.png` (glue, score 13.17), `test/thread/002.png` (thread, score 11.64), `test/glue/007.png` (glue, score 11.14), `test/metal_contamination/006.png` (metal_contamination, score 10.53); 3 more in the per-image CSV.

## Visual evidence

- [model comparison](../../figures/grid/model_comparison.png)
- [normal samples](../../figures/grid/eda/normal_training_samples.png)
- [anomalous samples](../../figures/grid/eda/anomalous_test_samples.png)
- [defect area](../../figures/grid/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/grid/patchcore/correct_detections_true_positives.png)
- [patchcore false negatives](../../figures/grid/patchcore/false_negatives.png)
- [padim true positives](../../figures/grid/padim/correct_detections_true_positives.png)
- [padim false negatives](../../figures/grid/padim/false_negatives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `bent` | 12 | 0.99 | 2.16 | 2-7 |
| `broken` | 12 | 0.51 | 2.32 | 2-4 |
| `glue` | 11 | 0.49 | 0.73 | 1-4 |
| `metal_contamination` | 11 | 0.41 | 0.82 | 1-5 |
| `thread` | 11 | 1.58 | 3.58 | 1-17 |

Observations:

- Training data contains 264 defect-free images; the test split has 21 normal and 57 anomalous images with 57 masks.
- Ground-truth defects cover 0.19% to 3.58% of an image (median 0.60%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: bent (12), broken (12), glue (11), metal_contamination (11), thread (11).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type thread covers a median 1.6% of the image, while the other types cover medians of 0.4-1.0%; pooled pixel metrics are therefore dominated by the large thread masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
