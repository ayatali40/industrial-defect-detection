# Category report: `capsule`

Generated: 2026-11-18T20:33:28+00:00

This report covers only the category named above. Thresholds, model state and metrics are never shared with another category.

## Dataset summary

- Dataset root: `transistor`
- Category directory: `capsule`
- Readiness: **PASS**
- Critical issues: 0

| Split / class | Images | Masks |
| --- | ---: | ---: |
| train/good | 219 | 0 |
| test/good | 23 | 0 |
| test/anomalous | 109 | 109 |

## Training and calibration

| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |
| --- | ---: | ---: | --- | --- | ---: |
| patchcore | 175 | 27.59 | cpu | projection=256, candidate patches=15000, coreset=1372 | 0 |
| padim | 175 | 27.26 | cpu | positions=784, features=150, epsilon=0.01 | 0 |

| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 44 | 12.6702 | 10.6310 | 99.0 | 99.5 | 0.0227 | 0.005000 |
| padim | 44 | 76.5921 | 12.6996 | 99.0 | 99.5 | 0.0227 | 0.005000 |

Thresholds are computed on held-out normal training images of this category only (`test_data_used: false` in the threshold record).

## Evaluation results

| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |
| --- | ---: | ---: | ---: | ---: | ---: |
| patchcore | 175 | 44 | 23 | 109 | 132 |
| padim | 175 | 44 | 23 | 109 | 132 |

| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patchcore | 0.9517 | 0.9884 | 0.9885 | 0.7890 | 0.8776 | 0.9874 | 0.4424 | 0.1688 | 0.9426 | 0.2863 | 0.9918 |
| padim | 0.9541 | 0.9881 | 1.0000 | 0.2844 | 0.4429 | 0.9811 | 0.3552 | 0.1280 | 0.9151 | 0.2246 | 0.9936 |

Image metrics use the calibrated image threshold; pixel metrics are pooled over all test pixels at the configured evaluation resolution; AUPRO integrates per-region overlap up to the configured false-positive limit. Metric definitions are unchanged from the single-category implementation.

## Per-defect behaviour

(localised = predicted pixel region overlaps the ground-truth mask)

| Defect type | Images | patchcore detected | patchcore localised | padim detected | padim localised |
| --- | ---: | ---: | ---: | ---: | ---: |
| `crack` | 23 | 18 | 21 | 8 | 22 |
| `faulty_imprint` | 22 | 18 | 21 | 2 | 18 |
| `poke` | 21 | 16 | 21 | 3 | 20 |
| `scratch` | 23 | 14 | 20 | 1 | 19 |
| `squeeze` | 20 | 20 | 20 | 17 | 20 |

## Decision outcomes

| Model | True positives | False positives | True negatives | False negatives |
| --- | ---: | ---: | ---: | ---: |
| patchcore | 86 | 1 | 22 | 23 |
| padim | 31 | 0 | 23 | 78 |

### Missed anomalies

- **patchcore** misses (23): `test/scratch/015.png` (scratch, score 12.61), `test/scratch/021.png` (scratch, score 12.56), `test/poke/018.png` (poke, score 12.47), `test/crack/008.png` (crack, score 12.40), `test/poke/017.png` (poke, score 12.37), `test/scratch/001.png` (scratch, score 12.30), `test/crack/022.png` (crack, score 12.23), `test/poke/003.png` (poke, score 12.21); 15 more in the per-image CSV.
- **padim** misses (78): `test/scratch/008.png` (scratch, score 75.16), `test/scratch/018.png` (scratch, score 73.09), `test/crack/006.png` (crack, score 72.72), `test/crack/009.png` (crack, score 72.53), `test/crack/011.png` (crack, score 63.63), `test/squeeze/012.png` (squeeze, score 60.96), `test/crack/003.png` (crack, score 59.59), `test/scratch/022.png` (scratch, score 59.13); 70 more in the per-image CSV.

## Visual evidence

- [model comparison](../../figures/capsule/model_comparison.png)
- [normal samples](../../figures/capsule/eda/normal_training_samples.png)
- [anomalous samples](../../figures/capsule/eda/anomalous_test_samples.png)
- [defect area](../../figures/capsule/eda/defect_area_distribution.png)
- [patchcore true positives](../../figures/capsule/patchcore/correct_detections_true_positives.png)
- [patchcore false negatives](../../figures/capsule/patchcore/false_negatives.png)
- [padim true positives](../../figures/capsule/padim/correct_detections_true_positives.png)
- [padim false negatives](../../figures/capsule/padim/false_negatives.png)

## Descriptive EDA

| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |
| --- | ---: | ---: | ---: | ---: |
| `crack` | 23 | 0.30 | 2.33 | 1-2 |
| `faulty_imprint` | 22 | 0.28 | 4.13 | 1-2 |
| `poke` | 21 | 0.23 | 1.23 | 1-2 |
| `scratch` | 23 | 0.64 | 1.53 | 1-3 |
| `squeeze` | 20 | 2.76 | 8.08 | 1-2 |

Observations:

- Training data contains 219 defect-free images; the test split has 23 normal and 109 anomalous images with 109 masks.
- Ground-truth defects cover 0.04% to 8.08% of an image (median 0.43%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: crack (23), faulty_imprint (22), poke (21), scratch (23), squeeze (20).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type squeeze covers a median 2.8% of the image, while the other types cover medians of 0.2-0.6%; pooled pixel metrics are therefore dominated by the large squeeze masks.

## Limitations

- Thresholds are calibrated on normal images only and may drift with lighting, camera or fixture changes.
- Pixel metrics use the configured evaluation resolution, not the original image resolution; numbers are not directly comparable to full-resolution literature.
- The models score deviation from normal; they do not classify defect types.
- Small test and calibration splits make threshold false-positive estimates coarse.
