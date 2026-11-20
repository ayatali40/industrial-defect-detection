# EDA summary - transistor

Descriptive analysis of the dataset layout and anomalous test annotations. No modelling, split or threshold decision was derived from this analysis; the test split is summarised only to characterise the evaluation set.

## Counts

| Split | Images |
| --- | ---: |
| train/good | 213 |
| test/good | 60 |
| test/bent_lead | 10 |
| test/cut_lead | 10 |
| test/damaged_case | 10 |
| test/misplaced | 10 |
| test/anomalous | 40 |

## Image properties

- **train image sizes**: 1024x1024: 213
- **test image sizes**: 1024x1024: 100
- **train colour modes**: RGB: 213
- **test colour modes**: RGB: 100
- **train formats**: PNG: 213
- **test formats**: PNG: 100

## Appearance statistics

- **normal train (sampled)** (22 sampled): grey mean 0.305, intra-image grey std mean 0.158, mean saturation 0.658
- **anomalous test (sampled)** (20 sampled): grey mean 0.315, intra-image grey std mean 0.160, mean saturation 0.647

## Ground-truth defect masks

| Defect type | Masks | Area fraction (min / median / max) | Components (min / median / max) |
| --- | ---: | --- | --- |
| bent_lead | 10 | 0.0022 / 0.0103 / 0.0365 | 1 / 1.0 / 2 |
| cut_lead | 10 | 0.0102 / 0.0148 / 0.0358 | 1 / 1.0 / 2 |
| damaged_case | 10 | 0.0063 / 0.0231 / 0.0535 | 1 / 1.0 / 2 |
| misplaced | 10 | 0.3773 / 0.4045 / 0.4932 | 1 / 1.0 / 1 |

## Observations

- Training data contains 213 defect-free images; the test split has 60 normal and 40 anomalous images with 40 masks.
- Ground-truth defects cover 0.22% to 49.32% of an image (median 2.03%), so image-level detection and pixel-level localisation are genuinely different problems on this category.
- Defect types: bent_lead (10), cut_lead (10), damaged_case (10), misplaced (10).
- All images share one acquisition setup, but illumination and viewpoint vary between captures.
- Defect type misplaced covers a median 40.5% of the image, while the other types cover medians of 1.0-2.3%; pooled pixel metrics are therefore dominated by the large misplaced masks.

## Generated figures

- `reports/figures/eda/normal_training_samples.png`
- `reports/figures/eda/anomalous_test_samples.png`
- `reports/figures/eda/split_and_defect_counts.png`
- `reports/figures/eda/defect_area_distribution.png`
- `reports/figures/eda/appearance_scatter.png`

## Notes

- Masks exist only for anomalous test images; normal images have no mask.
- Box-plot whiskers are matplotlib defaults (1.5 x IQR); the underlying per-image values are recorded in the JSON summary.
- The appearance scatter plots one point per sampled image (deterministic stride); it documents that normal and anomalous images share the same acquisition setup.
