# Dataset Audit - MVTec AD

- Generated: `2026-11-17T22:41:57+00:00`
- Dataset root: `transistor`
- Category: **transistor** (`transistor`)
- Categories available in root: n/a
- License: CC BY-NC-SA 4.0
- Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

## Counts (measured)

| Quantity | Value |
| --- | ---: |
| categories | 1 |
| defect types | 4 |
| train good images | 213 |
| test good images | 60 |
| test anomalous images | 40 |
| test total images | 100 |
| ground truth masks | 40 |
| anomalous with mask | 40 |

## Defect types

| Defect type | Test images | Masks | Mean mask area |
| --- | ---: | ---: | ---: |
| bent_lead | 10 | 10 | 1.5996% |
| cut_lead | 10 | 10 | 1.6389% |
| damaged_case | 10 | 10 | 2.8483% |
| misplaced | 10 | 10 | 41.8451% |

## Image properties

- Dimensions: {'1024x1024': 313}
- Formats: {'PNG': 313}
- Colour modes: {'RGB': 313}
- Mask dimensions: {'1024x1024': 40}
- Mask modes: {'L': 40}

## Checks

| Check | Result | Detail |
| --- | --- | --- |
| dataset_root_exists | PASS | transistor |
| category_layout | PASS | transistor |
| required_directories | PASS | {'train': True, 'test': True} |
| normal_training_labels | PASS | {'training_class_dirs': ['good'], 'loose_image_files': []} |
| test_good_split | PASS | ['bent_lead', 'cut_lead', 'damaged_case', 'good', 'misplaced'] |
| train_images_readable | PASS | 213/213 readable |
| test_images_readable | PASS | 100/100 readable |
| masks_cover_anomalies | PASS | 40/40 anomalous images have masks |
| mask_dimensions_match | PASS | 0 mismatches |
| masks_binary_and_nonempty | PASS | 0 non-binary, 0 empty |
| no_duplicate_paths | PASS | 0 duplicated relative paths |
| dataset_ready | PASS | 0 critical issue(s) |

## Issues

No issues were found.

## Provenance

Citation: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger, "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IEEE Conference on Computer Vision and Pattern Recognition, 2019
The dataset is not included in this repository and must not be redistributed. See data/raw/README.md for manual download instructions.
