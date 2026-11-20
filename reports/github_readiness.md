# GitHub readiness audit

Generated: 2026-11-18T20:33:28+00:00

## Repository status

- Git repository initialized: **False**. The project is prepared for a first commit; no commit was created by this audit.
- Files that would be tracked: 307 (242.7 MB).
- Dataset categories present locally: transistor; dataset files are excluded by ignore rules.
- Trained binaries present locally: 13 (ignored; regenerate with `scripts/04_train.py`).

## Ignore rules

| Pattern | Present in `.gitignore` |
| --- | --- |
| `bottle/` | yes |
| `cable/` | yes |
| `capsule/` | yes |
| `carpet/` | yes |
| `grid/` | yes |
| `hazelnut/` | yes |
| `leather/` | yes |
| `metal_nut/` | yes |
| `pill/` | yes |
| `screw/` | yes |
| `tile/` | yes |
| `toothbrush/` | yes |
| `transistor/` | yes |
| `wood/` | yes |
| `zipper/` | yes |
| `data/mvtec_ad/` | yes |
| `models/**` | yes |
| `*.pt` | yes |
| `reports/predictions/` | yes |
| `*.npz` | yes |
| `__pycache__/` | yes |
| `.pytest_cache/` | yes |
| `.env` | yes |

## Large files

| File | Size |
| --- | ---: |
| `reports/figures/leather/patchcore/correct_detections_true_negatives.png` | 5.5 MB |
| `reports/figures/leather/padim/correct_detections_true_negatives.png` | 5.5 MB |
| `reports/figures/leather/patchcore/false_positives.png` | 5.5 MB |
| `notebooks/02_anomaly_detection_experiments.ipynb` | 5.5 MB |
| `reports/figures/cable/patchcore/correct_detections_true_negatives.png` | 4.9 MB |
| `reports/figures/cable/patchcore/false_negatives.png` | 4.9 MB |
| `reports/figures/cable/patchcore/false_positives.png` | 4.8 MB |
| `reports/figures/leather/patchcore/correct_detections_true_positives.png` | 4.7 MB |
| `reports/figures/cable/padim/correct_detections_true_negatives.png` | 4.7 MB |
| `reports/figures/cable/padim/false_negatives.png` | 4.7 MB |
| `reports/figures/cable/patchcore/correct_detections_true_positives.png` | 4.4 MB |
| `reports/figures/leather/padim/correct_detections_true_positives.png` | 4.4 MB |
| `reports/figures/capsule/patchcore/correct_detections_true_negatives.png` | 4.4 MB |
| `reports/figures/capsule/patchcore/false_negatives.png` | 4.3 MB |
| `reports/figures/bottle/patchcore/correct_detections_true_negatives.png` | 4.1 MB |
| `reports/figures/transistor/patchcore/correct_detections_true_negatives.png` | 4.1 MB |
| `reports/figures/patchcore/correct_detections_true_negatives.png` | 4.1 MB |
| `reports/figures/capsule/patchcore/correct_detections_true_positives.png` | 4.0 MB |
| `reports/figures/capsule/padim/correct_detections_true_negatives.png` | 4.0 MB |
| `reports/figures/grid/patchcore/correct_detections_true_negatives.png` | 4.0 MB |

## Sensitivity scan

- Absolute user/home paths in text files: 0 
- Secret-like assignments: 0 
- Data raw files: README.md

## Checklist

- [x] all category dirs ignored
- [x] dataset root ignored
- [x] binaries ignored
- [x] predictions ignored
- [x] no absolute user paths
- [x] no secret like values
- [x] no file over 10mb
- [x] dataset files excluded
- [x] binaries excluded

Notebooks can be several MB each because executed figures are embedded; they remain below the 10 MB check threshold. Full anomaly maps and per-image predictions are git-ignored and produced by `scripts/06_evaluate.py`.
