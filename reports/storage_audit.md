# Storage audit - frozen six-category release

Generated: 2026-11-18

Scope: the repository root, its six mapped external datasets, and the local caches visible
to this checkout. Sizes were measured with `du -sb` / `stat`; duplicate checks used
`md5sum`. **No file was deleted or moved by this audit.**

System disk at audit time: 207 GB total, 205 GB used, **2.2 GB free (99% used)**. Storage is
tight, so the "safe to delete" rows below are advisory only and deliberately not executed.

## 1. Project totals

| Path | Bytes | Human | Content |
| --- | ---: | ---: | --- |
| `models/` | 446,261,007 | 426 MB | 12 fitted artifacts + 12 calibration/training records + smoke artifact |
| `reports/predictions/` | 1,216,020,570 | 1.2 GB | 12 prediction CSVs + 12 full anomaly-map NPZ files + CLI smoke outputs |
| `reports/figures/` | 242,146,316 | 232 MB | Category-scoped EDA/inspection figures + legacy single-category figures |
| `reports/metrics/` | 624,606 | 610 KB | Audits, EDA summaries, per-model metrics, cross-category metrics |
| `reports/categories/` | 33,932 | 33 KB | Six per-category Markdown reports |
| `data/processed/` | 45,696 | 45 KB | Six normal-split manifests + preprocessing records |
| `notebooks/` | 11,250,248 | 11 MB | Three executed notebooks with embedded outputs |
| Project total | - | 2.2 GB | Includes source, tests, docs and reports |

## 2. Model artifacts (all unique)

`md5sum models/*/*/artifact.pt` returned **13 distinct hashes**: the 12 supported
category/model artifacts are byte-distinct, plus the smoke artifact. No model artifact is a
duplicate of another.

| Category | PatchCore `artifact.pt` | PaDiM `artifact.pt` | Thresholds/training records |
| --- | ---: | ---: | --- |
| bottle | 2,918,853 | 71,044,165 | unique JSON per model |
| cable | 3,014,981 | 71,044,037 | unique JSON per model |
| capsule | 2,983,365 | 71,044,165 | unique JSON per model |
| grid | 3,272,133 | 71,044,165 | unique JSON per model |
| leather | 3,152,197 | 71,044,037 | unique JSON per model |
| transistor | 2,943,557 | 71,044,357 | unique JSON per model |
| transistor smoke | 1,642,757 | - | smoke training record only |

Classification:

| Item | Unique? | Required for inference | Required for reproducibility | Safe to delete |
| --- | --- | --- | --- | --- |
| `models/<cat>/<model>/artifact.pt` (12) | yes | **yes** | **yes** | **no** - unique fitted state |
| `models/<cat>/<model>/thresholds.json` (12) | yes | **yes** | **yes** | **no** - calibration record |
| `models/<cat>/<model>/training.json` (12) | yes | no | **yes** | **no** - training/reload evidence |
| `models/transistor/patchcore-smoke/` | yes | no | pipeline evidence only | no - unique smoke artifact (small) |

## 3. Predictions and metrics

| Path | Size | Unique? | Required for inference | Required for reproducibility | Safe to delete |
| --- | ---: | --- | --- | --- | --- |
| `reports/predictions/<cat>/<model>_test_predictions.csv` (12) | ~170 KB total | yes | no | **yes** - per-image evaluation record | no |
| `reports/predictions/<cat>/<model>_test_maps.npz` (12) | ~1.2 GB total | yes | no | **yes** - saved test predictions/maps | no |
| `reports/metrics/<cat>/` | included in 610 KB | yes | no | **yes** - audits, metrics, error analysis | no |
| `reports/metrics/cross_category_metrics.{csv,json}` | 8 KB / 24 KB | intentional duplicate formats | no | **yes** - aggregate evidence | no |
| `reports/metrics/{training,calibration,evaluation}_summary.json`, `artifact_migration.json` | small | yes | no | **yes** - run-level evidence | no |
| `reports/categories/<cat>.md` (6) | 33 KB | yes | no | **yes** - final per-category reports | no |
| `reports/final_report.md`, `reports/github_readiness.md` | 28 KB | yes | no | **yes** - release deliverables | no |

Full anomaly maps are the largest project component. They are git-ignored and are kept
because they are the saved test evidence behind the reported pixel metrics and AURPO.

## 4. Figures

| Path | Bytes | Unique? | Required for inference | Required for reproducibility | Safe to delete |
| --- | ---: | --- | --- | --- | --- |
| `reports/figures/<cat>/` (six categories) | 209,879,024 | yes | no | no (visual deliverables) | no - final committed figures |
| `reports/figures/eda/` | 3,945,576 | duplicate | no | no | **yes** - byte-identical to `reports/figures/transistor/eda/` (5 files) |
| `reports/figures/model_comparison.png` | 67,294 | duplicate | no | no | **yes** - byte-identical to `reports/figures/transistor/model_comparison.png` |
| `reports/figures/padim/` | 13,337,089 | superseded | no | no | **yes** - legacy single-category rendering |
| `reports/figures/patchcore/` | 14,917,333 | superseded | no | no | **yes** - legacy single-category rendering |

Duplicate evidence (MD5):

- `reports/figures/eda/*.png` matches `reports/figures/transistor/eda/*.png` for all five
  files (`anomalous_test_samples`, `appearance_scatter`, `defect_area_distribution`,
  `normal_training_samples`, `split_and_defect_counts`).
- `reports/figures/model_comparison.png` matches
  `reports/figures/transistor/model_comparison.png` (`e69834a0...`).
- `reports/figures/padim/` and `reports/figures/patchcore/` are older re-renderings; their
  file sizes differ from the nested transistor copies, so they are classified as superseded
  legacy outputs, **not** byte-identical duplicates.

## 5. Duplicate prediction CSVs and CLI smoke outputs

| Path | Bytes | Status | Safe to delete |
| --- | ---: | --- | --- |
| `reports/predictions/transistor_padim_test_predictions.csv` | 11,528 | byte-identical duplicate of `reports/predictions/transistor/padim_test_predictions.csv` | **yes** |
| `reports/predictions/transistor_patchcore_test_predictions.csv` | 11,938 | byte-identical duplicate of `reports/predictions/transistor/patchcore_test_predictions.csv` | **yes** |
| `reports/predictions/cli/` | 1,222,100 | demo/smoke inference outputs from CLI and Streamlit checks | yes (regenerable) |

## 6. Datasets

| Path | Size | Relationship | Required | Redistributable |
| --- | ---: | --- | --- | --- |
| `transistor/` inside the project | 386 MB | raw MVTec AD category, git-ignored | training/integration tests | **no** (CC BY-NC-SA 4.0) |
| `../cable/cable` | 483 MB | mapped through `data.category_roots` | training/reproducibility | no |
| `../leather/leather` | 503 MB | mapped through `data.category_roots` | training/reproducibility | no |
| `../bottle/bottle` | 151 MB | mapped through `data.category_roots` | training/reproducibility | no |
| `../capsule/capsule` | 387 MB | mapped through `data.category_roots` | training/reproducibility | no |
| `../grid/grid` | 155 MB | mapped through `data.category_roots` | training/reproducibility | no |

Total external category data: ~2.0 GB. There are no second copies under `data/raw/`; that
folder contains only `README.md`. `data/processed/` contains only the six split manifests
and preprocessing JSON records (45 KB), not image copies.

## 7. Caches and temporary files

| Path | Size | Classification | Safe to delete |
| --- | ---: | --- | --- |
| `.pytest_cache/` | 12 KB | test cache | yes, but not necessary |
| `.ruff_cache/` | 16 KB | lint cache | yes, but not necessary |
| `%LOCALAPPDATA%\pip\cache` | 714 KB | package download cache (already purged earlier, small regrowth) | yes |
| `%USERPROFILE%\.cache\torch` | 98 MB | pretrained ResNet-50 weights | only if re-downloading weights for retraining is acceptable |
| `%TEMP%` | 414 MB | OS-wide temp, outside the project | yes in principle, but not audited or touched here |

No `*.tmp`, `*.bak` or editor backup files were found inside the project. No cache was
cleaned during this audit.

## 8. Reclaim summary (not executed)

If a future cleanup is explicitly requested, the demonstrably disposable, non-required
items total about **32 MB** inside the project:

- legacy duplicate EDA figures: 3,945,576 bytes
- duplicate root `model_comparison.png`: 67,294 bytes
- legacy `figures/padim` + `figures/patchcore`: 28,254,422 bytes
- duplicate legacy prediction CSVs: 23,466 bytes
- CLI smoke outputs: 1,222,100 bytes

Do **not** delete raw datasets, the 12 unique model artifacts, calibration/training records,
saved test predictions/maps, final metrics, final reports, source code or notebooks. The
model binaries and prediction maps are git-ignored, so they do not affect repository size;
if they must be shared, publish them separately as release assets (documented in
`reports/final_release_checklist.md`) rather than committing them.
