# Final release checklist - six-category frozen scope

Generated: 2026-11-18

Release scope: exactly `transistor`, `cable`, `leather`, `bottle`, `capsule`, `grid`.
Target: GitHub source release plus separately hosted model/prediction artifacts.

## 1. Scope and configuration

| Check | Status | Evidence |
| --- | --- | --- |
| Six-category scope is explicit in central config | **PASS** | `data.supported_categories` in `configs/config.yaml` |
| Active scope matches supported scope | **PASS** | both lists contain the same six names |
| Nine other MVTec AD categories are out of scope | **PASS** | `out_of_scope_categories()` returns `carpet`, `hazelnut`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `wood`, `zipper` |
| `--categories all` resolves only to six | **PASS** | `select_categories("all", config)` returns the supported list |
| Out-of-scope category is rejected before model loading | **PASS** | `ensure_in_scope()`; `IndustrialInspector` raises `InferenceError: ... outside the frozen project scope` |
| App and CLI do not inventory or select out-of-scope categories | **PASS** | `category_inventory()` defaults to supported scope; `07_predict --list-categories` prints only six plus the out-of-scope note |

## 2. Model and evaluation evidence (preserved, not rebuilt)

| Check | Status | Evidence |
| --- | --- | --- |
| 12 category/model combinations exist | **PASS** | 6 categories x PatchCore + PaDiM |
| Each combination has `artifact.pt`, `thresholds.json`, `training.json` | **PASS** | all 36 files present under `models/<category>/<model>/` |
| All 12 artifacts load and validate their own category/model | **PASS** | constructed 12 `IndustrialInspector` instances; artifact metadata matched requested category/model |
| Thresholds match the artifact's category/model and are normal-only | **PASS** | 12/12 threshold records have matching `category`/`model` and `test_data_used: false` |
| Thresholds match the saved metrics records | **PASS** | 12/12 image and pixel thresholds equal values in `<model>_metrics.json` |
| Test predictions and metrics exist for all 12 | **PASS** | 12 prediction CSVs, 12 metric JSONs, 12 error-analysis JSONs |
| Six per-category reports exist | **PASS** | `reports/categories/{transistor,cable,leather,bottle,capsule,grid}.md` |
| No model was retrained or recalibrated for this release | **PASS** | no training/calibration/evaluation command was run; only report generation from saved files |
| Existing metrics were not changed | **PASS** | metric tables regenerated from the same saved metric files; cross-category values unchanged |
| Transistor artifacts and results preserved | **PASS** | same thresholds and saved predictions; browser/CLI integration test still matches |

Artifacts are byte-distinct: `md5sum` of the 12 supported artifacts returned 12 different hashes
(plus the separate smoke artifact).

## 3. Streamlit verification (lightweight one-image smoke)

| Check | Status | Evidence |
| --- | --- | --- |
| App shows exactly six selectable categories | **PASS** | expanded combobox listed `transistor, cable, leather, bottle, capsule, grid` |
| Out-of-scope categories are not offered | **PASS** | no out-of-scope name appears in the selectbox; caption lists the nine as not loaded |
| Both models are offered for a trained category | **PASS** | `cable` model options: `padim`, `patchcore` |
| Switching category loads that category's model/thresholds | **PASS** | `cable` / `padim` loaded; shown thresholds 26.403 / 13.344 |
| Single-image inference works | **PASS** | uploaded `cable/test/good/000.png`; 1024x1024 -> 512x512 in 167 ms on CPU |
| Result matches the saved evaluation | **PASS** | score 14.985 vs saved 14.9846; thresholds 26.403/13.344 vs saved 26.4028/13.3443; area 0.29% vs 0.0028915 |
| Missing artifact shows a clear error and never falls back | **PASS (tested)** | unit tests assert `No calibrated thresholds` and no cross-category fallback; app raises an explicit error for an untrained selection. Not exercised by deleting a real artifact. |
| App server stopped after the smoke test | **PASS** | Streamlit background process stopped |

## 4. Tests and lint

| Check | Status | Evidence |
| --- | --- | --- |
| Test suite passes | **PASS** | `python -m pytest -q`: 78 passed |
| Ruff passes | **PASS** | `python -m ruff check src scripts tests app`: All checks passed |
| Scope tests added | **PASS** | six-category partition, `"all"` resolution, all nine out-of-scope rejections, inventory default, inspector rejection |
| No expensive training/evaluation in tests | **PASS** | suite is lightweight; integration test only does a single inference against an existing artifact |

## 5. Documentation

| Check | Status | Evidence |
| --- | --- | --- |
| README declares six supported and nine out-of-scope categories | **PASS** | top section and Scope section |
| README states results apply only to six, not a 15-category benchmark | **PASS** | top section and Limitations |
| README reports frozen-scope rationale | **PASS** | reproducibility and CPU/storage budget |
| README metric tables match saved files | **PASS** | values unchanged from evaluated artifacts |
| Final report states frozen scope and limitations | **PASS** | `reports/final_report.md` sections 1, 2, 6 |
| Per-category model/calibration/evaluation isolation documented | **PASS** | README and final report |
| Storage audit exists | **PASS** | `reports/storage_audit.md` |
| No "add a category" instructions remain | **PASS** | replaced by the frozen-scope section |

## 6. Repository hygiene

| Check | Status | Evidence |
| --- | --- | --- |
| `.gitignore` present | **PASS** | ignores all 15 dataset directories, `data/raw/*`, `data/processed/*`, `models/**`, `*.pt`, `reports/predictions/`, `*.npz`, caches, `.env` |
| Raw datasets excluded from Git | **PASS** | 0 dataset candidate files; `transistor/` and external mappings are ignored |
| Model binaries excluded from Git | **PASS** | 13 `.pt` files present locally, all ignored |
| Full prediction collections excluded from Git | **PASS** | 12 NPZ maps (~1.2 GB) and prediction CSVs under the ignored predictions directory |
| No absolute user/home paths in tracked text | **PASS** | scan: 0 hits |
| No secret-like assignments | **PASS** | scan: 0 hits |
| No tracked file over 10 MB | **PASS** | largest candidate ~5.5 MB |
| Candidate repository size | **PASS (documented)** | 307 files, 242.7 MB; dominated by committed figures and executed notebooks |
| `requirements.txt` / `requirements-dev.txt` present and pinned | **PASS** | runtime and dev pins match `pyproject.toml` |
| `pyproject.toml` valid project metadata | **PASS** | setuptools src layout, pytest and Ruff config |
| Git repository initialized / commit created | **NOT DONE (intentional)** | audit only; no commit or push was requested or performed |

## 7. Artifact storage strategy

GitHub will contain source, configs, tests, docs, reports, metrics and figures (242.7 MB).
It will **not** contain raw datasets, model binaries or full prediction maps.

If exact inference/evaluation artifacts must be distributed:

1. Publish `models/` (~426 MB) and optionally `reports/predictions/` (~1.2 GB) as a separate
   release asset or object-storage archive.
2. Include a SHA-256 manifest so recipients can verify the 12 artifacts and prediction maps.
3. Alternatively, regenerate them with the documented pipeline from the licensed MVTec AD
   data; thresholds and splits are deterministic under the recorded seed.

Do not commit the dataset (CC BY-NC-SA 4.0, non-commercial, not redistributable here) and do
not commit the `.pt`/`.npz` binaries.

## 8. Known limitations (not blockers)

- Six categories are supported by design; this is not a complete 15-category benchmark.
- The dataset is external and must be obtained from the official MVTec AD source.
- Model artifacts and full prediction maps are not in Git; exact inference requires the
  external artifacts or retraining.
- Disk is at 99% (2.2 GB free); a full re-run needs more free space.
- Category test sets are small (78-150 images); threshold and metric uncertainty is wide.
- Pixel metrics use 512x512 evaluation, not native 1024x1024.
- Leather and capsule/PaDiM show known operating-point effects documented in the README.
- No Git commit, tag or push was created.

## 9. Verdict

**READY WITH DOCUMENTED LIMITATIONS**

All required six-category model artifacts, thresholds, metrics, predictions, reports and
scope checks are present and verified. There are **no blocking defects**. The documented
limitations are the intentional nine-category exclusion, external dataset/artifact hosting,
and the absence of a Git commit/push action.
