# Model artifacts

Trained artifacts are written here by `scripts/04_train.py`, isolated per category:

```
models/<category>/<model>/artifact.pt
models/<category>/<model>/training.json
models/<category>/<model>/thresholds.json
```

For example, the locally evaluated categories have:

```
models/bottle/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
models/cable/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
models/capsule/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
models/grid/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
models/leather/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
models/transistor/{patchcore,padim}/{artifact.pt,training.json,thresholds.json}
```

All six supported categories have both PatchCore and PaDiM artifacts. Inference refuses
out-of-scope categories, and it refuses an untrained category rather than substituting
another category's model or thresholds.

The `.pt` files contain the fitted memory bank / Gaussian statistics together with
preprocessing, category and environment metadata. They are **not** committed to Git
(binary, large, and reproducible from the scripts). Training and threshold JSON records are
small and document exactly how the artifact was fitted and calibrated; each record includes
the category name so cross-category reuse is detectable.
