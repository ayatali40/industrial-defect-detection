"""Migrate pre-expansion flat artifacts into models/<category>/<model>/ and verify them.

The move preserves the fitted tensors exactly; only artifact-path metadata is
rewritten. Verification reloads each migrated model and compares its scores on
the official test images against the saved per-image predictions that produced
the published metrics. No training or calibration is performed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from industrial_defect.categories import select_categories  # noqa: E402
from industrial_defect.config import (  # noqa: E402
    category_dir,
    configured_path,
    ensure_output_dirs,
    load_config,
)
from industrial_defect.dataset import MVTecDataset, discover_samples  # noqa: E402
from industrial_defect.training import (  # noqa: E402
    MODEL_NAMES,
    artifact_path,
    load_artifact,
    migrate_legacy_artifact,
)
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None, help="Single category")
    parser.add_argument("--categories", type=str, default=None, help="Comma list, 'active' or 'all'")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "both"], default="both")
    parser.add_argument("--skip-verification", action="store_true")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _prediction_frame(config: dict, category: str, model_name: str) -> pd.DataFrame | None:
    predictions_dir = configured_path(config, "predictions_dir")
    new_path = predictions_dir / category / f"{model_name}_test_predictions.csv"
    legacy_path = predictions_dir / f"{category}_{model_name}_test_predictions.csv"
    for path in (new_path, legacy_path):
        if path.is_file():
            return pd.read_csv(path)
    return None


def _verify(config: dict, category: str, model_name: str) -> dict:
    frame = _prediction_frame(config, category, model_name)
    if frame is None:
        return {"verified": False, "reason": "no saved test predictions available"}
    samples = {
        sample.relative_path: sample for sample in discover_samples(category_dir(config, category), "test")
    }
    missing = [path for path in frame["relative_path"] if path not in samples]
    if missing:
        return {"verified": False, "reason": f"test images missing: {missing[:3]}"}
    ordered = [samples[path] for path in frame["relative_path"]]
    dataset = MVTecDataset(
        category_dir(config, category),
        "test",
        samples=ordered,
        image_size=int(config["data"]["image_size"]),
    )
    loader = DataLoader(dataset, batch_size=int(config["data"].get("eval_batch_size", 8)))
    model, _ = load_artifact(artifact_path(config, category, model_name), device="cpu")
    scores = []
    for batch in loader:
        _, batch_scores = model.predict(
            batch["image"], out_size=int(config["evaluation"]["pixel_size"])
        )
        scores.extend(float(value) for value in batch_scores)
    expected = frame["score"].to_numpy()
    diff = (pd.Series(scores).to_numpy() - expected).__abs__()
    return {
        "verified": True,
        "images": int(len(scores)),
        "max_abs_score_diff": float(diff.max()),
        "mean_abs_score_diff": float(diff.mean()),
        "passed": bool(diff.max() <= 1e-4),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))
    categories = select_categories(args.categories or args.category, config)
    models = list(MODEL_NAMES) if args.model == "both" else [args.model]
    records: list[dict] = []
    status = 0
    for category in categories:
        for model_name in models:
            result = migrate_legacy_artifact(config, category, model_name)
            record = {
                "category": category,
                "model": model_name,
                "status": result["status"],
                "artifact": _relative(result["artifact"]) if result["artifact"].exists() else None,
            }
            if result["status"] == "migrated" and not args.skip_verification:
                verification = _verify(config, category, model_name)
                record["verification"] = verification
                if verification.get("verified") and not verification.get("passed", True):
                    status = 1
            records.append(record)
            print(f"[migrate] {category}/{model_name}: {record['status']}")
            if "verification" in record:
                verification = record["verification"]
                if verification.get("verified"):
                    print(
                        f"[migrate] {category}/{model_name}: {verification['images']} images, "
                        f"max score diff {verification['max_abs_score_diff']:.2e}, "
                        f"passed={verification['passed']}"
                    )
                else:
                    print(f"[migrate] {category}/{model_name}: not verified ({verification['reason']})")
    metrics_dir = configured_path(config, "metrics_dir")
    output = save_json(
        metrics_dir / "artifact_migration.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "Layout migration only; no model weights or thresholds were changed.",
            "records": records,
        },
    )
    print(f"[migrate] record: {output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
