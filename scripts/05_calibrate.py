"""Phase 7: calibrate image and pixel thresholds on normal validation images only."""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from industrial_defect.categories import select_categories  # noqa: E402
from industrial_defect.config import (  # noqa: E402
    category_dir,
    configured_path,
    ensure_output_dirs,
    get_device,
    load_config,
)
from industrial_defect.dataset import (  # noqa: E402
    DatasetLayoutError,
    DatasetNotFoundError,
    MVTecDataset,
    assert_normal_only,
    get_or_create_normal_split,
    resolve_manifest_path,
)
from industrial_defect.thresholding import calibrate_thresholds  # noqa: E402
from industrial_defect.training import (  # noqa: E402
    MODEL_NAMES,
    artifact_path,
    load_artifact,
    thresholds_path,
)
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    environment_info,
    load_json,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma list, 'active' or 'all'; categories are processed sequentially",
    )
    parser.add_argument("--model", choices=[*MODEL_NAMES, "both"], default="both")
    parser.add_argument("--tag", type=str, default=None, help="Artifact suffix, e.g. 'smoke'")
    parser.add_argument("--force", action="store_true", help="Recalibrate existing thresholds")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def calibrate_category(
    config: dict,
    category: str,
    models: list[str],
    args: argparse.Namespace,
    device,
) -> list[dict]:
    split = get_or_create_normal_split(
        category_dir(config, category),
        resolve_manifest_path(config, category),
        val_fraction=float(config["data"]["normal_split"]["val_fraction"]),
        seed=int(config["data"]["normal_split"]["seed"]),
    )
    assert_normal_only(split["val"])
    val_dataset = MVTecDataset(
        category_dir(config, category),
        "train",
        samples=split["val"],
        image_size=int(config["data"]["image_size"]),
        load_masks=False,
    )
    loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"].get("eval_batch_size", 8)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    records: list[dict] = []
    for model_name in models:
        artifact = artifact_path(config, category, model_name, tag=args.tag)
        output = thresholds_path(config, category, model_name, tag=args.tag)
        if not artifact.is_file():
            print(f"[calibrate] {category}/{model_name}: missing artifact {artifact}")
            records.append(
                {"category": category, "model": model_name, "status": "missing_artifact"}
            )
            continue
        if output.is_file() and not args.force:
            print(f"[calibrate] {category}/{model_name}: thresholds exist, skipping {output}")
            records.append(
                {
                    "category": category,
                    "model": model_name,
                    "status": "existing",
                    "thresholds": load_json(output),
                }
            )
            continue
        model, _ = load_artifact(artifact, device=device)
        record, scores, pixels = calibrate_thresholds(
            model,
            loader,
            eval_size=int(config["evaluation"]["pixel_size"]),
            image_percentile=float(config["thresholds"]["image_percentile"]),
            pixel_percentile=float(config["thresholds"]["pixel_percentile"]),
        )
        record.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "category": category,
                "model": model_name,
                "model_artifact": _relative(artifact),
                "split_manifest": _relative(resolve_manifest_path(config, category)),
                "seed": int(config["project"]["seed"]),
                "val_paths": [sample.relative_path for sample in split["val"]],
                "val_score_histogram": {
                    "edges": np.histogram(scores, bins=10)[1].tolist(),
                    "counts": np.histogram(scores, bins=10)[0].tolist(),
                },
                "environment": environment_info(),
            }
        )
        save_json(output, record)
        print(
            f"[calibrate] {category}/{model_name}: image_threshold={record['image_threshold']:.4f} "
            f"(p{record['image_percentile']}), pixel_threshold={record['pixel_threshold']:.4f} "
            f"(p{record['pixel_percentile']}) -> {output}"
        )
        print(
            f"[calibrate] {category}/{model_name}: val FPR image="
            f"{record['val_image_false_positive_rate_at_threshold']:.4f} pixel="
            f"{record['val_pixel_false_positive_rate_at_threshold']:.6f}"
        )
        records.append({"category": category, "model": model_name, "status": "calibrated"})
        del model
        gc.collect()
    return records


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    try:
        categories = select_categories(args.categories or args.category, config)
    except ValueError as exc:
        print(f"[calibrate] error: {exc}", file=sys.stderr)
        return 2
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))
    device = get_device(config)
    models = list(MODEL_NAMES) if args.model == "both" else [args.model]

    records: list[dict] = []
    failures: list[str] = []
    for category in categories:
        config["data"]["category"] = category
        try:
            records.extend(calibrate_category(config, category, models, args, device))
        except (DatasetNotFoundError, DatasetLayoutError) as exc:
            print(f"[calibrate] {category}: dataset unavailable -> {exc}")
            records.append(
                {"category": category, "status": "dataset_unavailable", "detail": str(exc)}
            )
            failures.append(category)
            continue
        except Exception as exc:
            print(f"[calibrate] {category}: FAILED -> {exc}")
            records.append({"category": category, "status": "error", "detail": str(exc)})
            failures.append(category)
            continue
        gc.collect()
    for record in records:
        if record.get("status") == "missing_artifact":
            failures.append(str(record["category"]))
    output = save_json(
        configured_path(config, "metrics_dir") / "calibration_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": models,
            "records": records,
        },
    )
    print(f"[calibrate] summary: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
