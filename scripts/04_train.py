"""Phase 5/6: fit PatchCore and/or PaDiM on normal training images only."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
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
from industrial_defect.training import (  # noqa: E402
    MODEL_NAMES,
    artifact_path,
    build_extractor,
    build_model,
    fit_model,
    load_artifact,
    save_artifact,
    training_record_path,
)
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    environment_info,
    human_bytes,
    load_json,
    save_json,
    seed_everything,
    timer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma list, 'active' or 'all'; models are trained sequentially",
    )
    parser.add_argument(
        "--model", choices=[*MODEL_NAMES, "both"], default="both", help="Model(s) to train"
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts exist")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit fit images (smoke test only)"
    )
    parser.add_argument("--tag", type=str, default=None, help="Artifact suffix, e.g. 'smoke'")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def train_one(
    config: dict,
    category: str,
    model_name: str,
    fit_samples,
    val_samples,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    if args.max_images:
        fit_samples = fit_samples[: args.max_images]
    assert_normal_only(fit_samples)
    artifact = artifact_path(config, category, model_name, tag=args.tag)
    metadata_path = training_record_path(config, category, model_name, tag=args.tag)
    if artifact.exists() and not args.force:
        print(f"[train] {model_name}: artifact exists, skipping ({artifact})")
        return 0

    fit_dataset = MVTecDataset(
        category_dir(config),
        "train",
        samples=fit_samples,
        image_size=int(config["data"]["image_size"]),
        load_masks=False,
    )
    batch_size = int(config["data"].get("train_batch_size", 8))
    loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    print(
        f"[train] {model_name}: fitting on {len(fit_dataset)} normal images "
        f"(batch={batch_size}, device={device.type})"
    )
    extractor = build_extractor(config)
    model = build_model(config, model_name, extractor)
    model.to(device)
    started = time.perf_counter()
    stats = fit_model(model, loader, config)
    fit_seconds = round(time.perf_counter() - started, 3)

    # Validate the artifact by reloading it and comparing predictions on held-out normal images.
    val_dataset = MVTecDataset(
        category_dir(config),
        "train",
        samples=val_samples[:2],
        image_size=int(config["data"]["image_size"]),
        load_masks=False,
    )
    val_batch = next(
        iter(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0))
    )
    reference_maps, reference_scores = model.predict(val_batch["image"], out_size=256)

    training = {
        "stage": "smoke" if args.tag else "full",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "model": model_name,
        "fit_images": len(fit_dataset),
        "fit_seconds": fit_seconds,
        "seed": int(config["project"]["seed"]),
        "batch_size": batch_size,
        "image_size": int(config["data"]["image_size"]),
        "device": device.type,
        "normal_only": True,
        "test_data_used": False,
        "split_manifest": _relative(resolve_manifest_path(config, category)),
        "stats": stats,
        "environment": environment_info(),
    }

    with timer() as save_timer:
        save_artifact(artifact, model, category, config, training)
    size = artifact.stat().st_size
    print(f"[train] {model_name}: artifact saved {artifact} ({human_bytes(size)})")

    reload_started = time.perf_counter()
    reloaded, payload = load_artifact(artifact, device="cpu")
    reload_seconds = round(time.perf_counter() - reload_started, 3)
    new_maps, new_scores = reloaded.predict(val_batch["image"], out_size=256)
    max_diff = float((new_scores - reference_scores).abs().max())
    map_diff = float((new_maps - reference_maps).abs().max())
    finite = bool(torch.isfinite(new_scores).all() and torch.isfinite(new_maps).all())
    artifact_validation = {
        "reload_seconds": reload_seconds,
        "scores_original": reference_scores.tolist(),
        "scores_reloaded": new_scores.tolist(),
        "max_abs_score_diff": max_diff,
        "max_abs_map_diff": map_diff,
        "finite_outputs": finite,
        "passed": bool(finite and max_diff <= 1e-4 and map_diff <= 1e-4),
        "artifact_bytes": int(size),
        "artifact": _relative(artifact),
        "artifact_metadata_model": payload.get("model_name"),
    }
    training["artifact_validation"] = artifact_validation
    training["save_seconds"] = save_timer["seconds"]
    save_artifact(
        artifact, model, category, config, training
    )  # persist validation metadata inside the artifact
    save_json(metadata_path, training)
    print(
        f"[train] {model_name}: reload check passed={artifact_validation['passed']} "
        f"(max score diff {max_diff:.2e}, max map diff {map_diff:.2e})"
    )
    print(f"[train] {model_name}: metadata {metadata_path}")
    return 0 if artifact_validation["passed"] else 1


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    try:
        categories = select_categories(args.categories or args.category, config)
    except ValueError as exc:
        print(f"[train] error: {exc}", file=sys.stderr)
        return 2
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))
    device = get_device(config)
    models = list(MODEL_NAMES) if args.model == "both" else [args.model]

    import psutil

    summary: list[dict] = []
    failures: list[str] = []
    for category in categories:
        config["data"]["category"] = category
        try:
            cat_path = category_dir(config, category)
            split = get_or_create_normal_split(
                cat_path,
                resolve_manifest_path(config, category),
                val_fraction=float(config["data"]["normal_split"]["val_fraction"]),
                seed=int(config["data"]["normal_split"]["seed"]),
            )
        except (DatasetNotFoundError, DatasetLayoutError) as exc:
            print(f"[train] {category}: dataset unavailable -> {exc}")
            summary.append(
                {
                    "category": category,
                    "status": "dataset_unavailable",
                    "detail": str(exc),
                }
            )
            failures.append(category)
            continue
        print(
            f"[train] category={category} models={models} device={device.type} "
            f"fit={len(split['fit'])} val={len(split['val'])}"
        )
        status = 0
        try:
            for model_name in models:
                status |= train_one(
                    config, category, model_name, split["fit"], split["val"], args, device
                )
                gc.collect()
        except Exception as exc:  # keep other categories running, but record the failure
            print(f"[train] {category}: FAILED -> {exc}")
            summary.append({"category": category, "status": "error", "detail": str(exc)})
            failures.append(category)
            continue
        records = {}
        for model_name in models:
            path = training_record_path(config, category, model_name, tag=args.tag)
            records[model_name] = (
                load_json(path) if path.is_file() else {"status": "missing"}
            )
        summary.append(
            {
                "category": category,
                "status": "ok" if status == 0 else "incomplete",
                "fit_images": len(split["fit"]),
                "val_images": len(split["val"]),
                "models": {
                    name: {
                        "artifact": _relative(artifact_path(config, category, name, tag=args.tag)),
                        "fit_seconds": record.get("fit_seconds"),
                        "reload_passed": record.get("artifact_validation", {}).get("passed"),
                    }
                    for name, record in records.items()
                },
                "process_rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
            }
        )
        if status != 0:
            failures.append(category)
    output = save_json(
        configured_path(config, "metrics_dir") / "training_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": models,
            "device": device.type,
            "smoke_tag": args.tag,
            "environment": environment_info(),
            "categories": summary,
        },
    )
    print(f"[train] summary: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
