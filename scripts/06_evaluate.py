"""Phase 8/9: final test evaluation, error analysis and visual examples."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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
)
from industrial_defect.evaluation import (  # noqa: E402
    image_level_metrics,
    pixel_level_metrics,
    pro_metric,
)
from industrial_defect.training import (  # noqa: E402
    MODEL_NAMES,
    artifact_path,
    load_artifact,
    thresholds_path,
)
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    human_bytes,
    load_json,
    save_json,
    seed_everything,
)
from industrial_defect.visualization import (  # noqa: E402
    save_contact_sheet,
    save_inspection_panel,
    save_metric_comparison,
    save_score_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma list, 'active' or 'all'; categories are evaluated sequentially",
    )
    parser.add_argument("--model", choices=[*MODEL_NAMES, "both"], default="both")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _build_dataframe(
    relative_paths: list[str],
    defect_types: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    maps: list[np.ndarray],
    masks: list[np.ndarray],
    image_threshold: float,
    pixel_threshold: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "relative_path": relative_paths,
            "defect_type": defect_types,
            "label": labels.astype(int),
            "score": scores.astype(float),
        }
    )
    frame["image_threshold"] = float(image_threshold)
    frame["predicted_label"] = (frame["score"] >= image_threshold).astype(int)
    frame["outcome"] = "true_negative"
    frame.loc[(frame.label == 0) & (frame.predicted_label == 1), "outcome"] = "false_positive"
    frame.loc[(frame.label == 1) & (frame.predicted_label == 1), "outcome"] = "true_positive"
    frame.loc[(frame.label == 1) & (frame.predicted_label == 0), "outcome"] = "false_negative"
    frame["correct"] = frame["label"] == frame["predicted_label"]
    frame["predicted_area_fraction"] = [float((item >= pixel_threshold).mean()) for item in maps]
    frame["gt_area_fraction"] = [float(item.mean()) for item in masks]
    hits: list[bool | None] = []
    for index, label in enumerate(labels):
        if label == 1:
            gt = masks[index] > 0.5
            hits.append(bool(((maps[index] >= pixel_threshold) & gt).any()))
        else:
            hits.append(None)
    frame["localization_hit"] = hits
    return frame


def _error_analysis(frame: pd.DataFrame) -> dict:
    anomalies = frame[frame.label == 1]
    per_defect: dict[str, dict] = {}
    for defect, group in anomalies.groupby("defect_type"):
        per_defect[str(defect)] = {
            "images": int(len(group)),
            "detected": int(group.predicted_label.sum()),
            "missed": int((group.predicted_label == 0).sum()),
            "detection_rate": float(group.predicted_label.mean()),
            "mean_score": float(group.score.mean()),
            "min_score": float(group.score.min()),
            "max_score": float(group.score.max()),
            "localization_hits": int(group["localization_hit"].astype(bool).sum()),
            "localization_rate": float(group["localization_hit"].astype(bool).mean()),
        }
    false_positives = frame[frame.outcome == "false_positive"]
    false_negatives = frame[frame.outcome == "false_negative"]
    return {
        "false_positives": false_positives[["relative_path", "score"]].to_dict("records"),
        "false_negatives": false_negatives[
            ["relative_path", "defect_type", "score", "localization_hit"]
        ].to_dict("records"),
        "per_defect_type": per_defect,
        "localization": {
            "anomalous_images": int(len(anomalies)),
            "images_with_overlap": int(anomalies["localization_hit"].astype(bool).sum()),
            "localization_rate": float(anomalies["localization_hit"].astype(bool).mean()),
            "note": "A hit means the predicted defect region overlaps the ground-truth mask.",
        },
    }


def evaluate_model(
    config: dict,
    category: str,
    model_name: str,
    device,
    args: argparse.Namespace,
) -> dict:
    threshold_file = thresholds_path(config, category, model_name)
    if not threshold_file.is_file():
        raise FileNotFoundError(
            f"Missing thresholds {threshold_file}. Run scripts/05_calibrate.py first."
        )
    artifact_file = artifact_path(config, category, model_name)
    if not artifact_file.is_file():
        raise FileNotFoundError(
            f"Missing model artifact {artifact_file}. Run scripts/04_train.py first."
        )
    threshold_record = load_json(threshold_file)
    image_threshold = float(threshold_record["image_threshold"])
    pixel_threshold = float(threshold_record["pixel_threshold"])
    pixel_size = int(config["evaluation"]["pixel_size"])

    model, artifact_metadata = load_artifact(
        artifact_path(config, category, model_name), device=device
    )
    cat_dir = category_dir(config)
    dataset = MVTecDataset(
        cat_dir,
        "test",
        image_size=int(config["data"]["image_size"]),
        load_masks=True,
        mask_size=pixel_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("eval_batch_size", 8)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    started = time.perf_counter()
    maps_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    labels: list[int] = []
    scores: list[float] = []
    relative_paths: list[str] = []
    defect_types: list[str] = []
    image_paths: list[Path] = []
    mask_paths: list[Path | None] = []
    for batch in loader:
        maps, batch_scores = model.predict(batch["image"], out_size=pixel_size)
        maps_numpy = maps.numpy()[:, 0].astype(np.float32)
        masks_numpy = (batch["mask"].numpy()[:, 0] > 0.5).astype(np.uint8)
        for index in range(len(maps_numpy)):
            maps_list.append(maps_numpy[index])
            masks_list.append(masks_numpy[index])
            labels.append(int(batch["label"][index]))
            scores.append(float(batch_scores[index]))
            relative_paths.append(str(batch["relative_path"][index]))
            defect_types.append(str(batch["defect_type"][index]))
            image_paths.append(cat_dir / str(batch["relative_path"][index]))
            raw_mask = str(batch["mask_path"][index])
            mask_paths.append(Path(raw_mask) if raw_mask else None)
    inference_seconds = round(time.perf_counter() - started, 2)

    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float32)
    image_metrics = image_level_metrics(label_array, score_array, image_threshold)
    pixel_metrics = pixel_level_metrics(maps_list, masks_list, pixel_threshold)
    normal_indices = [index for index, label in enumerate(labels) if label == 0]
    anomalous_indices = [index for index, label in enumerate(labels) if label == 1]
    pro = pro_metric(
        [maps_list[index] for index in anomalous_indices],
        [masks_list[index] for index in anomalous_indices],
        [maps_list[index] for index in normal_indices],
        fpr_limit=float(config["evaluation"]["pro"]["fpr_limit"]),
        steps=int(config["evaluation"]["pro"]["steps"]),
    )
    frame = _build_dataframe(
        relative_paths,
        defect_types,
        label_array,
        score_array,
        maps_list,
        masks_list,
        image_threshold,
        pixel_threshold,
    )
    error_analysis = _error_analysis(frame)

    predictions_dir = configured_path(config, "predictions_dir") / category
    predictions_dir.mkdir(parents=True, exist_ok=True)
    csv_path = predictions_dir / f"{model_name}_test_predictions.csv"
    frame.to_csv(csv_path, index=False)
    maps_path = predictions_dir / f"{model_name}_test_maps.npz"
    np.savez_compressed(
        maps_path,
        maps=np.stack(maps_list).astype(np.float32),
        masks=np.stack(masks_list).astype(np.uint8),
        scores=score_array,
        labels=label_array.astype(np.int8),
        relative_paths=np.asarray(relative_paths),
        defect_types=np.asarray(defect_types),
    )

    metrics_dir = configured_path(config, "metrics_dir") / category
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "category": category,
        "evaluation_split": "official test split",
        "counts": {
            "normal": int((label_array == 0).sum()),
            "anomalous": int((label_array == 1).sum()),
            "total": int(label_array.size),
        },
        "pixel_evaluation": {
            "resolution": pixel_size,
            "note": "Pixel metrics are pooled over all test images at the stated resolution; "
            "masks are downsampled with max-pooling.",
        },
        "threshold_calibration": threshold_record,
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "pro_metrics": pro,
        "error_analysis": error_analysis,
        "model_artifact": _relative(artifact_path(config, category, model_name)),
        "model_metadata": {
            key: artifact_metadata.get(key)
            for key in ("model_config", "training", "preprocessing", "backbone", "environment")
        },
        "inference_seconds": inference_seconds,
        "artifacts": {
            "predictions_csv": _relative(csv_path),
            "maps_npz": _relative(maps_path),
            "maps_bytes": int(maps_path.stat().st_size),
            "maps_note": "Full test anomaly maps are stored locally for independent metric "
            "verification; reports/predictions is git-ignored.",
        },
    }
    save_json(metrics_dir / f"{model_name}_metrics.json", metrics)
    save_json(metrics_dir / f"{model_name}_error_analysis.json", error_analysis)

    print(
        f"[evaluate] {model_name}: image AUROC={image_metrics['image_auroc']:.4f} "
        f"AP={image_metrics['image_average_precision']:.4f} "
        f"F1={image_metrics['image_f1']:.4f} | pixel AUROC={pixel_metrics['pixel_auroc']:.4f} "
        f"AP={pixel_metrics['pixel_average_precision']:.4f} "
        f"F1={pixel_metrics['pixel_f1']:.4f} | AUPRO={pro['pro_auc']:.4f}"
    )
    print(
        f"[evaluate] {model_name}: predictions={csv_path} "
        f"maps={maps_path} ({human_bytes(maps_path.stat().st_size)})"
    )

    if not args.no_figures:
        figures_dir = configured_path(config, "figures_dir") / category / model_name
        figures_dir.mkdir(parents=True, exist_ok=True)
        display_title = f"{category}/{model_name}"
        save_score_distribution(
            label_array,
            score_array,
            image_threshold,
            figures_dir / "image_score_distribution.png",
            f"{display_title} - image-level score distribution ({category})",
        )
        max_examples = int(config["evaluation"]["max_examples_per_error_type"])

        def items_for(subset: pd.DataFrame) -> list[dict]:
            items: list[dict] = []
            for row in subset.head(max_examples).itertuples():
                index = int(row.Index)
                items.append(
                    {
                        "image_path": image_paths[index],
                        "anomaly_map": maps_list[index],
                        "mask_path": mask_paths[index],
                        "title": f"{row.outcome} | score={row.score:.3f} | "
                        f"{row.relative_path}",
                    }
                )
            return items

        true_positives = frame[frame.outcome == "true_positive"].sort_values(
            "score", ascending=False
        )
        true_negatives = frame[frame.outcome == "true_negative"].sort_values("score")
        false_positives = frame[frame.outcome == "false_positive"].sort_values(
            "score", ascending=False
        )
        false_negatives = frame[frame.outcome == "false_negative"].sort_values(
            "score", ascending=False
        )
        generated: dict[str, str] = {}
        if not true_positives.empty:
            target = figures_dir / "correct_detections_true_positives.png"
            save_contact_sheet(
                items_for(true_positives),
                target,
                pixel_threshold,
                f"{display_title} - correct defect detections (true positives)",
            )
            generated["true_positives"] = _relative(target)
        if not true_negatives.empty:
            target = figures_dir / "correct_detections_true_negatives.png"
            save_contact_sheet(
                items_for(true_negatives),
                target,
                pixel_threshold,
                f"{display_title} - correct normal predictions (true negatives)",
            )
            generated["true_negatives"] = _relative(target)
        if not false_positives.empty:
            target = figures_dir / "false_positives.png"
            save_contact_sheet(
                items_for(false_positives),
                target,
                pixel_threshold,
                f"{display_title} - false positives",
            )
            generated["false_positives"] = _relative(target)
        if not false_negatives.empty:
            target = figures_dir / "false_negatives.png"
            save_contact_sheet(
                items_for(false_negatives),
                target,
                pixel_threshold,
                f"{display_title} - false negatives",
            )
            generated["false_negatives"] = _relative(target)
        for name, subset in (
            ("true_positive", true_positives),
            ("true_negative", true_negatives),
            ("false_negative", false_negatives),
        ):
            if subset.empty:
                continue
            index = int(subset.index[0])
            target = figures_dir / f"inspection_{name}.png"
            save_inspection_panel(
                image_paths[index],
                maps_list[index],
                mask_paths[index],
                image_threshold,
                pixel_threshold,
                float(score_array[index]),
                target,
                title=f"{display_title} - {relative_paths[index]}",
            )
            generated[f"inspection_{name}"] = _relative(target)
        metrics["figures"] = generated
        save_json(metrics_dir / f"{model_name}_metrics.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    try:
        categories = select_categories(args.categories or args.category, config)
    except ValueError as exc:
        print(f"[evaluate] error: {exc}", file=sys.stderr)
        return 2
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))
    device = get_device(config)
    models = list(MODEL_NAMES) if args.model == "both" else [args.model]

    summary: list[dict] = []
    failures: list[str] = []
    for category in categories:
        config["data"]["category"] = category
        try:
            results: dict[str, dict] = {}
            for model_name in models:
                results[model_name] = evaluate_model(config, category, model_name, device, args)
        except (DatasetNotFoundError, DatasetLayoutError, FileNotFoundError) as exc:
            print(f"[evaluate] {category}: not evaluated -> {exc}")
            summary.append({"category": category, "status": "not_evaluated", "detail": str(exc)})
            failures.append(category)
            continue
        except Exception as exc:
            print(f"[evaluate] {category}: FAILED -> {exc}")
            summary.append({"category": category, "status": "error", "detail": str(exc)})
            failures.append(category)
            continue

        metrics_dir = configured_path(config, "metrics_dir") / category
        metrics_dir.mkdir(parents=True, exist_ok=True)
        comparison_rows = []
        comparison: dict[str, dict] = {}
        for model_name, metrics in results.items():
            combined = {
                **{
                    key: value
                    for key, value in metrics["image_metrics"].items()
                    if key != "image_confusion"
                },
                **{
                    key: value
                    for key, value in metrics["pixel_metrics"].items()
                    if key != "pixel_confusion"
                },
                "pro_auc": metrics["pro_metrics"]["pro_auc"],
                "image_threshold": metrics["image_metrics"]["image_threshold"],
                "pixel_threshold": metrics["pixel_metrics"]["pixel_threshold"],
            }
            comparison[model_name] = combined
            row = {"model": model_name}
            row.update(
                {key: value for key, value in combined.items() if isinstance(value, (int, float))}
            )
            comparison_rows.append(row)
        save_json(metrics_dir / "model_comparison.json", comparison)
        pd.DataFrame(comparison_rows).to_csv(metrics_dir / "model_comparison.csv", index=False)
        if not args.no_figures:
            save_metric_comparison(
                {
                    model_name: {
                        **metrics["image_metrics"],
                        "pro_auc": metrics["pro_metrics"]["pro_auc"],
                    }
                    for model_name, metrics in results.items()
                },
                configured_path(config, "figures_dir") / category / "model_comparison.png",
            )
        summary.append(
            {
                "category": category,
                "status": "evaluated",
                "models": {
                    model_name: {
                        "image_auroc": metrics["image_metrics"]["image_auroc"],
                        "image_average_precision": metrics["image_metrics"][
                            "image_average_precision"
                        ],
                        "pixel_auroc": metrics["pixel_metrics"]["pixel_auroc"],
                        "pro_auc": metrics["pro_metrics"]["pro_auc"],
                        "image_threshold": metrics["image_metrics"]["image_threshold"],
                        "pixel_threshold": metrics["pixel_metrics"]["pixel_threshold"],
                        "counts": metrics["counts"],
                        "predictions_csv": metrics["artifacts"]["predictions_csv"],
                    }
                    for model_name, metrics in results.items()
                },
            }
        )
        print(f"[evaluate] {category}: comparison -> {metrics_dir / 'model_comparison.csv'}")
        del results
        gc.collect()

    output = save_json(
        configured_path(config, "metrics_dir") / "evaluation_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": models,
            "categories": summary,
        },
    )
    print(f"[evaluate] summary: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
