"""Cross-category metric collection and explicitly documented aggregation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .categories import supported_categories
from .config import configured_path, project_root
from .inventory import category_metrics
from .training import MODEL_NAMES
from .utils import save_json

METRIC_KEYS = (
    "image_auroc",
    "image_average_precision",
    "image_precision",
    "image_recall",
    "image_f1",
    "pixel_auroc",
    "pixel_average_precision",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pro_auc",
)

AGGREGATION_METHODS = {
    "macro_average": (
        "Unweighted arithmetic mean of the per-category values; every evaluated category "
        "contributes equally regardless of its test-set size."
    ),
    "pooled_confusions": (
        "Thresholded precision/recall/F1 recomputed from TP/FP/FN summed across categories. "
        "Image rows are weighted by image count; pixel rows by pixel count. AUROC/AP are "
        "ranking metrics and are NOT pooled from per-category summaries."
    ),
}


def build_row(config: dict[str, Any], category: str, model_name: str) -> dict[str, Any] | None:
    metrics = category_metrics(config, category, model_name)
    if metrics is None:
        return None
    image = metrics["image_metrics"]
    pixel = metrics["pixel_metrics"]
    pro = metrics["pro_metrics"]
    training = (metrics.get("model_metadata") or {}).get("training") or {}
    calibration = metrics.get("threshold_calibration") or {}
    counts = metrics.get("counts") or {}
    return {
        "row_type": "category",
        "category": category,
        "model": model_name,
        "n_fit": training.get("fit_images"),
        "n_calibration": calibration.get("val_images"),
        "n_test_normal": counts.get("normal"),
        "n_test_anomalous": counts.get("anomalous"),
        "n_test_total": counts.get("total"),
        "image_auroc": image.get("image_auroc"),
        "image_average_precision": image.get("image_average_precision"),
        "image_precision": image.get("image_precision"),
        "image_recall": image.get("image_recall"),
        "image_f1": image.get("image_f1"),
        "pixel_auroc": pixel.get("pixel_auroc"),
        "pixel_average_precision": pixel.get("pixel_average_precision"),
        "pixel_precision": pixel.get("pixel_precision"),
        "pixel_recall": pixel.get("pixel_recall"),
        "pixel_f1": pixel.get("pixel_f1"),
        "pro_auc": pro.get("pro_auc"),
        "image_threshold": image.get("image_threshold"),
        "pixel_threshold": pixel.get("pixel_threshold"),
        "fit_seconds": training.get("fit_seconds"),
        "inference_seconds": metrics.get("inference_seconds"),
        "image_confusion": image.get("image_confusion"),
        "pixel_confusion": pixel.get("pixel_confusion"),
    }


def collect_rows(
    config: dict[str, Any],
    categories: list[str] | tuple[str, ...] | None = None,
    models: tuple[str, ...] = MODEL_NAMES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in categories or supported_categories(config):
        for model_name in models:
            row = build_row(config, category, model_name)
            if row is not None:
                rows.append(row)
    return rows


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    for model_name in dict.fromkeys(row["model"] for row in rows):
        subset = [row for row in rows if row["model"] == model_name]
        macro = {key: _mean([row.get(key) for row in subset]) for key in METRIC_KEYS}
        image_conf = {
            key: sum((row.get("image_confusion") or {}).get(key, 0) for row in subset)
            for key in ("tp", "fp", "tn", "fn")
        }
        pixel_conf = {
            key: sum((row.get("pixel_confusion") or {}).get(key, 0) for row in subset)
            for key in ("tp", "fp", "tn", "fn")
        }
        image_p, image_r, image_f1 = _prf(
            image_conf["tp"], image_conf["fp"], image_conf["fn"]
        )
        pixel_p, pixel_r, pixel_f1 = _prf(
            pixel_conf["tp"], pixel_conf["fp"], pixel_conf["fn"]
        )
        per_model[model_name] = {
            "categories": [row["category"] for row in subset],
            "macro_average": macro,
            "pooled_confusions": {
                "image_confusion": image_conf,
                "image_precision": image_p,
                "image_recall": image_r,
                "image_f1": image_f1,
                "pixel_confusion": pixel_conf,
                "pixel_precision": pixel_p,
                "pixel_recall": pixel_r,
                "pixel_f1": pixel_f1,
                "n_test_normal": sum(int(row.get("n_test_normal") or 0) for row in subset),
                "n_test_anomalous": sum(
                    int(row.get("n_test_anomalous") or 0) for row in subset
                ),
            },
        }
    return {
        "aggregation_methods": AGGREGATION_METHODS,
        "evaluated_categories": sorted({row["category"] for row in rows}),
        "per_model": per_model,
    }


def aggregate_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    category_rows = pd.DataFrame(rows)
    aggregate = aggregate_rows(rows)
    extra: list[dict[str, Any]] = []
    for model_name, payload in aggregate["per_model"].items():
        macro = {
            "row_type": "macro_average",
            "category": "MACRO_AVERAGE",
            "model": model_name,
            "aggregation": "equal weight per category",
            **{key: payload["macro_average"].get(key) for key in METRIC_KEYS},
        }
        extra.append(macro)
        pooled = payload["pooled_confusions"]
        extra.append(
            {
                "row_type": "pooled_confusions",
                "category": "POOLED",
                "model": model_name,
                "aggregation": "summed TP/FP/TN/FN across categories",
                "image_precision": pooled["image_precision"],
                "image_recall": pooled["image_recall"],
                "image_f1": pooled["image_f1"],
                "pixel_precision": pooled["pixel_precision"],
                "pixel_recall": pooled["pixel_recall"],
                "pixel_f1": pooled["pixel_f1"],
                "n_test_normal": pooled["n_test_normal"],
                "n_test_anomalous": pooled["n_test_anomalous"],
            }
        )
    return pd.concat([category_rows, pd.DataFrame(extra)], ignore_index=True)


def category_ranking(
    rows: list[dict[str, Any]], metric: str = "image_auroc"
) -> dict[str, Any]:
    """Rank evaluated category/model pairs; refuses to rank a single category."""
    ranked = sorted(
        (row for row in rows if row.get(metric) is not None),
        key=lambda row: float(row[metric]),
        reverse=True,
    )
    if len({row["category"] for row in ranked}) < 2:
        return {
            "available": False,
            "reason": "Fewer than two categories have been evaluated; a ranking would be "
            "descriptive only and is not reported.",
            "metric": metric,
            "ranked": [
                {"category": row["category"], "model": row["model"], "value": row[metric]}
                for row in ranked
            ],
        }
    return {
        "available": True,
        "metric": metric,
        "easiest": {"category": ranked[0]["category"], "model": ranked[0]["model"], "value": ranked[0][metric]},
        "hardest": {"category": ranked[-1]["category"], "model": ranked[-1]["model"], "value": ranked[-1][metric]},
        "ranked": [
            {"category": row["category"], "model": row["model"], "value": row[metric]}
            for row in ranked
        ],
    }


def write_cross_category_metrics(config: dict[str, Any]) -> dict[str, Any]:
    rows = collect_rows(config)
    metrics_dir = configured_path(config, "metrics_dir")
    frame = aggregate_frame(rows)
    csv_path = metrics_dir / "cross_category_metrics.csv"
    frame.to_csv(csv_path, index=False)
    try:
        csv_display = csv_path.resolve().relative_to(project_root(config)).as_posix()
    except ValueError:
        csv_display = csv_path.name
    payload = {
        "aggregation_methods": AGGREGATION_METHODS,
        "rows": rows,
        "aggregates": aggregate_rows(rows),
        "rankings": {
            "image_auroc": category_ranking(rows, "image_auroc"),
            "pro_auc": category_ranking(rows, "pro_auc"),
        },
        "csv": csv_display,
    }
    json_path = save_json(metrics_dir / "cross_category_metrics.json", payload)
    return {"rows": rows, "aggregates": payload["aggregates"], "rankings": payload["rankings"],
            "csv": csv_path, "json": json_path, "frame": frame}
