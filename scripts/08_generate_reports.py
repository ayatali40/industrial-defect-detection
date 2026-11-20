"""Phase 7: generate per-category, cross-category and final reports plus GitHub readiness."""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from industrial_defect.categories import (  # noqa: E402
    MVTEC_CATEGORIES,
    out_of_scope_categories,
    supported_categories,
)
from industrial_defect.config import (  # noqa: E402
    configured_path,
    ensure_output_dirs,
    load_config,
)
from industrial_defect.cross_category import (  # noqa: E402
    aggregate_rows,
    category_ranking,
    collect_rows,
    write_cross_category_metrics,
)
from industrial_defect.inventory import category_inventory, category_status  # noqa: E402
from industrial_defect.training import (  # noqa: E402
    MODEL_NAMES,
    thresholds_path,
    training_record_path,
)
from industrial_defect.utils import human_bytes, load_json, save_json  # noqa: E402

MODELS = MODEL_NAMES
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".ipynb"}
SCAN_SKIP_PARTS = {
    "predictions",
    "processed",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _load_optional(path: Path) -> dict | None:
    return load_json(path) if path.is_file() else None


def _bundle(config: dict, category: str) -> dict[str, Any]:
    metrics_dir = configured_path(config, "metrics_dir") / category
    predictions_dir = configured_path(config, "predictions_dir") / category
    metrics = {
        model: _load_optional(metrics_dir / f"{model}_metrics.json") for model in MODELS
    }
    return {
        "category": category,
        "status": category_status(config, category),
        "audit": _load_optional(metrics_dir / "dataset_audit.json"),
        "eda": _load_optional(metrics_dir / "eda_summary.json"),
        "metrics": metrics,
        "training": {
            model: _load_optional(training_record_path(config, category, model))
            for model in MODELS
        },
        "thresholds": {
            model: _load_optional(thresholds_path(config, category, model))
            for model in MODELS
        },
        "frames": {
            model: (
                pd.read_csv(predictions_dir / f"{model}_test_predictions.csv")
                if (predictions_dir / f"{model}_test_predictions.csv").is_file()
                else None
            )
            for model in MODELS
        },
        "evaluated": [model for model, item in metrics.items() if item is not None],
    }


def _figure_path(config: dict, *parts: str) -> str | None:
    path = configured_path(config, "figures_dir").joinpath(*parts)
    if not path.is_file():
        return None
    return f"figures/{'/'.join(parts)}"


def _results_table(metrics: dict[str, dict]) -> list[str]:
    lines = [
        "| Model | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        item = metrics.get(model)
        if item is None:
            lines.append(f"| {model} | not evaluated | | | | | | | | | | |")
            continue
        image = item["image_metrics"]
        pixel = item["pixel_metrics"]
        lines.append(
            f"| {model} | {_fmt(image['image_auroc'])} | {_fmt(image['image_average_precision'])} | "
            f"{_fmt(image['image_precision'])} | {_fmt(image['image_recall'])} | {_fmt(image['image_f1'])} | "
            f"{_fmt(pixel['pixel_auroc'])} | {_fmt(pixel['pixel_average_precision'])} | "
            f"{_fmt(pixel['pixel_precision'])} | {_fmt(pixel['pixel_recall'])} | {_fmt(pixel['pixel_f1'])} | "
            f"{_fmt(item['pro_metrics']['pro_auc'])} |"
        )
    return lines


def _threshold_table(bundle: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Validation images | Image threshold | Pixel threshold | Image percentile | Pixel percentile | Val image FPR | Val pixel FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        record = bundle["thresholds"].get(model)
        if record is None:
            lines.append(f"| {model} | not calibrated | | | | | | |")
            continue
        lines.append(
            f"| {model} | {record['val_images']} | {_fmt(record['image_threshold'])} | "
            f"{_fmt(record['pixel_threshold'])} | {record['image_percentile']} | "
            f"{record['pixel_percentile']} | "
            f"{_fmt(record['val_image_false_positive_rate_at_threshold'])} | "
            f"{_fmt(record['val_pixel_false_positive_rate_at_threshold'], 6)} |"
        )
    return lines


def _training_table(bundle: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Fit images | Fit time (s) | Device | Model-specific configuration | Reload max score diff |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for model in MODELS:
        record = bundle["training"].get(model)
        if record is None:
            lines.append(f"| {model} | not trained | | | | |")
            continue
        stats = record["stats"]
        if model == "patchcore":
            model_info = (
                f"projection={stats['projected_dim']}, candidate patches={stats['candidate_patches']}, "
                f"coreset={stats['memory_bank_size']}"
            )
        else:
            model_info = (
                f"positions={stats['positions']}, features={stats['n_features']}, "
                f"epsilon={stats['covariance_epsilon']}"
            )
        lines.append(
            f"| {model} | {record['fit_images']} | {record['fit_seconds']:.2f} | "
            f"{record['device']} | {model_info} | "
            f"{record['artifact_validation']['max_abs_score_diff']:.3g} |"
        )
    return lines


def _defect_table(metrics: dict[str, dict]) -> list[str]:
    defects: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        item = metrics.get(model)
        if item is None:
            continue
        for defect, stats in item["error_analysis"]["per_defect_type"].items():
            defects.setdefault(defect, {})[model] = stats
    if not defects:
        return ["No per-defect statistics available."]
    lines = [
        "| Defect type | Images | "
        + " | ".join(f"{model} detected | {model} localised" for model in MODELS)
        + " |",
        "| --- | ---: | " + " | ".join("---: | ---:" for _ in MODELS) + " |",
    ]
    for defect in sorted(defects):
        cells = [f"`{defect}`"]
        patch = defects[defect].get("patchcore")
        cells.append(str(patch["images"]) if patch else "n/a")
        for model in MODELS:
            stats = defects[defect].get(model)
            cells.append(str(stats["detected"]) if stats else "n/a")
            cells.append(str(stats["localization_hits"]) if stats else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _outcome_table(frames: dict[str, pd.DataFrame | None]) -> list[str]:
    lines = [
        "| Model | True positives | False positives | True negatives | False negatives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        frame = frames.get(model)
        if frame is None:
            lines.append(f"| {model} | not available | not available | not available | not available |")
            continue
        counts = frame["outcome"].value_counts()
        lines.append(
            f"| {model} | {int(counts.get('true_positive', 0))} "
            f"| {int(counts.get('false_positive', 0))} "
            f"| {int(counts.get('true_negative', 0))} "
            f"| {int(counts.get('false_negative', 0))} |"
        )
    return lines


def _missed_examples(frames: dict[str, pd.DataFrame | None]) -> list[str]:
    lines = []
    for model in MODELS:
        frame = frames.get(model)
        if frame is None or frame.empty:
            continue
        missed = frame[frame.outcome == "false_negative"].sort_values("score", ascending=False)
        if missed.empty:
            lines.append(f"- **{model}**: no false negatives.")
            continue
        examples = ", ".join(
            f"`{row.relative_path}` ({row.defect_type}, score {row.score:.2f})"
            for row in missed.head(8).itertuples()
        )
        suffix = "" if len(missed) <= 8 else f"; {len(missed) - 8} more in the per-image CSV"
        lines.append(f"- **{model}** misses ({len(missed)}): {examples}{suffix}.")
    return lines


def _sample_rows(bundle: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Fit images | Calibration images | Test normal | Test anomalous | Test total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        item = bundle["metrics"].get(model)
        training = bundle["training"].get(model) or {}
        threshold = bundle["thresholds"].get(model) or {}
        counts = (item or {}).get("counts", {})
        lines.append(
            f"| {model} | {training.get('fit_images', 'n/a')} "
            f"| {threshold.get('val_images', 'n/a')} "
            f"| {counts.get('normal', 'n/a')} | {counts.get('anomalous', 'n/a')} "
            f"| {counts.get('total', 'n/a')} |"
        )
    return lines


def _render_category_report(bundle: dict[str, Any]) -> str:
    category = bundle["category"]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit = bundle["audit"]
    eda = bundle["eda"]
    metrics = bundle["metrics"]
    lines: list[str] = [
        f"# Category report: `{category}`",
        "",
        f"Generated: {generated}",
        "",
        "This report covers only the category named above. Thresholds, model state and "
        "metrics are never shared with another category.",
        "",
        "## Dataset summary",
        "",
    ]
    if audit:
        counts = audit.get("counts", {})
        lines += [
            f"- Dataset root: `{audit['dataset']['root']}`",
            f"- Category directory: `{audit['dataset']['category_dir']}`",
            f"- Readiness: **{'PASS' if audit['checks'].get('dataset_ready', {}).get('passed') else 'FAIL'}**",
            f"- Critical issues: {sum(1 for item in audit['issues'] if item['severity'] == 'critical')}",
            "",
            "| Split / class | Images | Masks |",
            "| --- | ---: | ---: |",
            f"| train/good | {counts.get('train_good_images', 0)} | 0 |",
            f"| test/good | {counts.get('test_good_images', 0)} | 0 |",
            f"| test/anomalous | {counts.get('test_anomalous_images', 0)} | "
            f"{counts.get('ground_truth_masks', 0)} |",
            "",
        ]
    else:
        lines += ["No dataset audit found; run `scripts/01_validate_dataset.py`.", ""]

    lines += ["## Training and calibration", ""]
    lines += _training_table(bundle)
    lines += [""]
    lines += _threshold_table(bundle)
    lines += [
        "",
        "Thresholds are computed on held-out normal training images of this category only "
        "(`test_data_used: false` in the threshold record).",
        "",
        "## Evaluation results",
        "",
    ]
    lines += _sample_rows(bundle)
    lines += [""]
    lines += _results_table(metrics)
    lines += [
        "",
        "Image metrics use the calibrated image threshold; pixel metrics are pooled over all "
        "test pixels at the configured evaluation resolution; AUPRO integrates per-region "
        "overlap up to the configured false-positive limit. Metric definitions are unchanged "
        "from the single-category implementation.",
        "",
        "## Per-defect behaviour",
        "",
        "(localised = predicted pixel region overlaps the ground-truth mask)",
        "",
    ]
    lines += _defect_table(metrics)
    lines += ["", "## Decision outcomes", ""]
    lines += _outcome_table(bundle["frames"])
    lines += ["", "### Missed anomalies", ""]
    lines += _missed_examples(bundle["frames"])
    lines += ["", "## Visual evidence", ""]
    figures = {
        "model comparison": _figure_path(bundle["config"], category, "model_comparison.png")
        if "config" in bundle
        else None,
        "normal samples": _figure_path(bundle["config"], category, "eda", "normal_training_samples.png")
        if "config" in bundle
        else None,
        "anomalous samples": _figure_path(bundle["config"], category, "eda", "anomalous_test_samples.png")
        if "config" in bundle
        else None,
        "defect area": _figure_path(bundle["config"], category, "eda", "defect_area_distribution.png")
        if "config" in bundle
        else None,
    }
    for model in MODELS:
        figures[f"{model} true positives"] = _figure_path(
            bundle["config"], category, model, "correct_detections_true_positives.png"
        ) if "config" in bundle else None
        figures[f"{model} false negatives"] = _figure_path(
            bundle["config"], category, model, "false_negatives.png"
        ) if "config" in bundle else None
    for label, path in figures.items():
        if path:
            lines.append(f"- [{label}](../../{path})")
    if eda:
        lines += ["", "## Descriptive EDA", ""]
        lines += [
            "| Defect type | Masks | Median area (% of image) | Max area (%) | Components (min-max) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for defect, stats in eda["defect_masks"].items():
            area = stats["area_fraction"]
            components = stats["components"]
            lines.append(
                f"| `{defect}` | {stats['masks']} | {area['median'] * 100:.2f} | "
                f"{area['max'] * 100:.2f} | {components['min']}-{components['max']} |"
            )
        lines += ["", "Observations:", ""]
        lines += [f"- {observation}" for observation in eda["observations"]]
    lines += [
        "",
        "## Limitations",
        "",
        "- Thresholds are calibrated on normal images only and may drift with lighting, "
        "camera or fixture changes.",
        "- Pixel metrics use the configured evaluation resolution, not the original image "
        "resolution; numbers are not directly comparable to full-resolution literature.",
        "- The models score deviation from normal; they do not classify defect types.",
        "- Small test and calibration splits make threshold false-positive estimates coarse.",
        "",
    ]
    return "\n".join(lines)


def _status_table(inventory: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Category | Dataset | Trained models | Evaluated models |",
        "| --- | --- | --- | --- |",
    ]
    for category, status in inventory.items():
        lines.append(
            f"| `{category}` | {'available' if status['data_available'] else 'missing'} "
            f"| {', '.join(status['trained_models']) or '-'} "
            f"| {', '.join(status['evaluated_models']) or '-'} |"
        )
    return lines


def _aggregate_tables(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        if not subset:
            continue
        lines += [
            f"### {model}",
            "",
            "| Category | Image AUROC | Image AP | Image P | Image R | Image F1 | Pixel AUROC | Pixel AP | Pixel P | Pixel R | Pixel F1 | AUPRO |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in subset:
            lines.append(
                f"| `{row['category']}` | {_fmt(row['image_auroc'])} | "
                f"{_fmt(row['image_average_precision'])} | {_fmt(row['image_precision'])} | "
                f"{_fmt(row['image_recall'])} | {_fmt(row['image_f1'])} | "
                f"{_fmt(row['pixel_auroc'])} | {_fmt(row['pixel_average_precision'])} | "
                f"{_fmt(row['pixel_precision'])} | {_fmt(row['pixel_recall'])} | "
                f"{_fmt(row['pixel_f1'])} | {_fmt(row['pro_auc'])} |"
            )
        lines.append("")
    return lines


def _macro_table(aggregates: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Categories | Image AUROC | Image AP | Image F1 | Pixel AUROC | Pixel AP | Pixel F1 | AUPRO |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, payload in aggregates["per_model"].items():
        macro = payload["macro_average"]
        lines.append(
            f"| {model} | {len(payload['categories'])} | {_fmt(macro['image_auroc'])} | "
            f"{_fmt(macro['image_average_precision'])} | {_fmt(macro['image_f1'])} | "
            f"{_fmt(macro['pixel_auroc'])} | {_fmt(macro['pixel_average_precision'])} | "
            f"{_fmt(macro['pixel_f1'])} | {_fmt(macro['pro_auc'])} |"
        )
    return lines


def _pooled_table(aggregates: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Test normal | Test anomalous | Image TP | Image FP | Image FN | Image P | Image R | Image F1 | Pixel P | Pixel R | Pixel F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, payload in aggregates["per_model"].items():
        pooled = payload["pooled_confusions"]
        image = pooled["image_confusion"]
        lines.append(
            f"| {model} | {pooled['n_test_normal']} | {pooled['n_test_anomalous']} "
            f"| {image['tp']} | {image['fp']} | {image['fn']} "
            f"| {_fmt(pooled['image_precision'])} | {_fmt(pooled['image_recall'])} "
            f"| {_fmt(pooled['image_f1'])} | {_fmt(pooled['pixel_precision'])} "
            f"| {_fmt(pooled['pixel_recall'])} | {_fmt(pooled['pixel_f1'])} |"
        )
    return lines


def _ranking_section(rankings: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for metric, payload in rankings.items():
        label = "image AUROC" if metric == "image_auroc" else "AUPRO"
        if not payload["available"]:
            lines.append(
                f"- **Easiest/hardest by {label}**: not reported. {payload['reason']}"
            )
            continue
        lines.append(
            f"- **{label}**: easiest `{payload['easiest']['category']}` / "
            f"{payload['easiest']['model']} ({_fmt(payload['easiest']['value'])}); hardest "
            f"`{payload['hardest']['category']}` / {payload['hardest']['model']} "
            f"({_fmt(payload['hardest']['value'])})."
        )
    return lines


def _threshold_differences(bundles: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Category | Model | Image threshold | Pixel threshold | Validation images |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for category, bundle in bundles.items():
        for model in MODELS:
            record = bundle["thresholds"].get(model)
            if record is None:
                continue
            lines.append(
                f"| `{category}` | {model} | {_fmt(record['image_threshold'])} | "
                f"{_fmt(record['pixel_threshold'])} | {record['val_images']} |"
            )
    lines += [
        "",
        "Thresholds are percentile-based per category/model, but raw score scales are not "
        "comparable across categories (different normal appearance and distance magnitudes). "
        "Only the resulting decisions and metrics are comparable.",
    ]
    return lines


def _defect_appearance(bundles: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Category | Defect type | Masks | Median area (% of image) | Max area (%) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    found = False
    for category, bundle in bundles.items():
        eda = bundle["eda"]
        if not eda:
            continue
        for defect, stats in eda["defect_masks"].items():
            found = True
            lines.append(
                f"| `{category}` | `{defect}` | {stats['masks']} "
                f"| {stats['area_fraction']['median'] * 100:.2f} "
                f"| {stats['area_fraction']['max'] * 100:.2f} |"
            )
    if not found:
        return ["No descriptive defect statistics available."]
    return lines


def _render_final_report(
    config: dict,
    inventory: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregates: dict[str, Any],
    rankings: dict[str, Any],
) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    supported = supported_categories(config)
    out_of_scope = out_of_scope_categories(config)
    evaluated = [name for name in supported if bundles.get(name, {}).get("evaluated")]
    trained_not_evaluated = [
        name
        for name in supported
        if inventory[name]["trained_models"] and not inventory[name]["evaluated_models"]
    ]
    data_not_trained = [
        name
        for name in supported
        if inventory[name]["data_available"] and not inventory[name]["trained_models"]
    ]
    unavailable = [name for name in supported if not inventory[name]["data_available"]]
    environment = None
    for bundle in bundles.values():
        for record in bundle["training"].values():
            if record and record.get("environment"):
                environment = record["environment"]
                break
        if environment:
            break

    lines: list[str] = [
        "# Industrial Defect Detection and Localization - Multi-Category Technical Report",
        "",
        f"Generated: {generated}",
        "",
        "Dataset: MVTec AD | Models: PatchCore (primary), PaDiM (secondary) | "
        "Normal-only training and calibration",
        "",
        "## 1. Executive summary",
        "",
        "The project scope is **final and frozen** at exactly six MVTec AD categories: "
        + ", ".join(f"`{name}`" for name in supported)
        + ". "
        + f"All **{len(evaluated)}** supported categories completed dataset validation, "
        "training, calibration and test evaluation: "
        + (", ".join(f"`{name}`" for name in evaluated) or "none")
        + ".",
        "",
        f"- Supported categories: {len(supported)} "
        f"({', '.join(f'`{name}`' for name in supported)}).",
        f"- Out of scope: {len(out_of_scope)} MVTec AD categories "
        f"({', '.join(f'`{name}`' for name in out_of_scope)}). They were not downloaded, "
        "trained, evaluated or made selectable.",
        f"- Trained but not evaluated: {', '.join(f'`{name}`' for name in trained_not_evaluated) or 'none'}.",
        f"- Dataset present but not trained: {', '.join(f'`{name}`' for name in data_not_trained) or 'none'}.",
        f"- Supported category with dataset unavailable locally: {len(unavailable)} "
        f"({', '.join(f'`{name}`' for name in unavailable) or 'none'}).",
        "",
        "The scope was frozen to keep the release reproducible and to stay within the local "
        "CPU/storage budget. This is a six-category multi-category prototype; it is **not** a "
        "complete 15-category MVTec AD benchmark. A category is reported as evaluated only when "
        "its metrics files exist in the category-scoped metrics directory.",
        "",
        "## 2. Category scope and status",
        "",
        "Only the six supported categories are inventoried. The nine out-of-scope categories "
        "are excluded from selection, training, evaluation, reports and the CLI/app; their "
        "absence is intentional and not a missing-data failure.",
        "",
    ]
    lines += _status_table(inventory)
    lines += [
        "",
        "Status is derived from the filesystem: dataset layout, nested artifacts "
        "(`models/<category>/<model>/`), calibrated thresholds and evaluated metrics. "
        "A supported category can be selected in the CLI/app only when it has both an "
        "artifact and calibrated thresholds. Out of scope: "
        + ", ".join(f"`{name}`" for name in out_of_scope)
        + ".",
        "",
        "## 3. Per-category results",
        "",
    ]
    if not evaluated:
        lines += ["No category has completed evaluation yet.", ""]
    for category in evaluated:
        bundle = bundles[category]
        lines += [
            f"### `{category}`",
            "",
            f"Detailed report: [`reports/categories/{category}.md`](categories/{category}.md)",
            "",
        ]
        lines += _sample_rows(bundle)
        lines += [""]
        lines += _results_table(bundle["metrics"])
        lines += [""]
        audit = bundle["audit"]
        if audit:
            counts = audit.get("counts", {})
            lines.append(
                f"Dataset readiness: **{'PASS' if audit['checks'].get('dataset_ready', {}).get('passed') else 'FAIL'}**; "
                f"train/good={counts.get('train_good_images', 0)}, "
                f"test/good={counts.get('test_good_images', 0)}, "
                f"test/anomalous={counts.get('test_anomalous_images', 0)}."
            )
            lines.append("")
        lines += ["Decision outcomes:", ""]
        lines += _outcome_table(bundle["frames"])
        lines += [""]

    lines += [
        "## 4. Cross-category analysis",
        "",
        "### 4.1 Aggregation methods",
        "",
        "- **Macro average**: unweighted arithmetic mean of the per-category values. Every "
        "evaluated category contributes equally, regardless of test-set size. It is labelled "
        "`MACRO_AVERAGE` in `reports/metrics/cross_category_metrics.csv`.",
        "- **Pooled confusions**: thresholded precision/recall/F1 recomputed from TP/FP/FN "
        "summed across categories. Image rows are therefore weighted by image count and "
        "pixel rows by pixel count. Labelled `POOLED` in the CSV.",
        "- **Ranking metrics (AUROC, AP, AUPRO) are not pooled** from per-category summaries. "
        "Only their unweighted macro average is reported; a pooled ranking metric requires "
        "the raw scores and is intentionally not computed here.",
        "",
        "### 4.2 Per-category metrics",
        "",
    ]
    lines += _aggregate_tables(rows)
    lines += [
        "### 4.3 Macro average (equal weight per category)",
        "",
    ]
    lines += _macro_table(aggregates)
    lines += [
        "",
        "### 4.4 Pooled thresholded metrics (weighted by sample count)",
        "",
    ]
    lines += _pooled_table(aggregates)
    lines += [
        "",
        "### 4.5 Easiest and hardest categories",
        "",
    ]
    lines += _ranking_section(rankings)
    lines += [
        "",
        "### 4.6 Model comparison",
        "",
    ]
    comparisons: list[str] = []
    for category in evaluated:
        bundle = bundles[category]
        available = [model for model in MODELS if bundle["metrics"].get(model)]
        if len(available) < 2:
            comparisons.append(
                f"- `{category}`: only {', '.join(available) or 'no model'} "
                "has metrics; no pairwise comparison."
            )
            continue
        parts = []
        for model in available:
            item = bundle["metrics"][model]
            parts.append(
                f"{model} image AUROC {_fmt(item['image_metrics']['image_auroc'])}, "
                f"AUPRO {_fmt(item['pro_metrics']['pro_auc'])}"
            )
        comparisons.append(f"- `{category}`: " + "; ".join(parts) + ".")
    if not comparisons:
        comparisons = ["- No evaluated category to compare."]
    lines += comparisons
    lines += [
        "",
        "### 4.7 Threshold differences",
        "",
    ]
    lines += _threshold_differences(bundles)
    lines += [
        "",
        "### 4.8 False positives and false negatives",
        "",
    ]
    fp_fn_found = False
    for category in evaluated:
        bundle = bundles[category]
        for model in MODELS:
            frame = bundle["frames"].get(model)
            if frame is None:
                continue
            fp_fn_found = True
            counts = frame["outcome"].value_counts()
            lines.append(
                f"- `{category}` / {model}: "
                f"{int(counts.get('false_positive', 0))} false positives, "
                f"{int(counts.get('false_negative', 0))} false negatives."
            )
    if not fp_fn_found:
        lines.append("- No per-image prediction files available.")
    lines += [
        "",
        "### 4.9 Defect appearance",
        "",
    ]
    lines += _defect_appearance(bundles)
    lines += [
        "",
        "### 4.10 Image-level versus pixel-level behaviour",
        "",
    ]
    for category in evaluated:
        bundle = bundles[category]
        for model in MODELS:
            item = bundle["metrics"].get(model)
            if item is None:
                continue
            lines.append(
                f"- `{category}` / {model}: image AUROC "
                f"{_fmt(item['image_metrics']['image_auroc'])} vs pixel AUROC "
                f"{_fmt(item['pixel_metrics']['pixel_auroc'])}; image recall "
                f"{_fmt(item['image_metrics']['image_recall'])} vs pixel recall "
                f"{_fmt(item['pixel_metrics']['pixel_recall'])} at their calibrated thresholds."
            )
    lines += [
        "",
        "## 5. Evaluation integrity disclosure",
        "",
        "- Training used only `train/good` images of the same category.",
        "- Thresholds used only held-out normal training images of the same category "
        "(`test_data_used: false`); no test label was used for fitting or calibration.",
        "- Test labels are used only for the final reported metrics and error-analysis "
        "figures, after models and thresholds were frozen.",
        "- Artifacts, thresholds, predictions, metrics and figures are category-scoped; no "
        "fitted state or threshold is reused across categories.",
        "- Descriptive EDA summarises test annotations but does not influence modelling.",
        "",
        "## 6. Limitations",
        "",
        f"- The report covers only the {len(supported)} supported categories. All results, "
        "aggregates and rankings apply to these six categories only; this is not a "
        "15-category benchmark, and the nine out-of-scope categories were intentionally "
        "excluded.",
        "- Pixel metrics use the configured evaluation resolution with pooled pixels; they "
        "are not directly comparable to full-resolution literature numbers.",
        "- AUPRO is capped at the configured false-positive limit and derives thresholds "
        "from normal-score quantiles.",
        "- Normal-only calibration cannot guarantee a good defect-type balance; operating "
        "points should follow business costs.",
        "- No robustness study for lighting, camera or fixture changes; recalibrate per site.",
        "- Models score deviation from normal; they do not classify defect types.",
        "",
        "## 7. Reproducibility",
        "",
        "```bash",
        "# Validate the six supported categories (out-of-scope categories are not touched)",
        "python scripts/01_validate_dataset.py --categories active",
        "",
        "# Per-category EDA, training, calibration and evaluation (sequential, CPU-friendly)",
        "python scripts/02_explore_dataset.py --categories active",
        "python scripts/04_train.py --categories active",
        "python scripts/05_calibrate.py --categories active",
        "python scripts/06_evaluate.py --categories active",
        "",
        "# Cross-category aggregation and reports",
        "python scripts/08_generate_reports.py",
        "",
        "# Category-aware inference",
        "python scripts/07_predict.py --list-categories",
        "python scripts/07_predict.py --category transistor --input path/to/image.png --model both",
        "streamlit run app/streamlit_app.py",
        "```",
        "",
        "Artifacts live under `models/<category>/<model>/` and outputs under "
        "`reports/{metrics,figures,predictions}/<category>/`. The six-category scope is "
        "permanently frozen for this release; no additional category is supported. See "
        "`README.md` for dataset acquisition and licensing instructions.",
        "",
        "## 8. Reference hardware and software",
        "",
    ]
    if environment:
        lines += [
            f"- Platform: {environment['platform']}",
            f"- Python {environment['python']}, torch {environment['torch']}, "
            f"torchvision {environment['torchvision']}, numpy {environment['numpy']}, "
            f"scikit-learn {environment['scikit_learn']}, scipy {environment['scipy']}",
            f"- {environment['cpu_count_logical']} logical CPUs, "
            f"{environment['ram_total_gb']} GB RAM, CUDA available: {environment['cuda_available']}",
            f"- Training device: CPU; torch threads: {environment['torch_threads']}",
        ]
    else:
        lines.append("- No training environment metadata available (no trained category).")
    lines += [
        "",
        "### 8.1 Pipeline smoke verification",
        "",
        "A short smoke fit on a small subset (`--max-images`, `--tag smoke`) exercised the "
        "train/save/reload path end to end. Smoke artifacts are never calibrated, evaluated "
        "or used by the CLI/app; they exist only to verify the pipeline wiring.",
        "",
    ]
    lines += _smoke_evidence(config)
    lines.append("")
    return "\n".join(lines)


def _smoke_evidence(config: dict) -> list[str]:
    models_dir = configured_path(config, "models_dir")
    records = []
    if models_dir.is_dir():
        for path in sorted(models_dir.glob("*/*/training.json")):
            if "smoke" not in path.parent.name:
                continue
            record = _load_optional(path)
            if record:
                records.append((path, record))
    if not records:
        return ["No smoke-training evidence found."]
    lines = [
        "| Artifact | Fit images | Fit time (s) | Reload max diff | Artifact size | Passed |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for path, record in records:
        try:
            label = path.parent.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            label = path.parent.name
        validation = record.get("artifact_validation", {})
        lines.append(
            f"| `{label}` | {record.get('fit_images', 'n/a')} | {record.get('fit_seconds', 0):.2f} | "
            f"{validation.get('max_abs_score_diff', 'n/a')} | "
            f"{human_bytes(validation.get('artifact_bytes', 0))} | "
            f"{'yes' if validation.get('passed') else 'no'} |"
        )
    return lines


def _dataset_directories() -> set[Path]:
    paths = {PROJECT_ROOT / category for category in MVTEC_CATEGORIES}
    paths |= {PROJECT_ROOT / "data" / "mvtec_ad", PROJECT_ROOT / "mvtec_ad"}
    return {path.resolve() for path in paths}


def _tracked_candidates() -> list[Path]:
    ignored = [
        "models/*.pt",
        "data/raw/*",
        "data/processed/*",
        "reports/predictions/*",
        "*.npz",
        "__pycache__/*",
        ".pytest_cache/*",
        ".ruff_cache/*",
    ]
    dataset_dirs = _dataset_directories()
    candidates = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(directory == resolved or directory in resolved.parents for directory in dataset_dirs):
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in ignored):
            continue
        if any(part in SCAN_SKIP_PARTS for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if relative.startswith("models/") and path.suffix.lower() not in {".json", ".md"}:
            continue
        if path.name == ".gitkeep":
            continue
        candidates.append(path)
    return sorted(candidates)


def _github_readiness(config: dict) -> tuple[dict, str]:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        *[f"{category}/" for category in MVTEC_CATEGORIES],
        "data/mvtec_ad/",
        "models/**",
        "*.pt",
        "reports/predictions/",
        "*.npz",
        "__pycache__/",
        ".pytest_cache/",
        ".env",
    ]
    patterns = {pattern: pattern in gitignore for pattern in required_patterns}
    candidates = _tracked_candidates()
    large_files = [
        (path.relative_to(PROJECT_ROOT).as_posix(), path.stat().st_size)
        for path in candidates
        if path.stat().st_size > 1_000_000
    ]
    large_files.sort(key=lambda item: item[1], reverse=True)
    total_bytes = sum(path.stat().st_size for path in candidates)

    absolute_pattern = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:Users|home)[\\/][^\\/\s]+")
    secret_pattern = re.compile(
        r"(api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{6,}",
        re.IGNORECASE,
    )
    absolute_hits: list[str] = []
    secret_hits: list[str] = []
    for path in candidates:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 8_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if absolute_pattern.search(text):
            absolute_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
        if path.suffix.lower() != ".ipynb" and secret_pattern.search(text):
            secret_hits.append(path.relative_to(PROJECT_ROOT).as_posix())

    dataset_present = [
        category
        for category in MVTEC_CATEGORIES
        if (PROJECT_ROOT / category / "train").is_dir()
        and (PROJECT_ROOT / category / "test").is_dir()
    ]
    dataset_candidates = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in candidates
        if path.relative_to(PROJECT_ROOT).parts[0] in set(MVTEC_CATEGORIES) | {"data"}
        and path.name != "README.md"
    ]
    binary_candidates = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in candidates
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".npz"}
    ]
    raw_files = (
        [path.name for path in (PROJECT_ROOT / "data" / "raw").glob("*") if path.is_file()]
        if (PROJECT_ROOT / "data" / "raw").is_dir()
        else []
    )
    artifacts = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "models").rglob("*")
        if path.is_file() and path.suffix in {".pt", ".pth"}
    )
    readiness = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_repository_initialized": (PROJECT_ROOT / ".git").is_dir(),
        "ignore_patterns": patterns,
        "candidate_files": len(candidates),
        "candidate_total_bytes": total_bytes,
        "large_files": [
            {"path": path, "bytes": size, "human": human_bytes(size)}
            for path, size in large_files
        ],
        "absolute_path_hits": absolute_hits,
        "secret_pattern_hits": secret_hits,
        "dataset_categories_present": dataset_present,
        "data_raw_files": raw_files,
        "trained_binaries_present_but_ignored": artifacts,
        "dataset_candidates": dataset_candidates,
        "binary_candidates": binary_candidates,
        "checks": {
            "all_category_dirs_ignored": all(
                patterns.get(f"{category}/", False) for category in MVTEC_CATEGORIES
            ),
            "dataset_root_ignored": patterns.get("data/mvtec_ad/", False),
            "binaries_ignored": patterns.get("models/**", False) and patterns.get("*.pt", False),
            "predictions_ignored": patterns.get("reports/predictions/", False)
            and patterns.get("*.npz", False),
            "no_absolute_user_paths": not absolute_hits,
            "no_secret_like_values": not secret_hits,
            "no_file_over_10mb": all(size < 10_000_000 for _, size in large_files),
            "dataset_files_excluded": not dataset_candidates,
            "binaries_excluded": not binary_candidates,
        },
    }
    lines = [
        "# GitHub readiness audit",
        "",
        f"Generated: {readiness['generated_at']}",
        "",
        "## Repository status",
        "",
        f"- Git repository initialized: **{readiness['git_repository_initialized']}**. "
        "The project is prepared for a first commit; no commit was created by this audit.",
        f"- Files that would be tracked: {readiness['candidate_files']} "
        f"({human_bytes(total_bytes)}).",
        f"- Dataset categories present locally: "
        f"{', '.join(dataset_present) or 'none'}; dataset files are excluded by ignore rules.",
        f"- Trained binaries present locally: {len(artifacts)} "
        "(ignored; regenerate with `scripts/04_train.py`).",
        "",
        "## Ignore rules",
        "",
        "| Pattern | Present in `.gitignore` |",
        "| --- | --- |",
    ]
    for pattern, present in patterns.items():
        lines.append(f"| `{pattern}` | {'yes' if present else 'no'} |")
    lines += [
        "",
        "## Large files",
        "",
        "| File | Size |",
        "| --- | ---: |",
    ]
    for item in readiness["large_files"][:20]:
        lines.append(f"| `{item['path']}` | {item['human']} |")
    if not readiness["large_files"]:
        lines.append("| none | - |")
    lines += [
        "",
        "## Sensitivity scan",
        "",
        f"- Absolute user/home paths in text files: {len(absolute_hits)} "
        + (f"({', '.join(absolute_hits[:5])})" if absolute_hits else ""),
        f"- Secret-like assignments: {len(secret_hits)} "
        + (f"({', '.join(secret_hits[:5])})" if secret_hits else ""),
        f"- Data raw files: {', '.join(raw_files) if raw_files else 'none'}",
        "",
        "## Checklist",
        "",
    ]
    for name, passed in readiness["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}")
    lines += [
        "",
        "Notebooks can be several MB each because executed figures are embedded; they remain "
        "below the 10 MB check threshold. Full anomaly maps and per-image predictions are "
        "git-ignored and produced by `scripts/06_evaluate.py`.",
        "",
    ]
    return readiness, "\n".join(lines)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    supported = supported_categories(config)
    inventory = category_inventory(config, supported)
    bundles = {category: _bundle(config, category) for category in supported}
    for payload in bundles.values():
        payload["config"] = config

    categories_dir = configured_path(config, "reports_dir") / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for category in supported:
        bundle = bundles[category]
        if not bundle["evaluated"]:
            continue
        path = categories_dir / f"{category}.md"
        path.write_text(_render_category_report(bundle), encoding="utf-8")
        written.append(path)

    rows = collect_rows(config)
    write_cross_category_metrics(config)
    aggregates = aggregate_rows(rows)
    rankings = {
        "image_auroc": category_ranking(rows, "image_auroc"),
        "pro_auc": category_ranking(rows, "pro_auc"),
    }
    report = _render_final_report(config, inventory, bundles, rows, aggregates, rankings)
    report_path = configured_path(config, "reports_dir") / "final_report.md"
    report_path.write_text(report, encoding="utf-8")

    readiness, readiness_markdown = _github_readiness(config)
    save_json(configured_path(config, "metrics_dir") / "github_readiness.json", readiness)
    readiness_path = configured_path(config, "reports_dir") / "github_readiness.md"
    readiness_path.write_text(readiness_markdown, encoding="utf-8")

    evaluated = [name for name in supported if bundles[name]["evaluated"]]
    print(f"[reports] evaluated categories: {', '.join(evaluated) or 'none'}")
    for path in written:
        print(f"[reports] category report : {path}")
    print(f"[reports] final report    : {report_path}")
    print(f"[reports] readiness       : {readiness_path}")
    print(
        f"[reports] candidate files: {readiness['candidate_files']} "
        f"({human_bytes(readiness['candidate_total_bytes'])}), "
        f"absolute-path hits={len(readiness['absolute_path_hits'])}, "
        f"secret-like hits={len(readiness['secret_pattern_hits'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
