"""Phase 3: descriptive EDA over the selected MVTec category.

Everything here is descriptive. No split, threshold or model decision is derived
from the test annotations; the test split is only summarised so that readers can
understand the evaluation data and its limitations.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from industrial_defect.categories import select_categories  # noqa: E402
from industrial_defect.config import (  # noqa: E402
    category_dir,
    configured_path,
    ensure_output_dirs,
    load_config,
)
from industrial_defect.dataset import (  # noqa: E402
    DatasetLayoutError,
    discover_samples,
    list_defect_types,
)
from industrial_defect.preprocessing import load_mask_binary  # noqa: E402
from industrial_defect.utils import save_json  # noqa: E402
from industrial_defect.visualization import save_dataset_grid  # noqa: E402

SPLIT_COLORS = {"train/good": "#2c7fb8", "test/good": "#7fcdbb", "test/anomalous": "#de2d26"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--categories", type=str, default=None, help="Comma list, 'active' or 'all'")
    parser.add_argument("--grid-samples", type=int, default=8)
    parser.add_argument("--intensity-stride", type=int, default=10)
    return parser.parse_args()


def _image_properties(paths: list[Path]) -> tuple[Counter, Counter, Counter]:
    sizes: Counter = Counter()
    modes: Counter = Counter()
    formats: Counter = Counter()
    for path in paths:
        with Image.open(path) as image:
            sizes[f"{image.width}x{image.height}"] += 1
            modes[image.mode] += 1
            formats[image.format or "unknown"] += 1
    return sizes, modes, formats


def _intensity_stats(paths: list[Path]) -> dict[str, float]:
    means: list[float] = []
    stds: list[float] = []
    saturations: list[float] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        grey = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        saturation = np.divide(minimum, maximum, out=np.zeros_like(minimum), where=maximum > 0)
        means.append(float(grey.mean()))
        stds.append(float(grey.std()))
        saturations.append(float(saturation.mean()))
    return {
        "images": len(means),
        "grey_mean": float(np.mean(means)),
        "grey_std_mean": float(np.mean(stds)),
        "grey_std_min": float(np.min(stds)),
        "grey_std_max": float(np.max(stds)),
        "saturation_mean": float(np.mean(saturations)),
        "per_image": {
            "grey_mean": means,
            "grey_std": stds,
            "saturation": saturations,
        },
    }


def _mask_stats(mask_paths: list[Path]) -> dict[str, dict]:
    records: dict[str, list[dict]] = {}
    for mask_path in mask_paths:
        defect = mask_path.parent.name
        mask = load_mask_binary(mask_path).numpy()[0] > 0.5
        height, width = mask.shape
        components, _ = ndimage.label(mask)
        component_count = int(components.max()) if components.size else 0
        sizes = np.bincount(components.ravel())[1:] if component_count > 0 else np.array([])
        boxes = ndimage.find_objects(components.astype(np.int32))
        box_fractions = [
            ((box[0].stop - box[0].start) / height) * ((box[1].stop - box[1].start) / width)
            for box in boxes
            if box is not None
        ]
        records.setdefault(defect, []).append(
            {
                "area_fraction": float(mask.mean()),
                "components": component_count,
                "largest_area_fraction": float(sizes.max() / mask.size) if sizes.size else 0.0,
                "largest_bbox_fraction": float(max(box_fractions)) if box_fractions else 0.0,
            }
        )
    summary: dict[str, dict] = {}
    for defect, items in sorted(records.items()):
        areas = np.array([item["area_fraction"] for item in items])
        components = np.array([item["components"] for item in items])
        largest = np.array([item["largest_area_fraction"] for item in items])
        summary[defect] = {
            "masks": len(items),
            "area_fraction": {
                "min": float(areas.min()),
                "median": float(np.median(areas)),
                "mean": float(areas.mean()),
                "max": float(areas.max()),
            },
            "components": {
                "min": int(components.min()),
                "median": float(np.median(components)),
                "max": int(components.max()),
            },
            "largest_component_area_fraction": {
                "median": float(np.median(largest)),
                "max": float(largest.max()),
            },
            "items": items,
        }
    return summary


def _save_dataset_overview(counts: dict[str, int], path: Path, category: str) -> None:
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = [SPLIT_COLORS.get(label, "#9467bd") for label in labels]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, padding=2, fontsize=9)
    axis.set_ylabel("Images")
    axis.set_title(f"{category} - sample counts by split and defect type")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def _save_area_distribution(mask_summary: dict[str, dict], path: Path, category: str) -> None:
    defects = list(mask_summary)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    areas = [[item["area_fraction"] * 100 for item in mask_summary[d]["items"]] for d in defects]
    components = [[item["components"] for item in mask_summary[d]["items"]] for d in defects]
    axes[0].boxplot(areas, tick_labels=defects, showmeans=True)
    axes[0].set_ylabel("Defect area (% of image)")
    axes[0].set_title("Ground-truth defect area per type")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].boxplot(components, tick_labels=defects, showmeans=True)
    axes[1].set_ylabel("Connected mask components")
    axes[1].set_title("Defect fragmentation per type")
    axes[1].tick_params(axis="x", rotation=15)
    figure.suptitle(f"{category} - anomalous test masks (n={sum(len(v['items']) for v in mask_summary.values())})")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def _save_intensity_scatter(
    normal_stats: dict, anomalous_stats: dict, path: Path, category: str
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(
        normal_stats["per_image"]["grey_std"],
        [value * 100 for value in normal_stats["per_image"]["grey_mean"]],
        s=28,
        alpha=0.75,
        color="#2c7fb8",
        label=f"normal ({normal_stats['images']} sampled)",
    )
    axis.scatter(
        anomalous_stats["per_image"]["grey_std"],
        [value * 100 for value in anomalous_stats["per_image"]["grey_mean"]],
        s=28,
        alpha=0.75,
        color="#de2d26",
        marker="^",
        label=f"anomalous ({anomalous_stats['images']} sampled)",
    )
    axis.set_xlabel("Mean intra-image grey standard deviation")
    axis.set_ylabel("Mean grey level (%)")
    axis.set_title(f"{category} - appearance summary (descriptive only)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def render_markdown(summary: dict) -> str:
    lines = [
        f"# EDA summary - {summary['category']}",
        "",
        "Descriptive analysis of the dataset layout and anomalous test annotations. "
        "No modelling, split or threshold decision was derived from this analysis; "
        "the test split is summarised only to characterise the evaluation set.",
        "",
        "## Counts",
        "",
        "| Split | Images |",
        "| --- | ---: |",
    ]
    for label, value in summary["counts"].items():
        lines.append(f"| {label} | {value} |")
    lines += ["", "## Image properties", ""]
    for name, values in summary["image_properties"].items():
        rendered = ", ".join(f"{key}: {value}" for key, value in values.items())
        lines.append(f"- **{name}**: {rendered}")
    lines += ["", "## Appearance statistics", ""]
    for name, stats in summary["appearance"].items():
        lines.append(
            f"- **{name}** ({stats['images']} sampled): grey mean {stats['grey_mean']:.3f}, "
            f"intra-image grey std mean {stats['grey_std_mean']:.3f}, "
            f"mean saturation {stats['saturation_mean']:.3f}"
        )
    lines += [
        "",
        "## Ground-truth defect masks",
        "",
        "| Defect type | Masks | Area fraction (min / median / max) | Components (min / median / max) |",
        "| --- | ---: | --- | --- |",
    ]
    for defect, stats in summary["defect_masks"].items():
        area = stats["area_fraction"]
        comp = stats["components"]
        lines.append(
            f"| {defect} | {stats['masks']} | {area['min']:.4f} / {area['median']:.4f} / "
            f"{area['max']:.4f} | {comp['min']} / {comp['median']:.1f} / {comp['max']} |"
        )
    lines += [
        "",
        "## Observations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["observations"])
    lines += ["", "## Generated figures", ""]
    lines.extend(f"- `{path}`" for path in summary["figures"].values())
    lines += [
        "",
        "## Notes",
        "",
        "- Masks exist only for anomalous test images; normal images have no mask.",
        "- Box-plot whiskers are matplotlib defaults (1.5 x IQR); the underlying per-image "
        "values are recorded in the JSON summary.",
        "- The appearance scatter plots one point per sampled image (deterministic stride); "
        "it documents that normal and anomalous images share the same acquisition setup.",
        "",
    ]
    return "\n".join(lines)


def _run_category(args: argparse.Namespace, config: dict) -> None:
    if args.category:
        config["data"]["category"] = args.category
    category = str(config["data"]["category"])
    ensure_output_dirs(config)
    cat_dir = category_dir(config)
    figures_dir = configured_path(config, "figures_dir") / category / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_samples = discover_samples(cat_dir, "train")
    test_samples = discover_samples(cat_dir, "test")
    anomalous = [sample for sample in test_samples if sample.label == 1]
    normal_test = [sample for sample in test_samples if sample.label == 0]
    masks = [sample.mask_path for sample in anomalous if sample.mask_path is not None]

    counts: dict[str, int] = {"train/good": len(train_samples)}
    defect_counts: Counter = Counter(sample.defect_type for sample in anomalous)
    counts["test/good"] = len(normal_test)
    for defect in list_defect_types(cat_dir):
        if defect != "good":
            counts[f"test/{defect}"] = int(defect_counts.get(defect, 0))
    counts["test/anomalous"] = len(anomalous)

    train_sizes, train_modes, train_formats = _image_properties(
        [sample.path for sample in train_samples]
    )
    test_sizes, test_modes, test_formats = _image_properties(
        [sample.path for sample in test_samples]
    )
    image_properties = {
        "train image sizes": dict(train_sizes),
        "test image sizes": dict(test_sizes),
        "train colour modes": dict(train_modes),
        "test colour modes": dict(test_modes),
        "train formats": dict(train_formats),
        "test formats": dict(test_formats),
    }

    stride = max(1, args.intensity_stride)
    normal_sample = [sample.path for sample in train_samples[::stride]]
    anomalous_sample = [sample.path for sample in anomalous[:: max(1, args.intensity_stride // 5)]]
    appearance = {
        "normal train (sampled)": _intensity_stats(normal_sample),
        "anomalous test (sampled)": _intensity_stats(anomalous_sample),
    }
    mask_summary = _mask_stats(masks)

    grid = max(1, min(args.grid_samples, 16))
    normal_items = [
        {
            "image_path": sample.path,
            "title": sample.relative_path,
        }
        for sample in train_samples[:: max(1, len(train_samples) // grid)][:grid]
    ]
    save_dataset_grid(
        normal_items,
        figures_dir / "normal_training_samples.png",
        f"{category} - normal training samples (every {max(1, len(train_samples) // grid)}th image)",
        columns=4,
    )

    defect_items: list[dict] = []
    for defect in sorted(defect_counts):
        subset = [sample for sample in anomalous if sample.defect_type == defect]
        take = max(1, grid // max(1, len(defect_counts)))
        for sample in subset[:take]:
            defect_items.append(
                {
                    "image_path": sample.path,
                    "title": f"{sample.defect_type}\n{sample.relative_path}",
                    "mask_path": sample.mask_path,
                }
            )
    save_dataset_grid(
        defect_items,
        figures_dir / "anomalous_test_samples.png",
        f"{category} - anomalous test samples with ground-truth contours",
        columns=4,
        show_masks=True,
    )
    _save_dataset_overview(counts, figures_dir / "split_and_defect_counts.png", category)
    _save_area_distribution(
        mask_summary, figures_dir / "defect_area_distribution.png", category
    )
    _save_intensity_scatter(
        appearance["normal train (sampled)"],
        appearance["anomalous test (sampled)"],
        figures_dir / "appearance_scatter.png",
        category,
    )
    plt.close("all")

    areas = [
        item["area_fraction"] for stats in mask_summary.values() for item in stats["items"]
    ]
    observations = [
        f"Training data contains {len(train_samples)} defect-free images; the test split has "
        f"{len(normal_test)} normal and {len(anomalous)} anomalous images with {len(masks)} masks.",
        "Ground-truth defects cover "
        f"{min(areas) * 100:.2f}% to {max(areas) * 100:.2f}% of an image "
        f"(median {np.median(areas) * 100:.2f}%), so image-level detection and pixel-level "
        "localisation are genuinely different problems on this category.",
        "Defect types: "
        + ", ".join(f"{name} ({count})" for name, count in sorted(defect_counts.items()))
        + ".",
        "All images share one acquisition setup, but illumination and viewpoint vary between "
        "captures.",
    ]
    medians = {
        defect: float(stats["area_fraction"]["median"])
        for defect, stats in mask_summary.items()
    }
    if len(medians) > 1:
        largest = max(medians, key=medians.__getitem__)
        others = [value for defect, value in medians.items() if defect != largest]
        observations.append(
            f"Defect type {largest} covers a median {medians[largest] * 100:.1f}% of the image, "
            f"while the other types cover medians of {min(others) * 100:.1f}-"
            f"{max(others) * 100:.1f}%; pooled pixel metrics are therefore dominated by the large "
            f"{largest} masks."
        )
    summary = {
        "category": category,
        "dataset_root": str(cat_dir.relative_to(PROJECT_ROOT)),
        "counts": counts,
        "per_defect_type": dict(sorted(defect_counts.items())),
        "masks": len(masks),
        "image_properties": image_properties,
        "appearance": appearance,
        "defect_masks": mask_summary,
        "observations": observations,
        "notes": [
            "Descriptive EDA only; no test annotation was used for training, calibration or "
            "model selection.",
            "Normal images have no ground-truth masks.",
            "The test split appears here for dataset characterisation and error-analysis context.",
        ],
        "figures": {
            name: f"{(figures_dir / name).relative_to(PROJECT_ROOT).as_posix()}"
            for name in (
                "normal_training_samples.png",
                "anomalous_test_samples.png",
                "split_and_defect_counts.png",
                "defect_area_distribution.png",
                "appearance_scatter.png",
            )
        },
    }
    metrics_dir = configured_path(config, "metrics_dir") / category
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_dir / "eda_summary.json", summary)
    (metrics_dir / "eda_summary.md").write_text(render_markdown(summary), encoding="utf-8")

    print(f"[eda] category={category} train={len(train_samples)} test={len(test_samples)}")
    for defect, count in sorted(defect_counts.items()):
        print(f"[eda] test/{defect}: {count}")
    print(f"[eda] masks={len(masks)} figures={figures_dir}")
    print(f"[eda] metrics={metrics_dir / 'eda_summary.json'}")


def main() -> int:
    args = parse_args()
    if args.category and args.categories:
        raise SystemExit("Use either --category or --categories, not both")
    config = load_config(args.config)
    categories = select_categories(args.categories or args.category, config)
    status = 0
    for category in categories:
        args.category = category
        try:
            _run_category(args, config)
        except DatasetLayoutError as exc:
            print(f"[eda] {category}: dataset unavailable -> {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
