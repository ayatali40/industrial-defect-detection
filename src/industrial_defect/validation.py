"""Dataset validation and audit for the MVTec AD layout."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PROJECT_ROOT
from .dataset import (
    DatasetLayoutError,
    DatasetNotFoundError,
    find_categories,
    resolve_category_dir,
)
from .preprocessing import is_image_file


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            return {
                "ok": True,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _read_mask(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image.convert("L"), dtype=np.uint8)
            unique = set(np.unique(array).tolist())
            positive = int((array > 127).sum())
            return {
                "ok": True,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "binary": unique.issubset({0, 1, 255}),
                "unique_values": sorted(unique)[:10],
                "positive_fraction": positive / float(array.size),
            }
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _issue(issues: list[dict[str, str]], severity: str, check: str, message: str) -> None:
    issues.append({"severity": severity, "check": check, "message": message})


def _extract_provenance(category_dir: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license": "Unknown (dataset readme/license files not found)",
        "citation": None,
        "files_present": [],
    }
    readme = category_dir / "readme.txt"
    license_file = category_dir / "license.txt"
    present = [path.name for path in (readme, license_file) if path.is_file()]
    provenance["files_present"] = present
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        citation = re.search(r"please cite:\s*(.+?)(?:\n\n|\nLicense)", text, re.S)
        if citation:
            provenance["citation"] = " ".join(citation.group(1).split())
        if "Attribution-NonCommercial-ShareAlike 4.0" in text:
            provenance["license"] = "CC BY-NC-SA 4.0"
    return provenance


def audit_dataset(
    data_root: str | Path, category: str, category_path: str | Path | None = None
) -> dict[str, Any]:
    """Inspect the real dataset and return an evidence-based audit dictionary.

    ``category_path`` allows callers to pass an explicitly resolved category
    directory (for example a ``data.category_roots`` override) while keeping
    ``data_root`` for display.
    """
    root = Path(data_root)
    issues: list[dict[str, str]] = []
    checks: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "root": _display(root),
            "root_exists": root.exists(),
            "category": category,
            "category_dir": None,
            "categories_available": [],
        },
        "counts": {},
        "defect_types": {},
        "image_summary": {"dimensions": {}, "formats": {}, "modes": {}, "masks": {}},
        "mask_alignment": {
            "anomalous_test_images": 0,
            "masks_present": 0,
            "aligned": 0,
            "missing_masks": [],
            "dimension_mismatches": [],
            "empty_masks": [],
            "non_binary_masks": [],
            "orphan_masks": [],
            "mask_positive_fraction": {},
        },
        "duplicates": {"duplicate_paths": [], "duplicate_content_groups": []},
        "checks": checks,
        "issues": issues,
        "provenance": {},
    }

    checks["dataset_root_exists"] = {"passed": root.exists(), "detail": _display(root)}
    if not root.exists():
        _issue(issues, "critical", "dataset_root_exists", f"Dataset root missing: {_display(root)}")
        return audit

    categories = find_categories(root) if root.name != category or (root / category).is_dir() else []
    audit["dataset"]["categories_available"] = categories
    checks["category_layout"] = {"passed": False, "detail": f"requested '{category}'"}
    try:
        category_dir = (
            Path(category_path)
            if category_path is not None
            else resolve_category_dir(root, category)
        )
    except (DatasetNotFoundError, DatasetLayoutError) as exc:
        _issue(issues, "critical", "category_layout", str(exc))
        return audit
    audit["dataset"]["category_dir"] = _display(category_dir)
    checks["category_layout"] = {"passed": True, "detail": _display(category_dir)}
    audit["provenance"] = _extract_provenance(category_dir)

    train_dir = category_dir / "train"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"
    checks["required_directories"] = {
        "passed": train_dir.is_dir() and test_dir.is_dir(),
        "detail": {"train": train_dir.is_dir(), "test": test_dir.is_dir()},
    }
    if not train_dir.is_dir() or not test_dir.is_dir():
        _issue(issues, "critical", "required_directories", "train/ and test/ must both exist")
        return audit

    # --- Normal-only training layout -------------------------------------------------
    train_classes = sorted(child.name for child in train_dir.iterdir() if child.is_dir())
    stray_train_files = [
        path.name for path in train_dir.iterdir() if path.is_file() and is_image_file(path)
    ]
    normal_training_ok = train_classes == ["good"] and not stray_train_files
    checks["normal_training_labels"] = {
        "passed": normal_training_ok,
        "detail": {"training_class_dirs": train_classes, "loose_image_files": stray_train_files},
    }
    if not normal_training_ok:
        _issue(
            issues,
            "critical",
            "normal_training_labels",
            f"Training split must contain only 'good' images; found {train_classes} "
            f"and {len(stray_train_files)} loose image files",
        )

    test_classes = sorted(child.name for child in test_dir.iterdir() if child.is_dir())
    if "good" not in test_classes:
        _issue(issues, "critical", "test_good_split", "test/good directory is missing")
    checks["test_good_split"] = {"passed": "good" in test_classes, "detail": test_classes}

    # --- Inventory -------------------------------------------------------------------
    train_images = sorted(
        path for path in train_dir.rglob("*") if path.is_file() and is_image_file(path)
    )
    test_images = sorted(
        path for path in test_dir.rglob("*") if path.is_file() and is_image_file(path)
    )
    mask_files = (
        sorted(path for path in gt_dir.rglob("*") if path.is_file() and is_image_file(path))
        if gt_dir.is_dir()
        else []
    )
    if not gt_dir.is_dir():
        _issue(issues, "critical", "ground_truth_dir", "ground_truth/ directory is missing")

    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    unreadable_images: list[str] = []
    unreadable_train = 0
    unreadable_test = 0
    train_set = set(train_images)
    for path in train_images + test_images:
        info = _read_image(path)
        if not info["ok"]:
            unreadable_images.append(f"{_display(path)}: {info['error']}")
            if path in train_set:
                unreadable_train += 1
            else:
                unreadable_test += 1
            continue
        dimensions[f"{info['width']}x{info['height']}"] += 1
        formats[str(info["format"])] += 1
        modes[str(info["mode"])] += 1
    checks["train_images_readable"] = {
        "passed": unreadable_train == 0,
        "detail": f"{len(train_images) - unreadable_train}/{len(train_images)} readable",
    }
    checks["test_images_readable"] = {
        "passed": unreadable_test == 0,
        "detail": f"{len(test_images) - unreadable_test}/{len(test_images)} readable",
    }
    for failed in unreadable_images:
        _issue(issues, "critical", "image_readability", f"Unreadable image: {failed}")

    normal_train = [path for path in train_images if path.parent.name == "good"]
    normal_test = [path for path in test_images if path.parent.name == "good"]
    anomalous_test = [path for path in test_images if path.parent.name != "good"]

    defect_counts: dict[str, dict[str, Any]] = {}
    for defect in sorted({path.parent.name for path in anomalous_test}):
        defect_images = [path for path in anomalous_test if path.parent.name == defect]
        defect_counts[defect] = {
            "test_images": len(defect_images),
            "masks": 0,
            "mask_area_fraction": [],
        }
    audit["defect_types"] = defect_counts

    # --- Masks: presence, pairing, alignment, validity -------------------------------
    mask_info_by_path: dict[Path, dict[str, Any]] = {}
    mask_dims: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    for mask_path in mask_files:
        info = _read_mask(mask_path)
        mask_info_by_path[mask_path] = info
        if not info["ok"]:
            _issue(issues, "critical", "mask_readability", f"Unreadable mask {_display(mask_path)}")
            continue
        mask_dims[f"{info['width']}x{info['height']}"] += 1
        mask_modes[str(info["mode"])] += 1
        defect = mask_path.parent.name
        if defect not in defect_counts:
            defect_counts[defect] = {"test_images": 0, "masks": 0, "mask_area_fraction": []}
        defect_counts[defect]["masks"] += 1
        defect_counts[defect]["mask_area_fraction"].append(info["positive_fraction"])
        paired = [
            path
            for path in anomalous_test
            if path.parent.name == defect and path.stem == mask_path.stem.removesuffix("_mask")
        ]
        if not paired:
            audit["mask_alignment"]["orphan_masks"].append(_display(mask_path))
            _issue(
                issues,
                "warning",
                "mask_pairing",
                f"Mask has no matching test image: {_display(mask_path)}",
            )
        if not info["binary"]:
            audit["mask_alignment"]["non_binary_masks"].append(_display(mask_path))
            _issue(
                issues,
                "critical",
                "mask_binary",
                f"Mask contains non-binary values {info['unique_values']}: {_display(mask_path)}",
            )
        if info["positive_fraction"] <= 0:
            audit["mask_alignment"]["empty_masks"].append(_display(mask_path))
            _issue(
                issues,
                "critical",
                "mask_nonempty",
                f"Mask has no positive anomaly pixels: {_display(mask_path)}",
            )
        if paired:
            image_info = _read_image(paired[0])
            if image_info["ok"] and (
                (image_info["width"], image_info["height"]) != (info["width"], info["height"])
            ):
                audit["mask_alignment"]["dimension_mismatches"].append(
                    {
                        "image": _display(paired[0]),
                        "mask": _display(mask_path),
                        "image_size": [image_info["width"], image_info["height"]],
                        "mask_size": [info["width"], info["height"]],
                    }
                )
                _issue(
                    issues,
                    "critical",
                    "mask_alignment",
                    f"Mask dimensions differ from image: {_display(mask_path)}",
                )

    anomalous_with_mask = 0
    for image_path in anomalous_test:
        expected = gt_dir / image_path.parent.name / f"{image_path.stem}_mask{image_path.suffix}"
        if not expected.is_file():
            audit["mask_alignment"]["missing_masks"].append(_display(image_path))
            _issue(
                issues,
                "critical",
                "mask_coverage",
                f"Missing ground-truth mask for anomalous image: {_display(image_path)}",
            )
        else:
            anomalous_with_mask += 1
    audit["mask_alignment"]["anomalous_test_images"] = len(anomalous_test)
    audit["mask_alignment"]["masks_present"] = len(mask_files)
    audit["mask_alignment"]["aligned"] = anomalous_with_mask
    checks["masks_cover_anomalies"] = {
        "passed": anomalous_with_mask == len(anomalous_test) and len(anomalous_test) > 0,
        "detail": f"{anomalous_with_mask}/{len(anomalous_test)} anomalous images have masks",
    }
    checks["mask_dimensions_match"] = {
        "passed": not audit["mask_alignment"]["dimension_mismatches"],
        "detail": f"{len(audit['mask_alignment']['dimension_mismatches'])} mismatches",
    }
    checks["masks_binary_and_nonempty"] = {
        "passed": not audit["mask_alignment"]["non_binary_masks"]
        and not audit["mask_alignment"]["empty_masks"],
        "detail": f"{len(audit['mask_alignment']['non_binary_masks'])} non-binary, "
        f"{len(audit['mask_alignment']['empty_masks'])} empty",
    }

    for defect, stats in defect_counts.items():
        fractions = stats.pop("mask_area_fraction")
        defect_counts[defect]["mean_mask_area_fraction"] = (
            float(np.mean(fractions)) if fractions else None
        )

    # --- Duplicate paths and contents -------------------------------------------------
    relative_keys = [
        path.relative_to(category_dir).as_posix().casefold()
        for path in train_images + test_images + mask_files
    ]
    duplicate_paths = sorted(
        {key for key, count in Counter(relative_keys).items() if count > 1}
    )
    audit["duplicates"]["duplicate_paths"] = duplicate_paths
    for duplicate in duplicate_paths:
        _issue(issues, "critical", "duplicate_paths", f"Duplicate path: {duplicate}")
    checks["no_duplicate_paths"] = {
        "passed": not duplicate_paths,
        "detail": f"{len(duplicate_paths)} duplicated relative paths",
    }

    hashes: dict[str, list[str]] = defaultdict(list)
    for path in train_images + test_images:
        hashes[_sha256(path)].append(_display(path))
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    audit["duplicates"]["duplicate_content_groups"] = duplicate_groups
    if duplicate_groups:
        _issue(
            issues,
            "warning",
            "duplicate_content",
            f"{len(duplicate_groups)} group(s) of byte-identical images",
        )

    audit["counts"] = {
        "categories": len(categories) if categories else 1,
        "defect_types": len([name for name in defect_counts if name != "good"]),
        "train_good_images": len(normal_train),
        "test_good_images": len(normal_test),
        "test_anomalous_images": len(anomalous_test),
        "test_total_images": len(test_images),
        "ground_truth_masks": len(mask_files),
        "anomalous_with_mask": anomalous_with_mask,
    }
    audit["image_summary"] = {
        "dimensions": dict(dimensions.most_common()),
        "formats": dict(formats.most_common()),
        "modes": dict(modes.most_common()),
        "masks": {
            "dimensions": dict(mask_dims.most_common()),
            "modes": dict(mask_modes.most_common()),
        },
    }
    audit["checks"]["dataset_ready"] = {
        "passed": not any(issue["severity"] == "critical" for issue in issues),
        "detail": f"{sum(issue['severity'] == 'critical' for issue in issues)} critical issue(s)",
    }
    return audit


def summarise_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Compact per-category status for combined multi-category audits."""
    checks = audit.get("checks", {})
    critical = [issue for issue in audit.get("issues", []) if issue["severity"] == "critical"]
    warnings = [issue for issue in audit.get("issues", []) if issue["severity"] == "warning"]
    return {
        "category": audit["dataset"]["category"],
        "dataset_root": audit["dataset"]["root"],
        "category_dir": audit["dataset"].get("category_dir"),
        "data_available": bool(checks.get("category_layout", {}).get("passed", False)),
        "ready": bool(checks.get("dataset_ready", {}).get("passed", False)),
        "counts": audit.get("counts", {}),
        "defect_types": sorted(audit.get("defect_types", {})),
        "critical_issues": [issue["message"] for issue in critical],
        "warnings": [issue["message"] for issue in warnings],
    }


def render_combined_audit_markdown(combined: dict[str, Any]) -> str:
    lines = [
        "# Multi-category dataset audit - MVTec AD",
        "",
        f"- Generated: `{combined['generated_at']}`",
        f"- Dataset root: `{combined['dataset_root']}`",
        f"- Categories requested: {len(combined['requested'])}",
        f"- Ready: {', '.join(combined['ready_categories']) or 'none'}",
        f"- Missing / failed: {', '.join(combined['missing_categories']) or 'none'}",
        "",
        "| Category | Data present | Audit ready | Train good | Test good | Test anomalous | Masks | Defects |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for category, summary in combined["per_category"].items():
        counts = summary["counts"]
        lines.append(
            f"| {category} | {'yes' if summary['data_available'] else 'no'} "
            f"| {'PASS' if summary['ready'] else 'FAIL'} "
            f"| {counts.get('train_good_images', 0)} | {counts.get('test_good_images', 0)} "
            f"| {counts.get('test_anomalous_images', 0)} | {counts.get('ground_truth_masks', 0)} "
            f"| {', '.join(summary['defect_types']) or '-'} |"
        )
    lines += ["", "## Per-category findings", ""]
    for category, summary in combined["per_category"].items():
        if not summary["critical_issues"]:
            lines.append(f"- **{category}**: no critical issues.")
            continue
        for message in summary["critical_issues"]:
            lines.append(f"- **{category}**: {message}")
    lines += [
        "",
        "Full per-category audit details are stored under `reports/metrics/<category>/dataset_audit.md`.",
        "Missing categories are reported, never simulated: no synthetic dataset samples are generated.",
        "",
    ]
    return "\n".join(lines)


def render_audit_markdown(audit: dict[str, Any]) -> str:
    dataset = audit["dataset"]
    counts = audit["counts"]
    lines = [
        "# Dataset Audit - MVTec AD",
        "",
        f"- Generated: `{audit['generated_at']}`",
        f"- Dataset root: `{dataset['root']}`",
        f"- Category: **{dataset['category']}** (`{dataset.get('category_dir')}`)",
        f"- Categories available in root: {', '.join(dataset['categories_available']) or 'n/a'}",
        f"- License: {audit['provenance'].get('license', 'unknown')}",
        f"- Source: {audit['provenance'].get('source', 'unknown')}",
        "",
        "## Counts (measured)",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
    ]
    for key, value in counts.items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines += [
        "",
        "## Defect types",
        "",
        "| Defect type | Test images | Masks | Mean mask area |",
        "| --- | ---: | ---: | ---: |",
    ]
    for defect, stats in audit["defect_types"].items():
        area = stats.get("mean_mask_area_fraction")
        area_text = f"{area:.4%}" if area is not None else "n/a"
        lines.append(f"| {defect} | {stats['test_images']} | {stats['masks']} | {area_text} |")
    summary = audit["image_summary"]
    masks_summary = summary.get("masks", {})
    lines += [
        "",
        "## Image properties",
        "",
        f"- Dimensions: {summary.get('dimensions', {})}",
        f"- Formats: {summary.get('formats', {})}",
        f"- Colour modes: {summary.get('modes', {})}",
        f"- Mask dimensions: {masks_summary.get('dimensions', {})}",
        f"- Mask modes: {masks_summary.get('modes', {})}",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for name, check in audit["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {name} | {status} | {check['detail']} |")
    lines += ["", "## Issues", ""]
    if not audit["issues"]:
        lines.append("No issues were found.")
    else:
        lines += ["| Severity | Check | Message |", "| --- | --- | --- |"]
        for issue in audit["issues"]:
            lines.append(f"| {issue['severity']} | {issue['check']} | {issue['message']} |")
    lines += ["", "## Provenance", ""]
    if audit["provenance"].get("citation"):
        lines.append(f"Citation: {audit['provenance']['citation']}")
    lines.append(
        "The dataset is not included in this repository and must not be redistributed. "
        "See data/raw/README.md for manual download instructions."
    )
    return "\n".join(lines) + "\n"
