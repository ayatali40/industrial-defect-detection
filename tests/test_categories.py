"""Multi-category registry, isolation, inventory and aggregation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from industrial_defect.categories import (
    MVTEC_CATEGORIES,
    OutOfScopeCategoryError,
    UnknownCategoryError,
    out_of_scope_categories,
    select_categories,
    supported_categories,
)
from industrial_defect.config import category_dir
from industrial_defect.cross_category import (
    aggregate_frame,
    aggregate_rows,
    category_ranking,
    collect_rows,
)
from industrial_defect.dataset import (
    DatasetLayoutError,
    get_or_create_normal_split,
    resolve_manifest_path,
)
from industrial_defect.inference import IndustrialInspector, InferenceError
from industrial_defect.inventory import category_inventory, category_status
from industrial_defect.training import (
    artifact_path,
    available_models,
    thresholds_path,
    trained_categories,
)


def _make_category(root, name: str, count: int = 6) -> str:
    category = root / name
    for split, label in (("train", "good"), ("test", "good")):
        directory = category / split / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (16, 16), color=(index * 10 % 255, 40, 80)).save(
                directory / f"{index:03d}.png"
            )
    return str(category)


def test_registry_lists_all_fifteen_categories():
    assert len(MVTEC_CATEGORIES) == 15
    assert set(MVTEC_CATEGORIES) == {
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    }


def test_frozen_scope_is_six_categories_and_partitions_the_registry(config):
    assert len(MVTEC_CATEGORIES) == 15
    supported = supported_categories(config)
    excluded = out_of_scope_categories(config)
    assert supported == ["transistor", "cable", "leather", "bottle", "capsule", "grid"]
    assert len(supported) == 6
    assert len(excluded) == 9
    assert set(supported).isdisjoint(excluded)
    assert set(supported) | set(excluded) == set(MVTEC_CATEGORIES)


def test_select_categories_supports_none_active_all_and_lists(config):
    assert select_categories(None, config) == [config["data"]["category"]]
    assert select_categories("all", config) == supported_categories(config)
    assert select_categories("active", config) == list(config["data"]["categories"])
    assert select_categories("bottle, capsule,bottle", config) == ["bottle", "capsule"]
    with pytest.raises(UnknownCategoryError):
        select_categories("transistor,nope", config)


def test_out_of_scope_categories_are_rejected(config):
    for category in out_of_scope_categories(config):
        with pytest.raises(OutOfScopeCategoryError, match="outside the frozen project scope"):
            select_categories(category, config)


def test_inventory_defaults_to_the_supported_scope(tmp_config):
    assert set(category_inventory(tmp_config)) == set(supported_categories(tmp_config))


def test_artifact_paths_are_nested_and_category_isolated(tmp_config):
    config = tmp_config
    first = artifact_path(config, "transistor", "patchcore")
    second = artifact_path(config, "bottle", "patchcore")
    assert first != second
    assert first.parts[-3:] == ("transistor", "patchcore", "artifact.pt")
    assert thresholds_path(config, "bottle", "padim").parts[-3:] == (
        "bottle",
        "padim",
        "thresholds.json",
    )


def test_untrained_category_has_no_models_and_reports_missing_dataset(tmp_config):
    config = tmp_config
    assert available_models(config, "bottle") == []
    status = category_status(config, "bottle")
    assert status["data_available"] is False
    assert status["trained_models"] == []
    assert status["evaluated_models"] == []
    assert "bottle" not in trained_categories(config)


def test_category_roots_override_and_invalid_override(tmp_config, tmp_path):
    config = tmp_config
    custom = _make_category(tmp_path / "custom", "bottle", count=2)
    config["data"]["category_roots"] = {"bottle": custom}
    assert category_dir(config, "bottle").as_posix() == custom.replace("\\", "/")

    config["data"]["category_roots"] = {"bottle": str(tmp_path / "not-a-category")}
    with pytest.raises(DatasetLayoutError):
        category_dir(config, "bottle")


def test_normal_split_manifests_are_category_scoped_and_deterministic(
    tmp_config, tmp_path
):
    config = tmp_config
    manifests = {}
    splits = {}
    for category in ("bottle", "capsule"):
        path = _make_category(tmp_path / "data", category)
        manifest_path = resolve_manifest_path(config, category)
        splits[category] = get_or_create_normal_split(path, manifest_path, 0.34, 42)
        manifests[category] = manifest_path
    assert manifests["bottle"] != manifests["capsule"]
    assert manifests["bottle"].name == "bottle.json"
    assert manifests["capsule"].name == "capsule.json"

    first = json.loads(manifests["bottle"].read_text())
    assert first["category"] == "bottle"
    assert set(first["fit_paths"]).isdisjoint(first["val_paths"])
    repeated = get_or_create_normal_split(
        _make_category(tmp_path / "data", "bottle"), manifests["bottle"], 0.34, 42
    )
    assert [sample.relative_path for sample in repeated["fit"]] == [
        sample.relative_path for sample in splits["bottle"]["fit"]
    ]
    assert all(sample.label == 0 for sample in repeated["fit"] + repeated["val"])


def test_inference_rejects_unknown_and_untrained_categories(tmp_config, tmp_path):
    config = tmp_config
    _make_category(tmp_path / "data", "bottle", count=2)
    with pytest.raises(InferenceError, match="Unknown MVTec AD category"):
        IndustrialInspector(config, "patchcore", "not_a_category")
    with pytest.raises(InferenceError, match="No calibrated thresholds"):
        IndustrialInspector(config, "patchcore", "bottle")


def _write_metrics(
    config,
    category: str,
    model: str,
    *,
    image_auroc: float,
    image_f1: float,
    pixel_auroc: float,
    pro_auc: float,
    tp: int,
    fn: int,
    fp: int,
    normal: int,
) -> None:
    metrics_dir = (
        Path(config["_project_root"]) / "reports" / "metrics" / category
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_metrics": {
            "image_auroc": image_auroc,
            "image_average_precision": image_auroc,
            "image_precision": tp / (tp + fp) if tp + fp else 0.0,
            "image_recall": tp / (tp + fn) if tp + fn else 0.0,
            "image_f1": image_f1,
            "image_threshold": 1.0,
            "image_confusion": {"tp": tp, "fp": fp, "tn": normal - fp, "fn": fn},
        },
        "pixel_metrics": {
            "pixel_auroc": pixel_auroc,
            "pixel_average_precision": pixel_auroc,
            "pixel_precision": 0.5,
            "pixel_recall": 0.5,
            "pixel_f1": 0.5,
            "pixel_threshold": 0.5,
            "pixel_confusion": {"tp": 10, "fp": 10, "tn": 80, "fn": 10},
        },
        "pro_metrics": {"pro_auc": pro_auc},
        "model_metadata": {
            "training": {"fit_images": 10, "fit_seconds": 1.0},
        },
        "threshold_calibration": {"val_images": 3},
        "counts": {"normal": normal, "anomalous": tp + fn, "total": normal + tp + fn},
    }
    (metrics_dir / f"{model}_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cross_category_aggregation_labels_macro_and_pooled(tmp_config):
    config = tmp_config
    _write_metrics(
        config,
        "bottle",
        "patchcore",
        image_auroc=0.9,
        image_f1=0.8,
        pixel_auroc=0.7,
        pro_auc=0.6,
        tp=8,
        fn=2,
        fp=1,
        normal=5,
    )
    _write_metrics(
        config,
        "capsule",
        "patchcore",
        image_auroc=0.7,
        image_f1=0.6,
        pixel_auroc=0.5,
        pro_auc=0.4,
        tp=4,
        fn=6,
        fp=1,
        normal=9,
    )
    rows = collect_rows(config, ["bottle", "capsule"])
    assert len(rows) == 2
    aggregates = aggregate_rows(rows)
    macro = aggregates["per_model"]["patchcore"]["macro_average"]
    assert macro["image_auroc"] == pytest.approx(0.8)
    pooled = aggregates["per_model"]["patchcore"]["pooled_confusions"]
    assert pooled["image_confusion"] == {"tp": 12, "fp": 2, "tn": 12, "fn": 8}
    assert pooled["n_test_normal"] == 14
    assert pooled["n_test_anomalous"] == 20

    frame = aggregate_frame(rows)
    assert set(frame["row_type"]) == {"category", "macro_average", "pooled_confusions"}
    ranking = category_ranking(rows)
    assert ranking["available"] is True
    assert ranking["easiest"]["category"] == "bottle"
    assert ranking["hardest"]["category"] == "capsule"


def test_cross_category_ranking_refuses_one_category(tmp_config):
    config = tmp_config
    _write_metrics(
        config,
        "bottle",
        "patchcore",
        image_auroc=0.9,
        image_f1=0.8,
        pixel_auroc=0.7,
        pro_auc=0.6,
        tp=8,
        fn=2,
        fp=1,
        normal=5,
    )
    rows = collect_rows(config, ["bottle"])
    ranking = category_ranking(rows)
    assert ranking["available"] is False
    assert "Fewer than two categories" in ranking["reason"]
