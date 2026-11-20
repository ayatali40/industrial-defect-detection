"""Phase 3: create the deterministic normal-only split and record preprocessing."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from industrial_defect.categories import ensure_in_scope  # noqa: E402
from industrial_defect.config import (  # noqa: E402
    category_dir,
    configured_path,
    ensure_output_dirs,
    load_config,
)
from industrial_defect.dataset import (  # noqa: E402
    assert_normal_only,
    discover_samples,
    get_or_create_normal_split,
    resolve_manifest_path,
)
from industrial_defect.preprocessing import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image_file,
)
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    category = ensure_in_scope(str(config["data"]["category"]), config)
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))

    cat_dir = category_dir(config)
    manifest_path = resolve_manifest_path(config, category)
    split = get_or_create_normal_split(
        cat_dir,
        manifest_path,
        val_fraction=float(config["data"]["normal_split"]["val_fraction"]),
        seed=int(config["data"]["normal_split"]["seed"]),
    )
    assert_normal_only(split["fit"])
    assert_normal_only(split["val"])

    test_samples = discover_samples(cat_dir, "test")
    fit_paths = {sample.relative_path for sample in split["fit"]}
    val_paths = {sample.relative_path for sample in split["val"]}
    test_paths = {sample.relative_path for sample in test_samples}
    overlap = (fit_paths | val_paths) & test_paths
    if overlap:
        raise RuntimeError(f"Split leakage detected between train and test: {sorted(overlap)[:5]}")

    sample_tensor = preprocess_image_file(split["fit"][0].path, int(config["data"]["image_size"]))
    sample_stats = {
        "shape": list(sample_tensor.shape),
        "dtype": str(sample_tensor.dtype),
        "mean_per_channel": np.round(sample_tensor.mean(dim=(1, 2)).numpy(), 6).tolist(),
        "std_per_channel": np.round(sample_tensor.std(dim=(1, 2)).numpy(), 6).tolist(),
    }
    processed_record = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "category_dir": _relative(cat_dir),
        "image_size": int(config["data"]["image_size"]),
        "resize": "bilinear",
        "normalization": {"name": "imagenet", "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "color_mode": "RGB (grayscale converted to RGB)",
        "mask_handling": "binary masks resized with max-pooling when the factor is integral",
        "split_manifest": _relative(manifest_path),
        "split": {
            "fit": len(split["fit"]),
            "val": len(split["val"]),
            "test": len(test_samples),
            "normal_only": True,
            "test_used_for_fitting_or_calibration": False,
        },
        "sample_preprocessed_stats": sample_stats,
        "images_copied": False,
        "note": "Original files are never modified or duplicated; only derived features and reports are written.",
    }
    record_path = save_json(
        configured_path(config, "processed_dir") / f"{category}_preprocessing.json", processed_record
    )

    print(f"[prepare] category        : {category}")
    print(f"[prepare] fit / val / test: {len(split['fit'])} / {len(split['val'])} / {len(test_samples)}")
    print(f"[prepare] split manifest  : {manifest_path}")
    print(f"[prepare] preprocessing   : {record_path}")
    print(f"[prepare] sample tensor   : {sample_tensor.shape} {sample_tensor.dtype}")
    print(f"[prepare] leakage check   : {'PASS' if not overlap else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
