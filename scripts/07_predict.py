"""Phase 10: run the trained models on one image or a folder of images."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from industrial_defect.categories import (  # noqa: E402
    out_of_scope_categories,
    supported_categories,
)
from industrial_defect.config import (  # noqa: E402
    ensure_output_dirs,
    get_device,
    load_config,
    resolve_path,
)
from industrial_defect.inference import (  # noqa: E402
    DISCLAIMER,
    IndustrialInspector,
    InferenceError,
    save_result,
)
from industrial_defect.preprocessing import is_image_file  # noqa: E402
from industrial_defect.training import MODEL_NAMES  # noqa: E402
from industrial_defect.utils import (  # noqa: E402
    configure_threads,
    environment_info,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Image file or folder")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--model", choices=[*MODEL_NAMES, "both"], default="patchcore")
    parser.add_argument("--output", type=Path, default=None, help="Override the output directory")
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List dataset/training status for the six supported categories and exit",
    )
    return parser.parse_args()


def _collect_inputs(target: Path) -> list[Path]:
    if target.is_file():
        if not is_image_file(target):
            raise InferenceError(f"Unsupported input file type: {target.suffix}")
        return [target]
    if target.is_dir():
        inputs = sorted(child for child in target.iterdir() if child.is_file() and is_image_file(child))
        if not inputs:
            raise InferenceError(f"No supported images found in {target}")
        return inputs
    raise InferenceError(f"Input path does not exist: {target}")


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
    if args.list_categories:
        from industrial_defect.inventory import category_inventory

        print(f"Supported categories (frozen scope): {', '.join(supported_categories(config))}")
        print(f"{'category':12s} {'data':6s} {'trained models':28s} evaluated")
        for name, status in category_inventory(config).items():
            print(
                f"{name:12s} {'yes' if status['data_available'] else 'no':6s} "
                f"{', '.join(status['trained_models']) or '-':28s} "
                f"{', '.join(status['evaluated_models']) or '-'}"
            )
        print(
            "Out of scope (not loaded): "
            + ", ".join(out_of_scope_categories(config))
        )
        return 0
    if args.input is None:
        print("[predict] error: --input is required unless --list-categories is used", file=sys.stderr)
        return 2
    category = str(config["data"]["category"])
    ensure_output_dirs(config)
    seed_everything(int(config["project"]["seed"]))
    configure_threads(int(config["project"].get("num_threads", 0)))
    device = get_device(config)
    inputs = _collect_inputs(args.input)
    if args.output is not None:
        output_root = resolve_path(config, args.output)
    else:
        output_root = resolve_path(config, config["inference"]["output_dir"]) / category
    output_root.mkdir(parents=True, exist_ok=True)
    models = list(MODEL_NAMES) if args.model == "both" else [args.model]

    print(f"[predict] inputs={len(inputs)} category={category} models={models}")
    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": category,
        "inputs": [path.name for path in inputs],
        "input_selection": _relative(args.input),
        "models": {},
        "environment": environment_info(),
        "disclaimer": DISCLAIMER,
    }
    for model_name in models:
        inspector = IndustrialInspector(config, model_name, category, device=device)
        model_dir = output_root / model_name
        records = []
        for input_path in inputs:
            result = inspector.inspect(input_path)
            outputs = save_result(result, model_dir, stem=input_path.stem)
            record = {
                **result.to_dict(),
                "outputs": {key: _relative(Path(value)) for key, value in outputs.items()},
            }
            records.append(record)
            print(
                f"[predict] {model_name} {input_path.name}: score={result.score:.3f} "
                f"threshold={result.image_threshold:.3f} decision={result.decision} "
                f"defect_area={result.defect_area_fraction * 100:.2f}% "
                f"({result.latency_ms:.0f} ms)"
            )
        latencies = np.array([record["latency_ms"] for record in records])
        manifest["models"][model_name] = {
            "inspector": {key: value for key, value in inspector.describe().items() if key != "artifact"},
            "artifact": _relative(inspector.artifact_path),
            "results": records,
            "summary": {
                "images": len(records),
                "defects": int(sum(record["decision"] == "defect" for record in records)),
                "passes": int(sum(record["decision"] == "pass" for record in records)),
                "latency_ms_mean": float(latencies.mean()),
                "latency_ms_min": float(latencies.min()),
                "latency_ms_max": float(latencies.max()),
            },
        }
    manifest_path = save_json(output_root / "inference_manifest.json", manifest)
    print(f"[predict] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InferenceError as exc:
        print(f"[predict] error: {exc}", file=sys.stderr)
        sys.exit(2)
