"""Phase 2: validate the MVTec AD layout and write evidence-based per-category audits."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_defect.categories import (  # noqa: E402
    UnknownCategoryError,
    select_categories,
)
from industrial_defect.config import (  # noqa: E402
    ConfigError,
    category_dir,
    configured_path,
    ensure_output_dirs,
    load_config,
)
from industrial_defect.utils import save_json  # noqa: E402
from industrial_defect.validation import (  # noqa: E402
    audit_dataset,
    render_audit_markdown,
    render_combined_audit_markdown,
    summarise_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML configuration")
    parser.add_argument("--category", type=str, default=None, help="Single dataset category")
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list, 'active' (config data.categories) or 'all' (six supported categories)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.category:
        config["data"]["category"] = args.category
    try:
        categories = select_categories(args.categories or args.category, config)
    except UnknownCategoryError as exc:
        print(f"[validate] error: {exc}", file=sys.stderr)
        return 2
    data_root = configured_path(config, "data_root")
    metrics_dir = configured_path(config, "metrics_dir")
    ensure_output_dirs(config)

    print(f"[validate] dataset root : {data_root}")
    print(f"[validate] categories   : {', '.join(categories)}")

    summaries = {}
    for category in categories:
        try:
            resolved = category_dir(config, category)
        except Exception:
            resolved = None
        audit = audit_dataset(data_root, category, category_path=resolved)
        category_metrics = metrics_dir / category
        category_metrics.mkdir(parents=True, exist_ok=True)
        save_json(category_metrics / "dataset_audit.json", audit)
        (category_metrics / "dataset_audit.md").write_text(
            render_audit_markdown(audit), encoding="utf-8"
        )
        summary = summarise_audit(audit)
        summaries[category] = summary
        critical = len(summary["critical_issues"])
        warnings = len(summary["warnings"])
        print(
            f"[audit] {category:12s} data={'yes' if summary['data_available'] else 'no ':3s} "
            f"ready={'PASS' if summary['ready'] else 'FAIL'} "
            f"critical={critical} warnings={warnings}"
        )
        for message in summary["critical_issues"][:3]:
            print(f"[audit]   CRITICAL {message}")

    ready_categories = [name for name, item in summaries.items() if item["ready"]]
    missing_categories = [name for name, item in summaries.items() if not item["ready"]]
    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(data_root.resolve().relative_to(PROJECT_ROOT))
        if data_root.is_relative_to(PROJECT_ROOT)
        else data_root.name,
        "requested": categories,
        "ready_categories": ready_categories,
        "missing_categories": missing_categories,
        "per_category": summaries,
        "note": "Each category is audited independently; missing categories are reported as "
        "unavailable and are never replaced with synthetic data.",
    }
    combined_json = save_json(metrics_dir / "dataset_audit_all.json", combined)
    combined_md = metrics_dir / "dataset_audit_all.md"
    markdown = render_combined_audit_markdown(combined)
    combined_md.write_text(markdown, encoding="utf-8")
    reports_dir = configured_path(config, "reports_dir")
    (reports_dir / "dataset_audit.md").write_text(markdown, encoding="utf-8")

    print(f"[audit] combined JSON: {combined_json}")
    print(f"[audit] combined MD  : {combined_md}")
    all_ready = len(missing_categories) == 0
    print(f"[audit] all requested categories ready: {all_ready}")
    return 0 if all_ready else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"[validate] configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
