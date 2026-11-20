"""MVTec AD category registry, frozen project scope and selection helpers."""

from __future__ import annotations

from typing import Any

MVTEC_CATEGORIES: tuple[str, ...] = (
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
)

DEFAULT_EXPANSION: tuple[str, ...] = ("bottle", "capsule", "metal_nut")


class UnknownCategoryError(ValueError):
    """Raised when a category is not part of the MVTec AD registry."""


class OutOfScopeCategoryError(UnknownCategoryError):
    """Raised for registry categories excluded by the frozen project scope."""


def validate_category(category: str) -> str:
    category = str(category).strip()
    if category not in MVTEC_CATEGORIES:
        raise UnknownCategoryError(
            f"Unknown MVTec AD category '{category}'. Registered categories: "
            + ", ".join(MVTEC_CATEGORIES)
        )
    return category


def supported_categories(config: dict[str, Any]) -> list[str]:
    """Return the frozen scope declared in ``data.supported_categories``."""
    configured = (config.get("data") or {}).get("supported_categories")
    if not configured:
        raise UnknownCategoryError(
            "No frozen project scope configured; set data.supported_categories in configs/config.yaml"
        )
    categories = list(
        dict.fromkeys(validate_category(item) for item in configured)
    )
    if not categories:
        raise UnknownCategoryError("data.supported_categories is empty")
    return categories


def out_of_scope_categories(config: dict[str, Any]) -> list[str]:
    supported = set(supported_categories(config))
    return [category for category in MVTEC_CATEGORIES if category not in supported]


def ensure_in_scope(category: str, config: dict[str, Any]) -> str:
    """Reject registry categories excluded from the frozen six-category scope."""
    category = validate_category(category)
    supported = supported_categories(config)
    if category not in supported:
        raise OutOfScopeCategoryError(
            f"Category '{category}' is outside the frozen project scope. Supported "
            f"categories: {', '.join(supported)}. The other MVTec AD categories are "
            "intentionally not supported."
        )
    return category


def select_categories(
    spec: str | list[str] | tuple[str, ...] | None, config: dict[str, Any]
) -> list[str]:
    """Resolve a CLI category specification inside the frozen scope.

    ``None`` selects ``data.category``; ``"all"`` selects every supported
    category; ``"active"`` selects ``data.categories`` from the configuration;
    otherwise a comma-separated list or sequence of names is accepted.
    """
    if spec is None:
        return [ensure_in_scope(str(config["data"]["category"]), config)]
    if isinstance(spec, str):
        token = spec.strip()
        if token == "all":
            return supported_categories(config)
        if token == "active":
            configured = config["data"].get("categories") or [config["data"]["category"]]
            return list(
                dict.fromkeys(ensure_in_scope(item, config) for item in configured)
            )
        items = [item.strip() for item in token.split(",") if item.strip()]
    else:
        items = list(spec)
    if not items:
        raise UnknownCategoryError("No categories supplied")
    return list(dict.fromkeys(ensure_in_scope(item, config) for item in items))


def active_categories(config: dict[str, Any]) -> list[str]:
    return select_categories("active", config)
