from __future__ import annotations

import ast
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

FEATURE_REGISTRY_PATH = Path("configs/features/feature_registry.yaml")

FEATURE_TYPES = ["transform", "formula", "pipeline", "base_column"]
FEATURE_STATUSES = ["draft", "active", "planned", "archived"]
BLOCKED_INPUT_NAMES = {
    "winner",
    "winner_id",
    "winner_is_red",
    "winner_is_blue",
    "red_won",
    "blue_won",
    "target",
    "label",
    "outcome",
    "result",
    "fight_result",
}
BLOCKED_INPUT_PREFIXES = ("winner_", "target_", "outcome_", "result_")
ALLOWED_FORMULA_FUNCTIONS = {"abs", "min", "max", "clip", "log", "sqrt", "where"}


class FeatureRegistryError(RuntimeError):
    """Raised when the Model Lab feature registry is invalid."""


def safe_feature_id(value: str) -> str:
    """Normalize user-entered labels into stable registry IDs."""

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise FeatureRegistryError(f"YAML file must contain a mapping: {path}")
    return payload


def dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _default_model_lab_feature_studio() -> dict[str, Any]:
    return {
        "rules": {
            "dashboard_edits_definitions_only": True,
            "pipeline_computes_feature_values": True,
            "bundles_contain_features": True,
            "features_define_build_method": True,
            "target_columns_are_blocked": True,
            "preserve_master_feature_families": True,
            "preserve_current_moneyline_v5_features": True,
        },
        "allowed_formula_functions": sorted(ALLOWED_FORMULA_FUNCTIONS),
        "features": {},
        "bundles": {},
    }


def load_feature_registry(path: str | Path = FEATURE_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_yaml(path)
    registry.setdefault("registry_name", "ufc_master_feature_registry")
    registry.setdefault("version", 1)
    registry.setdefault("status", "initial_family_registry")
    studio = registry.setdefault("model_lab_feature_studio", _default_model_lab_feature_studio())
    studio.setdefault("features", {})
    studio.setdefault("bundles", {})
    studio.setdefault("rules", {})
    studio.setdefault("allowed_formula_functions", sorted(ALLOWED_FORMULA_FUNCTIONS))
    return registry


def save_feature_registry(registry: dict[str, Any], path: str | Path = FEATURE_REGISTRY_PATH) -> None:
    Path(path).write_text(dump_yaml(registry), encoding="utf-8")


def studio_payload(registry: dict[str, Any]) -> dict[str, Any]:
    return registry.setdefault("model_lab_feature_studio", _default_model_lab_feature_studio())


def feature_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return studio_payload(registry).setdefault("features", {})


def bundle_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return studio_payload(registry).setdefault("bundles", {})


def feature_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_id, feature in feature_map(registry).items():
        rows.append(
            {
                "feature_id": feature_id,
                "label": feature.get("label", feature_id),
                "type": feature.get("type", ""),
                "family": feature.get("family", ""),
                "status": feature.get("status", ""),
                "inputs": ", ".join(str(item) for item in feature.get("inputs", []) or []),
                "formula": feature.get("formula", ""),
                "builder": feature.get("builder", ""),
                "leakage_safe": bool(feature.get("leakage_safe", False)),
            }
        )
    return sorted(rows, key=lambda row: row["feature_id"])


def bundle_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_id, bundle in bundle_map(registry).items():
        features = [str(item) for item in bundle.get("features", []) or []]
        rows.append(
            {
                "bundle_id": bundle_id,
                "label": bundle.get("label", bundle_id),
                "status": bundle.get("status", ""),
                "feature_count": len(features),
                "features": ", ".join(features),
            }
        )
    return sorted(rows, key=lambda row: row["bundle_id"])


def csv_to_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").replace("\n", ",").split(",") if item.strip()]


def has_blocked_input(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in BLOCKED_INPUT_NAMES or normalized.startswith(BLOCKED_INPUT_PREFIXES)


def formula_names(formula: str) -> set[str]:
    if not str(formula or "").strip():
        return set()
    tree = ast.parse(str(formula), mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def validate_formula_syntax(formula: str, allowed_functions: set[str] | None = None) -> list[str]:
    """Validate formula syntax without evaluating it."""

    issues: list[str] = []
    formula = str(formula or "").strip()
    if not formula:
        return ["Formula is empty."]

    allowed_functions = allowed_functions or ALLOWED_FORMULA_FUNCTIONS
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        return [f"Invalid formula syntax: {exc.msg}"]

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Compare,
        ast.IfExp,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            issues.append(f"Unsupported formula expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions:
                issues.append("Formula uses an unsupported function call.")
        if isinstance(node, ast.Name) and has_blocked_input(node.id):
            issues.append(f"Formula references blocked outcome/target field: {node.id}")
    return list(dict.fromkeys(issues))


def validate_registry(registry: dict[str, Any]) -> list[dict[str, str]]:
    """Return validation findings for the Model Lab feature studio section."""

    findings: list[dict[str, str]] = []
    features = feature_map(registry)
    bundles = bundle_map(registry)
    allowed_functions = set(studio_payload(registry).get("allowed_formula_functions") or ALLOWED_FORMULA_FUNCTIONS)

    for feature_id, feature in features.items():
        if safe_feature_id(feature_id) != feature_id:
            findings.append({"level": "error", "item": feature_id, "message": "Feature ID should be snake_case."})
        if feature.get("type") not in FEATURE_TYPES:
            findings.append({"level": "error", "item": feature_id, "message": "Feature has invalid type."})
        if feature.get("status") not in FEATURE_STATUSES:
            findings.append({"level": "warning", "item": feature_id, "message": "Feature has unknown status."})
        for input_name in feature.get("inputs", []) or []:
            if has_blocked_input(str(input_name)):
                findings.append({"level": "error", "item": feature_id, "message": f"Blocked leakage input: {input_name}"})
        if feature.get("type") == "formula":
            formula = str(feature.get("formula") or "")
            formula_issues = validate_formula_syntax(formula, allowed_functions)
            for issue in formula_issues:
                findings.append({"level": "error", "item": feature_id, "message": issue})
            names = set()
            if formula.strip() and not formula_issues:
                names = formula_names(formula)
            function_names = names.intersection(allowed_functions)
            referenced = sorted(names - function_names)
            declared = set(str(item) for item in feature.get("inputs", []) or [])
            missing_declared = [name for name in referenced if name not in declared]
            if missing_declared:
                findings.append({"level": "warning", "item": feature_id, "message": f"Formula references undeclared inputs: {missing_declared}"})
        if feature.get("type") == "pipeline" and not feature.get("builder"):
            findings.append({"level": "warning", "item": feature_id, "message": "Pipeline feature has no builder path."})

    for bundle_id, bundle in bundles.items():
        bundle_features = [str(item) for item in bundle.get("features", []) or []]
        if safe_feature_id(bundle_id) != bundle_id:
            findings.append({"level": "error", "item": bundle_id, "message": "Bundle ID should be snake_case."})
        if not bundle_features:
            findings.append({"level": "warning", "item": bundle_id, "message": "Bundle has no features."})
        for feature_id in bundle_features:
            if feature_id not in features:
                findings.append({"level": "error", "item": bundle_id, "message": f"Bundle references unknown feature: {feature_id}"})
            elif features[feature_id].get("status") == "archived":
                findings.append({"level": "warning", "item": bundle_id, "message": f"Bundle references archived feature: {feature_id}"})

    return findings


def upsert_feature(registry: dict[str, Any], feature_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(registry)
    feature_map(updated)[feature_id] = payload
    return updated


def archive_feature(registry: dict[str, Any], feature_id: str) -> dict[str, Any]:
    updated = deepcopy(registry)
    if feature_id in feature_map(updated):
        feature_map(updated)[feature_id]["status"] = "archived"
    return updated


def delete_feature(registry: dict[str, Any], feature_id: str) -> dict[str, Any]:
    """Remove a feature and drop it from any bundles that reference it."""

    updated = deepcopy(registry)
    feature_map(updated).pop(feature_id, None)
    for bundle in bundle_map(updated).values():
        bundle["features"] = [item for item in bundle.get("features", []) or [] if item != feature_id]
    return updated


def upsert_bundle(registry: dict[str, Any], bundle_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(registry)
    bundle_map(updated)[bundle_id] = payload
    return updated


def archive_bundle(registry: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    updated = deepcopy(registry)
    if bundle_id in bundle_map(updated):
        bundle_map(updated)[bundle_id]["status"] = "archived"
    return updated


def delete_bundle(registry: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    updated = deepcopy(registry)
    bundle_map(updated).pop(bundle_id, None)
    return updated
