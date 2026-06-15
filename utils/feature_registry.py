from __future__ import annotations

import ast
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

FEATURE_REGISTRY_PATH = Path("configs/features/feature_registry.yaml")
FEATURE_BUNDLE_REGISTRY_PATH = Path("configs/features/feature_bundles.yaml")
TRANSFORM_REGISTRY_PATH = Path("configs/features/transform_registry.yaml")
MONEYLINE_V5_CONTRACT_PATH = Path("configs/features/current_moneyline_v5_features.yaml")

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
    """Raised when a Model Lab feature or bundle registry operation is invalid."""


def safe_feature_id(value: str) -> str:
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


def labelize(value: str) -> str:
    return str(value).replace("_", " ").title()


def build_moneyline_v5_feature_definitions(
    contract_path: str | Path = MONEYLINE_V5_CONTRACT_PATH,
) -> dict[str, dict[str, Any]]:
    """Build canonical current Moneyline V5 feature definitions from the protected contract."""

    contract = load_yaml(contract_path)
    stats = [str(item) for item in contract.get("base_stat_names_used_for_diff_families", []) or []]
    engineered = [str(item) for item in contract.get("registered_engineered_features", []) or []]
    definitions: dict[str, dict[str, Any]] = {}

    for stat in stats:
        feature_id = f"{stat}_diff"
        definitions[feature_id] = {
            "label": f"{labelize(stat)} Difference",
            "type": "transform",
            "status": "active",
            "family": "career_differentials",
            "description": f"Red fighter pre-fight {stat} minus blue fighter pre-fight {stat}.",
            "inputs": [f"r_pre_{stat}", f"b_pre_{stat}"],
            "transform": "red_minus_blue",
            "source_columns": [stat],
            "output_column": feature_id,
            "model_input_allowed": True,
            "current_moneyline_v5": "included",
            "leakage_safe": True,
        }

    for stat in stats:
        feature_id = f"ewm_{stat}_diff"
        definitions[feature_id] = {
            "label": f"EWM {labelize(stat)} Difference",
            "type": "transform",
            "status": "active",
            "family": "ewm_differentials",
            "description": f"Red fighter EWM {stat} minus blue fighter EWM {stat}.",
            "inputs": [f"r_ewm_{stat}", f"b_ewm_{stat}"],
            "transform": "red_minus_blue",
            "source_columns": [f"ewm_{stat}"],
            "output_column": feature_id,
            "model_input_allowed": True,
            "current_moneyline_v5": "included",
            "leakage_safe": True,
        }

    for stat in stats:
        feature_id = f"recent_form_{stat}_diff"
        definitions[feature_id] = {
            "label": f"Recent Form {labelize(stat)} Difference",
            "type": "transform",
            "status": "active",
            "family": "recent_form_differentials",
            "description": f"Red fighter recent-form {stat} minus blue fighter recent-form {stat}.",
            "inputs": [f"r_pre_recent_{stat}", f"b_pre_recent_{stat}"],
            "transform": "red_minus_blue",
            "source_columns": [f"recent_form_{stat}"],
            "output_column": feature_id,
            "model_input_allowed": True,
            "current_moneyline_v5": "included",
            "leakage_safe": True,
        }

    for feature_id in engineered:
        definitions[feature_id] = {
            "label": labelize(feature_id),
            "type": "base_column",
            "status": "active",
            "family": "engineered_moneyline_matchup_features",
            "description": f"Registered engineered V5 moneyline matchup feature: {feature_id}.",
            "source_column": feature_id,
            "output_column": feature_id,
            "model_input_allowed": True,
            "current_moneyline_v5": "included",
            "leakage_safe": True,
        }

    return definitions


def expand_generated_feature_definitions(registry: dict[str, Any]) -> dict[str, Any]:
    generation = (registry.get("feature_definition_generation") or {}).get("current_moneyline_v5", {}) or {}
    if not generation.get("enabled", False):
        return registry
    contract_path = generation.get("contract_path") or MONEYLINE_V5_CONTRACT_PATH
    generated = build_moneyline_v5_feature_definitions(contract_path)
    explicit = registry.setdefault("feature_definitions", {})
    merged = {**generated, **explicit}
    registry["feature_definitions"] = merged
    return registry


def load_feature_registry(path: str | Path = FEATURE_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_yaml(path)
    registry.setdefault("registry_name", "ufc_master_feature_registry")
    registry.setdefault("version", 1)
    registry.setdefault("status", "initial_family_registry")
    registry.setdefault("feature_definitions", {})
    registry.setdefault("feature_families", {})
    registry.setdefault("validation_rules", [])
    registry.setdefault("allowed_formula_functions", sorted(ALLOWED_FORMULA_FUNCTIONS))
    return expand_generated_feature_definitions(registry)


def load_feature_bundle_registry(path: str | Path = FEATURE_BUNDLE_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_yaml(path)
    registry.setdefault("registry_name", "ufc_feature_bundles")
    registry.setdefault("version", 1)
    registry.setdefault("status", "draft_contract")
    registry.setdefault("bundle_rules", {})
    registry.setdefault("bundles", {})
    return registry


def load_transform_registry(path: str | Path = TRANSFORM_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_yaml(path)
    registry.setdefault("registry_name", "ufc_transform_registry")
    registry.setdefault("transforms", {})
    return registry


def transform_options(*, include_planned: bool = True) -> list[str]:
    transforms = load_transform_registry().get("transforms", {}) or {}
    options: list[str] = []
    for transform_id, transform in transforms.items():
        status = str((transform or {}).get("status", "active"))
        if include_planned or status == "active":
            options.append(str(transform_id))
    return sorted(options)


def save_feature_registry(registry: dict[str, Any], path: str | Path = FEATURE_REGISTRY_PATH) -> None:
    Path(path).write_text(dump_yaml(registry), encoding="utf-8")


def feature_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return registry.setdefault("feature_definitions", {})


def bundle_map(registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    if registry is not None and "bundles" in registry:
        return registry.setdefault("bundles", {})
    return load_feature_bundle_registry().get("bundles", {}) or {}


def feature_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_id, feature in feature_map(registry).items():
        rows.append({
            "feature_id": feature_id,
            "label": feature.get("label", feature_id),
            "type": feature.get("type", ""),
            "family": feature.get("family", ""),
            "status": feature.get("status", ""),
            "inputs": ", ".join(str(item) for item in feature.get("inputs", []) or []),
            "formula": feature.get("formula", ""),
            "builder": feature.get("builder", ""),
            "leakage_safe": bool(feature.get("leakage_safe", False)),
        })
    return sorted(rows, key=lambda row: row["feature_id"])


def bundle_rows(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_id, bundle in bundle_map(registry).items():
        candidate_columns = [str(item) for item in bundle.get("candidate_columns", []) or []]
        rows.append({
            "bundle_id": bundle_id,
            "description": bundle.get("description", bundle_id),
            "candidate_count": len(candidate_columns),
            "candidate_columns": ", ".join(candidate_columns),
            "markets": ", ".join(str(item) for item in bundle.get("markets", []) or []),
        })
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
    issues: list[str] = []
    formula = str(formula or "").strip()
    if not formula:
        return ["Formula is empty."]
    allowed_functions = allowed_functions or ALLOWED_FORMULA_FUNCTIONS
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        return [f"Invalid formula syntax: {exc.msg}"]
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Compare, ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
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
    findings: list[dict[str, str]] = []
    features = feature_map(registry)
    allowed_functions = set(registry.get("allowed_formula_functions") or ALLOWED_FORMULA_FUNCTIONS)
    valid_transforms = set(transform_options(include_planned=True))
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
        if feature.get("type") == "transform" and feature.get("transform") not in valid_transforms:
            findings.append({"level": "error", "item": feature_id, "message": "Transform ID is not registered."})
        if feature.get("type") == "formula":
            formula = str(feature.get("formula") or "")
            for issue in validate_formula_syntax(formula, allowed_functions):
                findings.append({"level": "error", "item": feature_id, "message": issue})
        if feature.get("type") == "pipeline" and not feature.get("builder"):
            findings.append({"level": "warning", "item": feature_id, "message": "Pipeline feature has no builder path."})
    return findings


def validate_bundle_registry(registry: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for bundle_id, bundle in bundle_map(registry).items():
        if safe_feature_id(bundle_id) != bundle_id:
            findings.append({"level": "error", "item": bundle_id, "message": "Bundle ID should be snake_case."})
        if not bundle.get("description"):
            findings.append({"level": "warning", "item": bundle_id, "message": "Bundle has no description."})
        if not bundle.get("candidate_columns"):
            findings.append({"level": "warning", "item": bundle_id, "message": "Bundle has no candidate columns."})
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
    updated = deepcopy(registry)
    feature_map(updated).pop(feature_id, None)
    return updated


def upsert_bundle(registry: dict[str, Any], bundle_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(registry)
    bundle_map(updated)[bundle_id] = payload
    return updated


def delete_bundle(registry: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    updated = deepcopy(registry)
    bundle_map(updated).pop(bundle_id, None)
    return updated
