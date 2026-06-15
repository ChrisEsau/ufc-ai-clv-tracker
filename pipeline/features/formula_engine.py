from __future__ import annotations

import ast
from typing import Any, Iterable

import numpy as np
import pandas as pd

ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "log": np.log,
    "sqrt": np.sqrt,
    "where": np.where,
}

ALLOWED_NODES = (
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


class FormulaFeatureError(RuntimeError):
    """Raised when a formula feature cannot be safely computed."""


def validate_formula_ast(formula: str) -> ast.Expression:
    """Parse and validate a formula expression for safe evaluation."""

    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError as exc:
        raise FormulaFeatureError(f"Invalid formula syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise FormulaFeatureError(f"Unsupported formula expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise FormulaFeatureError("Formula uses an unsupported function call.")
    return tree


def formula_column_names(formula: str) -> set[str]:
    tree = validate_formula_ast(formula)
    function_names = set(ALLOWED_FUNCTIONS)
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in function_names
    }


def compute_formula_feature(frame: pd.DataFrame, formula: str) -> pd.Series:
    """Compute one formula feature against an existing feature frame.

    The evaluator exposes only frame columns and a small allow-list of vectorized
    math functions. It does not expose Python builtins, imports, files, or network.
    """

    tree = validate_formula_ast(formula)
    missing = sorted(column for column in formula_column_names(formula) if column not in frame.columns)
    if missing:
        raise FormulaFeatureError(f"Formula references missing columns: {missing}")

    namespace: dict[str, Any] = {column: frame[column] for column in frame.columns}
    namespace.update(ALLOWED_FUNCTIONS)
    compiled = compile(tree, "<formula_feature>", "eval")
    result = eval(compiled, {"__builtins__": {}}, namespace)  # noqa: S307 - restricted AST and namespace
    if isinstance(result, pd.Series):
        return result
    return pd.Series(result, index=frame.index)


def _resolve_requested_feature_ids(
    *,
    registry: dict[str, Any],
    selected_bundles: Iterable[str] | None = None,
    selected_features: Iterable[str] | None = None,
) -> set[str] | None:
    """Resolve requested feature IDs from explicit features and bundle IDs.

    Returns None when no explicit selection is provided, which means callers want
    all eligible formula features. A concrete set means formulas are limited to
    the selected feature IDs.
    """

    studio = registry.get("model_lab_feature_studio", {}) or {}
    bundles = studio.get("bundles", {}) or {}
    requested: set[str] = set(str(item) for item in (selected_features or []) if str(item).strip())

    for bundle_id in selected_bundles or []:
        bundle = bundles.get(str(bundle_id), {}) or {}
        requested.update(str(item) for item in bundle.get("features", []) or [] if str(item).strip())

    return requested if requested else None


def apply_formula_features(
    frame: pd.DataFrame,
    registry: dict[str, Any],
    *,
    selected_bundles: Iterable[str] | None = None,
    selected_features: Iterable[str] | None = None,
    allowed_statuses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply formula features from the Model Lab feature registry section.

    Formula features are computed in registry order so a later formula may depend
    on an earlier formula. Callers can restrict computation by bundle or explicit
    feature IDs. By default, only active formulas are materialized into feature
    views; draft formulas remain editable in Model Lab without changing builds.
    """

    output = frame.copy()
    studio = registry.get("model_lab_feature_studio", {}) or {}
    features = studio.get("features", {}) or {}
    requested = _resolve_requested_feature_ids(
        registry=registry,
        selected_bundles=selected_bundles,
        selected_features=selected_features,
    )
    statuses = set(str(item) for item in (allowed_statuses or {"active"}))

    for feature_id, feature in features.items():
        feature_id = str(feature_id)
        if requested is not None and feature_id not in requested:
            continue
        if feature.get("type") != "formula" or feature.get("status") not in statuses:
            continue
        formula = feature.get("formula")
        if not formula:
            continue
        output[feature_id] = compute_formula_feature(output, str(formula))
    return output
