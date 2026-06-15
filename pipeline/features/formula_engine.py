from __future__ import annotations

import ast
from typing import Any

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


def apply_formula_features(frame: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    """Apply active formula features from the Model Lab feature registry section."""

    output = frame.copy()
    studio = registry.get("model_lab_feature_studio", {}) or {}
    for feature_id, feature in (studio.get("features", {}) or {}).items():
        if feature.get("type") != "formula" or feature.get("status") not in {"active", "draft"}:
            continue
        formula = feature.get("formula")
        if not formula:
            continue
        output[str(feature_id)] = compute_formula_feature(output, str(formula))
    return output
