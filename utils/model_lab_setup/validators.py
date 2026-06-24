from __future__ import annotations

from typing import Any

from utils.model_lab_setup.registry_io import is_active_primary

ValidationResult = dict[str, Any]


def _result(errors: list[str] | None = None, warnings: list[str] | None = None) -> ValidationResult:
    error_list = errors or []
    warning_list = warnings or []
    return {"ok": not error_list, "errors": error_list, "warnings": warning_list}


def validate_required_identity(payload: dict[str, Any]) -> ValidationResult:
    identity = payload.get("identity") or payload
    errors: list[str] = []
    for field in ["model_id", "display_name", "model_family", "market_key"]:
        if not str(identity.get(field) or "").strip():
            errors.append(f"Missing required identity field: {field}")
    return _result(errors)


def validate_training_dates(payload: dict[str, Any]) -> ValidationResult:
    training = payload.get("training") or payload
    errors: list[str] = []
    warnings: list[str] = []
    is_new = bool(payload.get("is_new_model"))

    if not str(training.get("train_start_date") or "").strip():
        if is_new:
            errors.append("Train start date is required for new model configs.")
        else:
            warnings.append("Train start date is missing on this existing config.")
    for field in ["train_end_date", "calibration_end_date"]:
        if not str(training.get(field) or "").strip():
            errors.append(f"Missing required training date: {field}")
    return _result(errors, warnings)


def validate_probability_settings(payload: dict[str, Any]) -> ValidationResult:
    behavior = payload.get("behavior") or payload
    errors: list[str] = []
    try:
        clip_low = float(behavior.get("clip_low"))
        clip_high = float(behavior.get("clip_high"))
    except (TypeError, ValueError):
        return _result(["Probability clip values must be numeric."])

    if not (0 <= clip_low < clip_high <= 1):
        errors.append("Probability clipping must satisfy 0 <= clip_low < clip_high <= 1.")
    if clip_low >= 0.5:
        errors.append("Probability clip low must be below 0.5.")
    if clip_high <= 0.5:
        errors.append("Probability clip high must be above 0.5.")
    return _result(errors)


def validate_feature_count(expected_count: int | None, resolved_count: int) -> ValidationResult:
    if expected_count is None:
        return _result(warnings=["Expected feature count is missing."])
    if int(expected_count) != int(resolved_count):
        return _result(errors=[f"Feature count mismatch: expected {expected_count}, resolved {resolved_count}."])
    return _result()


def validate_model_id_available(registry: dict[str, Any], model_id: str) -> ValidationResult:
    if model_id in (registry.get("models") or {}):
        return _result(errors=[f"Model ID already exists: {model_id}"])
    return _result()


def validate_save_allowed(context: dict[str, Any]) -> ValidationResult:
    if context.get("is_new_model") or str(context.get("status") or "").lower() == "draft":
        return _result()
    return _result(errors=["Save is only allowed for draft or new models."])


def validate_delete_allowed(context: dict[str, Any], registry: dict[str, Any]) -> ValidationResult:
    status = str(context.get("status") or "").lower()
    model_id = str(context.get("model_id") or "")
    model_family = str(context.get("model_family") or "")
    market_key = str(context.get("market_key") or "")
    errors: list[str] = []
    if status == "production":
        errors.append("Production models cannot be deleted.")
    if is_active_primary(registry, model_id, model_family, market_key):
        errors.append("Active production models cannot be deleted.")
    return _result(errors)


def combine_validation_results(*results: ValidationResult) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    for result in results:
        errors.extend(result.get("errors") or [])
        warnings.extend(result.get("warnings") or [])
    return _result(errors, warnings)


def validate_model_setup_form(
    context: dict[str, Any],
    registry: dict[str, Any],
    payload: dict[str, Any],
) -> ValidationResult:
    identity = payload.get("identity") or {}
    training = dict(payload.get("training") or {})
    training["is_new_model"] = bool(context.get("is_new_model"))
    behavior = payload.get("behavior") or {}
    features = payload.get("features") or {}

    results = [
        validate_save_allowed(context),
        validate_required_identity(identity),
        validate_training_dates(training),
        validate_probability_settings(behavior),
    ]

    if "resolved_feature_count" in features:
        results.append(
            validate_feature_count(
                features.get("expected_feature_count"),
                int(features.get("resolved_feature_count") or 0),
            )
        )

    return combine_validation_results(*results)
