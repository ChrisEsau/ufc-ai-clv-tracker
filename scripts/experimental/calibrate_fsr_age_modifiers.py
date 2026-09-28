"""Calibrate external FSR age modifiers from existing shadow audit artifacts.

No FSR replay and no Monte Carlo run occurs here.

Ordinary retained traits
-----------------------
Use the already-built consecutive-fight transition table:

    delta = next_prefight_FSR - current_prefight_FSR

Fit age-only population drift on the well-supported age 21-39 range. Candidate
model degrees are linear, quadratic, and cubic. Degree selection uses a
chronological 80/20 holdout and deliberately chooses the *simplest* model whose
holdout RMSE is within 1% of the best model. The selected model is then refit on
all calibration rows using centered age x = age - 30.

The configured modifier is a one-next-fight population trajectory prior. It is
not written back into stored FSR. Evaluation age is capped to 21-40 so sparse
age tails cannot create polynomial explosions.

Striking power
--------------
Stored striking_power is intentionally non-degrading, so its ordinary delta
curve is structurally unsuitable. Instead fit a logistic power-event model:

    power_event ~ stored_power + opportunity + prior_fights + age polynomial

The age polynomial lives on log-odds scale. Divide those age coefficients by the
fitted stored-power coefficient to express the age effect in equivalent
striking-power FSR points. Degree is again chosen chronologically with the same
1% simplicity rule, using log loss.

Outputs
-------
- config/fsr_age_modifiers_candidate.yaml
- data/experimental/fsr_age_modifier_calibration/age_modifier_calibration_summary.csv
- data/experimental/fsr_age_modifier_calibration/power_age_model_metrics.csv

The active config is NOT overwritten unless --apply is explicitly supplied.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from scripts.experimental import fsr_age_modifiers


TRANSITIONS_PATH = Path(
    "data/experimental/population_fsr_delta_vs_age_all_traits/"
    "population_fsr_age_transitions.parquet"
)
POWER_ROWS_PATH = Path(
    "data/experimental/striking_power_age_effect/"
    "striking_power_age_effect_rows.csv"
)
ACTIVE_CONFIG_PATH = Path("config/fsr_age_modifiers.yaml")
CANDIDATE_CONFIG_PATH = Path("config/fsr_age_modifiers_candidate.yaml")
OUTPUT_DIR = Path("data/experimental/fsr_age_modifier_calibration")
SUMMARY_PATH = OUTPUT_DIR / "age_modifier_calibration_summary.csv"
POWER_METRICS_PATH = OUTPUT_DIR / "power_age_model_metrics.csv"

AGE_CENTER = 30.0
FIT_AGE_MIN = 21.0
FIT_AGE_MAX = 39.0
CONFIG_AGE_MIN = 21.0
CONFIG_AGE_MAX = 40.0
SIMPLICITY_TOLERANCE = 0.01  # lowest degree within 1% of best holdout score

RETAINED_ORDINARY = (
    "knockdown_resistance",
    "damage_durability",
    "control_imposition",
    "reversal_ability",
    "wrestling_conversion",
    "submission_conversion",
    "submission_resistance",
    "submission_pressure",
    "td_defense",
    "distance_striking_pressure",
    "distance_striking_defense",
    "adversity_resistance",
    "ground_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
)

LABEL = {1: "linear", 2: "quadratic", 3: "cubic"}


def _chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.sort(pd.to_datetime(frame["date"], errors="coerce").dropna().unique())
    if len(dates) < 5:
        raise RuntimeError("not enough dates for chronological holdout")
    idx = max(1, int(len(dates) * 0.80)) - 1
    split = pd.Timestamp(dates[idx])
    train = frame.loc[pd.to_datetime(frame["date"]) <= split].copy()
    test = frame.loc[pd.to_datetime(frame["date"]) > split].copy()
    if train.empty or test.empty:
        raise RuntimeError("empty chronological train/test split")
    return train, test, split


def _choose_simple_degree(scores: dict[int, float]) -> int:
    best = min(scores.values())
    threshold = best * (1.0 + SIMPLICITY_TOLERANCE)
    eligible = [degree for degree in (1, 2, 3) if scores[degree] <= threshold]
    return min(eligible)


def _centered_poly_coefficients(age: np.ndarray, y: np.ndarray, degree: int) -> dict[str, float]:
    x = np.asarray(age, dtype=float) - AGE_CENTER
    raw = np.polyfit(x, np.asarray(y, dtype=float), degree)
    # np.polyfit returns highest power first. Normalize to explicit YAML slots.
    slots = {"intercept": 0.0, "linear": 0.0, "quadratic": 0.0, "cubic": 0.0}
    power_to_name = {0: "intercept", 1: "linear", 2: "quadratic", 3: "cubic"}
    for coeff, power in zip(raw, range(degree, -1, -1)):
        slots[power_to_name[power]] = float(coeff)
    return slots


def _eval_coeff(coeff: dict[str, float], age: np.ndarray | float) -> np.ndarray:
    x = np.asarray(age, dtype=float) - AGE_CENTER
    return (
        float(coeff["intercept"])
        + float(coeff["linear"]) * x
        + float(coeff["quadratic"]) * x * x
        + float(coeff["cubic"]) * x * x * x
    )


def _ordinary_trait_calibration(transitions: pd.DataFrame, trait: str) -> dict[str, object]:
    frame = transitions.loc[transitions["trait"].eq(trait)].copy()
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["delta"] = pd.to_numeric(frame["delta"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["age", "delta", "date"])
    frame = frame.loc[frame["age"].between(FIT_AGE_MIN, FIT_AGE_MAX)].copy()
    if len(frame) < 100:
        raise RuntimeError(f"too few calibration transitions for {trait}: {len(frame)}")

    train, test, split = _chronological_split(frame)
    scores: dict[int, float] = {}
    for degree in (1, 2, 3):
        coeff = _centered_poly_coefficients(
            train["age"].to_numpy(float), train["delta"].to_numpy(float), degree
        )
        pred = _eval_coeff(coeff, test["age"].to_numpy(float))
        scores[degree] = float(np.sqrt(np.mean((test["delta"].to_numpy(float) - pred) ** 2)))

    degree = _choose_simple_degree(scores)
    coeff = _centered_poly_coefficients(
        frame["age"].to_numpy(float), frame["delta"].to_numpy(float), degree
    )

    grid = np.linspace(CONFIG_AGE_MIN, CONFIG_AGE_MAX, 500)
    grid_pred = _eval_coeff(coeff, grid)
    probes = {age: float(_eval_coeff(coeff, age)) for age in (25.0, 30.0, 35.0, 38.0, 40.0)}

    return {
        "trait": trait,
        "degree": degree,
        "label": LABEL[degree],
        "coefficients": coeff,
        "split_date": split,
        "train_rows": len(train),
        "test_rows": len(test),
        "rows": len(frame),
        "holdout_rmse": scores[degree],
        "best_holdout_rmse": min(scores.values()),
        "min_adjustment": float(np.min(grid_pred)),
        "max_adjustment": float(np.max(grid_pred)),
        "probes": probes,
    }


def _power_design(frame: pd.DataFrame, degree: int) -> tuple[np.ndarray, list[str]]:
    age_x = frame["age"].to_numpy(float) - AGE_CENTER
    columns = [
        frame["striking_power"].to_numpy(float),
        frame["sig_str_landed"].to_numpy(float),
        np.log1p(frame["prior_ufc_fights"].to_numpy(float)),
    ]
    names = ["stored_power", "r1_sig_landed", "log1p_prior_ufc_fights"]
    for power in range(1, degree + 1):
        columns.append(age_x ** power)
        names.append(f"age_x_pow_{power}")
    return np.column_stack(columns), names


def _fit_power_model(frame: pd.DataFrame, degree: int) -> tuple[LogisticRegression, list[str]]:
    X, names = _power_design(frame, degree)
    y = frame["power_event_int"].to_numpy(int)
    # Deliberately weak regularization so coefficient ratios retain interpretable scale.
    model = LogisticRegression(C=1000.0, max_iter=10000, solver="lbfgs")
    model.fit(X, y)
    return model, names


def _power_calibration(power_rows: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    frame = power_rows.copy()
    for col in ("age", "striking_power", "sig_str_landed", "prior_ufc_fights", "power_event_int"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.dropna(
        subset=["age", "striking_power", "sig_str_landed", "prior_ufc_fights", "power_event_int", "event_date"]
    )
    frame = frame.loc[frame["age"].between(FIT_AGE_MIN, FIT_AGE_MAX)].copy()
    frame = frame.rename(columns={"event_date": "date"})
    train, test, split = _chronological_split(frame)

    rows = []
    score: dict[int, float] = {}
    for degree in (1, 2, 3):
        model, names = _fit_power_model(train, degree)
        X_test, _ = _power_design(test, degree)
        p = model.predict_proba(X_test)[:, 1]
        loss = float(log_loss(test["power_event_int"].to_numpy(int), p, labels=[0, 1]))
        score[degree] = loss
        power_beta = float(model.coef_[0, names.index("stored_power")])
        rows.append({
            "degree": degree,
            "label": LABEL[degree],
            "split_date": split,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_logloss": loss,
            "stored_power_beta": power_beta,
        })

    degree = _choose_simple_degree(score)
    model, names = _fit_power_model(frame, degree)
    beta = {name: float(model.coef_[0, i]) for i, name in enumerate(names)}
    power_beta = beta["stored_power"]
    if abs(power_beta) < 1e-6:
        raise RuntimeError(f"stored striking-power coefficient is too small for FSR conversion: {power_beta}")
    if power_beta <= 0.0:
        raise RuntimeError(f"stored striking-power coefficient unexpectedly non-positive: {power_beta}")

    coeff = {"intercept": 0.0, "linear": 0.0, "quadratic": 0.0, "cubic": 0.0}
    for power in range(1, degree + 1):
        # Equivalent FSR-point shift: age logit contribution / beta_power.
        coeff[{1: "linear", 2: "quadratic", 3: "cubic"}[power]] = (
            beta[f"age_x_pow_{power}"] / power_beta
        )

    # Center age 30 at zero by construction. The model's intercept is not an age effect.
    grid = np.linspace(CONFIG_AGE_MIN, CONFIG_AGE_MAX, 500)
    pred = _eval_coeff(coeff, grid)
    probes = {age: float(_eval_coeff(coeff, age)) for age in (25.0, 30.0, 35.0, 38.0, 40.0)}

    result = {
        "trait": "striking_power",
        "degree": degree,
        "label": LABEL[degree],
        "coefficients": coeff,
        "split_date": split,
        "train_rows": len(train),
        "test_rows": len(test),
        "rows": len(frame),
        "holdout_logloss": score[degree],
        "best_holdout_logloss": min(score.values()),
        "stored_power_beta": power_beta,
        "min_adjustment": float(np.min(pred)),
        "max_adjustment": float(np.max(pred)),
        "probes": probes,
    }
    metrics = pd.DataFrame(rows).sort_values(["test_logloss", "degree"]).reset_index(drop=True)
    return result, metrics


def _update_rule(rule: dict, result: dict[str, object], *, source: str, model: str) -> None:
    rule["enabled"] = True
    rule["calibrated"] = True
    rule["status"] = "population_age_calibrated_candidate"
    rule["model"] = model
    rule["age_center"] = AGE_CENTER
    rule["coefficients"] = {k: float(v) for k, v in result["coefficients"].items()}
    # Below minimum age, cap to the age-21 value by using max_age/min_age handling
    # in the config contract. The generic evaluator currently returns 0 below min_age,
    # so ordinary use should occur in the supported UFC age window; max tail is capped.
    rule["max_age"] = CONFIG_AGE_MAX
    rule.pop("min_age", None)
    rule["min_adjustment"] = float(result["min_adjustment"])
    rule["max_adjustment"] = float(result["max_adjustment"])
    rule["source"] = source
    rule["fit_degree"] = int(result["degree"])
    rule["calibration_rows"] = int(result["rows"])
    rule["notes"] = (
        f"Centered at age {AGE_CENTER:.0f}; {result['label']} selected with chronological holdout "
        f"and 1% simplicity tolerance. Fit on ages {FIT_AGE_MIN:.0f}-{FIT_AGE_MAX:.0f}."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="overwrite active config after writing candidate")
    args = p.parse_args()

    if not TRANSITIONS_PATH.exists():
        raise FileNotFoundError(f"missing transition artifact: {TRANSITIONS_PATH}")
    if not POWER_ROWS_PATH.exists():
        raise FileNotFoundError(f"missing striking-power age rows: {POWER_ROWS_PATH}")
    if not ACTIVE_CONFIG_PATH.exists():
        raise FileNotFoundError(ACTIVE_CONFIG_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[age-cal] loading transitions: {TRANSITIONS_PATH}", flush=True)
    transitions = pd.read_parquet(TRANSITIONS_PATH)
    print(f"[age-cal] transition rows={len(transitions):,}", flush=True)

    with ACTIVE_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        active_cfg = yaml.safe_load(fh)
    candidate = deepcopy(active_cfg)
    candidate["name"] = "fsr_age_modifiers_population_candidate_v1"
    candidate["calibration"] = {
        "age_center": AGE_CENTER,
        "fit_age_min": FIT_AGE_MIN,
        "fit_age_max": FIT_AGE_MAX,
        "configured_age_max": CONFIG_AGE_MAX,
        "degree_selection": "simplest degree within 1% of best chronological holdout score",
        "ordinary_target": "next_prefight_fsr_minus_current_prefight_fsr",
        "stored_fsr_mutated": False,
    }

    summary_rows: list[dict[str, object]] = []
    for i, trait in enumerate(RETAINED_ORDINARY, start=1):
        print(f"[age-cal] ordinary trait {i:02d}/{len(RETAINED_ORDINARY)}: {trait}", flush=True)
        result = _ordinary_trait_calibration(transitions, trait)
        rule = candidate["traits"][trait]
        _update_rule(
            rule,
            result,
            source="population_next_fight_delta_vs_age_v1",
            model="polynomial",
        )
        probes = result["probes"]
        summary_rows.append({
            "trait": trait,
            "model": result["label"],
            "rows": result["rows"],
            "holdout_metric": result["holdout_rmse"],
            "metric_name": "rmse",
            "modifier_age_25": probes[25.0],
            "modifier_age_30": probes[30.0],
            "modifier_age_35": probes[35.0],
            "modifier_age_38": probes[38.0],
            "modifier_age_40": probes[40.0],
        })

    print(f"[age-cal] loading striking-power rows: {POWER_ROWS_PATH}", flush=True)
    power_rows = pd.read_csv(POWER_ROWS_PATH)
    power_result, power_metrics = _power_calibration(power_rows)
    _update_rule(
        candidate["traits"]["striking_power"],
        power_result,
        source="controlled_power_event_logit_equivalent_fsr_v1",
        model="power_residual_polynomial",
    )
    candidate["traits"]["striking_power"]["stored_power_beta"] = float(power_result["stored_power_beta"])
    candidate["traits"]["striking_power"]["notes"] += (
        " Age coefficients are logistic-event logit coefficients divided by the stored-power "
        "logit coefficient, yielding equivalent striking-power FSR points."
    )
    pprobe = power_result["probes"]
    summary_rows.append({
        "trait": "striking_power",
        "model": power_result["label"],
        "rows": power_result["rows"],
        "holdout_metric": power_result["holdout_logloss"],
        "metric_name": "logloss",
        "modifier_age_25": pprobe[25.0],
        "modifier_age_30": pprobe[30.0],
        "modifier_age_35": pprobe[35.0],
        "modifier_age_38": pprobe[38.0],
        "modifier_age_40": pprobe[40.0],
    })

    # Weak/rejected traits remain exactly as configured in the active file.
    with CANDIDATE_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(candidate, fh, sort_keys=False, width=110)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    power_metrics.to_csv(POWER_METRICS_PATH, index=False)

    print("\n" + "=" * 150)
    print("FSR AGE MODIFIER CALIBRATION CANDIDATE")
    print("=" * 150)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"\nwrote: {CANDIDATE_CONFIG_PATH}", flush=True)
    print(f"wrote: {SUMMARY_PATH}", flush=True)
    print(f"wrote: {POWER_METRICS_PATH}", flush=True)

    if args.apply:
        with ACTIVE_CONFIG_PATH.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate, fh, sort_keys=False, width=110)
        fsr_age_modifiers.load_age_modifier_config.cache_clear()
        print(f"APPLIED candidate -> {ACTIVE_CONFIG_PATH}", flush=True)
    else:
        print("NOT APPLIED. Review candidate first; rerun with --apply only after approval.", flush=True)


if __name__ == "__main__":
    main()
