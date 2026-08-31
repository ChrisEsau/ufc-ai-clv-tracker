"""Search shadow age-decay curves for KD resistance and damage durability.

Purpose
-------
Determine whether a single age-adjusted *effective* FSR trait can recover the
predictive gain previously obtained by adding age explicitly to controlled
historical models.

This is intentionally designed around simulator use:

    historical FSR -> deterministic age adjustment -> effective simulator trait

The stored leakage-safe FSR is never modified.  Candidate transformations are
pre-specified and evaluated out-of-fold on the same 2020+ mature-fighter cohort.

For KD absorption, all models also control for opponent striking power and
current-fight significant-strike exposure.  For KO/TKO loss, all models also
control for opponent striking power and current-fight damage exposure.  Those
current-fight exposure fields are research-only controls and are NOT proposed
as simulator inputs.

Candidate curve families
------------------------
1. No decay.
2. Linear hinge: trait - slope * max(age - onset, 0).
3. Quadratic hinge: trait - slope * max(age - onset, 0)^2 / 5.
4. Two-stage hinge: modest decline after onset plus extra decline after age 37.

The script also prints the effective Rodriguez ratings at age 39.58 for the
best candidate as a concrete sanity check; this does not change any artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as controlled

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "age_decay_curve_search_kd_durability_2020plus_mature.csv"
)

CV_SEED = 20260810
N_SPLITS = 5
N_REPEATS = 5

# Candidate search space is intentionally compact and interpretable.  We are
# not tuning hundreds of values against one cohort.
LINEAR_ONSETS = (30.0, 32.0, 34.0, 35.0)
LINEAR_SLOPES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
QUADRATIC_ONSETS = (30.0, 32.0, 34.0, 35.0)
QUADRATIC_SLOPES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
TWO_STAGE_ONSETS = (32.0, 34.0, 35.0)
TWO_STAGE_BASE_SLOPES = (0.25, 0.50, 0.75, 1.00)
TWO_STAGE_EXTRA_SLOPES = (0.25, 0.50, 0.75, 1.00)

RODRIGUEZ_AGE = 39.58
RODRIGUEZ_KD_RESISTANCE = 62.533
RODRIGUEZ_DURABILITY = 70.616


@dataclass(frozen=True)
class Curve:
    family: str
    onset: float = 0.0
    slope: float = 0.0
    extra_slope: float = 0.0

    @property
    def label(self) -> str:
        if self.family == "none":
            return "none"
        if self.family == "linear":
            return f"linear_on{self.onset:g}_s{self.slope:g}"
        if self.family == "quadratic":
            return f"quadratic_on{self.onset:g}_s{self.slope:g}"
        if self.family == "two_stage":
            return (
                f"two_stage_on{self.onset:g}_s{self.slope:g}_"
                f"extra37_{self.extra_slope:g}"
            )
        raise ValueError(f"unknown curve family: {self.family}")


def _curves() -> list[Curve]:
    out = [Curve("none")]
    for onset in LINEAR_ONSETS:
        for slope in LINEAR_SLOPES:
            out.append(Curve("linear", onset=onset, slope=slope))
    for onset in QUADRATIC_ONSETS:
        for slope in QUADRATIC_SLOPES:
            out.append(Curve("quadratic", onset=onset, slope=slope))
    for onset in TWO_STAGE_ONSETS:
        for slope in TWO_STAGE_BASE_SLOPES:
            for extra in TWO_STAGE_EXTRA_SLOPES:
                out.append(
                    Curve(
                        "two_stage",
                        onset=onset,
                        slope=slope,
                        extra_slope=extra,
                    )
                )
    return out


def _apply_curve(trait: pd.Series, age: pd.Series, curve: Curve) -> pd.Series:
    trait = pd.to_numeric(trait, errors="coerce").astype(float)
    age = pd.to_numeric(age, errors="coerce").astype(float)

    if curve.family == "none":
        penalty = np.zeros(len(trait), dtype=float)
    elif curve.family == "linear":
        penalty = curve.slope * np.maximum(age - curve.onset, 0.0)
    elif curve.family == "quadratic":
        years = np.maximum(age - curve.onset, 0.0)
        # Divide by five to keep slope values interpretable near the onset while
        # allowing acceleration later in a fighter's career.
        penalty = curve.slope * (years ** 2) / 5.0
    elif curve.family == "two_stage":
        penalty = (
            curve.slope * np.maximum(age - curve.onset, 0.0)
            + curve.extra_slope * np.maximum(age - 37.0, 0.0)
        )
    else:
        raise ValueError(f"unknown curve family: {curve.family}")

    # Keep the effective value inside the established FSR rating contract.
    return pd.Series(np.clip(trait.to_numpy() - penalty, 10.0, 90.0), index=trait.index)


def _oof_metrics(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, float]:
    work = frame[[target] + features].dropna(subset=[target]).copy()
    y = work[target].astype(int).to_numpy()
    X = work[features]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")),
        ]
    )
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=CV_SEED,
    )
    pred_sum = np.zeros(len(work), dtype=float)
    pred_count = np.zeros(len(work), dtype=int)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred_sum[test_idx] += pred
        pred_count[test_idx] += 1

    if (pred_count == 0).any():
        raise RuntimeError("OOF prediction coverage failure")
    p = pred_sum / pred_count
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def _explicit_age_benchmark(frame: pd.DataFrame, target_kind: str) -> dict[str, float]:
    work = frame.copy()
    work["age_over_35"] = np.maximum(work["age"] - 35.0, 0.0)
    if target_kind == "kd":
        return _oof_metrics(
            work,
            "any_kd_absorbed",
            [
                "knockdown_resistance",
                "opponent_striking_power",
                "sig_absorbed",
                "age",
                "age_over_35",
            ],
        )
    if target_kind == "durability":
        return _oof_metrics(
            work,
            "ko_tko_loss",
            [
                "damage_durability",
                "opponent_striking_power",
                "damage_exposure",
                "age",
                "age_over_35",
            ],
        )
    raise ValueError(target_kind)


def _evaluate_trait(frame: pd.DataFrame, target_kind: str) -> pd.DataFrame:
    if target_kind == "kd":
        trait = "knockdown_resistance"
        target = "any_kd_absorbed"
        context = ["opponent_striking_power", "sig_absorbed"]
    elif target_kind == "durability":
        trait = "damage_durability"
        target = "ko_tko_loss"
        context = ["opponent_striking_power", "damage_exposure"]
    else:
        raise ValueError(target_kind)

    rows: list[dict[str, object]] = []
    curves = _curves()
    for idx, curve in enumerate(curves, start=1):
        work = frame.copy()
        work["effective_trait"] = _apply_curve(work[trait], work["age"], curve)
        metrics = _oof_metrics(work, target, ["effective_trait"] + context)
        rows.append(
            {
                "trait": trait,
                "target": target,
                "curve": curve.label,
                "family": curve.family,
                "onset": curve.onset if curve.family != "none" else np.nan,
                "slope": curve.slope if curve.family != "none" else np.nan,
                "extra_slope": curve.extra_slope if curve.family == "two_stage" else np.nan,
                **metrics,
            }
        )
        if idx % 25 == 0 or idx == len(curves):
            print(
                f"[age curve search] {trait}: {idx:,}/{len(curves):,} candidates",
                flush=True,
            )
    return pd.DataFrame(rows)


def _print_ranking(results: pd.DataFrame, trait: str, benchmark: dict[str, float]) -> None:
    subset = results.loc[results["trait"].eq(trait)].copy()
    baseline = subset.loc[subset["family"].eq("none")].iloc[0]
    subset["auc_gain_vs_none"] = subset["auc"] - float(baseline["auc"])
    subset["logloss_gain_vs_none"] = float(baseline["logloss"]) - subset["logloss"]
    subset["brier_gain_vs_none"] = float(baseline["brier"]) - subset["brier"]
    subset = subset.sort_values(["logloss", "brier", "auc"], ascending=[True, True, False])

    print("\n" + "=" * 122)
    print(f"{trait.upper()} — TOP AGE-ADJUSTED EFFECTIVE-TRAIT CURVES")
    print("=" * 122)
    cols = [
        "curve", "family", "onset", "slope", "extra_slope",
        "auc", "logloss", "brier", "auc_gain_vs_none",
        "logloss_gain_vs_none", "brier_gain_vs_none",
    ]
    print(subset[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nBASELINE / EXPLICIT-AGE BENCHMARK")
    print(
        f"no-decay effective trait: AUC={baseline['auc']:.5f}  "
        f"logloss={baseline['logloss']:.5f}  brier={baseline['brier']:.5f}"
    )
    print(
        f"trait + explicit age:    AUC={benchmark['auc']:.5f}  "
        f"logloss={benchmark['logloss']:.5f}  brier={benchmark['brier']:.5f}"
    )

    best = subset.iloc[0]
    curve = Curve(
        family=str(best["family"]),
        onset=0.0 if pd.isna(best["onset"]) else float(best["onset"]),
        slope=0.0 if pd.isna(best["slope"]) else float(best["slope"]),
        extra_slope=0.0 if pd.isna(best["extra_slope"]) else float(best["extra_slope"]),
    )
    if trait == "knockdown_resistance":
        original = RODRIGUEZ_KD_RESISTANCE
    else:
        original = RODRIGUEZ_DURABILITY
    effective = float(_apply_curve(
        pd.Series([original]), pd.Series([RODRIGUEZ_AGE]), curve
    ).iloc[0])
    print(
        f"Rodriguez sanity check at age {RODRIGUEZ_AGE:.2f}: "
        f"historical={original:.3f} -> candidate effective={effective:.3f} "
        f"using {curve.label}"
    )


def main() -> None:
    frame = controlled._prepare_frame()
    print("\n" + "=" * 122)
    print("AGE-DECAY CURVE SEARCH — KD RESISTANCE / DAMAGE DURABILITY")
    print("=" * 122)
    print(f"fighter-side rows: {len(frame):,}")
    print(f"candidate curves per trait: {len(_curves()):,}")
    print("Stored FSR ratings remain unchanged; only shadow effective traits are evaluated.")

    kd = _evaluate_trait(frame, "kd")
    dur = _evaluate_trait(frame, "durability")
    results = pd.concat([kd, dur], ignore_index=True)

    kd_benchmark = _explicit_age_benchmark(frame, "kd")
    dur_benchmark = _explicit_age_benchmark(frame, "durability")
    _print_ranking(results, "knockdown_resistance", kd_benchmark)
    _print_ranking(results, "damage_durability", dur_benchmark)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(results):,} curve-evaluation rows to {OUTPUT_PATH}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
