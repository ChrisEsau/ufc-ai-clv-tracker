"""Decompose which pre-fight FSR traits and interactions drive R1 KO signal.

This diagnostic uses the established 2020+ mature-fighter cohort and the exact
symmetric E_all_traits_interactions feature construction from the joint-signal
study. It does not change FSR values or simulator constants.

Outputs
-------
1. Baseline repeated out-of-fold AUC for the full E bundle.
2. Family-only AUC: how much R1-KO signal each mechanic family carries alone.
3. Leave-one-family-out AUC: how much unique signal is lost when a family is removed.
4. Standardized coefficient stability across repeated CV folds for individual
   features and explicit interactions.

The purpose is to convert statistical signal into simulator design guidance:
we want the simulator mechanics to reproduce the same matchup interactions,
not inject a black-box R1-KO probability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv as joint
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_r1_ko_signal_contribution_decomposition_2020plus_mature.parquet"
)
DEFAULT_SPLITS = 5
DEFAULT_REPEATS = 10
DEFAULT_SEED = 20260810


def _family_for_feature(name: str) -> str:
    """Map exact E-bundle columns to simulator-relevant mechanic families."""
    lower = name.lower()
    if lower.startswith("interaction__"):
        return "explicit_interactions"
    if "power_minus_kd_resistance" in lower:
        return "power_x_kd_resistance"
    if "power_minus_durability" in lower:
        return "power_x_durability"
    if "distance_pressure_minus_defense" in lower or "distance_precision_minus_defense" in lower:
        return "distance_offense_x_defense"
    if "clinch_pressure_minus_defense" in lower or "clinch_precision_minus_defense" in lower:
        return "clinch_offense_x_defense"
    if "ground_pressure_minus_defense" in lower or "ground_precision_minus_defense" in lower:
        return "ground_offense_x_defense"

    # Raw symmetric trait summaries.
    if "trait_" in lower:
        if "striking_power" in lower:
            return "raw_power"
        if "knockdown_resistance" in lower:
            return "raw_kd_resistance"
        if "damage_durability" in lower:
            return "raw_durability"
        if "distance_" in lower:
            return "raw_distance"
        if "clinch_" in lower:
            return "raw_clinch"
        if "ground_" in lower:
            return "raw_ground"
    return "other"


def _oof_auc(frame: pd.DataFrame, features: list[str], y: np.ndarray, *, splits: int, repeats: int, seed: int) -> float:
    if not features:
        return float("nan")
    pred, _, _ = joint._repeated_oof(
        frame[features],
        y,
        n_splits=splits,
        n_repeats=repeats,
        seed=seed,
        label=f"R1 contribution ({len(features)} features)",
    )
    return float(roc_auc_score(y, pred))


def _coefficient_stability(
    frame: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    *,
    splits: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Fit the same standardized regularized learner and summarize fold coefficients."""
    cv = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=repeats,
        random_state=seed,
    )
    values: dict[str, list[float]] = defaultdict(list)
    total = splits * repeats
    for fold_no, (train_idx, _) in enumerate(cv.split(frame[features], y), start=1):
        model = joint._model()
        model.fit(frame.iloc[train_idx][features], y[train_idx])
        coef = model.named_steps["model"].coef_[0]
        for feature, value in zip(features, coef):
            values[feature].append(float(value))
        if fold_no % 10 == 0 or fold_no == total:
            print(f"[R1 contribution] coefficient folds {fold_no}/{total}", flush=True)

    rows: list[dict[str, object]] = []
    for feature, coeffs in values.items():
        arr = np.asarray(coeffs, dtype=float)
        mean = float(arr.mean())
        positive_rate = float((arr > 0).mean())
        negative_rate = float((arr < 0).mean())
        sign_consistency = max(positive_rate, negative_rate)
        rows.append(
            {
                "record_type": "feature_coefficient",
                "name": feature,
                "family": _family_for_feature(feature),
                "feature_count": 1,
                "auc": np.nan,
                "auc_delta_vs_full": np.nan,
                "coef_mean": mean,
                "coef_abs_mean": float(np.abs(arr).mean()),
                "coef_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "coef_sign_consistency": sign_consistency,
                "coef_positive_rate": positive_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["coef_sign_consistency", "coef_abs_mean"], ascending=[False, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose modern mature-fighter R1-KO FSR signal")
    parser.add_argument("--master", type=Path, default=modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=modern.FSR_PATH)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    master = modern._load_master(args.master)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(args.fsr_path, cohort)
    frame = modern._build_joint_frame(cohort, pairs)
    bundles = joint._feature_bundles(frame)
    full_features = bundles["E_all_traits_interactions"]
    y = frame["actual_r1_ko"].to_numpy(dtype=int)

    print(
        f"[R1 contribution] bouts={len(frame):,}; R1_KO={int(y.sum()):,} ({y.mean():.2%}); "
        f"features={len(full_features)}; folds={args.splits * args.repeats}",
        flush=True,
    )

    family_map: dict[str, list[str]] = defaultdict(list)
    for feature in full_features:
        family_map[_family_for_feature(feature)].append(feature)

    print("\nFEATURE FAMILIES")
    for family, features in sorted(family_map.items()):
        print(f"{family}: {len(features)}")

    full_auc = _oof_auc(
        frame, full_features, y,
        splits=args.splits, repeats=args.repeats, seed=args.seed,
    )
    print(f"\nFULL E-BUNDLE R1-KO OOF AUC: {full_auc:.4f}")

    summary_rows: list[dict[str, object]] = [
        {
            "record_type": "bundle",
            "name": "FULL_E_all_traits_interactions",
            "family": "all",
            "feature_count": len(full_features),
            "auc": full_auc,
            "auc_delta_vs_full": 0.0,
            "coef_mean": np.nan,
            "coef_abs_mean": np.nan,
            "coef_std": np.nan,
            "coef_sign_consistency": np.nan,
            "coef_positive_rate": np.nan,
        }
    ]

    print("\nFAMILY-ONLY AND LEAVE-ONE-FAMILY-OUT R1-KO AUC")
    display_rows: list[dict[str, object]] = []
    for family, features in sorted(family_map.items()):
        only_auc = _oof_auc(
            frame, features, y,
            splits=args.splits, repeats=args.repeats, seed=args.seed + 101,
        )
        without = [f for f in full_features if f not in set(features)]
        without_auc = _oof_auc(
            frame, without, y,
            splits=args.splits, repeats=args.repeats, seed=args.seed + 202,
        )
        unique_loss = full_auc - without_auc
        display_rows.append(
            {
                "family": family,
                "features": len(features),
                "family_only_auc": only_auc,
                "full_without_family_auc": without_auc,
                "unique_auc_loss": unique_loss,
            }
        )
        summary_rows.extend(
            [
                {
                    "record_type": "family_only",
                    "name": family,
                    "family": family,
                    "feature_count": len(features),
                    "auc": only_auc,
                    "auc_delta_vs_full": only_auc - full_auc,
                    "coef_mean": np.nan,
                    "coef_abs_mean": np.nan,
                    "coef_std": np.nan,
                    "coef_sign_consistency": np.nan,
                    "coef_positive_rate": np.nan,
                },
                {
                    "record_type": "family_ablation",
                    "name": family,
                    "family": family,
                    "feature_count": len(without),
                    "auc": without_auc,
                    "auc_delta_vs_full": without_auc - full_auc,
                    "coef_mean": np.nan,
                    "coef_abs_mean": np.nan,
                    "coef_std": np.nan,
                    "coef_sign_consistency": np.nan,
                    "coef_positive_rate": np.nan,
                },
            ]
        )

    display = pd.DataFrame(display_rows).sort_values("unique_auc_loss", ascending=False)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    coef = _coefficient_stability(
        frame, full_features, y,
        splits=args.splits, repeats=args.repeats, seed=args.seed + 303,
    )

    interactions = coef[coef["family"].eq("explicit_interactions")].copy()
    interactions = interactions.sort_values("coef_abs_mean", ascending=False)
    print("\nEXPLICIT INTERACTION COEFFICIENT STABILITY")
    if interactions.empty:
        print("No explicit interaction columns found.")
    else:
        print(
            interactions[
                ["name", "coef_mean", "coef_abs_mean", "coef_std", "coef_sign_consistency", "coef_positive_rate"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    stable = coef[coef["coef_sign_consistency"].ge(0.80)].copy()
    stable = stable.sort_values("coef_abs_mean", ascending=False).head(25)
    print("\nTOP STABLE FEATURES (>=80% SAME SIGN ACROSS FOLDS)")
    print(
        stable[
            ["name", "family", "coef_mean", "coef_abs_mean", "coef_std", "coef_sign_consistency"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    out = pd.concat([pd.DataFrame(summary_rows), coef], ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\n[R1 contribution] wrote {args.output}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
