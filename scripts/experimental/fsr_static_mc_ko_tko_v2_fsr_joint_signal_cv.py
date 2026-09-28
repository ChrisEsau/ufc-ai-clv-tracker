"""Cross-validated diagnostic for joint KO/TKO signal in pre-fight FSR traits.

Purpose
-------
Individual KO-relevant FSR edges showed only modest standalone discrimination.
This study asks whether the *combination* of leakage-safe pre-fight FSR traits
contains stronger KO/TKO or Round-1 KO/TKO signal than any one edge alone.

This is deliberately not a production model. It uses a fixed, regularized
logistic regression with repeated stratified cross-validation and no parameter
tuning. Every reported prediction is out-of-fold.

Feature bundles
---------------
A_power_kd:
    Power vs knockdown-resistance danger/separation only.
B_power_kd_durability:
    A plus power vs damage-durability danger/separation.
C_distance_striking:
    Distance pressure/precision vs defense danger/separation.
D_all_finish_edges:
    All eight symmetric KO-relevant matchup edge families.
E_all_traits_interactions:
    D plus symmetric summaries of all available KO-relevant raw FSR traits and
    a small set of explicit interaction terms among the strongest finish edges.

All occurrence features are symmetric with respect to red/blue corner. The
actual winner is never used to orient a feature. Draws/no-contests can therefore
remain in the negative outcome cohort.

No simulator constants or FSR values are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_kd_audit as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_historical_300_actual_validation as validation
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_signal_diagnostic as signal


VALIDATION_PATH = validation.OUTPUT_PATH
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv.parquet"
)
DEFAULT_SPLITS = 5
DEFAULT_REPEATS = 20
DEFAULT_SEED = 20260810

# Raw KO-relevant FSR traits. Only traits present in the current artifact are used.
KO_TRAITS = (
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "distance_striking_pressure",
    "distance_precision",
    "distance_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
)


def _numeric(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _load_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical KO validation artifact not found: {path}. "
            "Run the 300-bout KO validation first."
        )
    frame = pd.read_parquet(path).copy()
    required = {"bout_id", "actual_ko_tko", "actual_finish_round", "mc_p_ko_tko"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Historical KO validation missing required columns: {missing}")
    frame["bout_id"] = frame["bout_id"].astype(str)
    frame["actual_ko_tko"] = frame["actual_ko_tko"].astype(int)
    frame["actual_finish_round"] = pd.to_numeric(frame["actual_finish_round"], errors="coerce")
    frame["actual_r1_ko"] = (
        frame["actual_ko_tko"].eq(1) & frame["actual_finish_round"].eq(1)
    ).astype(int)
    return frame


def _load_pairs(path: Path, bout_ids: set[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    frame = pd.read_parquet(path)
    bout_key = hist._resolve_bout_key(frame, None)
    frame[bout_key] = frame[bout_key].astype(str)
    frame = frame[frame[bout_key].isin(bout_ids)].copy()
    bouts, _ = hist._prepare_historical_bouts(frame, bout_key=bout_key)
    pairs = {str(bout_id): (red, blue) for bout_id, red, blue in bouts}
    missing = sorted(bout_ids - set(pairs))
    if missing:
        raise ValueError(f"Missing leakage-safe FSR pairs for {len(missing)} bouts; first={missing[:10]}")
    return pairs


def _build_features(
    validation_frame: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, bout in validation_frame.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        row: dict[str, object] = {
            "bout_id": bout_id,
            "actual_ko_tko": int(bout["actual_ko_tko"]),
            "actual_r1_ko": int(bout["actual_r1_ko"]),
            "mc_p_ko_tko": float(bout["mc_p_ko_tko"]),
        }

        # Symmetric raw-trait summaries: these do not reveal which corner won.
        for trait in KO_TRAITS:
            if trait not in red.index or trait not in blue.index:
                continue
            rv = _numeric(red, trait)
            bv = _numeric(blue, trait)
            row[f"trait_max_{trait}"] = max(rv, bv)
            row[f"trait_min_{trait}"] = min(rv, bv)
            row[f"trait_absdiff_{trait}"] = abs(rv - bv)
            row[f"trait_mean_{trait}"] = 0.5 * (rv + bv)

        # Symmetric attacker-vs-defender edge summaries.
        for edge_name, attacker_trait, defender_trait in signal.EDGE_SPECS:
            if (
                attacker_trait not in red.index
                or attacker_trait not in blue.index
                or defender_trait not in red.index
                or defender_trait not in blue.index
            ):
                continue
            red_to_blue = _numeric(red, attacker_trait) - _numeric(blue, defender_trait)
            blue_to_red = _numeric(blue, attacker_trait) - _numeric(red, defender_trait)
            row[f"danger_{edge_name}"] = max(red_to_blue, blue_to_red)
            row[f"separation_{edge_name}"] = abs(red_to_blue - blue_to_red)

        rows.append(row)

    frame = pd.DataFrame(rows)

    # A few explicit, pre-declared interactions. These let a simple linear model
    # detect combinations such as high distance pressure + poor durability.
    interaction_pairs = (
        ("danger_power_minus_durability", "danger_distance_pressure_minus_defense"),
        ("danger_power_minus_durability", "danger_distance_precision_minus_defense"),
        ("danger_power_minus_kd_resistance", "danger_distance_pressure_minus_defense"),
        ("danger_power_minus_kd_resistance", "danger_distance_precision_minus_defense"),
        ("danger_distance_pressure_minus_defense", "danger_distance_precision_minus_defense"),
    )
    for left, right in interaction_pairs:
        if left in frame.columns and right in frame.columns:
            frame[f"interaction__{left}__x__{right}"] = frame[left] * frame[right]

    return frame


def _feature_bundles(frame: pd.DataFrame) -> dict[str, list[str]]:
    def existing(cols: list[str]) -> list[str]:
        return [c for c in cols if c in frame.columns]

    a = existing([
        "danger_power_minus_kd_resistance",
        "separation_power_minus_kd_resistance",
    ])
    b = existing(a + [
        "danger_power_minus_durability",
        "separation_power_minus_durability",
    ])
    c = existing([
        "danger_distance_pressure_minus_defense",
        "separation_distance_pressure_minus_defense",
        "danger_distance_precision_minus_defense",
        "separation_distance_precision_minus_defense",
    ])

    all_edge_cols = sorted(
        c for c in frame.columns if c.startswith("danger_") or c.startswith("separation_")
    )
    all_trait_cols = sorted(c for c in frame.columns if c.startswith("trait_"))
    interaction_cols = sorted(c for c in frame.columns if c.startswith("interaction__"))

    bundles = {
        "A_power_kd": a,
        "B_power_kd_durability": b,
        "C_distance_striking": c,
        "D_all_finish_edges": all_edge_cols,
        "E_all_traits_interactions": sorted(set(all_edge_cols + all_trait_cols + interaction_cols)),
    }
    empty = [name for name, cols in bundles.items() if not cols]
    if empty:
        raise ValueError(f"Feature bundles unexpectedly empty: {empty}")
    return bundles


def _model() -> Pipeline:
    # Fixed model/no tuning: the purpose is measuring available signal, not finding
    # the best production learner for this small 300-bout cohort.
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")),
        ]
    )


def _repeated_oof(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    label: str,
) -> tuple[np.ndarray, list[float], list[float]]:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    prediction_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    fold_auc: list[float] = []
    fold_ap: list[float] = []
    total_folds = n_splits * n_repeats

    for fold_no, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        model = _model()
        model.fit(x.iloc[train_idx], y[train_idx])
        pred = model.predict_proba(x.iloc[test_idx])[:, 1]
        prediction_sum[test_idx] += pred
        prediction_count[test_idx] += 1

        y_test = y[test_idx]
        if len(np.unique(y_test)) == 2:
            fold_auc.append(float(roc_auc_score(y_test, pred)))
            fold_ap.append(float(average_precision_score(y_test, pred)))

        if fold_no % 20 == 0 or fold_no == total_folds:
            print(f"[joint FSR CV] {label}: folds {fold_no}/{total_folds}", flush=True)

    if np.any(prediction_count == 0):
        raise RuntimeError("Repeated CV left at least one bout without an out-of-fold prediction.")
    return prediction_sum / prediction_count, fold_auc, fold_ap


def _evaluate_bundle(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    bundle_name: str,
    target_col: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    y = frame[target_col].to_numpy(dtype=int)
    x = frame[feature_cols].astype(float)
    pred, fold_auc, fold_ap = _repeated_oof(
        x,
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        label=f"{target_col}/{bundle_name}",
    )

    row: dict[str, object] = {
        "target": target_col,
        "bundle": bundle_name,
        "features": len(feature_cols),
        "positive_bouts": int(y.sum()),
        "prevalence": float(y.mean()),
        "oof_auc": float(roc_auc_score(y, pred)),
        "oof_average_precision": float(average_precision_score(y, pred)),
        "oof_brier": float(brier_score_loss(y, pred)),
        "fold_auc_mean": float(np.mean(fold_auc)),
        "fold_auc_std": float(np.std(fold_auc, ddof=1)),
        "fold_ap_mean": float(np.mean(fold_ap)),
        "fold_ap_std": float(np.std(fold_ap, ddof=1)),
    }
    return row, pred


def _print_summary(results: pd.DataFrame, frame: pd.DataFrame) -> None:
    print("\n" + "=" * 128)
    print("JOINT KO-RELEVANT FSR SIGNAL — REPEATED OUT-OF-FOLD CROSS-VALIDATION")
    print("=" * 128)
    print(f"bouts: {len(frame):,}")
    print(f"actual KO/TKO: {int(frame['actual_ko_tko'].sum()):,} ({frame['actual_ko_tko'].mean():.2%})")
    print(f"actual R1 KO/TKO: {int(frame['actual_r1_ko'].sum()):,} ({frame['actual_r1_ko'].mean():.2%})")
    print("fixed learner: standardized regularized logistic regression; no hyperparameter tuning")
    print("all reported probabilities are repeated out-of-fold predictions")

    display = [
        "target", "bundle", "features", "positive_bouts", "prevalence",
        "oof_auc", "oof_average_precision", "oof_brier", "fold_auc_mean", "fold_auc_std",
    ]
    print("\nCROSS-VALIDATED SIGNAL BY FEATURE BUNDLE")
    print(results[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compare against the already-existing simulator p(KO) on the same cohort.
    print("\nREFERENCE — EXISTING STRONG-CANDIDATE MC P(KO)")
    print(
        f"any KO ROC-AUC: {roc_auc_score(frame['actual_ko_tko'], frame['mc_p_ko_tko']):.4f}"
    )
    print(
        f"R1 KO ROC-AUC using overall MC p(KO) only: "
        f"{roc_auc_score(frame['actual_r1_ko'], frame['mc_p_ko_tko']):.4f}"
    )

    print("\nPLAIN-ENGLISH DECISION GUIDE")
    print("- If A/B stay near 0.50 but C/D/E improve, KO information exists mostly in combinations/style matchup rather than power alone.")
    print("- If D/E reach roughly 0.65+ out of fold, the FSR family contains useful joint finish signal and simulator translation becomes the larger suspect.")
    print("- If D/E remain around 0.55-0.60, the current KO-relevant FSR family itself is missing substantial finish information.")
    print("- If E jumps far above D, interactions/raw trait context matter; simple edge formulas are throwing away useful information.")
    print("- This is a 300-bout research cohort, so fold-to-fold spread matters alongside the headline AUC.")
    print("- No FSR values or simulator constants are changed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate joint KO signal in leakage-safe FSR traits")
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation_frame = _load_validation(args.validation)
    bout_ids = set(validation_frame["bout_id"].astype(str))
    print(f"[joint FSR CV] validation bouts={len(validation_frame):,}", flush=True)
    pairs = _load_pairs(args.fsr_path, bout_ids)
    print(f"[joint FSR CV] matched leakage-safe FSR pairs={len(pairs):,}", flush=True)

    frame = _build_features(validation_frame, pairs)
    bundles = _feature_bundles(frame)
    print(
        "[joint FSR CV] bundles="
        + ", ".join(f"{name}:{len(cols)}" for name, cols in bundles.items()),
        flush=True,
    )
    print(
        f"[joint FSR CV] repeated CV={args.splits} folds x {args.repeats} repeats; "
        f"fits={len(bundles) * 2 * args.splits * args.repeats:,}",
        flush=True,
    )

    result_rows: list[dict[str, object]] = []
    prediction_cols: dict[str, np.ndarray] = {}
    targets = ("actual_ko_tko", "actual_r1_ko")
    for target_index, target in enumerate(targets):
        for bundle_index, (bundle_name, feature_cols) in enumerate(bundles.items()):
            row, pred = _evaluate_bundle(
                frame,
                feature_cols,
                bundle_name=bundle_name,
                target_col=target,
                n_splits=args.splits,
                n_repeats=args.repeats,
                seed=args.seed + target_index * 1000 + bundle_index,
            )
            result_rows.append(row)
            prediction_cols[f"oof_{target}__{bundle_name}"] = pred

    results = pd.DataFrame(result_rows)
    artifact = frame[["bout_id", "actual_ko_tko", "actual_r1_ko", "mc_p_ko_tko"]].copy()
    for col, values in prediction_cols.items():
        artifact[col] = values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(args.output, index=False)
    _print_summary(results, frame)
    print(f"\n[joint FSR CV] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
