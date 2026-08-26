"""Historical validation for KO/TKO V2 strong shock-collapse candidate.

Uses the exact historical bout IDs from the prior 300-bout KD audit, reruns those
real leakage-safe FSR matchups through the KO/TKO V2 strong KD-collapse shadow
candidate, then joins canonical actual results from ufc_master.parquet.

Questions answered
------------------
1. When MC assigns high KO/TKO probability, did the real bout end by KO/TKO?
2. For real KO/TKO bouts, does MC put finish mass in the correct round?
3. When MC predicts a KO side, is that side the actual KO/TKO winner?

This is a diagnostic only. It does not change production or locked KD constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_kd_audit as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse


KD_AUDIT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_kd_audit.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_historical_300_actual_validation.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810
HEARTBEAT_PATHS = 1000

STRONG = next(c for c in collapse.CANDIDATES if c.name == "strong")


def _is_ko_tko(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if "doctor" in text:
        return False
    return "ko/tko" in text or text == "tko" or text == "ko" or text.startswith("tko ") or text.startswith("ko ")


def _load_selected_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prior 300-bout KD audit not found: {path}. Run the historical KD audit first."
        )
    frame = pd.read_parquet(path)
    if "bout_id" not in frame.columns:
        raise ValueError("Prior KD audit is missing bout_id.")
    # Preserve original audit order rather than resampling a new cohort.
    return frame["bout_id"].astype(str).drop_duplicates().tolist()


def _load_prefight_pairs(fsr_path: Path, selected_ids: list[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    frame = pd.read_parquet(fsr_path)
    bout_key = hist._resolve_bout_key(frame, None)
    frame[bout_key] = frame[bout_key].astype(str)
    frame = frame[frame[bout_key].isin(set(selected_ids))].copy()

    bouts, _ = hist._prepare_historical_bouts(frame, bout_key=bout_key)
    pairs = {bout_id: (red, blue) for bout_id, red, blue in bouts}
    missing = [bout_id for bout_id in selected_ids if bout_id not in pairs]
    if missing:
        raise ValueError(
            f"FSR pre-fight pair missing for {len(missing)} selected bouts; first={missing[:10]}"
        )
    return pairs


def _load_actual(master_path: Path, selected_ids: list[str]) -> pd.DataFrame:
    master = pd.read_parquet(master_path)
    required = {"fight_id", "method", "finish_round", "winner_id", "r_id", "b_id", "total_rounds"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master missing required columns: {missing}")

    work = master.copy()
    work["fight_id"] = work["fight_id"].astype(str)
    work = work[work["fight_id"].isin(set(selected_ids))].copy()
    if work["fight_id"].duplicated().any():
        dupes = work.loc[work["fight_id"].duplicated(False), "fight_id"].astype(str).tolist()
        raise ValueError(f"Master has duplicate selected fight_id rows; first={dupes[:10]}")

    work["actual_ko_tko"] = work["method"].map(_is_ko_tko).astype(int)
    work["actual_finish_round"] = pd.to_numeric(work["finish_round"], errors="coerce")
    work["actual_winner_id"] = work["winner_id"].astype(str)
    keep = [
        "fight_id", "method", "actual_ko_tko", "actual_finish_round", "actual_winner_id",
        "r_id", "b_id", "total_rounds",
    ]
    actual = work[keep].rename(columns={"fight_id": "bout_id"})

    missing_ids = sorted(set(selected_ids) - set(actual["bout_id"]))
    if missing_ids:
        raise ValueError(f"Master results missing for {len(missing_ids)} selected bouts; first={missing_ids[:10]}")
    return actual


def _scheduled_rounds(actual_row: pd.Series) -> int:
    value = pd.to_numeric(pd.Series([actual_row.get("total_rounds")]), errors="coerce").iloc[0]
    if pd.notna(value):
        rounds = int(round(float(value)))
        if rounds in (3, 5):
            return rounds
    return 3


def _run_mc(
    selected_ids: list[str],
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    actual: pd.DataFrame,
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    actual_by_id = actual.set_index("bout_id")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(selected_ids) * paths_per_bout
    path_counter = 0
    finish_counter = 0

    print(
        f"[historical KO validation] candidate=strong scale={STRONG.collapse_scale:.2f} "
        f"curve={STRONG.shock_curvature:.2f}; bouts={len(selected_ids)}; "
        f"paths_per_bout={paths_per_bout}; total_paths={total_paths:,}",
        flush=True,
    )

    for bout_index, bout_id in enumerate(selected_ids, start=1):
        red, blue = pairs[bout_id]
        actual_row = actual_by_id.loc[bout_id]
        rounds = _scheduled_rounds(actual_row)

        for path_index in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = collapse.StaticFSRMCKOTKOV2KDCollapse(
                red,
                blue,
                collapse=STRONG,
                rounds=rounds,
                seed=path_seed,
            )
            path = sim.run()
            finish = path.finish
            did_finish = finish is not None
            if did_finish:
                finish_counter += 1

            winner_id = None
            finish_round = None
            if finish is not None:
                winner_row = red if finish.winner == 0 else blue
                winner_id = str(winner_row["fighter_id"])
                finish_round = int(finish.round) if finish.round is not None else None

            rows.append(
                {
                    "bout_id": bout_id,
                    "bout_index": bout_index,
                    "path_index": path_index,
                    "path_seed": path_seed,
                    "scheduled_rounds": rounds,
                    "mc_ko_tko": int(did_finish),
                    "mc_finish_round": finish_round,
                    "mc_ko_winner_id": winner_id,
                }
            )

            path_counter += 1
            if path_counter % HEARTBEAT_PATHS == 0 or path_counter == total_paths:
                print(
                    f"[historical KO validation] paths {path_counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_index}/{len(selected_ids)}; "
                    f"KO/TKO paths={finish_counter:,} ({finish_counter/path_counter:.2%})",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _aggregate(paths: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    base_rows: list[dict[str, object]] = []
    for bout_id, g in paths.groupby("bout_id", sort=False):
        ko = g[g["mc_ko_tko"] == 1]
        p_ko = float(g["mc_ko_tko"].mean())
        round_probs = {r: float((g["mc_finish_round"] == r).mean()) for r in range(1, 6)}
        conditional_round_probs = {
            r: (float((ko["mc_finish_round"] == r).mean()) if len(ko) else 0.0)
            for r in range(1, 6)
        }

        if len(ko):
            predicted_round = int(ko["mc_finish_round"].value_counts().idxmax())
            expected_round = float(pd.to_numeric(ko["mc_finish_round"], errors="coerce").mean())
            winner_probs = ko["mc_ko_winner_id"].value_counts(normalize=True)
            predicted_winner = str(winner_probs.index[0])
            predicted_winner_prob = float(winner_probs.iloc[0])
        else:
            predicted_round = np.nan
            expected_round = np.nan
            predicted_winner = None
            predicted_winner_prob = np.nan

        row: dict[str, object] = {
            "bout_id": bout_id,
            "mc_paths": len(g),
            "mc_p_ko_tko": p_ko,
            "mc_predicted_ko_round_conditional": predicted_round,
            "mc_expected_ko_round_conditional": expected_round,
            "mc_predicted_ko_winner_id": predicted_winner,
            "mc_predicted_ko_winner_prob_conditional": predicted_winner_prob,
        }
        for r in range(1, 6):
            row[f"mc_p_ko_round_{r}_unconditional"] = round_probs[r]
            row[f"mc_p_round_{r}_given_ko"] = conditional_round_probs[r]
        base_rows.append(row)

    agg = pd.DataFrame(base_rows)
    merged = agg.merge(actual, on="bout_id", how="left", validate="one_to_one")
    merged["actual_ko_tko"] = merged["actual_ko_tko"].astype(int)
    merged["actual_finish_round"] = pd.to_numeric(merged["actual_finish_round"], errors="coerce")
    merged["predicted_ko_winner_correct"] = (
        merged["mc_predicted_ko_winner_id"].astype(str) == merged["actual_winner_id"].astype(str)
    ).astype(int)
    return merged


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    # Mann-Whitney formulation; handles ties and avoids a sklearn dependency.
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(pos) * len(neg)))


def _print_summary(frame: pd.DataFrame) -> None:
    y = frame["actual_ko_tko"].to_numpy(dtype=int)
    p = frame["mc_p_ko_tko"].to_numpy(dtype=float)
    actual_ko = frame[frame["actual_ko_tko"] == 1].copy()

    print("\n" + "=" * 118)
    print("HISTORICAL 300-BOUT KO/TKO V2 VALIDATION — STRONG SHOCK-COLLAPSE")
    print("=" * 118)
    print(f"bouts: {len(frame):,}")
    print(f"actual KO/TKO rate: {y.mean():.2%}")
    print(f"mean MC p(KO/TKO): {p.mean():.2%}")
    print(f"KO discrimination ROC-AUC: {_auc(y, p):.4f}")
    print(f"Brier score: {np.mean((p-y)**2):.6f}")
    print(
        f"mean MC p(KO) actual KO bouts={frame.loc[frame.actual_ko_tko.eq(1), 'mc_p_ko_tko'].mean():.3f}; "
        f"actual non-KO bouts={frame.loc[frame.actual_ko_tko.eq(0), 'mc_p_ko_tko'].mean():.3f}"
    )

    print("\nIF MC CALLS KO — THRESHOLD PRECISION / RECALL")
    threshold_rows = []
    for threshold in (0.20, 0.30, 0.40, 0.50):
        pred = p >= threshold
        n_pred = int(pred.sum())
        true_pos = int(((y == 1) & pred).sum())
        precision = true_pos / n_pred if n_pred else float("nan")
        recall = true_pos / int(y.sum()) if y.sum() else float("nan")
        threshold_rows.append((threshold, n_pred, true_pos, precision, recall))
    print(
        pd.DataFrame(
            threshold_rows,
            columns=["pKO_threshold", "predicted_KO_bouts", "actual_KO_among_them", "precision", "recall"],
        ).to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nROUND ACCURACY — ACTUAL KO/TKO BOUTS")
    if actual_ko.empty:
        print("No actual KO/TKO bouts in cohort.")
    else:
        valid = actual_ko.dropna(
            subset=["actual_finish_round", "mc_predicted_ko_round_conditional", "mc_expected_ko_round_conditional"]
        ).copy()
        valid["round_exact"] = (
            valid["actual_finish_round"] == valid["mc_predicted_ko_round_conditional"]
        )
        valid["round_abs_error_mode"] = (
            valid["mc_predicted_ko_round_conditional"] - valid["actual_finish_round"]
        ).abs()
        valid["round_abs_error_expected"] = (
            valid["mc_expected_ko_round_conditional"] - valid["actual_finish_round"]
        ).abs()
        print(f"actual KO/TKO bouts: {len(actual_ko):,}")
        print(f"exact modal-round hit rate: {valid['round_exact'].mean():.2%}")
        print(f"modal-round within ±1: {(valid['round_abs_error_mode'] <= 1).mean():.2%}")
        print(f"modal-round MAE: {valid['round_abs_error_mode'].mean():.3f} rounds")
        print(f"conditional expected-round MAE: {valid['round_abs_error_expected'].mean():.3f} rounds")

        actual_round_dist = actual_ko["actual_finish_round"].value_counts(normalize=True).sort_index()
        print("\nACTUAL VS MC KO ROUND DISTRIBUTION")
        dist_rows = []
        for r in range(1, 6):
            dist_rows.append(
                {
                    "round": r,
                    "actual_KO_share": float(actual_round_dist.get(r, 0.0)),
                    "mean_MC_P_round_given_KO": float(actual_ko[f"mc_p_round_{r}_given_ko"].mean()),
                }
            )
        print(pd.DataFrame(dist_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        side_valid = actual_ko[actual_ko["mc_predicted_ko_winner_id"].notna()]
        print("\nKO WINNER SIDE — ACTUAL KO/TKO BOUTS")
        print(
            f"predicted KO winner correct: {side_valid['predicted_ko_winner_correct'].mean():.2%} "
            f"({int(side_valid['predicted_ko_winner_correct'].sum())}/{len(side_valid)})"
        )

    print("\nRESEARCH BOUNDARY")
    print("- Exact same historical cohort as prior 300-bout KD audit; no resampling.")
    print("- Leakage-safe pre-fight FSR profiles only.")
    print("- Actual method/round/winner come from canonical ufc_master.parquet.")
    print("- Strong KD collapse only: scale=5.0, curvature=2.0; locked KD constants unchanged.")
    print("- Submission/fatigue/judging/dynamic-state mechanics are not yet connected, so this is structural validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strong KO V2 candidate against actual historical KO/TKO and round")
    parser.add_argument("--kd-audit", type=Path, default=KD_AUDIT_PATH)
    parser.add_argument("--fsr-path", type=Path, default=damage.FSR_PATH)
    parser.add_argument("--master-path", type=Path, default=Path(MASTER_PATH))
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    selected_ids = _load_selected_ids(args.kd_audit)
    print(f"[historical KO validation] selected prior-audit bouts={len(selected_ids):,}", flush=True)
    pairs = _load_prefight_pairs(args.fsr_path, selected_ids)
    print(f"[historical KO validation] matched leakage-safe FSR pairs={len(pairs):,}", flush=True)
    actual = _load_actual(args.master_path, selected_ids)
    print(f"[historical KO validation] matched canonical actual results={len(actual):,}", flush=True)

    paths = _run_mc(
        selected_ids,
        pairs,
        actual,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    result = _aggregate(paths, actual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    _print_summary(result)
    print(f"\n[historical KO validation] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
