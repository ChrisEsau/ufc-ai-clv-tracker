"""Historical validation for the proposed FSR KO damage-reservoir model.

Shadow-only research.  This script does not modify FSR traits or simulator logic.

Questions tested in V0
----------------------
1. Does damage absorbed in PRIOR rounds predict greater knockdown susceptibility
   in a later round?
2. Does absorbing a knockdown strongly increase the probability of losing the
   fight by KO/TKO, and does that risk escalate for multiple knockdowns?

The study deliberately reuses the existing RFS Finish State damage-exposure
components:
- knockdowns absorbed;
- head strikes absorbed;
- ground strikes absorbed;
- opponent control.

Important data limitation
-------------------------
UFCStats provides round aggregates rather than exact within-round event timing.
Therefore this study can support a reservoir-depletion relationship and measure
KD/KO association, but it cannot prove that same-round follow-up strikes occurred
after a knockdown.  That ordering question belongs to the later KD-round study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from pipeline.round_stats.build_round_fighter_finish_state import (
    KO_TKO_METHODS,
    attach_opponent_round_values,
    standardize_outcomes,
    standardize_round_stats,
)


OUTPUT_ROUNDS = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_historical_study_v0_rounds.parquet"
)
OUTPUT_FIGHTS = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_historical_study_v0_fights.parquet"
)

OUTCOME_COLUMNS = [
    "fight_id",
    "winner",
    "winner_id",
    "method",
    "finish_round",
]


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"a": a[mask], "b": b[mask]}).corr(method="spearman").iloc[0, 1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _load_inputs(round_stats_path: Path, master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[KO reservoir study] loading round stats: {round_stats_path}", flush=True)
    raw_rounds = pd.read_parquet(round_stats_path)
    print(f"[KO reservoir study] raw fighter-round rows: {len(raw_rounds):,}", flush=True)

    # Reuse the exact reciprocal-round normalization used by Finish State.
    rounds = standardize_round_stats(raw_rounds)
    rounds = attach_opponent_round_values(rounds)
    print(f"[KO reservoir study] standardized reciprocal rows: {len(rounds):,}", flush=True)

    print(f"[KO reservoir study] loading outcomes: {master_path}", flush=True)
    outcomes = pd.read_parquet(master_path, columns=OUTCOME_COLUMNS)
    outcomes = standardize_outcomes(outcomes)
    print(f"[KO reservoir study] authoritative outcomes: {len(outcomes):,}", flush=True)

    for frame in (rounds, outcomes):
        frame["fight_id"] = frame["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    rounds["opponent_id"] = rounds["opponent_id"].astype(str)
    outcomes["winner_id"] = outcomes["winner_id"].astype("string")

    return rounds, outcomes


def _prior_round_damage_table(rounds: pd.DataFrame) -> pd.DataFrame:
    """Create later-round observations using ONLY damage from earlier rounds."""
    work = rounds.sort_values(["fight_id", "fighter_id", "round"]).copy()

    absorbed_sources = {
        "sig": "opponent_sig_str_landed",
        "head": "opponent_head_landed",
        "ground": "opponent_ground_landed",
        "kd": "opponent_kd",
        "control": "opponent_ctrl_sec",
    }

    grouped = work.groupby(["fight_id", "fighter_id"], sort=False)

    # Cumulative prior totals exclude the current round by construction.
    for label, column in absorbed_sources.items():
        values = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        cumulative = values.groupby([work["fight_id"], work["fighter_id"]]).cumsum()
        work[f"prior_{label}_absorbed"] = cumulative - values

    work["prior_rounds"] = grouped.cumcount().astype(float)

    # Current-round outcomes used to measure later vulnerability.
    work["current_kd_absorbed"] = pd.to_numeric(
        work["opponent_kd"], errors="coerce"
    ).fillna(0.0)
    work["current_sig_absorbed"] = pd.to_numeric(
        work["opponent_sig_str_landed"], errors="coerce"
    ).fillna(0.0)
    work["current_head_absorbed"] = pd.to_numeric(
        work["opponent_head_landed"], errors="coerce"
    ).fillna(0.0)
    work["current_ground_absorbed"] = pd.to_numeric(
        work["opponent_ground_landed"], errors="coerce"
    ).fillna(0.0)
    work["current_kd_indicator"] = (work["current_kd_absorbed"] > 0).astype(int)

    # Study 1 is explicitly later-round susceptibility, so Round 1 is excluded.
    later = work[work["prior_rounds"] >= 1].copy()

    # Mirror the existing Finish State damage-exposure components, but use only
    # PRIOR rounds.  The index is intentionally uncalibrated; quintile ranking is
    # more important here than its absolute numerical scale.
    prior_rounds = later["prior_rounds"].to_numpy(dtype=float)
    later["prior_kd_per_round"] = later["prior_kd_absorbed"] / prior_rounds
    later["prior_head_per_round"] = later["prior_head_absorbed"] / prior_rounds
    later["prior_ground_per_round"] = later["prior_ground_absorbed"] / prior_rounds
    later["prior_control_per_round_min"] = (
        later["prior_control_absorbed"] / (prior_rounds * 60.0)
    )
    later["prior_damage_exposure"] = later[
        [
            "prior_kd_per_round",
            "prior_head_per_round",
            "prior_ground_per_round",
            "prior_control_per_round_min",
        ]
    ].mean(axis=1)

    # Quantile buckets are formed with rank first so repeated zero/low values do
    # not make qcut fail.  Every row receives one ordered damage-exposure bucket.
    ranks = later["prior_damage_exposure"].rank(method="first", pct=True)
    later["prior_damage_quintile"] = pd.cut(
        ranks,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )
    return later


def _study1_bucket_table(later: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for bucket, group in later.groupby("prior_damage_quintile", observed=True, sort=False):
        kd = float(group["current_kd_absorbed"].sum())
        sig = float(group["current_sig_absorbed"].sum())
        rows.append(
            {
                "prior_damage_quintile": str(bucket),
                "fighter_rounds": int(len(group)),
                "mean_prior_damage_exposure": float(group["prior_damage_exposure"].mean()),
                "current_round_kd_probability": float(group["current_kd_indicator"].mean()),
                "current_kd_per_sig_absorbed": _safe_ratio(kd, sig),
                "current_sig_absorbed_mean": float(group["current_sig_absorbed"].mean()),
                "current_head_absorbed_mean": float(group["current_head_absorbed"].mean()),
                "current_ground_absorbed_mean": float(group["current_ground_absorbed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _study1_round_strata(later: pd.DataFrame) -> pd.DataFrame:
    """Repeat the core relationship within current-round strata."""
    work = later.copy()
    current_round = pd.to_numeric(work["round"], errors="coerce").astype(int)
    work["round_stratum"] = np.select(
        [current_round == 2, current_round == 3, current_round >= 4],
        ["Round 2", "Round 3", "Round 4+"],
        default="Other",
    )

    rows: list[dict[str, float | int | str]] = []
    for stratum, group in work.groupby("round_stratum", sort=False):
        if stratum == "Other" or len(group) < 20:
            continue
        rows.append(
            {
                "round_stratum": stratum,
                "fighter_rounds": int(len(group)),
                "spearman_prior_damage_vs_kd_indicator": _safe_spearman(
                    group["prior_damage_exposure"], group["current_kd_indicator"]
                ),
                "spearman_prior_damage_vs_current_kd_count": _safe_spearman(
                    group["prior_damage_exposure"], group["current_kd_absorbed"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _fight_level_kd_table(rounds: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Build one row per fighter-fight for Study 2."""
    grouped = rounds.groupby(["fight_id", "fighter_id"], sort=False)
    fights = grouped.agg(
        fighter_name=("fighter_name", "first"),
        opponent_id=("opponent_id", "first"),
        opponent_name=("opponent_name", "first"),
        kd_absorbed=("opponent_kd", "sum"),
        sig_absorbed=("opponent_sig_str_landed", "sum"),
        head_absorbed=("opponent_head_landed", "sum"),
        ground_absorbed=("opponent_ground_landed", "sum"),
        opponent_control_seconds=("opponent_ctrl_sec", "sum"),
        rounds_observed=("round", "nunique"),
    ).reset_index()

    fights = fights.merge(outcomes, on="fight_id", how="inner", validate="many_to_one")
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    winner = fights["winner_id"].astype("string")
    fights["ko_tko_loss"] = (
        (winner.notna())
        & (fights["fighter_id"].astype("string") != winner)
        & fights["method"].isin(KO_TKO_METHODS)
    ).astype(int)

    kd = pd.to_numeric(fights["kd_absorbed"], errors="coerce").fillna(0.0)
    fights["kd_group"] = np.select(
        [kd <= 0, kd == 1, kd >= 2],
        ["0 KD", "1 KD", "2+ KD"],
        default="unknown",
    )
    return fights


def _study2_table(fights: pd.DataFrame) -> pd.DataFrame:
    baseline = fights.loc[fights["kd_group"] == "0 KD", "ko_tko_loss"].mean()
    rows: list[dict[str, float | int | str]] = []
    for label in ("0 KD", "1 KD", "2+ KD"):
        group = fights[fights["kd_group"] == label]
        if group.empty:
            continue
        risk = float(group["ko_tko_loss"].mean())
        rows.append(
            {
                "kd_absorbed_group": label,
                "fighter_fights": int(len(group)),
                "ko_tko_losses": int(group["ko_tko_loss"].sum()),
                "ko_tko_loss_probability": risk,
                "relative_risk_vs_0_kd": _safe_ratio(risk, float(baseline)),
                "mean_sig_absorbed": float(group["sig_absorbed"].mean()),
                "mean_head_absorbed": float(group["head_absorbed"].mean()),
                "mean_ground_absorbed": float(group["ground_absorbed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _print_study1(later: pd.DataFrame) -> None:
    print("\n" + "=" * 116)
    print("STUDY 1 — PRIOR-ROUND DAMAGE EXPOSURE -> LATER-ROUND KD SUSCEPTIBILITY")
    print("=" * 116)
    print(f"later-round fighter observations: {len(later):,}")
    print(
        "Spearman prior damage vs current KD indicator: "
        f"{_safe_spearman(later['prior_damage_exposure'], later['current_kd_indicator']):.4f}"
    )
    print(
        "Spearman prior damage vs current KD count    : "
        f"{_safe_spearman(later['prior_damage_exposure'], later['current_kd_absorbed']):.4f}"
    )

    bucket = _study1_bucket_table(later)
    print("\nPRIOR DAMAGE QUINTILES")
    print(bucket.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    strata = _study1_round_strata(later)
    print("\nWITHIN CURRENT-ROUND STRATA")
    print(strata.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def _print_study2(fights: pd.DataFrame) -> None:
    print("\n" + "=" * 116)
    print("STUDY 2 — KNOCKDOWN ABSORBED -> KO/TKO LOSS")
    print("=" * 116)
    print(f"fighter-fight observations: {len(fights):,}")
    table = _study2_table(fights)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nSpearman KD absorbed vs KO/TKO loss: "
        f"{_safe_spearman(fights['kd_absorbed'], fights['ko_tko_loss']):.4f}"
    )
    print(
        "NOTE: this association includes KDs occurring in the finish sequence. "
        "It is a calibration relationship, not proof of within-round causal ordering."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical Study 1/2 for the proposed KO damage reservoir"
    )
    parser.add_argument("--round-stats-path", type=Path, default=ROUND_STATS_PATH)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--round-output", type=Path, default=OUTPUT_ROUNDS)
    parser.add_argument("--fight-output", type=Path, default=OUTPUT_FIGHTS)
    args = parser.parse_args()

    rounds, outcomes = _load_inputs(args.round_stats_path, args.master_path)

    later = _prior_round_damage_table(rounds)
    fights = _fight_level_kd_table(rounds, outcomes)

    _print_study1(later)
    _print_study2(fights)

    args.round_output.parent.mkdir(parents=True, exist_ok=True)
    later.to_parquet(args.round_output, index=False)
    fights.to_parquet(args.fight_output, index=False)
    print(f"\n[KO reservoir study] wrote {args.round_output}", flush=True)
    print(f"[KO reservoir study] wrote {args.fight_output}", flush=True)
    print(
        "\nNEXT: interpret these two studies before implementing any KO/KD simulator mechanics."
    )


if __name__ == "__main__":
    main()
