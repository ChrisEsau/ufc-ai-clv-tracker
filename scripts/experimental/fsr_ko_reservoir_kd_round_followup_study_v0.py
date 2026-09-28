"""Study 4: knockdown rounds versus non-knockdown rounds for KO reservoir research.

Shadow-only research. No simulator or FSR trait logic is modified.

Purpose
-------
Test whether fighter-rounds in which a fighter absorbs a knockdown also contain
substantially greater damaging exposure, especially head and ground strikes.
This is evidence for the proposed post-KD vulnerability/follow-up mechanism.

Critical limitation
-------------------
UFCStats is round-aggregated. It does not identify whether same-round strikes
occurred before or after the knockdown. Therefore this study measures the
association expected from a follow-up sequence; it cannot prove within-round
causal ordering.

The script reports:
1. All KD rounds vs all non-KD rounds.
2. The same comparison within round-number strata.
3. A cleaner "survived KD" cohort excluding the fighter's KO/TKO losing finish
   round, reducing contamination from the terminal finishing sequence.
4. Within-fighter-fight paired comparisons when the same fighter has both a KD
   round and a non-KD round in the same fight.
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

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_kd_round_followup_study_v0.parquet"
)

OUTCOME_COLUMNS = ["fight_id", "winner", "winner_id", "method", "finish_round"]

# Keep Study 4 inside the existing Finish State reciprocal-opponent contract.
# body_landed / leg_landed are not exposed by attach_opponent_round_values(),
# and they are not required for the post-KD hypotheses we are testing here.
METRICS = [
    "sig_absorbed",
    "head_absorbed",
    "ground_absorbed",
]


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def _load(round_stats_path: Path, master_path: Path) -> pd.DataFrame:
    print(f"[KD follow-up] loading round stats: {round_stats_path}", flush=True)
    rounds = pd.read_parquet(round_stats_path)
    print(f"[KD follow-up] raw fighter-round rows: {len(rounds):,}", flush=True)

    rounds = standardize_round_stats(rounds)
    rounds = attach_opponent_round_values(rounds)

    outcomes = pd.read_parquet(master_path, columns=OUTCOME_COLUMNS)
    outcomes = standardize_outcomes(outcomes)

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    outcomes["fight_id"] = outcomes["fight_id"].astype(str)
    outcomes["winner_id"] = outcomes["winner_id"].astype("string")

    work = rounds.merge(outcomes, on="fight_id", how="inner", validate="many_to_one")

    work["kd_absorbed"] = pd.to_numeric(work["opponent_kd"], errors="coerce").fillna(0.0)
    work["kd_round"] = (work["kd_absorbed"] > 0).astype(int)
    work["sig_absorbed"] = pd.to_numeric(
        work["opponent_sig_str_landed"], errors="coerce"
    ).fillna(0.0)
    work["head_absorbed"] = pd.to_numeric(
        work["opponent_head_landed"], errors="coerce"
    ).fillna(0.0)
    work["ground_absorbed"] = pd.to_numeric(
        work["opponent_ground_landed"], errors="coerce"
    ).fillna(0.0)

    fighter_lost = work["fighter_id"].astype("string") != work["winner_id"]
    ko_loss = fighter_lost & work["method"].isin(KO_TKO_METHODS)
    finish_round = pd.to_numeric(work["finish_round"], errors="coerce")
    round_no = pd.to_numeric(work["round"], errors="coerce")
    work["terminal_ko_loss_round"] = (ko_loss & (round_no == finish_round)).astype(int)

    work["round_stratum"] = np.select(
        [round_no == 1, round_no == 2, round_no == 3, round_no >= 4],
        ["Round 1", "Round 2", "Round 3", "Round 4+"],
        default="Other",
    )

    print(f"[KD follow-up] standardized fighter-round rows: {len(work):,}", flush=True)
    print(f"[KD follow-up] KD rounds: {int(work['kd_round'].sum()):,}", flush=True)
    return work


def _comparison(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    kd = frame[frame["kd_round"] == 1]
    non = frame[frame["kd_round"] == 0]

    for metric in METRICS:
        kd_mean = float(kd[metric].mean()) if len(kd) else float("nan")
        non_mean = float(non[metric].mean()) if len(non) else float("nan")
        rows.append(
            {
                "cohort": label,
                "metric": metric,
                "kd_rounds": int(len(kd)),
                "non_kd_rounds": int(len(non)),
                "kd_round_mean": kd_mean,
                "non_kd_round_mean": non_mean,
                "mean_difference": kd_mean - non_mean,
                "ratio_kd_vs_non_kd": _safe_ratio(kd_mean, non_mean),
            }
        )
    return pd.DataFrame(rows)


def _round_strata(frame: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for stratum in ("Round 1", "Round 2", "Round 3", "Round 4+"):
        group = frame[frame["round_stratum"] == stratum]
        if group.empty or group["kd_round"].nunique() < 2:
            continue
        tables.append(_comparison(group, stratum))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _paired_within_fight(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare KD vs non-KD rounds inside the same fighter-fight."""
    rows: list[dict[str, float | int | str]] = []
    eligible = 0
    diffs = {metric: [] for metric in METRICS}

    for _, group in frame.groupby(["fight_id", "fighter_id"], sort=False):
        kd = group[group["kd_round"] == 1]
        non = group[group["kd_round"] == 0]
        if kd.empty or non.empty:
            continue
        eligible += 1
        for metric in METRICS:
            diffs[metric].append(float(kd[metric].mean() - non[metric].mean()))

    for metric, values in diffs.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "eligible_fighter_fights": eligible,
                "mean_within_fight_difference": float(np.mean(arr)) if len(arr) else float("nan"),
                "median_within_fight_difference": float(np.median(arr)) if len(arr) else float("nan"),
                "share_positive_difference": float(np.mean(arr > 0)) if len(arr) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 4: KD-round follow-up damage association")
    parser.add_argument("--round-stats-path", type=Path, default=ROUND_STATS_PATH)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.round_stats_path, args.master_path)

    print("\n" + "=" * 118)
    print("STUDY 4A — ALL KD ROUNDS VS NON-KD ROUNDS")
    print("=" * 118)
    all_table = _comparison(work, "All rounds")
    print(all_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n" + "=" * 118)
    print("STUDY 4B — WITHIN ROUND-NUMBER STRATA")
    print("=" * 118)
    strata = _round_strata(work)
    print(strata.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # Exclude the exact round in which this fighter lost by KO/TKO. KD rounds
    # remaining here were survived through the end of the round/fight outcome,
    # so the comparison is less dominated by terminal finishing flurries.
    survived = work[work["terminal_ko_loss_round"] == 0].copy()
    print("\n" + "=" * 118)
    print("STUDY 4C — EXCLUDING TERMINAL KO/TKO LOSS ROUNDS")
    print("=" * 118)
    survived_table = _comparison(survived, "Non-terminal rounds")
    print(survived_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n" + "=" * 118)
    print("STUDY 4D — WITHIN-FIGHT PAIRED KD VS NON-KD ROUNDS")
    print("=" * 118)
    paired = _paired_within_fight(work)
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nNOTE: UFCStats has round aggregates only. Elevated same-round damage "
        "supports a KD/follow-up association but does not establish whether each "
        "strike occurred before or after the knockdown."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(args.output, index=False)
    print(f"\n[KD follow-up] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
