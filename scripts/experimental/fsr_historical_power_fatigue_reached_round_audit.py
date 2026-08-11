"""Matched-cohort historical power-fatigue proxy audit.

The raw by-round power audit changes population every round because fights that
finish early disappear. This audit holds the represented fights/fighters fixed
within each comparison by conditioning on a fight having reached a target
round, then recomputing KD-per-landed-strike potency for all earlier rounds in
that exact same cohort.

This does not prove physiological fatigue: tactics, damage, opponent behavior,
and selection into reaching a later round still matter. It is a stronger target
than the raw all-round comparison because changing fighter composition across
rounds is removed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_historical_power_fatigue_audit as raw


OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow")
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "historical_power_fatigue_reached_round_summary.csv"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _aggregate(detail: pd.DataFrame, cohort_name: str, target_round: int) -> pd.DataFrame:
    max_round = detail.groupby("fight_id")["round"].max()
    eligible_ids = set(max_round[max_round.ge(target_round)].index.astype(str))
    cohort = detail[
        detail["fight_id"].isin(eligible_ids) & detail["round"].between(1, target_round)
    ].copy()

    rows: list[dict[str, float | int | str]] = []
    for round_no, group in cohort.groupby("round", sort=True):
        kd_total = float(group["kd"].sum())
        head_total = float(group["head_landed"].sum())
        sig_total = float(group["sig_str_landed"].sum())
        rows.append(
            {
                "cohort": cohort_name,
                "target_round": target_round,
                "round": int(round_no),
                "fights": int(group["fight_id"].nunique()),
                "fighter_rounds": int(len(group)),
                "knockdowns": kd_total,
                "head_landed": head_total,
                "sig_str_landed": sig_total,
                "kd_per_100_head_landed": 100.0 * _safe_div(kd_total, head_total),
                "kd_per_100_sig_landed": 100.0 * _safe_div(kd_total, sig_total),
                "fighter_round_kd_rate": float(group["kd_any"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    r1 = summary.loc[summary["round"].eq(1)]
    if r1.empty:
        raise ValueError(f"No R1 rows for {cohort_name}")

    for metric in (
        "kd_per_100_head_landed",
        "kd_per_100_sig_landed",
        "fighter_round_kd_rate",
    ):
        baseline = float(r1.iloc[0][metric])
        summary[f"{metric}_vs_r1"] = summary[metric] / baseline if baseline > 0 else np.nan

    return summary


def _print_block(summary: pd.DataFrame, title: str) -> None:
    print(title)
    cols = [
        "round",
        "fights",
        "fighter_rounds",
        "knockdowns",
        "head_landed",
        "sig_str_landed",
        "kd_per_100_head_landed",
        "kd_per_100_head_landed_vs_r1",
        "kd_per_100_sig_landed",
        "kd_per_100_sig_landed_vs_r1",
        "fighter_round_kd_rate",
        "fighter_round_kd_rate_vs_r1",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()


def main() -> None:
    rounds = raw._load_round_stats()
    master = raw._load_master()
    detail = raw._prepare_detail(rounds, master)

    summaries = []
    for target_round in (2, 3, 5):
        label = f"fights_reaching_r{target_round}"
        summaries.append(_aggregate(detail, label, target_round))

    combined = pd.concat(summaries, ignore_index=True)

    print("=" * 150)
    print("MATCHED-COHORT HISTORICAL POWER-FATIGUE PROXY")
    print("=" * 150)
    print(
        "Each block holds the fight population fixed: only fights that reached the target round are included, "
        "and their earlier rounds are recomputed using the same fighters."
    )
    print(
        "Interpretation remains associative, not causal: these are historical potency proxies, not direct punch-force measurements."
    )
    print()

    for target_round in (2, 3, 5):
        block = combined[combined["target_round"].eq(target_round)].copy()
        _print_block(block, f"FIGHTS REACHING ROUND {target_round}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"Wrote matched-cohort summary to {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
