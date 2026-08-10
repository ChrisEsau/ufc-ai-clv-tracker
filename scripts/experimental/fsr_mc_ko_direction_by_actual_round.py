"""Break down historical KO/TKO winner-direction accuracy by actual finish round.

Reads the saved mature 2020+ round-recovery population audit. No Monte Carlo
paths are rerun; this is a cheap post-hoc diagnostic on the 1,565 bout rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_round_recovery_audit.csv"
)


def _direction_margin(row: pd.Series) -> float:
    """Positive means the MC favored the actual KO winner over the loser."""
    winner = str(row.get("winner_id", ""))
    if winner == str(row.get("r_id", "")):
        return float(row["p_r_ko"] - row["p_b_ko"])
    if winner == str(row.get("b_id", "")):
        return float(row["p_b_ko"] - row["p_r_ko"])
    return np.nan


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Population audit not found: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)
    required = {
        "actual_ko_tko", "actual_finish_round", "winner_id", "r_id", "b_id",
        "p_r_ko", "p_b_ko", "ko_winner_direction_hit", "ko_winner_direction_tie",
        "p_actual_winner_ko",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Audit missing required columns: {missing}")

    ko = frame.loc[frame["actual_ko_tko"].eq(1)].copy()
    ko["actual_finish_round"] = pd.to_numeric(ko["actual_finish_round"], errors="coerce")
    ko = ko.dropna(subset=["actual_finish_round"]).copy()
    ko["actual_finish_round"] = ko["actual_finish_round"].astype(int)
    ko["direction_margin"] = ko.apply(_direction_margin, axis=1)

    rows: list[dict[str, object]] = []
    for rnd, g in ko.groupby("actual_finish_round", sort=True):
        non_tie = g["ko_winner_direction_hit"].notna()
        rows.append(
            {
                "actual_finish_round": int(rnd),
                "ko_bouts": len(g),
                "non_tie_calls": int(non_tie.sum()),
                "direction_hit_rate_non_ties": float(
                    g.loc[non_tie, "ko_winner_direction_hit"].mean()
                ) if non_tie.any() else np.nan,
                "direction_tie_rate": float(g["ko_winner_direction_tie"].mean()),
                "mean_p_actual_ko_winner": float(g["p_actual_winner_ko"].mean()),
                "mean_direction_margin": float(g["direction_margin"].mean()),
                "median_direction_margin": float(g["direction_margin"].median()),
                "actual_winner_favored_rate": float((g["direction_margin"] > 0).mean()),
                "actual_loser_favored_rate": float((g["direction_margin"] < 0).mean()),
            }
        )

    result = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("KO/TKO WINNER DIRECTION BY ACTUAL FINISH ROUND — ROUND-RECOVERY MC AUDIT")
    print("=" * 120)
    print(f"historical KO/TKO bouts: {len(ko):,}")
    print(
        result.to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nINTERPRETATION")
    print("- direction_hit_rate_non_ties asks: when the MC chose one fighter, how often was it the actual KO winner?")
    print("- direction_tie_rate matters because 10 paths/bout creates many coarse ties.")
    print("- mean_direction_margin > 0 means the MC, on average, assigned more KO probability to the actual winner.")
    print("- If R1 is much worse than R2/R3, the direction problem is acute-finish specific.")
    print("- If all rounds are near 50%, the directional problem is broader than finish timing.")


if __name__ == "__main__":
    main()
