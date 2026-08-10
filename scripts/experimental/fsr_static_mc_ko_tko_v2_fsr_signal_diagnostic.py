"""Diagnose whether KO-relevant FSR traits contain historical finish signal.

Uses the existing 300-bout historical KO validation cohort and leakage-safe FSR
pre-fight profiles. No simulator rerun is required.

Questions answered
------------------
1. Are KO-relevant FSR matchup edges stronger in actual KO/TKO bouts than non-KO bouts?
2. Are those edges stronger in actual Round-1 KOs than later KOs?
3. Do directional FSR edges discriminate the actual KO winner from the loser?
4. How do the raw FSR edges relate to the simulator's existing p(KO/TKO)?

This script is diagnostic only. It changes no simulator constants or FSR values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_kd_audit as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_historical_300_actual_validation as validation


VALIDATION_PATH = validation.OUTPUT_PATH
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_fsr_signal_diagnostic.parquet"
)

EDGE_SPECS = (
    ("power_minus_kd_resistance", "striking_power", "knockdown_resistance"),
    ("power_minus_durability", "striking_power", "damage_durability"),
    ("distance_pressure_minus_defense", "distance_striking_pressure", "distance_defense"),
    ("distance_precision_minus_defense", "distance_precision", "distance_defense"),
    ("clinch_pressure_minus_defense", "clinch_striking_pressure", "clinch_striking_defense"),
    ("clinch_precision_minus_defense", "clinch_striking_precision", "clinch_striking_defense"),
    ("ground_pressure_minus_defense", "ground_striking_pressure", "ground_striking_defense"),
    ("ground_precision_minus_defense", "ground_striking_precision", "ground_striking_defense"),
)


def _numeric(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = np.asarray(y, dtype=int)[mask]
    score = np.asarray(score, dtype=float)[mask]
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * ties) / (len(pos) * len(neg)))


def _load_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical KO validation artifact not found: {path}. Run the 300-bout KO validation first."
        )
    frame = pd.read_parquet(path).copy()
    required = {"bout_id", "actual_ko_tko", "actual_finish_round", "actual_winner_id", "mc_p_ko_tko"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Historical KO validation missing required columns: {missing}")
    frame["bout_id"] = frame["bout_id"].astype(str)
    frame["actual_ko_tko"] = frame["actual_ko_tko"].astype(int)
    frame["actual_finish_round"] = pd.to_numeric(frame["actual_finish_round"], errors="coerce")
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


def _timing_group(row: pd.Series) -> str:
    if int(row["actual_ko_tko"]) == 0:
        return "non-KO"
    r = row["actual_finish_round"]
    if r == 1:
        return "actual R1 KO"
    if r == 2:
        return "actual R2 KO"
    return "actual R3+ KO"


def _build_frame(validation_frame: pd.DataFrame, pairs: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, bout in validation_frame.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        winner_id = str(bout["actual_winner_id"])
        red_id = str(red["fighter_id"])
        blue_id = str(blue["fighter_id"])
        if winner_id == red_id:
            winner, loser = red, blue
        elif winner_id == blue_id:
            winner, loser = blue, red
        else:
            raise ValueError(
                f"Actual winner {winner_id} does not match FSR pair {red_id} vs {blue_id} for bout {bout_id}."
            )

        row: dict[str, object] = {
            "bout_id": bout_id,
            "actual_ko_tko": int(bout["actual_ko_tko"]),
            "actual_finish_round": bout["actual_finish_round"],
            "actual_winner_id": winner_id,
            "timing_group": _timing_group(bout),
            "mc_p_ko_tko": float(bout["mc_p_ko_tko"]),
        }

        for edge_name, attacker_trait, defender_trait in EDGE_SPECS:
            if attacker_trait not in winner.index or defender_trait not in loser.index:
                continue
            winner_edge = _numeric(winner, attacker_trait) - _numeric(loser, defender_trait)
            loser_edge = _numeric(loser, attacker_trait) - _numeric(winner, defender_trait)
            row[f"winner_edge_{edge_name}"] = winner_edge
            row[f"loser_edge_{edge_name}"] = loser_edge
            row[f"directional_advantage_{edge_name}"] = winner_edge - loser_edge

        rows.append(row)

    return pd.DataFrame(rows)


def _print_group_summary(frame: pd.DataFrame, edge_cols: list[str]) -> None:
    print("\nFSR EDGE LEVEL BY ACTUAL OUTCOME/TIMING")
    rows: list[dict[str, object]] = []
    order = ["non-KO", "actual R1 KO", "actual R2 KO", "actual R3+ KO"]
    for group in order:
        g = frame[frame["timing_group"] == group]
        if g.empty:
            continue
        row: dict[str, object] = {"group": group, "bouts": len(g)}
        for col in edge_cols:
            row[col.replace("winner_edge_", "")] = float(g[col].mean())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_discrimination(frame: pd.DataFrame, edge_cols: list[str]) -> None:
    y_ko = frame["actual_ko_tko"].to_numpy(dtype=int)
    y_r1 = ((frame["actual_ko_tko"] == 1) & (frame["actual_finish_round"] == 1)).astype(int).to_numpy()

    print("\nFSR EDGE DISCRIMINATION")
    rows = []
    for col in edge_cols:
        score = frame[col].to_numpy(dtype=float)
        directional_col = col.replace("winner_edge_", "directional_advantage_")
        directional = frame[directional_col].to_numpy(dtype=float)
        rows.append(
            {
                "edge": col.replace("winner_edge_", ""),
                "auc_actual_KO": _auc(y_ko, score),
                "auc_actual_R1_KO": _auc(y_r1, score),
                "mean_KO": float(frame.loc[frame.actual_ko_tko.eq(1), col].mean()),
                "mean_non_KO": float(frame.loc[frame.actual_ko_tko.eq(0), col].mean()),
                "mean_directional_adv_KO": float(
                    frame.loc[frame.actual_ko_tko.eq(1), directional_col].mean()
                ),
                "corr_with_mc_pKO": float(pd.Series(score).corr(frame["mc_p_ko_tko"], method="spearman")),
                "corr_directional_with_mc_pKO": float(
                    pd.Series(directional).corr(frame["mc_p_ko_tko"], method="spearman")
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_quintiles(frame: pd.DataFrame, edge_cols: list[str]) -> None:
    print("\nACTUAL KO RATE BY WINNER-EDGE QUINTILE")
    for col in edge_cols:
        work = frame[[col, "actual_ko_tko", "actual_finish_round"]].dropna().copy()
        if work[col].nunique() < 5:
            continue
        work["q"] = pd.qcut(work[col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        table = (
            work.groupby("q", observed=True, as_index=False)
            .agg(
                bouts=("actual_ko_tko", "size"),
                mean_edge=(col, "mean"),
                actual_KO_rate=("actual_ko_tko", "mean"),
                actual_R1_KO_rate=(
                    "actual_finish_round",
                    lambda s: float(((s == 1) & work.loc[s.index, "actual_ko_tko"].eq(1)).mean()),
                ),
            )
        )
        print(f"\n{col.replace('winner_edge_', '')}")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_summary(frame: pd.DataFrame) -> None:
    edge_cols = [c for c in frame.columns if c.startswith("winner_edge_")]
    print("\n" + "=" * 128)
    print("HISTORICAL 300-BOUT KO FSR SIGNAL DIAGNOSTIC")
    print("=" * 128)
    print(f"bouts: {len(frame):,}")
    print(f"actual KO/TKO bouts: {int(frame['actual_ko_tko'].sum()):,} ({frame['actual_ko_tko'].mean():.2%})")
    print(f"actual R1 KO/TKO bouts: {int(((frame['actual_ko_tko'] == 1) & (frame['actual_finish_round'] == 1)).sum()):,}")
    print(f"FSR matchup edges evaluated: {len(edge_cols)}")

    _print_group_summary(frame, edge_cols)
    _print_discrimination(frame, edge_cols)
    _print_quintiles(frame, edge_cols)

    print("\nINTERPRETATION GUIDE")
    print("- AUC near 0.50 -> little/no standalone outcome signal in that edge.")
    print("- AUC >0.50 -> higher edge tends to occur more often in the target outcome.")
    print("- Strong R1-vs-non-KO separation would support the FSR as an early-finish signal.")
    print("- Weak edge signal plus good MC p(KO) would imply the MC is leaning on other mechanics/features.")
    print("- Strong edge signal plus weak MC timing would point to translation/mechanics rather than FSR construction.")
    print("- No simulator constants or FSR values are changed by this diagnostic.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KO-relevant FSR signal on the historical 300-bout cohort")
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation_frame = _load_validation(args.validation)
    bout_ids = set(validation_frame["bout_id"].astype(str))
    print(f"[KO FSR signal] validation bouts={len(validation_frame):,}", flush=True)
    pairs = _load_pairs(args.fsr_path, bout_ids)
    print(f"[KO FSR signal] matched leakage-safe FSR pairs={len(pairs):,}", flush=True)

    frame = _build_frame(validation_frame, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame)
    print(f"\n[KO FSR signal] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
