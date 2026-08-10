"""Diagnose whether KO-relevant FSR traits contain historical finish signal.

Uses the existing 300-bout historical KO validation cohort and leakage-safe FSR
pre-fight profiles. No simulator rerun is required.

Primary KO-occurrence diagnostics use a symmetric pre-fight ``max danger edge``:
for each attacker-vs-defender trait pairing, compute Red->Blue and Blue->Red and
retain the larger edge. This avoids using the eventual winner to orient a feature
when asking whether the bout ends by KO/TKO.

Winner-oriented edges are retained only for resolved bouts so we can separately
ask whether the actual KO winner had the stronger pre-fight finish signal. Draws,
no contests, and rows with missing winner IDs remain valid negative examples for
KO occurrence and no longer crash the study.

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
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
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


def _valid_winner_id(value: object, red_id: str, blue_id: str) -> str | None:
    if pd.isna(value):
        return None
    winner_id = str(value)
    return winner_id if winner_id in {red_id, blue_id} else None


def _build_frame(validation_frame: pd.DataFrame, pairs: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, bout in validation_frame.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        red_id = str(red["fighter_id"])
        blue_id = str(blue["fighter_id"])
        winner_id = _valid_winner_id(bout["actual_winner_id"], red_id, blue_id)

        row: dict[str, object] = {
            "bout_id": bout_id,
            "actual_ko_tko": int(bout["actual_ko_tko"]),
            "actual_finish_round": bout["actual_finish_round"],
            "actual_winner_id": winner_id,
            "resolved_winner": int(winner_id is not None),
            "timing_group": _timing_group(bout),
            "mc_p_ko_tko": float(bout["mc_p_ko_tko"]),
        }

        for edge_name, attacker_trait, defender_trait in EDGE_SPECS:
            if (
                attacker_trait not in red.index
                or attacker_trait not in blue.index
                or defender_trait not in red.index
                or defender_trait not in blue.index
            ):
                continue

            red_to_blue = _numeric(red, attacker_trait) - _numeric(blue, defender_trait)
            blue_to_red = _numeric(blue, attacker_trait) - _numeric(red, defender_trait)
            row[f"red_to_blue_edge_{edge_name}"] = red_to_blue
            row[f"blue_to_red_edge_{edge_name}"] = blue_to_red
            row[f"max_danger_edge_{edge_name}"] = max(red_to_blue, blue_to_red)
            row[f"edge_separation_{edge_name}"] = abs(red_to_blue - blue_to_red)

            if winner_id == red_id:
                winner_edge, loser_edge = red_to_blue, blue_to_red
            elif winner_id == blue_id:
                winner_edge, loser_edge = blue_to_red, red_to_blue
            else:
                winner_edge = float("nan")
                loser_edge = float("nan")

            row[f"winner_edge_{edge_name}"] = winner_edge
            row[f"loser_edge_{edge_name}"] = loser_edge
            row[f"directional_advantage_{edge_name}"] = winner_edge - loser_edge

        rows.append(row)

    return pd.DataFrame(rows)


def _print_group_summary(frame: pd.DataFrame, danger_cols: list[str]) -> None:
    print("\nMAX PRE-FIGHT DANGER EDGE BY ACTUAL OUTCOME/TIMING")
    rows: list[dict[str, object]] = []
    order = ["non-KO", "actual R1 KO", "actual R2 KO", "actual R3+ KO"]
    for group in order:
        g = frame[frame["timing_group"] == group]
        if g.empty:
            continue
        row: dict[str, object] = {"group": group, "bouts": len(g)}
        for col in danger_cols:
            row[col.replace("max_danger_edge_", "")] = float(g[col].mean())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_discrimination(frame: pd.DataFrame, danger_cols: list[str]) -> None:
    y_ko = frame["actual_ko_tko"].to_numpy(dtype=int)
    y_r1 = ((frame["actual_ko_tko"] == 1) & (frame["actual_finish_round"] == 1)).astype(int).to_numpy()

    print("\nFSR MAX-DANGER EDGE DISCRIMINATION")
    rows = []
    for col in danger_cols:
        score = frame[col].to_numpy(dtype=float)
        edge_name = col.replace("max_danger_edge_", "")
        directional_col = f"directional_advantage_{edge_name}"
        ko_directional = frame.loc[frame["actual_ko_tko"].eq(1), directional_col]
        rows.append(
            {
                "edge": edge_name,
                "auc_actual_KO": _auc(y_ko, score),
                "auc_actual_R1_KO": _auc(y_r1, score),
                "mean_KO": float(frame.loc[frame.actual_ko_tko.eq(1), col].mean()),
                "mean_non_KO": float(frame.loc[frame.actual_ko_tko.eq(0), col].mean()),
                "mean_winner_directional_adv_KO": float(ko_directional.mean()),
                "KO_winner_positive_direction_rate": float((ko_directional.dropna() > 0).mean()),
                "corr_with_mc_pKO": float(frame[col].corr(frame["mc_p_ko_tko"], method="spearman")),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_quintiles(frame: pd.DataFrame, danger_cols: list[str]) -> None:
    print("\nACTUAL KO RATE BY MAX-DANGER EDGE QUINTILE")
    for col in danger_cols:
        work = frame[[col, "actual_ko_tko", "actual_finish_round"]].dropna().copy()
        if work[col].nunique() < 5:
            continue
        work["q"] = pd.qcut(work[col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        work["actual_r1_ko"] = ((work["actual_ko_tko"] == 1) & (work["actual_finish_round"] == 1)).astype(int)
        table = (
            work.groupby("q", observed=True, as_index=False)
            .agg(
                bouts=("actual_ko_tko", "size"),
                mean_edge=(col, "mean"),
                actual_KO_rate=("actual_ko_tko", "mean"),
                actual_R1_KO_rate=("actual_r1_ko", "mean"),
            )
        )
        print(f"\n{col.replace('max_danger_edge_', '')}")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_summary(frame: pd.DataFrame) -> None:
    danger_cols = [c for c in frame.columns if c.startswith("max_danger_edge_")]
    unresolved = int((frame["resolved_winner"] == 0).sum())
    unresolved_ko = int(((frame["resolved_winner"] == 0) & frame["actual_ko_tko"].eq(1)).sum())

    print("\n" + "=" * 128)
    print("HISTORICAL 300-BOUT KO FSR SIGNAL DIAGNOSTIC")
    print("=" * 128)
    print(f"bouts: {len(frame):,}")
    print(f"actual KO/TKO bouts: {int(frame['actual_ko_tko'].sum()):,} ({frame['actual_ko_tko'].mean():.2%})")
    print(f"actual R1 KO/TKO bouts: {int(((frame['actual_ko_tko'] == 1) & (frame['actual_finish_round'] == 1)).sum()):,}")
    print(f"unresolved/missing-winner bouts retained for occurrence analysis: {unresolved:,}")
    print(f"unresolved actual KO bouts: {unresolved_ko:,}")
    print(f"FSR matchup edges evaluated: {len(danger_cols)}")

    _print_group_summary(frame, danger_cols)
    _print_discrimination(frame, danger_cols)
    _print_quintiles(frame, danger_cols)

    print("\nINTERPRETATION GUIDE")
    print("- max_danger_edge is pre-fight and symmetric: max(Red->Blue edge, Blue->Red edge).")
    print("- AUC near 0.50 -> little/no standalone KO occurrence signal in that FSR edge.")
    print("- AUC >0.50 -> larger pre-fight danger edge tends to occur more often in the target outcome.")
    print("- Rising Q1->Q5 actual KO/R1-KO rates would support useful monotonic finish signal.")
    print("- Winner directional advantage is evaluated only where an actual winner is resolved.")
    print("- Strong max-danger signal plus weak MC timing points to simulator translation/mechanics.")
    print("- Weak max-danger signal points toward revisiting the FSR finish-trait construction.")
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
    print(
        f"[KO FSR signal] built rows={len(frame):,}; unresolved winners="
        f"{int((frame['resolved_winner'] == 0).sum()):,}",
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    _print_summary(frame)
    print(f"\n[KO FSR signal] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
