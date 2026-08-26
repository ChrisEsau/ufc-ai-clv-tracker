"""Historical round-level power-fatigue proxy audit.

Purpose
-------
Estimate how knockdown-producing threat changes as UFC fights progress, without
using simulator assumptions. UFCStats does not measure punch force, so this
script treats knockdowns per landed-strike exposure as an empirical proxy for
striking potency.

This is intentionally separate from the rolling-stamina simulator. It does not
change FSR values or MC constants.

Cohort
------
All authoritative UFCStats fighter-round rows that can be matched to the UFC
master fight table. A fighter-round is included only when that round was
actually reached. We do NOT restrict to decisions, because doing so would
remove the KO/TKO signal we are trying to measure.

Outputs
-------
1. Console summary by round.
2. Fighter-round CSV with exposure and KD indicators.
3. Aggregate CSV with raw and normalized round potency proxies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow")
DETAIL_OUTPUT_PATH = OUTPUT_DIR / "historical_power_fatigue_fighter_rounds.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "historical_power_fatigue_summary.csv"


def _resolve_date_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("event_date", "date", "fight_date"):
        if candidate in frame.columns:
            return candidate
    return None


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _load_round_stats() -> pd.DataFrame:
    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(f"Round-stats artifact not found: {ROUND_STATS_PATH}")

    frame = pd.read_parquet(ROUND_STATS_PATH).copy()
    required = {
        "fight_id",
        "fighter_id",
        "round",
        "kd",
        "sig_str_landed",
        "head_landed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Round stats missing required columns: {missing}")

    for column in ("round", "kd", "sig_str_landed", "head_landed"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(
        subset=["fight_id", "fighter_id", "round", "kd", "sig_str_landed", "head_landed"]
    ).copy()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    frame["round"] = frame["round"].astype(int)

    if (frame[["kd", "sig_str_landed", "head_landed"]] < 0).any().any():
        raise ValueError("Round stats contain negative KD/strike counts")

    dupes = frame.duplicated(["fight_id", "fighter_id", "round"], keep=False)
    if dupes.any():
        raise ValueError(
            "Round stats violate fighter-round grain: "
            f"{int(dupes.sum())} duplicate rows"
        )

    return frame


def _load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master artifact not found: {MASTER_PATH}")

    master = pd.read_parquet(MASTER_PATH).copy()
    required = {"fight_id", "method", "finish_round"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master missing required columns: {missing}")

    keep = ["fight_id", "method", "finish_round"]
    date_col = _resolve_date_column(master)
    if date_col is not None:
        keep.append(date_col)
    if "weight_class" in master.columns:
        keep.append("weight_class")
    elif "division" in master.columns:
        keep.append("division")

    master = master[keep].copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["finish_round"] = pd.to_numeric(master["finish_round"], errors="coerce")
    master = master.drop_duplicates("fight_id", keep="last")
    return master


def _is_ko_tko(method: object) -> int:
    if pd.isna(method):
        return 0
    text = str(method).strip().lower()
    return int("ko" in text or "tko" in text)


def _prepare_detail(rounds: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    detail = rounds.merge(master, on="fight_id", how="inner", validate="many_to_one")
    detail["fight_ko_tko"] = detail["method"].map(_is_ko_tko).astype(int)
    detail["kd_any"] = detail["kd"].gt(0).astype(int)

    # Exposure proxies. Head-landed is the more power-specific denominator;
    # sig-landed is broader and much denser, so report both.
    detail["kd_per_head_landed_row"] = np.where(
        detail["head_landed"].gt(0),
        detail["kd"] / detail["head_landed"],
        np.nan,
    )
    detail["kd_per_sig_landed_row"] = np.where(
        detail["sig_str_landed"].gt(0),
        detail["kd"] / detail["sig_str_landed"],
        np.nan,
    )
    return detail


def _aggregate_by_round(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for round_no, group in detail.groupby("round", sort=True):
        n_rows = len(group)
        n_fights = group["fight_id"].nunique()
        kd_total = float(group["kd"].sum())
        head_total = float(group["head_landed"].sum())
        sig_total = float(group["sig_str_landed"].sum())
        rows.append(
            {
                "round": int(round_no),
                "fighter_rounds": int(n_rows),
                "fights_reached_round": int(n_fights),
                "knockdowns": kd_total,
                "head_landed": head_total,
                "sig_str_landed": sig_total,
                "kd_per_100_head_landed": 100.0 * _safe_div(kd_total, head_total),
                "kd_per_100_sig_landed": 100.0 * _safe_div(kd_total, sig_total),
                "fighter_round_kd_rate": float(group["kd_any"].mean()),
                "mean_kd_per_fighter_round": float(group["kd"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    if summary.empty:
        return summary

    r1 = summary.loc[summary["round"].eq(1)]
    if r1.empty:
        raise ValueError("No Round 1 rows available for normalization")

    for metric in (
        "kd_per_100_head_landed",
        "kd_per_100_sig_landed",
        "fighter_round_kd_rate",
        "mean_kd_per_fighter_round",
    ):
        baseline = float(r1.iloc[0][metric])
        summary[f"{metric}_vs_r1"] = (
            summary[metric] / baseline if baseline > 0 else np.nan
        )

    return summary


def _print_summary(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    print("=" * 150)
    print("HISTORICAL POWER-FATIGUE PROXY — ALL UFC FIGHTER-ROUNDS")
    print("=" * 150)
    print(f"matched fights: {detail['fight_id'].nunique():,}")
    print(f"fighter-round rows: {len(detail):,}")
    print(
        "Interpretation: KD/landed-strike is a historical proxy for knockdown-producing potency, "
        "not a direct measurement of punch force or physiological stamina."
    )
    print()

    display_cols = [
        "round",
        "fights_reached_round",
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
    print("ROUND POTENCY PROXIES")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # Also show KO/TKO-ending-fight composition among the fights represented in
    # each round. This is descriptive only; the KD exposure metrics above are the
    # calibration targets.
    composition = (
        detail.groupby("round", as_index=False)
        .agg(
            fights=("fight_id", "nunique"),
            ko_tko_fighter_round_share=("fight_ko_tko", "mean"),
        )
    )
    print("DESCRIPTIVE KO/TKO COMPOSITION")
    print(composition.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    rounds = _load_round_stats()
    master = _load_master()
    detail = _prepare_detail(rounds, master)
    summary = _aggregate_by_round(detail)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    _print_summary(summary, detail)
    print()
    print(f"Wrote fighter-round rows to {DETAIL_OUTPUT_PATH}")
    print(f"Wrote round summary to {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
