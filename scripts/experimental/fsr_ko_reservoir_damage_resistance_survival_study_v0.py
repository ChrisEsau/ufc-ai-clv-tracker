"""Study 5: pre-fight damage resistance versus KO/TKO survival under adversity.

Shadow-only research. No simulator or FSR trait logic is modified.

Purpose
-------
Test whether leakage-safe pre-fight ``damage_resistance`` predicts survival after
comparable damaging exposure.  The study deliberately conditions on realized
adversity rather than asking only whether damage_resistance correlates with KO/TKO
loss in the full population.

Primary questions
-----------------
1. Among fighters who absorbed at least one knockdown, does higher pre-fight
   damage_resistance correspond to lower KO/TKO loss probability?
2. Within similar realized damage-exposure strata, does damage_resistance still
   separate survivors from KO/TKO losses?
3. Within the same KD-count stratum (1 KD versus 2+ KD), is the direction stable?

Important interpretation limitation
-----------------------------------
Conditioning on realized fight adversity is useful for simulator calibration but
is not a causal estimate.  Exposure is itself affected by matchup style, fight
length, and the path by which a finish occurred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
FIGHT_STUDY_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_historical_study_v0_fights.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_damage_resistance_survival_study_v0.parquet"
)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"a": a[mask], "b": b[mask]}).corr(method="spearman").iloc[0, 1])


def _rank_quintile(series: pd.Series, labels: list[str]) -> pd.Series:
    ranks = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=labels,
        include_lowest=True,
    )


def _load(fsr_path: Path, fight_path: Path) -> pd.DataFrame:
    print(f"[damage resistance study] loading FSR snapshots: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path)
    print(f"[damage resistance study] FSR rows: {len(fsr):,}", flush=True)
    print(f"[damage resistance study] loading fight study rows: {fight_path}", flush=True)
    fights = pd.read_parquet(fight_path)
    print(f"[damage resistance study] fight rows: {len(fights):,}", flush=True)

    required_fsr = {"fight_id", "fighter_id", "damage_resistance"}
    missing_fsr = sorted(required_fsr - set(fsr.columns))
    if missing_fsr:
        raise ValueError(f"FSR artifact missing required columns: {missing_fsr}")

    required_fight = {
        "fight_id",
        "fighter_id",
        "kd_absorbed",
        "sig_absorbed",
        "head_absorbed",
        "ground_absorbed",
        "opponent_control_seconds",
        "rounds_observed",
        "ko_tko_loss",
    }
    missing_fight = sorted(required_fight - set(fights.columns))
    if missing_fight:
        raise ValueError(f"fight study artifact missing required columns: {missing_fight}")

    for frame in (fsr, fights):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)

    keep = ["fight_id", "fighter_id", "damage_resistance"]
    if "damage_resistance_updates" in fsr.columns:
        keep.append("damage_resistance_updates")

    if fsr.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("FSR snapshots violate fighter-fight grain")

    work = fights.merge(
        fsr[keep],
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    print(f"[damage resistance study] joined fighter-fights: {len(work):,}", flush=True)

    numeric = [
        "damage_resistance",
        "kd_absorbed",
        "sig_absorbed",
        "head_absorbed",
        "ground_absorbed",
        "opponent_control_seconds",
        "rounds_observed",
        "ko_tko_loss",
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["damage_resistance", "ko_tko_loss", "rounds_observed"]).copy()
    rounds = work["rounds_observed"].clip(lower=1.0)

    # Mirror the Finish State damage-exposure components at fighter-fight grain.
    work["kd_per_round"] = work["kd_absorbed"] / rounds
    work["head_per_round"] = work["head_absorbed"] / rounds
    work["ground_per_round"] = work["ground_absorbed"] / rounds
    work["opponent_control_per_round_min"] = work["opponent_control_seconds"] / (rounds * 60.0)
    work["damage_exposure"] = work[
        [
            "kd_per_round",
            "head_per_round",
            "ground_per_round",
            "opponent_control_per_round_min",
        ]
    ].mean(axis=1)

    kd = work["kd_absorbed"].fillna(0.0)
    work["kd_group"] = np.select(
        [kd <= 0, kd == 1, kd >= 2],
        ["0 KD", "1 KD", "2+ KD"],
        default="unknown",
    )
    work["damage_resistance_quintile"] = _rank_quintile(
        work["damage_resistance"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    work["damage_exposure_quintile"] = _rank_quintile(
        work["damage_exposure"],
        ["E1 lowest", "E2", "E3", "E4", "E5 highest"],
    )
    return work


def _resistance_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    order = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    for q in order:
        group = frame[frame["damage_resistance_quintile"].astype(str) == q]
        if group.empty:
            continue
        rows.append(
            {
                "cohort": label,
                "damage_resistance_quintile": q,
                "fighter_fights": int(len(group)),
                "mean_damage_resistance": float(group["damage_resistance"].mean()),
                "mean_kd_absorbed": float(group["kd_absorbed"].mean()),
                "mean_damage_exposure": float(group["damage_exposure"].mean()),
                "ko_tko_loss_probability": float(group["ko_tko_loss"].mean()),
                "survival_probability": float(1.0 - group["ko_tko_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _matched_exposure_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    exposure_order = ["E1 lowest", "E2", "E3", "E4", "E5 highest"]
    for exposure in exposure_order:
        group = frame[frame["damage_exposure_quintile"].astype(str) == exposure].copy()
        if len(group) < 40:
            continue
        # Re-form resistance tertiles WITHIN each exposure stratum so the
        # comparison is genuinely conditional on similar realized adversity.
        ranks = group["damage_resistance"].rank(method="first", pct=True)
        group["resistance_band"] = pd.cut(
            ranks,
            bins=[0.0, 1/3, 2/3, 1.0],
            labels=["low", "middle", "high"],
            include_lowest=True,
        )
        for band in ("low", "middle", "high"):
            sub = group[group["resistance_band"].astype(str) == band]
            if sub.empty:
                continue
            rows.append(
                {
                    "damage_exposure_quintile": exposure,
                    "resistance_band_within_exposure": band,
                    "fighter_fights": int(len(sub)),
                    "mean_damage_resistance": float(sub["damage_resistance"].mean()),
                    "mean_damage_exposure": float(sub["damage_exposure"].mean()),
                    "mean_kd_absorbed": float(sub["kd_absorbed"].mean()),
                    "ko_tko_loss_probability": float(sub["ko_tko_loss"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _kd_strata_table(frame: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for kd_group in ("1 KD", "2+ KD"):
        group = frame[frame["kd_group"] == kd_group]
        if group.empty:
            continue
        # Quintiles should be cohort-local here; global quintiles can leave tiny
        # tails in the 2+ KD sample.
        group = group.copy()
        group["damage_resistance_quintile"] = _rank_quintile(
            group["damage_resistance"],
            ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        )
        tables.append(_resistance_table(group, kd_group))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 5: damage resistance and KO/TKO survival")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--fight-path", type=Path, default=FIGHT_STUDY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.fsr_path, args.fight_path)

    print("\n" + "=" * 118)
    print("STUDY 5A — DAMAGE RESISTANCE VS KO/TKO LOSS, ALL FIGHTS")
    print("=" * 118)
    print(
        f"Spearman damage_resistance vs KO/TKO loss: "
        f"{_safe_spearman(work['damage_resistance'], work['ko_tko_loss']):.5f}"
    )
    print(_resistance_table(work, "All fights").to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    kd_exposed = work[work["kd_absorbed"] >= 1].copy()
    kd_exposed["damage_resistance_quintile"] = _rank_quintile(
        kd_exposed["damage_resistance"],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    print("\n" + "=" * 118)
    print("STUDY 5B — AMONG FIGHTERS WHO ABSORBED >=1 KD")
    print("=" * 118)
    print(f"KD-exposed fighter-fights: {len(kd_exposed):,}")
    print(
        f"Spearman damage_resistance vs KO/TKO loss: "
        f"{_safe_spearman(kd_exposed['damage_resistance'], kd_exposed['ko_tko_loss']):.5f}"
    )
    print(_resistance_table(kd_exposed, ">=1 KD").to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n" + "=" * 118)
    print("STUDY 5C — WITHIN KD-COUNT STRATA")
    print("=" * 118)
    print(_kd_strata_table(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n" + "=" * 118)
    print("STUDY 5D — RESISTANCE WITHIN MATCHED DAMAGE-EXPOSURE QUINTILES")
    print("=" * 118)
    print(_matched_exposure_table(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nNOTE: Study 5 is a conditional calibration study, not a causal estimate. "
        "The strongest evidence would be a stable decrease in KO/TKO loss as "
        "damage_resistance rises within >=1 KD and matched-exposure cohorts."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(args.output, index=False)
    print(f"\n[damage resistance study] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
