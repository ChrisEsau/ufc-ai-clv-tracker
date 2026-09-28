"""Study 6: pre-fight striking power / chin resistance versus realized knockdowns.

Shadow-only research. No simulator or FSR trait logic is modified.

Purpose
-------
Establish the empirical matchup relationships needed before calibrating the
latent damage-reservoir shock -> knockdown curve.

This study uses leakage-safe pre-fight FSR-26 snapshots and CURRENT-fight RFS
realized evidence. It tests:

1. Does higher pre-fight striking_power predict more knockdowns scored,
   especially per landed significant strike?
2. Does higher pre-fight chin_resistance predict fewer knockdowns absorbed?
   IMPORTANT: chin_resistance was originally trained from survival AFTER a KD,
   not first-KD avoidance. This is therefore a generalization test, not an
   assumption baked into the simulator.
3. Does striking_power - opponent chin_resistance improve realized KD
   separation?
4. Diagnostic: does striking_power - opponent damage_resistance separate KDs
   better than the chin edge?
5. Do the relationships persist within landed-strike exposure strata, reducing
   the chance that the result is just "more volume -> more KDs"?

No reservoir percentages are estimated here. The reservoir shock scale is
latent and must later be calibrated so simulated KD frequencies reproduce these
observable historical relationships.
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
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_power_chin_kd_study_v0.parquet"
)

RFS_COLUMNS = {
    "kd_scored": "rfs_finish_state_fight_knockdowns_scored",
    "kd_absorbed": "rfs_finish_state_fight_knockdowns_absorbed",
    "sig_landed": "rfs_finish_state_fight_sig_strikes_landed",
    "sig_absorbed": "rfs_finish_state_fight_sig_strikes_absorbed",
    "rounds": "rfs_finish_state_fight_rounds_observed",
    "ko_loss": "rfs_finish_state_fight_ko_tko_loss_indicator",
}

REQUIRED_FSR = {
    "fight_id",
    "fighter_id",
    "striking_power",
    "chin_resistance",
    "damage_resistance",
}
REQUIRED_RFS = {
    "fight_id",
    "fighter_id",
    "opponent_id",
    *RFS_COLUMNS.values(),
}


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(num) / float(den)


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"x": x[mask], "y": y[mask]}).corr(method="spearman").iloc[0, 1])


def _quintile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranks = numeric.rank(method="first", pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )


def _load(fsr_path: Path, rfs_path: Path) -> pd.DataFrame:
    print(f"[power/chin KD study] loading FSR-26: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path).copy()
    print(f"[power/chin KD study] FSR rows: {len(fsr):,}", flush=True)

    print(f"[power/chin KD study] loading RFS: {rfs_path}", flush=True)
    rfs = pd.read_parquet(rfs_path).copy()
    print(f"[power/chin KD study] RFS rows: {len(rfs):,}", flush=True)

    missing_fsr = sorted(REQUIRED_FSR - set(fsr.columns))
    missing_rfs = sorted(REQUIRED_RFS - set(rfs.columns))
    if missing_fsr:
        raise ValueError(f"FSR-26 missing required columns: {missing_fsr}")
    if missing_rfs:
        raise ValueError(f"RFS history missing required columns: {missing_rfs}")

    for frame in (fsr, rfs):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)
    rfs["opponent_id"] = rfs["opponent_id"].astype(str)

    keys = ["fight_id", "fighter_id"]
    if fsr.duplicated(keys).any():
        raise ValueError("FSR-26 violates fighter-fight grain")
    if rfs.duplicated(keys).any():
        raise ValueError("RFS history violates fighter-fight grain")

    own = fsr[[*keys, "striking_power", "chin_resistance", "damage_resistance"]].copy()
    own = own.rename(
        columns={
            "striking_power": "prefight_striking_power",
            "chin_resistance": "prefight_chin_resistance",
            "damage_resistance": "prefight_damage_resistance",
        }
    )

    opp = fsr[[*keys, "striking_power", "chin_resistance", "damage_resistance"]].copy()
    opp = opp.rename(
        columns={
            "fighter_id": "opponent_id",
            "striking_power": "opponent_prefight_striking_power",
            "chin_resistance": "opponent_prefight_chin_resistance",
            "damage_resistance": "opponent_prefight_damage_resistance",
        }
    )

    keep = ["fight_id", "fighter_id", "opponent_id", *RFS_COLUMNS.values()]
    work = rfs[keep].merge(own, on=keys, how="inner", validate="one_to_one")
    work = work.merge(
        opp,
        on=["fight_id", "opponent_id"],
        how="inner",
        validate="one_to_one",
    )

    for label, col in RFS_COLUMNS.items():
        work[label] = pd.to_numeric(work[col], errors="coerce")
    for col in (
        "prefight_striking_power",
        "prefight_chin_resistance",
        "prefight_damage_resistance",
        "opponent_prefight_striking_power",
        "opponent_prefight_chin_resistance",
        "opponent_prefight_damage_resistance",
    ):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    numeric_required = [
        "kd_scored", "kd_absorbed", "sig_landed", "sig_absorbed",
        "prefight_striking_power", "prefight_chin_resistance",
        "prefight_damage_resistance", "opponent_prefight_striking_power",
        "opponent_prefight_chin_resistance", "opponent_prefight_damage_resistance",
    ]
    if work[numeric_required].isna().any().any():
        counts = work[numeric_required].isna().sum()
        raise ValueError(f"Study frame contains missing required numeric values:\n{counts[counts > 0]}")

    work["kd_scored_indicator"] = (work["kd_scored"] > 0).astype(int)
    work["kd_absorbed_indicator"] = (work["kd_absorbed"] > 0).astype(int)
    work["kd_per_sig_landed"] = np.divide(
        work["kd_scored"], work["sig_landed"],
        out=np.full(len(work), np.nan),
        where=work["sig_landed"].to_numpy(dtype=float) > 0,
    )
    work["kd_absorbed_per_sig_absorbed"] = np.divide(
        work["kd_absorbed"], work["sig_absorbed"],
        out=np.full(len(work), np.nan),
        where=work["sig_absorbed"].to_numpy(dtype=float) > 0,
    )

    work["power_vs_chin_edge"] = (
        work["prefight_striking_power"] - work["opponent_prefight_chin_resistance"]
    )
    work["power_vs_damage_edge"] = (
        work["prefight_striking_power"] - work["opponent_prefight_damage_resistance"]
    )
    work["incoming_power_vs_chin_edge"] = (
        work["opponent_prefight_striking_power"] - work["prefight_chin_resistance"]
    )

    work["power_quintile"] = _quintile(work["prefight_striking_power"])
    work["chin_quintile"] = _quintile(work["prefight_chin_resistance"])
    work["power_chin_edge_quintile"] = _quintile(work["power_vs_chin_edge"])
    work["power_damage_edge_quintile"] = _quintile(work["power_vs_damage_edge"])
    work["sig_landed_exposure_quintile"] = _quintile(work["sig_landed"])

    print(f"[power/chin KD study] matched fighter-fight rows: {len(work):,}", flush=True)
    return work


def _bucket_table(frame: pd.DataFrame, bucket_col: str, rating_col: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for bucket, group in frame.groupby(bucket_col, observed=True, sort=False):
        kd_sum = float(group["kd_scored"].sum())
        sig_sum = float(group["sig_landed"].sum())
        rows.append(
            {
                "cohort": label,
                "bucket": str(bucket),
                "fighter_fights": int(len(group)),
                "mean_rating_or_edge": float(group[rating_col].mean()),
                "kd_scored_probability": float(group["kd_scored_indicator"].mean()),
                "mean_kd_scored": float(group["kd_scored"].mean()),
                "pooled_kd_per_sig_landed": _safe_ratio(kd_sum, sig_sum),
                "mean_sig_landed": float(group["sig_landed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _chin_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for bucket, group in frame.groupby("chin_quintile", observed=True, sort=False):
        kd_sum = float(group["kd_absorbed"].sum())
        sig_sum = float(group["sig_absorbed"].sum())
        rows.append(
            {
                "chin_quintile": str(bucket),
                "fighter_fights": int(len(group)),
                "mean_chin_resistance": float(group["prefight_chin_resistance"].mean()),
                "kd_absorbed_probability": float(group["kd_absorbed_indicator"].mean()),
                "mean_kd_absorbed": float(group["kd_absorbed"].mean()),
                "pooled_kd_per_sig_absorbed": _safe_ratio(kd_sum, sig_sum),
                "mean_sig_absorbed": float(group["sig_absorbed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _within_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Power/KD relationship within realized landed-strike exposure quintiles."""
    rows: list[dict[str, float | int | str]] = []
    for exposure, group in frame.groupby("sig_landed_exposure_quintile", observed=True, sort=False):
        if len(group) < 100:
            continue
        # Within each exposure stratum, split power into low/middle/high thirds.
        ranks = group["prefight_striking_power"].rank(method="first", pct=True)
        band = pd.cut(
            ranks,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["low", "middle", "high"],
            include_lowest=True,
        )
        temp = group.copy()
        temp["power_band"] = band
        for power_band, sub in temp.groupby("power_band", observed=True, sort=False):
            rows.append(
                {
                    "sig_landed_exposure_quintile": str(exposure),
                    "power_band_within_exposure": str(power_band),
                    "fighter_fights": int(len(sub)),
                    "mean_striking_power": float(sub["prefight_striking_power"].mean()),
                    "mean_sig_landed": float(sub["sig_landed"].mean()),
                    "kd_scored_probability": float(sub["kd_scored_indicator"].mean()),
                    "pooled_kd_per_sig_landed": _safe_ratio(
                        float(sub["kd_scored"].sum()), float(sub["sig_landed"].sum())
                    ),
                }
            )
    return pd.DataFrame(rows)


def _print_correlations(work: pd.DataFrame) -> None:
    print("\nCORE SPEARMAN RELATIONSHIPS")
    print("-" * 104)
    pairs = [
        ("striking_power vs KD scored indicator", "prefight_striking_power", "kd_scored_indicator"),
        ("striking_power vs KD count", "prefight_striking_power", "kd_scored"),
        ("striking_power vs KD / sig landed", "prefight_striking_power", "kd_per_sig_landed"),
        ("chin_resistance vs KD absorbed indicator", "prefight_chin_resistance", "kd_absorbed_indicator"),
        ("chin_resistance vs KD absorbed / sig absorbed", "prefight_chin_resistance", "kd_absorbed_per_sig_absorbed"),
        ("power - opp chin vs KD indicator", "power_vs_chin_edge", "kd_scored_indicator"),
        ("power - opp chin vs KD / sig landed", "power_vs_chin_edge", "kd_per_sig_landed"),
        ("power - opp damage vs KD indicator", "power_vs_damage_edge", "kd_scored_indicator"),
        ("power - opp damage vs KD / sig landed", "power_vs_damage_edge", "kd_per_sig_landed"),
    ]
    for label, a, b in pairs:
        print(f"{label:<52} {_safe_spearman(work[a], work[b]): .5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 6: FSR power/chin vs historical knockdowns")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--rfs-path", type=Path, default=RFS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.fsr_path, args.rfs_path)

    print("\n" + "=" * 122)
    print("STUDY 6 — STRIKING POWER / CHIN RESISTANCE / MATCHUP EDGE -> KNOCKDOWNS")
    print("=" * 122)
    print(f"fighter-fight observations: {len(work):,}")
    _print_correlations(work)

    print("\nSTRIKING POWER QUINTILES")
    print(_bucket_table(work, "power_quintile", "prefight_striking_power", "striking_power").to_string(
        index=False, float_format=lambda x: f"{x:.5f}"
    ))

    print("\nCHIN RESISTANCE QUINTILES — FIRST-KD GENERALIZATION TEST")
    print(_chin_table(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nPOWER - OPPONENT CHIN EDGE QUINTILES")
    print(_bucket_table(work, "power_chin_edge_quintile", "power_vs_chin_edge", "power - opp chin").to_string(
        index=False, float_format=lambda x: f"{x:.5f}"
    ))

    print("\nPOWER - OPPONENT DAMAGE RESISTANCE EDGE QUINTILES — DIAGNOSTIC")
    print(_bucket_table(work, "power_damage_edge_quintile", "power_vs_damage_edge", "power - opp damage").to_string(
        index=False, float_format=lambda x: f"{x:.5f}"
    ))

    print("\nSTRIKING POWER WITHIN LANDED-STRIKE EXPOSURE STRATA")
    print(_within_exposure(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nINTERPRETATION RULE: striking_power should show increasing KD probability and "
        "KD-per-landed-strike. A useful initial-KD resistance trait should show the "
        "opposite pattern. Because chin_resistance was trained on post-KD survival, "
        "failure to predict first-KD avoidance would mean we should NOT use it as the "
        "initial shock threshold modifier."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(args.output, index=False)
    print(f"\n[power/chin KD study] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
