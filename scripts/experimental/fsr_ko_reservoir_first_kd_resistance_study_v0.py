"""Study 7: discover leakage-safe first-knockdown resistance signals.

Shadow-only research. No production FSR or simulator logic is modified.

Purpose
-------
The current ``chin_resistance`` trait was trained on survival after a knockdown,
not avoidance of the first knockdown. Study 6 showed it does not generalize to
initial-KD susceptibility. This study therefore builds simple leakage-safe
historical candidate signals directly from PRIOR fights and tests whether they
predict current-fight knockdown absorption.

Candidate signals
-----------------
1. prior_kd_avoidance_per_sig
   Smoothed inverse KD rate under prior significant-strike exposure.
2. prior_kd_free_high_exposure_rate
   Share of prior high-damage-exposure fights survived without a KD.
3. prior_kd_free_fight_rate
   Share of all prior fights with zero KD absorbed.
4. candidate_knockdown_resistance
   Equal-weight percentile blend of the three signals above.

This is an ontology-discovery study, not a final trait builder. A useful signal
should show lower current KD probability and KD-per-sig-absorbed as resistance
rises, and ``striking_power - opponent candidate resistance`` should separate
KD production better than striking power alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_first_kd_resistance_study_v0.parquet"
)

KD_COL = "rfs_finish_state_fight_knockdowns_absorbed"
SIG_ABS_COL = "rfs_finish_state_fight_sig_strikes_absorbed"
KD_SCORED_COL = "rfs_finish_state_fight_knockdowns_scored"
SIG_LANDED_COL = "rfs_finish_state_fight_sig_strikes_landed"


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    return float(a) / float(b)


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"x": x[mask], "y": y[mask]}).corr(method="spearman").iloc[0, 1])


def _percentile_rank(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)


def _load(rfs_path: Path, fsr_path: Path) -> pd.DataFrame:
    print(f"[first-KD resistance] loading RFS: {rfs_path}", flush=True)
    rfs = pd.read_parquet(rfs_path).copy()
    print(f"[first-KD resistance] RFS rows: {len(rfs):,}", flush=True)

    print(f"[first-KD resistance] loading FSR-26: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path).copy()
    print(f"[first-KD resistance] FSR rows: {len(fsr):,}", flush=True)

    required_rfs = {"fight_id", "fighter_id", "opponent_id", "date", KD_COL, SIG_ABS_COL, KD_SCORED_COL, SIG_LANDED_COL}
    missing_rfs = sorted(required_rfs - set(rfs.columns))
    if missing_rfs:
        raise ValueError(f"RFS missing required columns: {missing_rfs}")

    required_fsr = {"fight_id", "fighter_id", "striking_power"}
    missing_fsr = sorted(required_fsr - set(fsr.columns))
    if missing_fsr:
        raise ValueError(f"FSR-26 missing required columns: {missing_fsr}")

    for frame in (rfs, fsr):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)
    rfs["opponent_id"] = rfs["opponent_id"].astype(str)
    rfs["date"] = pd.to_datetime(rfs["date"], errors="coerce")

    keep_fsr = ["fight_id", "fighter_id", "striking_power"]
    if "damage_resistance" in fsr.columns:
        keep_fsr.append("damage_resistance")
    if "chin_resistance" in fsr.columns:
        keep_fsr.append("chin_resistance")

    merged = rfs.merge(
        fsr[keep_fsr],
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    print(f"[first-KD resistance] merged fighter-fights: {len(merged):,}", flush=True)
    return merged.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)


def _build_prior_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["kd_absorbed"] = pd.to_numeric(work[KD_COL], errors="coerce").fillna(0.0)
    work["sig_absorbed"] = pd.to_numeric(work[SIG_ABS_COL], errors="coerce").fillna(0.0)
    work["kd_scored"] = pd.to_numeric(work[KD_SCORED_COL], errors="coerce").fillna(0.0)
    work["sig_landed"] = pd.to_numeric(work[SIG_LANDED_COL], errors="coerce").fillna(0.0)
    work["current_kd_indicator"] = (work["kd_absorbed"] > 0).astype(int)

    # High exposure is defined from the historical fighter-fight distribution,
    # then only prior fights contribute to each fighter's state.
    high_exposure_threshold = float(work["sig_absorbed"].quantile(0.67))
    print(
        f"[first-KD resistance] high-exposure threshold: "
        f"{high_exposure_threshold:.1f} sig strikes absorbed",
        flush=True,
    )
    work["high_exposure_fight"] = (work["sig_absorbed"] >= high_exposure_threshold).astype(int)
    work["kd_free_fight"] = (work["kd_absorbed"] <= 0).astype(int)
    work["kd_free_high_exposure"] = (
        (work["high_exposure_fight"] == 1) & (work["kd_free_fight"] == 1)
    ).astype(int)

    group_keys = work["fighter_id"]

    def prior_cumsum(series: pd.Series) -> pd.Series:
        return series.groupby(group_keys).cumsum() - series

    work["prior_fights"] = work.groupby("fighter_id").cumcount().astype(float)
    work["prior_kd_absorbed"] = prior_cumsum(work["kd_absorbed"])
    work["prior_sig_absorbed"] = prior_cumsum(work["sig_absorbed"])
    work["prior_kd_free_fights"] = prior_cumsum(work["kd_free_fight"])
    work["prior_high_exposure_fights"] = prior_cumsum(work["high_exposure_fight"])
    work["prior_kd_free_high_exposure"] = prior_cumsum(work["kd_free_high_exposure"])

    # Smoothed KD avoidance under strike exposure. The pseudocount keeps very
    # low-history fighters from producing extreme ratios.
    raw_kd_rate = (work["prior_kd_absorbed"] + 0.5) / (work["prior_sig_absorbed"] + 50.0)
    work["prior_kd_avoidance_per_sig"] = -raw_kd_rate

    work["prior_kd_free_fight_rate"] = np.divide(
        work["prior_kd_free_fights"],
        work["prior_fights"],
        out=np.full(len(work), np.nan),
        where=work["prior_fights"].to_numpy(dtype=float) > 0,
    )
    work["prior_kd_free_high_exposure_rate"] = np.divide(
        work["prior_kd_free_high_exposure"],
        work["prior_high_exposure_fights"],
        out=np.full(len(work), np.nan),
        where=work["prior_high_exposure_fights"].to_numpy(dtype=float) > 0,
    )

    # Candidate composite requires at least two prior fights. Missing high-
    # exposure evidence is allowed; available components are averaged.
    components = pd.DataFrame(
        {
            "a": _percentile_rank(work["prior_kd_avoidance_per_sig"]),
            "b": _percentile_rank(work["prior_kd_free_fight_rate"]),
            "c": _percentile_rank(work["prior_kd_free_high_exposure_rate"]),
        }
    )
    work["candidate_knockdown_resistance"] = components.mean(axis=1, skipna=True) * 100.0
    work.loc[work["prior_fights"] < 2, "candidate_knockdown_resistance"] = np.nan

    return work


def _attach_opponent_candidate(work: pd.DataFrame) -> pd.DataFrame:
    opponent = work[
        ["fight_id", "fighter_id", "candidate_knockdown_resistance"]
    ].rename(
        columns={
            "fighter_id": "opponent_id",
            "candidate_knockdown_resistance": "opponent_candidate_knockdown_resistance",
        }
    )
    out = work.merge(
        opponent,
        on=["fight_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
    out["power_minus_opponent_candidate"] = (
        pd.to_numeric(out["striking_power"], errors="coerce")
        - pd.to_numeric(out["opponent_candidate_knockdown_resistance"], errors="coerce")
    )
    out["kd_per_sig_absorbed"] = np.divide(
        out["kd_absorbed"],
        out["sig_absorbed"],
        out=np.full(len(out), np.nan),
        where=out["sig_absorbed"].to_numpy(dtype=float) > 0,
    )
    out["kd_scored_per_sig_landed"] = np.divide(
        out["kd_scored"],
        out["sig_landed"],
        out=np.full(len(out), np.nan),
        where=out["sig_landed"].to_numpy(dtype=float) > 0,
    )
    return out


def _bucket_table(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    eligible = frame[frame[column].notna()].copy()
    ranks = eligible[column].rank(method="first", pct=True)
    eligible["bucket"] = pd.cut(
        ranks,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in eligible.groupby("bucket", observed=True, sort=False):
        rows.append(
            {
                "cohort": label,
                "bucket": str(bucket),
                "fighter_fights": int(len(group)),
                "mean_signal": float(group[column].mean()),
                "current_kd_absorbed_probability": float(group["current_kd_indicator"].mean()),
                "mean_kd_absorbed": float(group["kd_absorbed"].mean()),
                "pooled_kd_per_sig_absorbed": _safe_ratio(
                    float(group["kd_absorbed"].sum()), float(group["sig_absorbed"].sum())
                ),
                "mean_sig_absorbed": float(group["sig_absorbed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _power_edge_table(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["power_minus_opponent_candidate"].notna()].copy()
    ranks = eligible["power_minus_opponent_candidate"].rank(method="first", pct=True)
    eligible["bucket"] = pd.cut(
        ranks,
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        include_lowest=True,
    )
    rows = []
    for bucket, group in eligible.groupby("bucket", observed=True, sort=False):
        rows.append(
            {
                "bucket": str(bucket),
                "fighter_fights": int(len(group)),
                "mean_power_minus_opponent_candidate": float(group["power_minus_opponent_candidate"].mean()),
                "kd_scored_probability": float((group["kd_scored"] > 0).mean()),
                "mean_kd_scored": float(group["kd_scored"].mean()),
                "pooled_kd_per_sig_landed": _safe_ratio(
                    float(group["kd_scored"].sum()), float(group["sig_landed"].sum())
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 7: first-KD resistance candidate discovery")
    parser.add_argument("--rfs-path", type=Path, default=RFS_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.rfs_path, args.fsr_path)
    work = _build_prior_candidates(work)
    work = _attach_opponent_candidate(work)

    eligible = work[work["candidate_knockdown_resistance"].notna()].copy()
    print("\n" + "=" * 118)
    print("STUDY 7 — FIRST-KD RESISTANCE CANDIDATE DISCOVERY")
    print("=" * 118)
    print(f"eligible fighter-fights with >=2 prior fights: {len(eligible):,}")

    candidates = [
        "prior_kd_avoidance_per_sig",
        "prior_kd_free_fight_rate",
        "prior_kd_free_high_exposure_rate",
        "candidate_knockdown_resistance",
    ]
    print("\nCORE SPEARMAN RELATIONSHIPS")
    for column in candidates:
        print(
            f"{column:42s} vs KD indicator  "
            f"{_safe_spearman(eligible[column], eligible['current_kd_indicator']): .5f}"
        )
        print(
            f"{column:42s} vs KD/sig abs  "
            f"{_safe_spearman(eligible[column], eligible['kd_per_sig_absorbed']): .5f}"
        )

    print("\nCANDIDATE KNOCKDOWN RESISTANCE QUINTILES")
    print(
        _bucket_table(
            eligible,
            "candidate_knockdown_resistance",
            "candidate resistance",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nPOWER - OPPONENT CANDIDATE RESISTANCE EDGE")
    edge = work[work["power_minus_opponent_candidate"].notna()].copy()
    print(
        "Spearman power-candidate edge vs KD scored indicator: "
        f"{_safe_spearman(edge['power_minus_opponent_candidate'], (edge['kd_scored'] > 0).astype(int)):.5f}"
    )
    print(
        "Spearman power-candidate edge vs KD/sig landed   : "
        f"{_safe_spearman(edge['power_minus_opponent_candidate'], edge['kd_scored_per_sig_landed']):.5f}"
    )
    print(_power_edge_table(edge).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nINTERPRETATION RULE: a useful first-KD resistance signal should show "
        "monotonically LOWER KD absorption as resistance rises. The power-minus-"
        "opponent-resistance edge should ideally improve separation over power alone."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(args.output, index=False)
    print(f"\n[first-KD resistance] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
