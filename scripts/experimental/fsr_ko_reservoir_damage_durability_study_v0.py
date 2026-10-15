"""Study 8: discover leakage-safe damage durability / reservoir-capacity signals.

Shadow-only research. No simulator or production FSR logic is modified.

Purpose
-------
Build candidate pre-fight durability signals from PRIOR fighter-fight history and
ask whether they predict future KO/TKO survival under damaging exposure better
than the current ``damage_resistance`` rating.

The candidates intentionally target accumulated-damage tolerance rather than
first-knockdown avoidance.  Study 7 already addressed first-KD resistance.

Candidate evidence
------------------
1. Prior high-exposure survival rate.
2. Prior high-exposure no-KO rate weighted by exposure severity.
3. Prior cumulative punishment survived per fight.
4. A blended candidate durability score on a 0-100 research scale.

Leakage safety
--------------
Every current-fight candidate is computed only from fights that occurred earlier
for that fighter. Current-fight outcomes are used only for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FIGHT_STUDY_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_historical_study_v0_fights.parquet"
)
FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_damage_durability_study_v0.parquet"
)

MIN_PRIOR_FIGHTS = 2
HIGH_EXPOSURE_QUANTILE = 0.70


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


def _load(fight_path: Path, fsr_path: Path) -> pd.DataFrame:
    print(f"[damage durability] loading fighter-fight study rows: {fight_path}", flush=True)
    fights = pd.read_parquet(fight_path)
    print(f"[damage durability] fight rows: {len(fights):,}", flush=True)

    print(f"[damage durability] loading FSR snapshots: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path)
    print(f"[damage durability] FSR rows: {len(fsr):,}", flush=True)

    required_fights = {
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
    missing = sorted(required_fights - set(fights.columns))
    if missing:
        raise ValueError(f"fight study artifact missing required columns: {missing}")

    required_fsr = {"fight_id", "fighter_id", "damage_resistance"}
    missing = sorted(required_fsr - set(fsr.columns))
    if missing:
        raise ValueError(f"FSR artifact missing required columns: {missing}")

    for frame in (fights, fsr):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)

    if fsr.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("FSR snapshots violate fighter-fight grain")

    keep = ["fight_id", "fighter_id", "damage_resistance"]
    for date_col in ("date", "event_date", "fight_date"):
        if date_col in fsr.columns:
            keep.append(date_col)
            break

    work = fights.merge(
        fsr[keep],
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    print(f"[damage durability] joined fighter-fights: {len(work):,}", flush=True)

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

    work = work.dropna(subset=["rounds_observed", "ko_tko_loss", "damage_resistance"]).copy()
    rounds = work["rounds_observed"].clip(lower=1.0)

    work["kd_per_round"] = work["kd_absorbed"].fillna(0.0) / rounds
    work["head_per_round"] = work["head_absorbed"].fillna(0.0) / rounds
    work["ground_per_round"] = work["ground_absorbed"].fillna(0.0) / rounds
    work["opp_ctrl_per_round_min"] = work["opponent_control_seconds"].fillna(0.0) / (rounds * 60.0)
    work["damage_exposure"] = work[
        ["kd_per_round", "head_per_round", "ground_per_round", "opp_ctrl_per_round_min"]
    ].mean(axis=1)

    # Use the FSR snapshot date if available. This is already the historical
    # fighter-fight grain used by the shadow pipeline.
    date_col = next((c for c in ("date", "event_date", "fight_date") if c in work.columns), None)
    if date_col is None:
        raise ValueError("FSR artifact needs date/event_date/fight_date for leakage-safe ordering")
    work["fight_date_order"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["fight_date_order"]).copy()

    return work.sort_values(["fighter_id", "fight_date_order", "fight_id"]).reset_index(drop=True)


def _build_prior_candidates(work: pd.DataFrame) -> pd.DataFrame:
    # Exposure threshold is global and fixed from the full historical distribution.
    # It defines what counts as materially high adversity; candidate values remain
    # leakage-safe because only prior fighter fights contribute to each row.
    high_threshold = float(work["damage_exposure"].quantile(HIGH_EXPOSURE_QUANTILE))
    print(
        f"[damage durability] high-exposure threshold (q={HIGH_EXPOSURE_QUANTILE:.2f}): "
        f"{high_threshold:.5f}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    total = len(work)
    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        fighter = row["fighter_id"]
        prior = work[
            (work["fighter_id"] == fighter)
            & (work["fight_date_order"] < row["fight_date_order"])
        ]

        if len(prior) < MIN_PRIOR_FIGHTS:
            continue

        high = prior[prior["damage_exposure"] >= high_threshold].copy()
        prior_count = int(len(prior))
        high_count = int(len(high))

        # 1) Fraction of high-adversity fights survived without KO/TKO.
        if high_count > 0:
            high_survival_rate = float(1.0 - high["ko_tko_loss"].mean())
        else:
            high_survival_rate = float("nan")

        # 2) Severity-weighted survival: surviving the largest prior exposures
        # provides stronger durability evidence than surviving borderline cases.
        if high_count > 0:
            sev = high["damage_exposure"].to_numpy(dtype=float)
            survived = (1.0 - high["ko_tko_loss"].to_numpy(dtype=float))
            denom = float(np.sum(sev))
            severity_weighted_survival = (
                float(np.sum(sev * survived) / denom) if denom > 0 else float("nan")
            )
        else:
            severity_weighted_survival = float("nan")

        # 3) Punishment tolerance: average damage exposure in survived fights,
        # normalized by the global high-exposure threshold. This rewards fighters
        # who have demonstrated survival at genuinely large exposure.
        survived_prior = prior[prior["ko_tko_loss"] < 0.5]
        if len(survived_prior) > 0 and high_threshold > 0:
            punishment_tolerance = float(
                np.clip(survived_prior["damage_exposure"].mean() / high_threshold, 0.0, 2.0)
                / 2.0
            )
        else:
            punishment_tolerance = float("nan")

        # 4) Penalize historical KO/TKO failure rate under exposure. This keeps
        # the blend from labeling high-volume but repeatedly finished fighters as
        # durable simply because they have large exposure totals.
        ko_failure_rate = float(prior["ko_tko_loss"].mean())
        overall_survival = 1.0 - ko_failure_rate

        parts = []
        for weight, value in (
            (0.35, high_survival_rate),
            (0.30, severity_weighted_survival),
            (0.20, punishment_tolerance),
            (0.15, overall_survival),
        ):
            if np.isfinite(value):
                parts.append((weight, float(value)))

        if not parts:
            candidate = float("nan")
        else:
            weight_sum = sum(w for w, _ in parts)
            candidate = 100.0 * sum(w * v for w, v in parts) / weight_sum

        rows.append(
            {
                "fight_id": row["fight_id"],
                "fighter_id": fighter,
                "fight_date_order": row["fight_date_order"],
                "prior_fights": prior_count,
                "prior_high_exposure_fights": high_count,
                "prior_high_exposure_survival_rate": high_survival_rate,
                "prior_severity_weighted_survival": severity_weighted_survival,
                "prior_punishment_tolerance": punishment_tolerance,
                "prior_overall_survival_rate": overall_survival,
                "candidate_damage_durability": candidate,
                "current_damage_resistance": float(row["damage_resistance"]),
                "current_damage_exposure": float(row["damage_exposure"]),
                "current_kd_absorbed": float(row["kd_absorbed"]),
                "current_sig_absorbed": float(row["sig_absorbed"]),
                "current_ko_tko_loss": float(row["ko_tko_loss"]),
            }
        )

        if idx % 2000 == 0 or idx == total:
            print(f"[damage durability] processed {idx:,}/{total:,} rows", flush=True)

    return pd.DataFrame(rows)


def _quintile_table(frame: pd.DataFrame, signal: str, label: str) -> pd.DataFrame:
    work = frame.dropna(subset=[signal]).copy()
    work["bucket"] = _rank_quintile(
        work[signal],
        ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
    )
    rows = []
    for bucket in ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]:
        g = work[work["bucket"].astype(str) == bucket]
        if g.empty:
            continue
        rows.append(
            {
                "cohort": label,
                "bucket": bucket,
                "fighter_fights": int(len(g)),
                "mean_signal": float(g[signal].mean()),
                "mean_current_damage_exposure": float(g["current_damage_exposure"].mean()),
                "mean_current_kd_absorbed": float(g["current_kd_absorbed"].mean()),
                "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
                "survival_probability": float(1.0 - g["current_ko_tko_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _matched_current_exposure(frame: pd.DataFrame, signal: str, label: str) -> pd.DataFrame:
    work = frame.dropna(subset=[signal, "current_damage_exposure"]).copy()
    work["exposure_bucket"] = _rank_quintile(
        work["current_damage_exposure"],
        ["E1 lowest", "E2", "E3", "E4", "E5 highest"],
    )
    rows = []
    for exposure in ["E1 lowest", "E2", "E3", "E4", "E5 highest"]:
        group = work[work["exposure_bucket"].astype(str) == exposure].copy()
        if len(group) < 60:
            continue
        ranks = group[signal].rank(method="first", pct=True)
        group["durability_band"] = pd.cut(
            ranks,
            bins=[0.0, 1/3, 2/3, 1.0],
            labels=["low", "middle", "high"],
            include_lowest=True,
        )
        for band in ["low", "middle", "high"]:
            g = group[group["durability_band"].astype(str) == band]
            if g.empty:
                continue
            rows.append(
                {
                    "signal": label,
                    "current_exposure_quintile": exposure,
                    "durability_band": band,
                    "fighter_fights": int(len(g)),
                    "mean_signal": float(g[signal].mean()),
                    "mean_current_damage_exposure": float(g["current_damage_exposure"].mean()),
                    "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 8: damage durability / reservoir capacity discovery")
    parser.add_argument("--fight-path", type=Path, default=FIGHT_STUDY_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.fight_path, args.fsr_path)
    study = _build_prior_candidates(work)

    print("\n" + "=" * 118)
    print("STUDY 8 — DAMAGE DURABILITY / RESERVOIR-CAPACITY DISCOVERY")
    print("=" * 118)
    print(f"eligible fighter-fights with >={MIN_PRIOR_FIGHTS} prior fights: {len(study):,}")

    signals = [
        "prior_high_exposure_survival_rate",
        "prior_severity_weighted_survival",
        "prior_punishment_tolerance",
        "prior_overall_survival_rate",
        "candidate_damage_durability",
        "current_damage_resistance",
    ]

    print("\nCORE SPEARMAN RELATIONSHIPS")
    for signal in signals:
        print(
            f"{signal:40s} vs KO/TKO loss  "
            f"{_safe_spearman(study[signal], study['current_ko_tko_loss']): .5f}"
        )

    print("\nCANDIDATE DAMAGE DURABILITY QUINTILES")
    print(
        _quintile_table(
            study,
            "candidate_damage_durability",
            "candidate durability",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nCURRENT DAMAGE_RESISTANCE QUINTILES — COMPARATOR")
    print(
        _quintile_table(
            study,
            "current_damage_resistance",
            "current damage_resistance",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nCANDIDATE DURABILITY WITHIN MATCHED CURRENT DAMAGE-EXPOSURE QUINTILES")
    print(
        _matched_current_exposure(
            study,
            "candidate_damage_durability",
            "candidate durability",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nCURRENT DAMAGE_RESISTANCE WITHIN MATCHED EXPOSURE — COMPARATOR")
    print(
        _matched_current_exposure(
            study,
            "current_damage_resistance",
            "current damage_resistance",
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print(
        "\nINTERPRETATION RULE: a useful reservoir-capacity/durability signal should "
        "show LOWER future KO/TKO loss as durability rises, especially inside "
        "matched current-damage-exposure strata. It should also outperform or "
        "meaningfully complement the existing damage_resistance rating."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    study.to_parquet(args.output, index=False)
    print(f"\n[damage durability] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
