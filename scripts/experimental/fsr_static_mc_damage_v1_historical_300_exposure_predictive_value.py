"""Measure matchup-level significant-strike exposure predictive value for Damage V1.

This research-only audit uses the exact 300 historical bouts from the existing
actual-vs-MC KD validation artifact. It reruns the current locked Damage V1 / KD=80
simulator and asks a separate question from KD calibration:

    Does the MC correctly rank which historical matchups will generate high
    significant-strike exposure?

For each bout it compares actual UFCStats significant strikes per minute with
simulated significant strikes per minute, then reports:
- aggregate bias and MAE;
- Pearson and Spearman matchup correlations;
- high-exposure ROC-AUC (historical top quartile as the positive class);
- actual exposure by quartile of MC-predicted exposure;
- largest under- and over-predicted matchups;
- error summaries by FSR distance pressure, wrestling tendency, clinch pressure;
- weight-class summaries when a supported master column is available.

Actual rates use actual elapsed fight time. MC rates use scheduled fight time
because KO/TKO stoppages remain disabled in the shadow simulator.

No simulator constants or architecture are changed by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from pipeline.common.fight_time import repair_elapsed_match_time
from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH
from scripts.experimental import fsr_static_mc_damage_v1 as damage


VALIDATION_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_kd_actual_vs_mc.parquet"
)
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_exposure_predictive_value.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810

WEIGHT_CLASS_CANDIDATES = (
    "weight_class",
    "weight_class_name",
    "division",
    "weightclass",
)


def _load_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical actual-vs-MC artifact not found: {path}. "
            "Run the 300-bout KD validation first."
        )
    frame = pd.read_parquet(path).copy()
    required = {"bout_id", "actual_any_kd", "mc_p_any_kd"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Validation artifact missing required columns: {missing}")
    frame["bout_id"] = frame["bout_id"].astype(str)
    if frame["bout_id"].nunique() != len(frame):
        raise ValueError("Validation artifact must contain exactly one row per historical bout.")
    return frame


def _load_fsr_pairs(
    path: Path,
    bout_ids: set[str],
) -> tuple[dict[str, tuple[pd.Series, pd.Series]], pd.DataFrame]:
    frame = pd.read_parquet(path)
    bout_key = "fight_id" if "fight_id" in frame.columns else "bout_id"
    required = (
        {bout_key, "fighter_id"}
        | set(damage.base.REQUIRED_COLUMNS)
        | damage.REQUIRED_DAMAGE_COLUMNS
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR artifact missing required columns: {missing}")

    work = frame.copy()
    work[bout_key] = work[bout_key].astype(str)
    work["fighter_id"] = work["fighter_id"].astype(str)
    work = work[work[bout_key].isin(bout_ids)].copy()

    pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    feature_rows: list[dict[str, object]] = []

    for key, group in work.groupby(bout_key, sort=False):
        group = group.reset_index(drop=True)
        if len(group) != 2 or group["fighter_id"].nunique() != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        bout_id = str(key)
        pairs[bout_id] = (a, b)

        distance_pressure = np.mean([
            float(a.get("distance_striking_pressure", 50.0)),
            float(b.get("distance_striking_pressure", 50.0)),
        ])
        clinch_pressure = np.mean([
            float(a.get("clinch_striking_pressure", 50.0)),
            float(b.get("clinch_striking_pressure", 50.0)),
        ])
        wrestling_entry = np.mean([
            float(a.get("wrestling_entry", 50.0)),
            float(b.get("wrestling_entry", 50.0)),
        ])
        control_imposition = np.mean([
            float(a.get("control_imposition", 50.0)),
            float(b.get("control_imposition", 50.0)),
        ])

        # A compact bout-level style tendency. Higher values indicate more
        # wrestling/control pull relative to distance striking pressure.
        wrestling_tendency = (
            0.70 * wrestling_entry
            + 0.30 * control_imposition
            - distance_pressure
        )

        feature_rows.append(
            {
                "bout_id": bout_id,
                "mean_distance_pressure": distance_pressure,
                "mean_clinch_pressure": clinch_pressure,
                "mean_wrestling_entry": wrestling_entry,
                "mean_control_imposition": control_imposition,
                "wrestling_tendency": wrestling_tendency,
            }
        )

    return pairs, pd.DataFrame(feature_rows)


def _actual_exposure(
    round_stats_path: Path,
    master_path: Path,
    bout_ids: set[str],
) -> pd.DataFrame:
    rounds = pd.read_parquet(round_stats_path)
    required_round = {"fight_id", "sig_str_landed", "sig_str_attempted"}
    missing = sorted(required_round - set(rounds.columns))
    if missing:
        raise ValueError(f"Round stats missing exposure columns: {missing}")

    rounds = rounds.copy()
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds = rounds[rounds["fight_id"].isin(bout_ids)].copy()
    rounds["sig_str_landed"] = pd.to_numeric(
        rounds["sig_str_landed"], errors="coerce"
    ).fillna(0.0)
    rounds["sig_str_attempted"] = pd.to_numeric(
        rounds["sig_str_attempted"], errors="coerce"
    ).fillna(0.0)

    actual = (
        rounds.groupby("fight_id", as_index=False)
        .agg(
            actual_sig_landed=("sig_str_landed", "sum"),
            actual_sig_attempted=("sig_str_attempted", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    master = pd.read_parquet(master_path)
    required_master = {"fight_id", "finish_round", "match_time_sec"}
    missing_master = sorted(required_master - set(master.columns))
    if missing_master:
        raise ValueError(f"Master artifact missing exposure columns: {missing_master}")

    master = master.copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master = master[master["fight_id"].isin(bout_ids)].copy()
    master["finish_round"] = pd.to_numeric(master["finish_round"], errors="coerce")
    master["match_time_sec"] = pd.to_numeric(master["match_time_sec"], errors="coerce")
    master = repair_elapsed_match_time(master)

    keep = ["fight_id", "match_time_sec"]
    weight_col = next((c for c in WEIGHT_CLASS_CANDIDATES if c in master.columns), None)
    if weight_col:
        keep.append(weight_col)

    master = master[keep].drop_duplicates("fight_id")
    rename = {
        "fight_id": "bout_id",
        "match_time_sec": "actual_elapsed_sec",
    }
    if weight_col:
        rename[weight_col] = "weight_class"
    master = master.rename(columns=rename)

    actual = actual.merge(master, on="bout_id", how="left", validate="one_to_one")
    if actual["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing elapsed fight time for one or more sampled bouts.")

    actual_minutes = actual["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    actual["actual_sig_landed_per_min"] = actual["actual_sig_landed"] / actual_minutes
    actual["actual_sig_attempted_per_min"] = actual["actual_sig_attempted"] / actual_minutes
    return actual


def _rounds_for_bout(row: pd.Series) -> int:
    value = row.get("rounds")
    try:
        rounds = int(round(float(value)))
    except (TypeError, ValueError):
        rounds = 3
    return rounds if rounds in (3, 5) else 3


def _simulate_exposure(
    validation: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(validation) * paths_per_bout
    counter = 0

    for bout_number, (_, bout) in enumerate(validation.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        if bout_id not in pairs:
            raise ValueError(f"Missing leakage-safe FSR pair for bout {bout_id}")
        red, blue = pairs[bout_id]
        rounds = _rounds_for_bout(bout)
        scheduled_minutes = rounds * 5.0

        landed: list[float] = []
        attempted: list[float] = []
        kds: list[float] = []

        for _ in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = damage.StaticFSRMCDamageV1(red, blue, rounds=rounds, seed=path_seed)
            sim.run()

            landed.append(float(sim.stats[0].sig_landed + sim.stats[1].sig_landed))
            # Base simulator contract is sig_att, not sig_attempted.
            attempted.append(float(sim.stats[0].sig_att + sim.stats[1].sig_att))
            kds.append(float(sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored))

            counter += 1
            if counter % 1000 == 0 or counter == total_paths:
                print(
                    f"[300 exposure audit] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_number:,}/{len(validation):,}",
                    flush=True,
                )

        rows.append(
            {
                "bout_id": bout_id,
                "sim_paths": paths_per_bout,
                "rounds": rounds,
                "sim_sig_landed": float(np.mean(landed)),
                "sim_sig_attempted": float(np.mean(attempted)),
                "sim_sig_landed_per_min": float(np.mean(landed) / scheduled_minutes),
                "sim_sig_attempted_per_min": float(np.mean(attempted) / scheduled_minutes),
                "sim_p_any_kd_rerun": float(np.mean(np.asarray(kds) > 0)),
                "sim_expected_total_kd_rerun": float(np.mean(kds)),
            }
        )

    return pd.DataFrame(rows)


def _quartile_summary(frame: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    work = frame[[score_col, "actual_sig_landed_per_min", "sim_sig_landed_per_min"]].dropna().copy()
    if work[score_col].nunique() < 4:
        return pd.DataFrame()
    work["quartile"] = pd.qcut(work[score_col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    out = (
        work.groupby("quartile", observed=True, as_index=False)
        .agg(
            bouts=(score_col, "size"),
            score_mean=(score_col, "mean"),
            actual_landed_per_min=("actual_sig_landed_per_min", "mean"),
            mc_landed_per_min=("sim_sig_landed_per_min", "mean"),
        )
    )
    out["bias_mc_minus_actual"] = out["mc_landed_per_min"] - out["actual_landed_per_min"]
    out.insert(0, "group", label)
    return out


def _print_summary(frame: pd.DataFrame) -> None:
    a_land = frame["actual_sig_landed_per_min"].astype(float)
    s_land = frame["sim_sig_landed_per_min"].astype(float)
    a_att = frame["actual_sig_attempted_per_min"].astype(float)
    s_att = frame["sim_sig_attempted_per_min"].astype(float)

    frame = frame.copy()
    frame["landed_bias_mc_minus_actual"] = s_land - a_land
    frame["attempt_bias_mc_minus_actual"] = s_att - a_att
    frame["landed_abs_error"] = (s_land - a_land).abs()
    frame["landed_ratio_actual_to_mc"] = a_land / s_land.replace(0.0, np.nan)

    pearson_land = float(a_land.corr(s_land, method="pearson"))
    spearman_land = float(a_land.corr(s_land, method="spearman"))
    pearson_att = float(a_att.corr(s_att, method="pearson"))
    spearman_att = float(a_att.corr(s_att, method="spearman"))

    actual_high_cut = float(a_land.quantile(0.75))
    high_actual = (a_land >= actual_high_cut).astype(int)
    high_auc = float(roc_auc_score(high_actual, s_land))

    print("\n" + "=" * 120)
    print("HISTORICAL 300-BOUT MC STRIKING-EXPOSURE PREDICTIVE VALUE")
    print("=" * 120)
    print(f"historical bouts: {len(frame):,}")
    print(f"rerun MC paths: {int(frame['sim_paths'].sum()):,}")
    print(f"KD shock coefficient unchanged: {damage.KD_SHOCK_COEFFICIENT:g}")

    print("\nAGGREGATE EXPOSURE")
    print(
        f"sig landed/min: actual={a_land.mean():.4f}; MC={s_land.mean():.4f}; "
        f"MC-actual={s_land.mean()-a_land.mean():+.4f}; actual/MC={a_land.mean()/s_land.mean():.4f}x"
    )
    print(
        f"sig attempted/min: actual={a_att.mean():.4f}; MC={s_att.mean():.4f}; "
        f"MC-actual={s_att.mean()-a_att.mean():+.4f}; actual/MC={a_att.mean()/s_att.mean():.4f}x"
    )

    print("\nMATCHUP-LEVEL EXPOSURE PREDICTIVE VALUE")
    print(f"landed/min Pearson r:  {pearson_land:.6f}")
    print(f"landed/min Spearman r: {spearman_land:.6f}")
    print(f"attempt/min Pearson r: {pearson_att:.6f}")
    print(f"attempt/min Spearman r:{spearman_att:.6f}")
    print(f"landed/min MAE:        {mean_absolute_error(a_land, s_land):.6f}")
    print(f"attempt/min MAE:       {mean_absolute_error(a_att, s_att):.6f}")
    print(
        f"ROC-AUC for identifying historical top-quartile landed exposure "
        f"(cut >= {actual_high_cut:.4f}/min): {high_auc:.6f}"
    )

    # Ranking check: if MC exposure has matchup value, actual exposure should
    # rise monotonically from the lowest to highest predicted MC quartile.
    work = frame.copy()
    work["mc_exposure_quartile"] = pd.qcut(
        work["sim_sig_landed_per_min"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    ranking = (
        work.groupby("mc_exposure_quartile", observed=True, as_index=False)
        .agg(
            bouts=("bout_id", "size"),
            mean_mc_landed_min=("sim_sig_landed_per_min", "mean"),
            mean_actual_landed_min=("actual_sig_landed_per_min", "mean"),
            actual_kd_rate=("actual_any_kd", "mean"),
        )
    )
    print("\nACTUAL OUTCOMES BY MC-PREDICTED EXPOSURE QUARTILE")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSTYLE / TRAIT ERROR BREAKDOWNS")
    summaries = []
    for col, label in [
        ("mean_distance_pressure", "distance pressure"),
        ("wrestling_tendency", "wrestling tendency"),
        ("mean_clinch_pressure", "clinch pressure"),
    ]:
        if col in frame.columns:
            q = _quartile_summary(frame, col, label)
            if not q.empty:
                summaries.append(q)
    if summaries:
        print(pd.concat(summaries, ignore_index=True).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "weight_class" in frame.columns:
        wc = frame.dropna(subset=["weight_class"]).groupby("weight_class", as_index=False).agg(
            bouts=("bout_id", "size"),
            actual_landed_per_min=("actual_sig_landed_per_min", "mean"),
            mc_landed_per_min=("sim_sig_landed_per_min", "mean"),
            mae=("landed_abs_error", "mean"),
        )
        wc["bias_mc_minus_actual"] = wc["mc_landed_per_min"] - wc["actual_landed_per_min"]
        wc = wc.sort_values("bouts", ascending=False)
        print("\nBY WEIGHT CLASS")
        print(wc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nBY WEIGHT CLASS")
        print("weight-class column not present in the master artifact; skipped without guessing.")

    display = [c for c in [
        "bout_id", "event_date", "red_name", "blue_name", "actual_any_kd", "mc_p_any_kd",
        "actual_sig_landed_per_min", "sim_sig_landed_per_min", "landed_ratio_actual_to_mc",
        "mean_distance_pressure", "wrestling_tendency", "mean_clinch_pressure",
    ] if c in frame.columns]

    print("\n20 LARGEST UNDERPREDICTED-EXPOSURE MATCHUPS")
    print(
        frame.sort_values("landed_bias_mc_minus_actual").head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n20 LARGEST OVERPREDICTED-EXPOSURE MATCHUPS")
    print(
        frame.sort_values("landed_bias_mc_minus_actual", ascending=False).head(20)[display]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nINTERPRETATION GUIDE")
    print("- Correlations near 0 imply little matchup-level exposure ranking value.")
    print("- Positive correlations and rising actual exposure across MC quartiles imply real matchup signal.")
    print("- Persistent negative aggregate bias means the MC is systematically too low-volume.")
    print("- Style-quartile bias identifies where phase/opportunity generation is most wrong.")
    print("- This audit does not retune strike rates, KD, reservoir, or KO/TKO mechanics.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure historical matchup-level striking exposure predictive value for Damage V1"
    )
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--round-stats", type=Path, default=Path(ROUND_STATS_PATH))
    parser.add_argument("--master", type=Path, default=Path(MASTER_PATH))
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation = _load_validation(args.validation)
    bout_ids = set(validation["bout_id"])
    print(
        f"[300 exposure audit] historical bouts={len(validation):,}; "
        f"paths_per_bout={args.paths_per_bout:,}; "
        f"total_paths={len(validation) * args.paths_per_bout:,}",
        flush=True,
    )

    pairs, style_features = _load_fsr_pairs(args.fsr_path, bout_ids)
    if len(pairs) != len(validation):
        missing = sorted(bout_ids - set(pairs))
        raise ValueError(
            f"Leakage-safe FSR pairs available for {len(pairs)}/{len(validation)} bouts. "
            f"First missing IDs: {missing[:10]}"
        )

    actual = _actual_exposure(args.round_stats, args.master, bout_ids)
    if len(actual) != len(validation):
        raise ValueError(
            f"Actual exposure available for {len(actual)}/{len(validation)} sampled bouts; "
            "refusing partial validation."
        )

    simulated = _simulate_exposure(
        validation,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )

    merged = validation.merge(actual, on="bout_id", how="left", validate="one_to_one")
    merged = merged.merge(simulated, on="bout_id", how="left", validate="one_to_one")
    merged = merged.merge(style_features, on="bout_id", how="left", validate="one_to_one")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    _print_summary(merged)
    print(f"\n[300 exposure audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
