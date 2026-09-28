"""Diagnose the ground-opportunity chain in Damage Reservoir V1.

Research-only diagnostic over the same 300 historical validation bouts used by
the KD/exposure audits. No simulator constants or mechanics are changed.

The goal is to separate the current ground-opportunity miss into stages:

    TD attempt -> TD success -> ground entry -> ground persistence

Historical observability is incomplete. UFCStats provides takedown attempts,
takedowns landed, control time, and phase-specific strikes, but does not provide
exact phase timestamps or exact ground residence seconds. Therefore:

- TD attempt rate is observed exactly from UFCStats;
- TD conversion is observed exactly as TD landed / TD attempted;
- successful TD rate is observed exactly as TD landed per minute;
- historical persistence uses transparent proxies:
    * control seconds per landed TD (control can include clinch control), and
    * ground significant-strike attempts per landed TD.

The MC side is instrumented only inside this script and records true simulated
TD attempts, TD successes, ground entries, ground episode count, and ground
residence time. This lets us determine whether the simulator's ground deficit is
primarily caused by entry opportunity, conversion, or persistence before any
shadow calibration is proposed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_predictive_value as prior
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_time_matched as tm


OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_ground_chain_diagnostic.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810


class GroundChainTrackingSim(damage.StaticFSRMCDamageV1):
    """Damage V1 with diagnostic-only ground-chain counters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ground_entries = 0
        self.ground_episode_starts = 0
        self.ground_seconds = 0.0
        self.ground_segments = 0

    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        before_landed = self.stats[attacker].td_landed
        note = super()._attempt_takedown(attacker, source_phase)
        if self.stats[attacker].td_landed > before_landed:
            self.ground_entries += 1
            self.ground_episode_starts += 1
        return note


def _reset_round(sim: GroundChainTrackingSim) -> None:
    sim.phase = "DISTANCE"
    sim.ground_controller = None
    sim.clinch_controller = None
    sim.clinch_initiator = None


def _run_full_segment(sim: GroundChainTrackingSim) -> None:
    phase_start = sim.phase
    for stats in sim.stats:
        stats.phase_segments[phase_start] += 1

    if phase_start == "GROUND":
        sim.ground_seconds += damage.base.SEGMENT_SECONDS
        sim.ground_segments += 1

    sim._generate_striking(phase_start)
    if phase_start == "DISTANCE":
        sim._distance_transition()
    elif phase_start == "CLINCH":
        sim._clinch_transition()
    else:
        sim._ground_transition()


def _run_partial_segment(sim: GroundChainTrackingSim, seconds: float) -> None:
    if seconds <= 0.0:
        return

    fraction = float(seconds / damage.base.SEGMENT_SECONDS)
    sim._advance_damage_timers()
    phase = sim.phase
    if phase == "GROUND":
        sim.ground_seconds += float(seconds)

    if phase == "GROUND" and sim.ground_controller is not None:
        top = sim.ground_controller
        bottom = sim._other(top)
        sim._generate_strikes_for_fighter(top, "GROUND", rate_multiplier=fraction)
        sim._generate_strikes_for_fighter(
            bottom,
            "GROUND",
            rate_multiplier=damage.base.BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER * fraction,
        )
    else:
        for fighter in (0, 1):
            sim._generate_strikes_for_fighter(fighter, phase, rate_multiplier=fraction)


def _simulate_one(
    red: pd.Series,
    blue: pd.Series,
    *,
    elapsed_sec: float,
    rounds: int,
    seed: int,
) -> GroundChainTrackingSim:
    maximum = float(rounds * 300)
    elapsed_sec = float(np.clip(elapsed_sec, 0.0, maximum))
    sim = GroundChainTrackingSim(red, blue, rounds=rounds, seed=seed)

    full = int(elapsed_sec // damage.base.SEGMENT_SECONDS)
    remainder = elapsed_sec - full * damage.base.SEGMENT_SECONDS
    current_round = 0

    for idx in range(full):
        round_no = idx // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
            current_round = round_no
        _run_full_segment(sim)

    if remainder > 1e-9:
        round_no = full // damage.base.SEGMENTS_PER_ROUND + 1
        if round_no != current_round:
            _reset_round(sim)
        _run_partial_segment(sim, remainder)

    return sim


def _actual_ground_chain(
    round_stats_path: Path,
    master_path: Path,
    bout_ids: set[str],
) -> pd.DataFrame:
    rounds = pd.read_parquet(round_stats_path).copy()
    required = {
        "fight_id",
        "td_attempted",
        "td_landed",
        "ctrl_sec",
        "ground_attempted",
    }
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise ValueError(f"Round stats missing ground-chain columns: {missing}")

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds = rounds[rounds["fight_id"].isin(bout_ids)].copy()
    for col in required - {"fight_id"}:
        rounds[col] = pd.to_numeric(rounds[col], errors="coerce").fillna(0.0)

    agg = (
        rounds.groupby("fight_id", as_index=False)
        .agg(
            actual_td_attempted=("td_attempted", "sum"),
            actual_td_landed=("td_landed", "sum"),
            actual_ctrl_sec=("ctrl_sec", "sum"),
            actual_ground_attempted=("ground_attempted", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    elapsed = prior._actual_exposure(round_stats_path, master_path, bout_ids)[
        ["bout_id", "actual_elapsed_sec"]
    ].drop_duplicates("bout_id")
    agg = agg.merge(elapsed, on="bout_id", how="left", validate="one_to_one")
    if agg["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing actual elapsed time in ground-chain history.")

    minutes = agg["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    agg["actual_td_att_per_min"] = agg["actual_td_attempted"] / minutes
    agg["actual_td_land_per_min"] = agg["actual_td_landed"] / minutes
    agg["actual_td_conversion"] = np.where(
        agg["actual_td_attempted"] > 0,
        agg["actual_td_landed"] / agg["actual_td_attempted"],
        np.nan,
    )
    agg["actual_any_td_attempt"] = (agg["actual_td_attempted"] > 0).astype(int)
    agg["actual_any_td_landed"] = (agg["actual_td_landed"] > 0).astype(int)
    agg["actual_ctrl_sec_per_td_landed"] = np.where(
        agg["actual_td_landed"] > 0,
        agg["actual_ctrl_sec"] / agg["actual_td_landed"],
        np.nan,
    )
    agg["actual_ground_att_per_td_landed"] = np.where(
        agg["actual_td_landed"] > 0,
        agg["actual_ground_attempted"] / agg["actual_td_landed"],
        np.nan,
    )
    return agg


def _simulate_ground_chain(
    validation: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(validation) * paths_per_bout
    done = 0

    for bout_no, (_, bout) in enumerate(validation.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        elapsed_sec = float(bout["actual_elapsed_sec"])
        elapsed_min = max(elapsed_sec, 1.0) / 60.0
        rounds = prior._rounds_for_bout(bout)

        path_rows: list[dict[str, float]] = []
        for _ in range(paths_per_bout):
            sim = _simulate_one(
                red,
                blue,
                elapsed_sec=elapsed_sec,
                rounds=rounds,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            td_att = float(sim.stats[0].td_att + sim.stats[1].td_att)
            td_land = float(sim.stats[0].td_landed + sim.stats[1].td_landed)
            ground_control = float(
                sim.stats[0].ground_control_seconds + sim.stats[1].ground_control_seconds
            )
            path_rows.append(
                {
                    "td_att": td_att,
                    "td_land": td_land,
                    "ground_entries": float(sim.ground_entries),
                    "ground_seconds": float(sim.ground_seconds),
                    "ground_control_seconds": ground_control,
                    "any_td_attempt": float(td_att > 0),
                    "any_td_landed": float(td_land > 0),
                }
            )

            done += 1
            if done % 1000 == 0 or done == total_paths:
                print(
                    f"[ground-chain] paths {done:,}/{total_paths:,}; "
                    f"bouts_started={bout_no:,}/{len(validation):,}",
                    flush=True,
                )

        pf = pd.DataFrame(path_rows)
        mean_td_att = float(pf["td_att"].mean())
        mean_td_land = float(pf["td_land"].mean())
        mean_entries = float(pf["ground_entries"].mean())
        mean_ground_sec = float(pf["ground_seconds"].mean())
        mean_ground_ctrl = float(pf["ground_control_seconds"].mean())

        rows.append(
            {
                "bout_id": bout_id,
                "sim_paths": int(paths_per_bout),
                "sim_td_attempted": mean_td_att,
                "sim_td_landed": mean_td_land,
                "sim_td_att_per_min": mean_td_att / elapsed_min,
                "sim_td_land_per_min": mean_td_land / elapsed_min,
                "sim_td_conversion": (
                    mean_td_land / mean_td_att if mean_td_att > 1e-12 else np.nan
                ),
                "sim_p_any_td_attempt": float(pf["any_td_attempt"].mean()),
                "sim_p_any_td_landed": float(pf["any_td_landed"].mean()),
                "sim_ground_entries": mean_entries,
                "sim_ground_entries_per_min": mean_entries / elapsed_min,
                "sim_ground_seconds": mean_ground_sec,
                "sim_ground_share": mean_ground_sec / max(elapsed_sec, 1.0),
                "sim_ground_control_seconds": mean_ground_ctrl,
                "sim_ground_sec_per_entry": (
                    mean_ground_sec / mean_entries if mean_entries > 1e-12 else np.nan
                ),
                "sim_ground_control_sec_per_td_landed": (
                    mean_ground_ctrl / mean_td_land if mean_td_land > 1e-12 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _safe_auc(labels: pd.Series, score: pd.Series) -> float:
    work = pd.DataFrame({"y": labels, "s": score}).dropna()
    if work["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(work["y"].astype(int), work["s"].astype(float)))


def _top_quartile_auc(actual: pd.Series, sim: pd.Series) -> float:
    work = pd.DataFrame({"a": actual, "s": sim}).dropna()
    if len(work) < 8:
        return float("nan")
    cut = float(work["a"].quantile(0.75))
    labels = (work["a"] >= cut).astype(int)
    return _safe_auc(labels, work["s"])


def _metric_row(
    frame: pd.DataFrame,
    label: str,
    actual_col: str,
    sim_col: str,
) -> dict[str, object]:
    work = frame[[actual_col, sim_col]].dropna().copy()
    a = work[actual_col].astype(float)
    s = work[sim_col].astype(float)
    return {
        "metric": label,
        "bouts": len(work),
        "actual_mean": float(a.mean()),
        "mc_mean": float(s.mean()),
        "mc_minus_actual": float(s.mean() - a.mean()),
        "pearson": float(a.corr(s, method="pearson")),
        "spearman": float(a.corr(s, method="spearman")),
        "mae": float((a - s).abs().mean()),
        "top_quartile_auc": _top_quartile_auc(a, s),
    }


def _quartile_table(
    frame: pd.DataFrame,
    label: str,
    actual_col: str,
    sim_col: str,
) -> pd.DataFrame:
    work = frame[[actual_col, sim_col]].dropna().copy()
    if len(work) < 8:
        return pd.DataFrame()
    work["mc_quartile"] = pd.qcut(
        work[sim_col],
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    out = (
        work.groupby("mc_quartile", observed=True, as_index=False)
        .agg(
            bouts=(actual_col, "size"),
            mc_mean=(sim_col, "mean"),
            actual_mean=(actual_col, "mean"),
        )
    )
    out.insert(0, "metric", label)
    return out


def _print_summary(frame: pd.DataFrame) -> None:
    print("\n" + "=" * 124)
    print("GROUND-CHAIN DIAGNOSTIC: TD ATTEMPT -> TD SUCCESS -> GROUND ENTRY -> PERSISTENCE")
    print("=" * 124)
    print(f"bouts: {len(frame):,}; paths: {int(frame['sim_paths'].sum()):,}")
    print(
        "Historical caveat: UFCStats has no exact ground-entry timestamps or exact ground residence time; "
        "persistence comparisons therefore use control/ground-strike proxies."
    )

    print("\nSTAGE 1 — TAKEDOWN ATTEMPT OPPORTUNITY")
    print(
        f"TD attempts/min: actual={frame['actual_td_att_per_min'].mean():.4f}; "
        f"MC={frame['sim_td_att_per_min'].mean():.4f}; "
        f"Spearman={frame['actual_td_att_per_min'].corr(frame['sim_td_att_per_min'], method='spearman'):.4f}; "
        f"top-Q AUC={_top_quartile_auc(frame['actual_td_att_per_min'], frame['sim_td_att_per_min']):.4f}"
    )
    print(
        f"any TD attempt: actual={frame['actual_any_td_attempt'].mean():.4f}; "
        f"MC mean probability={frame['sim_p_any_td_attempt'].mean():.4f}; "
        f"AUC={_safe_auc(frame['actual_any_td_attempt'], frame['sim_p_any_td_attempt']):.4f}"
    )

    print("\nSTAGE 2 — TAKEDOWN CONVERSION")
    conversion = frame[frame["actual_td_attempted"] > 0].copy()
    print(
        f"among bouts with historical TD attempts (n={len(conversion)}): "
        f"actual conversion={conversion['actual_td_conversion'].mean():.4f}; "
        f"MC={conversion['sim_td_conversion'].mean():.4f}; "
        f"Spearman={conversion['actual_td_conversion'].corr(conversion['sim_td_conversion'], method='spearman'):.4f}"
    )
    print(
        f"TD landed/min: actual={frame['actual_td_land_per_min'].mean():.4f}; "
        f"MC={frame['sim_td_land_per_min'].mean():.4f}; "
        f"Spearman={frame['actual_td_land_per_min'].corr(frame['sim_td_land_per_min'], method='spearman'):.4f}; "
        f"top-Q AUC={_top_quartile_auc(frame['actual_td_land_per_min'], frame['sim_td_land_per_min']):.4f}"
    )
    print(
        f"any TD landed: actual={frame['actual_any_td_landed'].mean():.4f}; "
        f"MC mean probability={frame['sim_p_any_td_landed'].mean():.4f}; "
        f"AUC={_safe_auc(frame['actual_any_td_landed'], frame['sim_p_any_td_landed']):.4f}"
    )

    print("\nSTAGE 3 — GROUND ENTRY")
    print(
        "In the current MC architecture, every successful TD creates a ground entry. "
        "Therefore simulated ground-entry rate equals simulated TD-landed rate by construction."
    )
    print(
        f"MC ground entries/min={frame['sim_ground_entries_per_min'].mean():.4f}; "
        f"historical TD landed/min proxy={frame['actual_td_land_per_min'].mean():.4f}"
    )

    print("\nSTAGE 4 — GROUND PERSISTENCE")
    hist_td = frame[frame["actual_td_landed"] > 0].copy()
    sim_td = frame[frame["sim_td_landed"] > 1e-12].copy()
    print(
        f"historical control sec / landed TD proxy (n={len(hist_td)}): "
        f"{hist_td['actual_ctrl_sec_per_td_landed'].mean():.2f}s"
    )
    print(
        f"MC ground-control sec / landed TD (n={len(sim_td)}): "
        f"{sim_td['sim_ground_control_sec_per_td_landed'].mean():.2f}s"
    )
    print(
        f"MC true ground residence sec / ground entry: "
        f"{sim_td['sim_ground_sec_per_entry'].mean():.2f}s"
    )
    print(
        f"historical ground sig attempts / landed TD proxy: "
        f"{hist_td['actual_ground_att_per_td_landed'].mean():.4f}"
    )

    metrics = pd.DataFrame(
        [
            _metric_row(frame, "TD attempts/min", "actual_td_att_per_min", "sim_td_att_per_min"),
            _metric_row(frame, "TD landed/min", "actual_td_land_per_min", "sim_td_land_per_min"),
        ]
    )
    print("\nMATCHUP RANKING SUMMARY")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nACTUAL TD ACTIVITY BY MC-PREDICTED QUARTILE")
    quartiles = pd.concat(
        [
            _quartile_table(frame, "TD attempts/min", "actual_td_att_per_min", "sim_td_att_per_min"),
            _quartile_table(frame, "TD landed/min", "actual_td_land_per_min", "sim_td_land_per_min"),
        ],
        ignore_index=True,
    )
    print(quartiles.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION GUIDE")
    print("- Low TD-attempt level + useful ranking => entry-opportunity calibration issue is plausible.")
    print("- Weak TD-attempt ranking => matchup selection is also wrong; do not fix with a global multiplier alone.")
    print("- Wrong TD conversion => wrestling_conversion / td_defense mapping needs separate diagnosis.")
    print("- Reasonable entries but too little persistence => ground-exit/control persistence becomes the primary target.")
    print("- No simulator constants were changed by this audit.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the MC takedown-to-ground chain")
    parser.add_argument("--validation", type=Path, default=prior.VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=prior.FSR_PATH)
    parser.add_argument("--round-stats", type=Path, default=Path(prior.ROUND_STATS_PATH))
    parser.add_argument("--master", type=Path, default=Path(prior.MASTER_PATH))
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    validation = prior._load_validation(args.validation)
    bout_ids = set(validation["bout_id"].astype(str))
    print(f"[ground-chain] loading {len(validation):,} historical bouts", flush=True)

    pairs, style = prior._load_fsr_pairs(args.fsr_path, bout_ids)
    actual = _actual_ground_chain(args.round_stats, args.master, bout_ids)
    base = validation.merge(actual, on="bout_id", how="left", validate="one_to_one")
    base = base.merge(style, on="bout_id", how="left", validate="one_to_one")

    if base["actual_elapsed_sec"].isna().any():
        raise ValueError("Missing elapsed time after historical ground-chain join.")
    if len(pairs) != len(validation):
        raise ValueError("Leakage-safe FSR pair count does not match validation cohort.")

    sim = _simulate_ground_chain(
        base,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    out = base.merge(sim, on="bout_id", how="left", validate="one_to_one")
    if out["sim_td_att_per_min"].isna().any():
        raise ValueError("Missing simulated ground-chain values after join.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    _print_summary(out)
    print(f"\n[ground-chain] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
