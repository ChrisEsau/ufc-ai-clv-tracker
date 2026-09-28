"""Diagnose whether historical exposure errors come from phase mix or strike rate.

Research-only audit over the same 300 historical validation bouts.  Each MC path
is time-matched to the historical elapsed fight time.  UFCStats does not expose
exact phase residence time, so historical phase mix is represented by
significant-strike attempt shares at distance/clinch/ground plus control time.
The MC records both phase residence shares and phase-specific strike attempts.

No simulator constants or architecture are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_predictive_value as prior
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_exposure_time_matched as timed

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_phase_mix_diagnostic.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810
PHASES = ("DISTANCE", "CLINCH", "GROUND")


class InstrumentedDamageV1(damage.StaticFSRMCDamageV1):
    """Damage V1 with diagnostic counters only; mechanics are unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_seconds = {p: 0.0 for p in PHASES}
        self.phase_sig_att = {p: 0.0 for p in PHASES}
        self.phase_sig_landed = {p: 0.0 for p in PHASES}

    def _generate_strikes_for_fighter(self, fighter, phase, *, rate_multiplier=1.0):
        before_att = self.stats[fighter].sig_att
        before_land = self.stats[fighter].sig_landed
        note = super()._generate_strikes_for_fighter(
            fighter, phase, rate_multiplier=rate_multiplier
        )
        self.phase_sig_att[phase] += self.stats[fighter].sig_att - before_att
        self.phase_sig_landed[phase] += self.stats[fighter].sig_landed - before_land
        return note


def _actual_phase_data(round_stats_path: Path, bout_ids: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(round_stats_path)
    required = {
        "fight_id", "distance_landed", "distance_attempted",
        "clinch_landed", "clinch_attempted", "ground_landed", "ground_attempted",
        "td_attempted", "ctrl_sec",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Round stats missing phase diagnostic columns: {missing}")

    df = df.copy()
    df["fight_id"] = df["fight_id"].astype(str)
    df = df[df["fight_id"].isin(bout_ids)].copy()
    numeric = sorted(required - {"fight_id"})
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    agg = df.groupby("fight_id", as_index=False)[numeric].sum().rename(columns={"fight_id": "bout_id"})
    total_att = agg[["distance_attempted", "clinch_attempted", "ground_attempted"]].sum(axis=1)
    total_land = agg[["distance_landed", "clinch_landed", "ground_landed"]].sum(axis=1)
    for phase, prefix in [("distance", "distance"), ("clinch", "clinch"), ("ground", "ground")]:
        agg[f"actual_{phase}_att_share"] = agg[f"{prefix}_attempted"] / total_att.replace(0.0, np.nan)
        agg[f"actual_{phase}_land_share"] = agg[f"{prefix}_landed"] / total_land.replace(0.0, np.nan)
    return agg


def _reset_round(sim: InstrumentedDamageV1) -> None:
    sim.phase = "DISTANCE"
    sim.ground_controller = None
    sim.clinch_controller = None
    sim.clinch_initiator = None


def _run_full_segment(sim: InstrumentedDamageV1) -> None:
    phase = sim.phase
    sim.phase_seconds[phase] += damage.base.SEGMENT_SECONDS
    for stats in sim.stats:
        stats.phase_segments[phase] += 1
    sim._generate_striking(phase)
    if phase == "DISTANCE":
        sim._distance_transition()
    elif phase == "CLINCH":
        sim._clinch_transition()
    else:
        sim._ground_transition()


def _run_partial_segment(sim: InstrumentedDamageV1, seconds: float) -> None:
    if seconds <= 0:
        return
    fraction = seconds / damage.base.SEGMENT_SECONDS
    sim._advance_damage_timers()
    phase = sim.phase
    sim.phase_seconds[phase] += seconds
    if phase == "GROUND" and sim.ground_controller is not None:
        top = sim.ground_controller
        bottom = sim._other(top)
        sim._generate_strikes_for_fighter(top, "GROUND", rate_multiplier=fraction)
        sim._generate_strikes_for_fighter(
            bottom, "GROUND",
            rate_multiplier=damage.base.BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER * fraction,
        )
    else:
        for fighter in (0, 1):
            sim._generate_strikes_for_fighter(fighter, phase, rate_multiplier=fraction)


def _simulate_one(red, blue, *, elapsed_sec: float, rounds: int, seed: int) -> InstrumentedDamageV1:
    maximum = rounds * 300.0
    elapsed_sec = float(np.clip(elapsed_sec, 0.0, maximum))
    sim = InstrumentedDamageV1(red, blue, rounds=rounds, seed=seed)
    full = int(elapsed_sec // 10)
    rem = elapsed_sec - full * 10
    current_round = 0
    for idx in range(full):
        round_no = idx // 30 + 1
        if round_no != current_round:
            _reset_round(sim)
            current_round = round_no
        _run_full_segment(sim)
    if rem > 1e-9:
        round_no = full // 30 + 1
        if round_no != current_round:
            _reset_round(sim)
        _run_partial_segment(sim, rem)
    return sim


def _simulate(validation: pd.DataFrame, pairs, *, paths_per_bout: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    total_paths = len(validation) * paths_per_bout
    done = 0
    for bout_no, (_, bout) in enumerate(validation.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        elapsed = float(bout["actual_elapsed_sec"])
        rounds = prior._rounds_for_bout(bout)
        path_rows = []
        for _ in range(paths_per_bout):
            sim = _simulate_one(
                red, blue, elapsed_sec=elapsed, rounds=rounds,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            total_att = sum(sim.phase_sig_att.values())
            total_land = sum(sim.phase_sig_landed.values())
            row = {
                "sim_sig_att": total_att,
                "sim_sig_landed": total_land,
                "sim_control_sec": float(sim.stats[0].control_seconds + sim.stats[1].control_seconds),
                "sim_td_att": float(sim.stats[0].td_att + sim.stats[1].td_att),
            }
            for phase in PHASES:
                p = phase.lower()
                row[f"sim_{p}_sec"] = sim.phase_seconds[phase]
                row[f"sim_{p}_att"] = sim.phase_sig_att[phase]
                row[f"sim_{p}_land"] = sim.phase_sig_landed[phase]
            path_rows.append(row)
            done += 1
            if done % 1000 == 0 or done == total_paths:
                print(
                    f"[phase-mix diagnostic] paths {done:,}/{total_paths:,}; "
                    f"bouts_started={bout_no:,}/{len(validation):,}", flush=True
                )
        pf = pd.DataFrame(path_rows)
        out = {"bout_id": bout_id, "sim_paths": paths_per_bout}
        for c in pf.columns:
            out[c] = float(pf[c].mean())
        sec_total = sum(out[f"sim_{p.lower()}_sec"] for p in PHASES)
        att_total = sum(out[f"sim_{p.lower()}_att"] for p in PHASES)
        land_total = sum(out[f"sim_{p.lower()}_land"] for p in PHASES)
        for phase in PHASES:
            p = phase.lower()
            out[f"sim_{p}_time_share"] = out[f"sim_{p}_sec"] / sec_total if sec_total else np.nan
            out[f"sim_{p}_att_share"] = out[f"sim_{p}_att"] / att_total if att_total else np.nan
            out[f"sim_{p}_land_share"] = out[f"sim_{p}_land"] / land_total if land_total else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame) -> None:
    elapsed_min = df["actual_elapsed_sec"].clip(lower=1.0) / 60.0
    df = df.copy()
    df["actual_ctrl_share"] = df["ctrl_sec"] / df["actual_elapsed_sec"].clip(lower=1.0)
    df["sim_ctrl_share"] = df["sim_control_sec"] / df["actual_elapsed_sec"].clip(lower=1.0)
    df["actual_td_att_per_min"] = df["td_attempted"] / elapsed_min
    df["sim_td_att_per_min"] = df["sim_td_att"] / elapsed_min
    df["pace_gap"] = df["actual_sig_landed_per_min"] - df["sim_sig_landed"] / elapsed_min

    print("\n" + "=" * 120)
    print("HISTORICAL 300-BOUT PHASE-MIX VS STRIKE-RATE DIAGNOSTIC")
    print("=" * 120)
    print(f"bouts: {len(df):,}; paths: {int(df['sim_paths'].sum()):,}")

    print("\nPHASE-SPECIFIC SIGNIFICANT-STRIKE ATTEMPT SHARE")
    for p in ("distance", "clinch", "ground"):
        a = df[f"actual_{p}_att_share"]
        s = df[f"sim_{p}_att_share"]
        print(
            f"{p:8s}: actual={a.mean():.4f}; MC={s.mean():.4f}; "
            f"bias={s.mean()-a.mean():+.4f}; Spearman={a.corr(s, method='spearman'):.4f}"
        )

    print("\nCONTROL / TAKEDOWN OPPORTUNITY")
    print(
        f"control share: actual={df['actual_ctrl_share'].mean():.4f}; "
        f"MC={df['sim_ctrl_share'].mean():.4f}; "
        f"Spearman={df['actual_ctrl_share'].corr(df['sim_ctrl_share'], method='spearman'):.4f}"
    )
    print(
        f"TD att/min: actual={df['actual_td_att_per_min'].mean():.4f}; "
        f"MC={df['sim_td_att_per_min'].mean():.4f}; "
        f"Spearman={df['actual_td_att_per_min'].corr(df['sim_td_att_per_min'], method='spearman'):.4f}"
    )

    phase_error = sum((df[f"sim_{p}_att_share"] - df[f"actual_{p}_att_share"]).abs() for p in ("distance", "clinch", "ground")) / 2.0
    df["phase_share_error"] = phase_error
    print("\nPACE ERROR RELATIONSHIPS")
    print(f"corr(|pace gap|, phase-share error): {df['pace_gap'].abs().corr(df['phase_share_error'], method='spearman'):.4f}")
    print(f"corr(pace gap, control-share error MC-actual): {df['pace_gap'].corr(df['sim_ctrl_share']-df['actual_ctrl_share'], method='spearman'):.4f}")

    display = [
        "bout_id", "event_date", "red_name", "blue_name", "weight_class", "pace_gap",
        "phase_share_error", "actual_distance_att_share", "sim_distance_att_share",
        "actual_clinch_att_share", "sim_clinch_att_share", "actual_ground_att_share",
        "sim_ground_att_share", "actual_ctrl_share", "sim_ctrl_share",
        "actual_td_att_per_min", "sim_td_att_per_min",
    ]
    display = [c for c in display if c in df.columns]
    print("\n20 LARGEST ABSOLUTE PACE ERRORS WITH PHASE DIAGNOSTICS")
    print(df.reindex(df['pace_gap'].abs().sort_values(ascending=False).index).head(20)[display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION")
    print("- Large phase-share error tied to pace error => phase-transition / control generation problem.")
    print("- Small phase-share error but large pace error => within-phase strike-rate / pressure problem.")
    print("- UFCStats phase mix here is strike-share based; exact historical phase residence time is unavailable.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose phase mix vs strike-rate exposure errors")
    ap.add_argument("--validation", type=Path, default=prior.VALIDATION_PATH)
    ap.add_argument("--fsr-path", type=Path, default=prior.FSR_PATH)
    ap.add_argument("--round-stats", type=Path, default=Path(prior.ROUND_STATS_PATH))
    ap.add_argument("--master", type=Path, default=Path(prior.MASTER_PATH))
    ap.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    validation = prior._load_validation(args.validation)
    bout_ids = set(validation["bout_id"].astype(str))
    pairs, style = prior._load_fsr_pairs(args.fsr_path, bout_ids)
    actual_exp = prior._actual_exposure(args.round_stats, args.master, bout_ids)
    actual_phase = _actual_phase_data(args.round_stats, bout_ids)
    base = validation.merge(actual_exp, on="bout_id", how="left", validate="one_to_one")
    base = base.merge(actual_phase, on="bout_id", how="left", validate="one_to_one")
    base = base.merge(style, on="bout_id", how="left", validate="one_to_one")
    if len(pairs) != len(validation):
        raise ValueError(f"Expected {len(validation)} FSR pairs, found {len(pairs)}")

    sim = _simulate(base, pairs, paths_per_bout=args.paths_per_bout, seed=args.seed)
    out = base.merge(sim, on="bout_id", how="left", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    _print_summary(out)
    print(f"\n[phase-mix diagnostic] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
