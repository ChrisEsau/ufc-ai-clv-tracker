"""Diagnose why KO/TKO V2 predicts historical finishes too late.

This is a read-only historical diagnostic for the current strong shock-collapse
candidate. It uses the exact 300-bout historical validation artifact, selects
only bouts that actually ended by KO/TKO, and reruns their leakage-safe pre-fight
FSR profiles while tracing first-round mechanics.

Primary questions
-----------------
1. Why are many actual Round-1 KO/TKO bouts receiving Round-3 MC finish mass?
2. Did the leakage-safe FSR traits themselves contain the correct KO-side signal?

The diagnostic separates five possible bottlenecks:
1. first-round strike opportunity/exposure;
2. first-round knockdown generation;
3. severe-shock / KD-collapse generation;
4. correct-side lethality / reservoir depletion;
5. fighter-trait / matchup signal quality in the pre-fight FSR profile.

No simulator constants are changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_historical_300_kd_audit as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_historical_300_actual_validation as validation
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse


VALIDATION_PATH = validation.OUTPUT_PATH
FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_ko_tko_v2_actual_ko_timing_diagnostic.parquet"
)
DEFAULT_PATHS_PER_BOUT = 100
DEFAULT_SEED = 20260810
HEARTBEAT_PATHS = 1000
STRONG = validation.STRONG

# KO-relevant FSR traits. Only columns present in the current FSR artifact are
# emitted; missing optional traits never get silently substituted.
KO_FSR_TRAITS = (
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "distance_striking_pressure",
    "distance_precision",
    "distance_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
)

# Directional attacker-vs-defender matchup edges. Positive means the actual KO
# winner entered with the more favorable offensive-vs-defensive FSR matchup.
KO_EDGE_SPECS = (
    ("power_minus_kd_resistance", "striking_power", "knockdown_resistance"),
    ("power_minus_durability", "striking_power", "damage_durability"),
    ("distance_pressure_minus_defense", "distance_striking_pressure", "distance_defense"),
    ("distance_precision_minus_defense", "distance_precision", "distance_defense"),
    ("clinch_pressure_minus_defense", "clinch_striking_pressure", "clinch_striking_defense"),
    ("clinch_precision_minus_defense", "clinch_striking_precision", "clinch_striking_defense"),
    ("ground_pressure_minus_defense", "ground_striking_pressure", "ground_striking_defense"),
    ("ground_precision_minus_defense", "ground_striking_precision", "ground_striking_defense"),
)


class TracedStrongKOSim(collapse.StaticFSRMCKOTKOV2KDCollapse):
    """Strong KO simulator with first-round pathway tracing only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, collapse=STRONG, **kwargs)
        self._trace_segment_calls = 0
        self.r1_sig_att = 0
        self.r1_sig_landed = 0
        self.r1_kd = 0
        self.r1_damage = 0.0
        self.r1_collapse_damage = 0.0
        self.r1_min_reservoir_fraction = 1.0
        self.r1_end_reservoir_fraction = 1.0
        self.r1_max_effective_shock_fraction = 0.0

    def _generate_striking(self, phase: str) -> list[str]:
        self._trace_segment_calls += 1
        round_no = (self._trace_segment_calls - 1) // damage.base.SEGMENTS_PER_ROUND + 1

        if round_no == 1:
            before_att = sum(int(s.sig_att) for s in self.stats)
            before_landed = sum(int(s.sig_landed) for s in self.stats)
            before_kd = sum(int(s.knockdowns_scored) for s in self.stats)
            before_damage = sum(float(s.damage_dealt) for s in self.stats)
            before_collapse = sum(float(x) for x in self.kd_collapse_damage_dealt)

        notes = super()._generate_striking(phase)

        if round_no == 1:
            after_att = sum(int(s.sig_att) for s in self.stats)
            after_landed = sum(int(s.sig_landed) for s in self.stats)
            after_kd = sum(int(s.knockdowns_scored) for s in self.stats)
            after_damage = sum(float(s.damage_dealt) for s in self.stats)
            after_collapse = sum(float(x) for x in self.kd_collapse_damage_dealt)

            self.r1_sig_att += after_att - before_att
            self.r1_sig_landed += after_landed - before_landed
            self.r1_kd += after_kd - before_kd
            self.r1_damage += after_damage - before_damage
            self.r1_collapse_damage += after_collapse - before_collapse

            fractions = [float(state.reservoir_fraction) for state in self.damage_state]
            self.r1_min_reservoir_fraction = min(
                self.r1_min_reservoir_fraction,
                min(fractions),
            )
            self.r1_end_reservoir_fraction = min(fractions)

            red_max = float(self.stats[0].max_single_strike_damage)
            blue_max = float(self.stats[1].max_single_strike_damage)
            red_to_blue = red_max / float(self.damage_state[1].reservoir_capacity)
            blue_to_red = blue_max / float(self.damage_state[0].reservoir_capacity)
            self.r1_max_effective_shock_fraction = max(
                self.r1_max_effective_shock_fraction,
                red_to_blue,
                blue_to_red,
            )

        return notes


def _load_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical KO validation artifact not found: {path}. "
            "Run fsr_static_mc_ko_tko_v2_historical_300_actual_validation.py first."
        )
    frame = pd.read_parquet(path).copy()
    required = {
        "bout_id", "actual_ko_tko", "actual_finish_round", "actual_winner_id",
        "total_rounds", "mc_p_ko_tko",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Historical KO validation missing columns: {missing}")
    frame["bout_id"] = frame["bout_id"].astype(str)
    frame["actual_finish_round"] = pd.to_numeric(frame["actual_finish_round"], errors="coerce")
    return frame


def _load_pairs(fsr_path: Path, bout_ids: set[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    frame = pd.read_parquet(fsr_path)
    bout_key = hist._resolve_bout_key(frame, None)
    frame[bout_key] = frame[bout_key].astype(str)
    frame = frame[frame[bout_key].isin(bout_ids)].copy()
    bouts, _ = hist._prepare_historical_bouts(frame, bout_key=bout_key)
    pairs = {str(bout_id): (red, blue) for bout_id, red, blue in bouts}
    missing = sorted(bout_ids - set(pairs))
    if missing:
        raise ValueError(f"Missing leakage-safe FSR pairs for {len(missing)} KO bouts; first={missing[:10]}")
    return pairs


def _scheduled_rounds(row: pd.Series) -> int:
    value = pd.to_numeric(pd.Series([row.get("total_rounds")]), errors="coerce").iloc[0]
    if pd.notna(value):
        rounds = int(round(float(value)))
        if rounds in (3, 5):
            return rounds
    return 3


def _numeric_value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _fsr_signal_row(
    red: pd.Series,
    blue: pd.Series,
    actual_winner_id: str,
) -> dict[str, object]:
    """Build leakage-safe KO FSR values and actual-winner directional deltas."""
    red_id = str(red["fighter_id"])
    blue_id = str(blue["fighter_id"])
    if actual_winner_id == red_id:
        winner, loser = red, blue
    elif actual_winner_id == blue_id:
        winner, loser = blue, red
    else:
        raise ValueError(
            f"Actual winner {actual_winner_id} does not match FSR pair {red_id} vs {blue_id}."
        )

    out: dict[str, object] = {
        "actual_ko_winner_fighter_id": str(winner["fighter_id"]),
        "actual_ko_loser_fighter_id": str(loser["fighter_id"]),
    }

    for trait in KO_FSR_TRAITS:
        if trait not in winner.index and trait not in loser.index:
            continue
        winner_value = _numeric_value(winner, trait)
        loser_value = _numeric_value(loser, trait)
        out[f"winner_fsr_{trait}"] = winner_value
        out[f"loser_fsr_{trait}"] = loser_value
        out[f"winner_minus_loser_{trait}"] = winner_value - loser_value

    for edge_name, attacker_trait, defender_trait in KO_EDGE_SPECS:
        if attacker_trait not in winner.index or defender_trait not in loser.index:
            continue
        attacker_value = _numeric_value(winner, attacker_trait)
        defender_value = _numeric_value(loser, defender_trait)
        out[f"fsr_edge_{edge_name}"] = attacker_value - defender_value

    return out


def _run_paths(
    actual_ko: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(actual_ko) * paths_per_bout
    counter = 0

    for bout_number, (_, bout) in enumerate(actual_ko.iterrows(), start=1):
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        rounds = _scheduled_rounds(bout)
        actual_winner_id = str(bout["actual_winner_id"])

        for path_index in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = TracedStrongKOSim(red, blue, rounds=rounds, seed=path_seed)
            path = sim.run()
            finish = path.finish

            mc_winner_id = None
            if finish is not None:
                winner_row = red if finish.winner == 0 else blue
                mc_winner_id = str(winner_row["fighter_id"])

            rows.append(
                {
                    "bout_id": bout_id,
                    "actual_finish_round": int(bout["actual_finish_round"]),
                    "actual_winner_id": actual_winner_id,
                    "path_index": path_index,
                    "path_seed": path_seed,
                    "mc_ko_tko": int(finish is not None),
                    "mc_finish_round": int(finish.round) if finish is not None and finish.round is not None else np.nan,
                    "mc_ko_winner_correct": int(mc_winner_id == actual_winner_id) if mc_winner_id is not None else 0,
                    "r1_sig_att": int(sim.r1_sig_att),
                    "r1_sig_landed": int(sim.r1_sig_landed),
                    "r1_any_kd": int(sim.r1_kd > 0),
                    "r1_total_kd": int(sim.r1_kd),
                    "r1_damage": float(sim.r1_damage),
                    "r1_collapse_damage": float(sim.r1_collapse_damage),
                    "r1_min_reservoir_fraction": float(sim.r1_min_reservoir_fraction),
                    "r1_end_reservoir_fraction": float(sim.r1_end_reservoir_fraction),
                    "r1_max_effective_shock_fraction": float(sim.r1_max_effective_shock_fraction),
                }
            )

            counter += 1
            if counter % HEARTBEAT_PATHS == 0 or counter == total_paths:
                recent = pd.DataFrame(rows[-min(HEARTBEAT_PATHS, len(rows)):])
                print(
                    f"[KO timing diagnostic] paths {counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_number}/{len(actual_ko)}; "
                    f"recent R1-KO={float((recent['mc_finish_round'] == 1).mean()):.2%}; "
                    f"recent R1-KD={float(recent['r1_any_kd'].mean()):.2%}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _aggregate(
    paths: pd.DataFrame,
    validation_frame: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bout_id, g in paths.groupby("bout_id", sort=False):
        ko_paths = g[g["mc_ko_tko"] == 1]
        actual_winner_id = str(g["actual_winner_id"].iloc[0])
        red, blue = pairs[str(bout_id)]
        row: dict[str, object] = {
            "bout_id": bout_id,
            "rerun_paths": len(g),
            "rerun_p_ko": float(g["mc_ko_tko"].mean()),
            "rerun_p_r1_ko": float((g["mc_finish_round"] == 1).mean()),
            "rerun_p_r2_ko": float((g["mc_finish_round"] == 2).mean()),
            "rerun_p_r3plus_ko": float((g["mc_finish_round"] >= 3).mean()),
            "rerun_p_r1_kd": float(g["r1_any_kd"].mean()),
            "rerun_mean_r1_kd": float(g["r1_total_kd"].mean()),
            "rerun_mean_r1_sig_att": float(g["r1_sig_att"].mean()),
            "rerun_mean_r1_sig_landed": float(g["r1_sig_landed"].mean()),
            "rerun_mean_r1_damage": float(g["r1_damage"].mean()),
            "rerun_mean_r1_collapse_damage": float(g["r1_collapse_damage"].mean()),
            "rerun_mean_r1_min_reservoir_fraction": float(g["r1_min_reservoir_fraction"].mean()),
            "rerun_mean_r1_end_reservoir_fraction": float(g["r1_end_reservoir_fraction"].mean()),
            "rerun_mean_r1_max_shock_fraction": float(g["r1_max_effective_shock_fraction"].mean()),
            "rerun_p_correct_ko_winner_unconditional": float(g["mc_ko_winner_correct"].mean()),
            "rerun_p_correct_ko_winner_given_ko": (
                float(ko_paths["mc_ko_winner_correct"].mean()) if len(ko_paths) else np.nan
            ),
        }
        row.update(_fsr_signal_row(red, blue, actual_winner_id))
        rows.append(row)

    agg = pd.DataFrame(rows)
    keep = [
        "bout_id", "actual_finish_round", "actual_winner_id", "method",
        "mc_p_ko_tko", "mc_predicted_ko_round_conditional",
        "mc_expected_ko_round_conditional", "mc_predicted_ko_winner_id",
    ]
    keep = [c for c in keep if c in validation_frame.columns]
    return validation_frame[keep].merge(agg, on="bout_id", how="inner", validate="one_to_one")


def _timing_group(round_value: float) -> str:
    if round_value == 1:
        return "actual R1 KO"
    if round_value == 2:
        return "actual R2 KO"
    return "actual R3+ KO"


def _print_fsr_signal_summary(work: pd.DataFrame) -> None:
    edge_cols = [c for c in work.columns if c.startswith("fsr_edge_")]
    if not edge_cols:
        print("\nFSR KO SIGNAL")
        print("No configured KO edge columns were available in the current FSR artifact.")
        return

    print("\nFSR KO SIGNAL — ACTUAL WINNER VS ACTUAL LOSER")
    timing = work.copy()
    timing["actual_timing_group"] = timing["actual_finish_round"].map(_timing_group)
    grouped = timing.groupby("actual_timing_group", sort=False)

    rows: list[dict[str, object]] = []
    for group_name, g in grouped:
        row: dict[str, object] = {"actual_timing_group": group_name, "bouts": len(g)}
        for col in edge_cols:
            row[col] = float(pd.to_numeric(g[col], errors="coerce").mean())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nFSR EDGE ASSOCIATION WITH SIMULATED KO SIGNAL — ACTUAL KO BOUTS")
    assoc_rows = []
    for col in edge_cols:
        s = pd.to_numeric(work[col], errors="coerce")
        valid = s.notna() & work["rerun_p_ko"].notna()
        if valid.sum() < 3:
            continue
        assoc_rows.append(
            {
                "fsr_edge": col,
                "n": int(valid.sum()),
                "spearman_vs_mc_pKO": float(s[valid].corr(work.loc[valid, "rerun_p_ko"], method="spearman")),
                "spearman_vs_mc_pR1KO": float(s[valid].corr(work.loc[valid, "rerun_p_r1_ko"], method="spearman")),
                "mean_edge": float(s[valid].mean()),
                "positive_edge_share": float((s[valid] > 0).mean()),
            }
        )
    if assoc_rows:
        print(
            pd.DataFrame(assoc_rows)
            .sort_values("spearman_vs_mc_pKO", ascending=False)
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )


def _print_summary(frame: pd.DataFrame) -> None:
    work = frame.copy()
    work["actual_timing_group"] = work["actual_finish_round"].map(_timing_group)

    print("\n" + "=" * 124)
    print("ACTUAL KO/TKO TIMING DIAGNOSTIC — STRONG SHOCK-COLLAPSE")
    print("=" * 124)
    print(f"actual KO/TKO bouts traced: {len(work):,}")
    print(f"paths per bout: {int(work['rerun_paths'].iloc[0]) if len(work) else 0:,}")
    print(f"strong candidate: collapse scale={STRONG.collapse_scale:.2f}; curvature={STRONG.shock_curvature:.2f}")

    summary = (
        work.groupby("actual_timing_group", sort=False, as_index=False)
        .agg(
            bouts=("bout_id", "size"),
            mc_p_ko=("rerun_p_ko", "mean"),
            mc_p_r1_ko=("rerun_p_r1_ko", "mean"),
            mc_p_r1_kd=("rerun_p_r1_kd", "mean"),
            mc_r1_sig_att=("rerun_mean_r1_sig_att", "mean"),
            mc_r1_sig_landed=("rerun_mean_r1_sig_landed", "mean"),
            mc_r1_damage=("rerun_mean_r1_damage", "mean"),
            mc_r1_collapse=("rerun_mean_r1_collapse_damage", "mean"),
            mc_r1_max_shock=("rerun_mean_r1_max_shock_fraction", "mean"),
            mc_r1_min_res_frac=("rerun_mean_r1_min_reservoir_fraction", "mean"),
            correct_ko_side_given_ko=("rerun_p_correct_ko_winner_given_ko", "mean"),
        )
    )
    print("\nPATHWAY BY ACTUAL KO ROUND")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _print_fsr_signal_summary(work)

    r1 = work[work["actual_finish_round"] == 1].copy()
    if not r1.empty:
        print("\nACTUAL ROUND-1 KO/TKO BOUTS — KEY FAILURE COUNTS")
        for threshold in (0.10, 0.20, 0.30, 0.40):
            count = int((r1["rerun_p_r1_ko"] >= threshold).sum())
            print(f"MC p(R1 KO) >= {threshold:.0%}: {count}/{len(r1)} ({count/len(r1):.2%})")
        low_kd = int((r1["rerun_p_r1_kd"] < 0.20).sum())
        low_ko_despite_kd = int(((r1["rerun_p_r1_kd"] >= 0.30) & (r1["rerun_p_r1_ko"] < 0.10)).sum())
        wrong_side = int((r1["rerun_p_correct_ko_winner_given_ko"] < 0.50).sum())
        print(f"R1 KO bouts with MC p(R1 KD) <20%: {low_kd}/{len(r1)}")
        print(f"R1 KO bouts with p(R1 KD)>=30% but p(R1 KO)<10%: {low_ko_despite_kd}/{len(r1)}")
        print(f"R1 KO bouts where actual winner gets <50% of simulated KO wins: {wrong_side}/{len(r1)}")

        display = [
            "bout_id", "actual_finish_round", "rerun_p_ko", "rerun_p_r1_ko",
            "rerun_p_r1_kd", "rerun_mean_r1_sig_att", "rerun_mean_r1_sig_landed",
            "rerun_mean_r1_max_shock_fraction", "rerun_mean_r1_collapse_damage",
            "rerun_mean_r1_min_reservoir_fraction", "rerun_p_correct_ko_winner_given_ko",
            "winner_fsr_striking_power", "loser_fsr_knockdown_resistance",
            "loser_fsr_damage_durability", "fsr_edge_power_minus_kd_resistance",
            "fsr_edge_power_minus_durability", "fsr_edge_distance_pressure_minus_defense",
            "fsr_edge_distance_precision_minus_defense",
        ]
        display = [c for c in display if c in r1.columns]
        print("\n15 WORST-MISSED ACTUAL R1 KO BOUTS — LOWEST MC P(R1 KO)")
        print(
            r1.sort_values("rerun_p_r1_ko", ascending=True)
            .head(15)[display]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\nDIAGNOSTIC GUIDE")
    print("- Low R1 strike attempts/landed -> opportunity/exposure bottleneck.")
    print("- Reasonable R1 exposure but low p(R1 KD) -> shock/KD-generation bottleneck.")
    print("- Reasonable p(R1 KD) but low p(R1 KO) -> KD-collapse/follow-up lethality bottleneck.")
    print("- Positive KO FSR edge but weak MC output -> simulator translation/mechanics bottleneck.")
    print("- Weak/negative KO FSR edge for an actual KO winner -> underlying FSR signal miss.")
    print("- Good R1 KO probability but wrong KO side -> fighter-trait / matchup-direction bottleneck.")
    print("- This script changes no simulator constants and is not a calibration sweep.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose late KO timing on actual historical KO/TKO bouts")
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    frame = _load_validation(args.validation)
    actual_ko = frame[frame["actual_ko_tko"] == 1].copy()
    actual_ko = actual_ko[actual_ko["actual_finish_round"].notna()].copy()
    bout_ids = set(actual_ko["bout_id"].astype(str))

    print(
        f"[KO timing diagnostic] actual KO/TKO bouts={len(actual_ko):,}; "
        f"R1={int((actual_ko['actual_finish_round'] == 1).sum())}; "
        f"R2={int((actual_ko['actual_finish_round'] == 2).sum())}; "
        f"R3+={int((actual_ko['actual_finish_round'] >= 3).sum())}",
        flush=True,
    )
    pairs = _load_pairs(args.fsr_path, bout_ids)
    print(f"[KO timing diagnostic] matched leakage-safe FSR pairs={len(pairs):,}", flush=True)

    available_traits = sorted(
        trait for trait in KO_FSR_TRAITS
        if any(trait in row.index for pair in pairs.values() for row in pair)
    )
    print(
        f"[KO timing diagnostic] KO-relevant FSR traits found={len(available_traits)}: "
        f"{', '.join(available_traits)}",
        flush=True,
    )
    print(
        f"[KO timing diagnostic] paths_per_bout={args.paths_per_bout}; "
        f"total_paths={len(actual_ko) * args.paths_per_bout:,}",
        flush=True,
    )

    paths = _run_paths(
        actual_ko,
        pairs,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed,
    )
    result = _aggregate(paths, frame, pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    _print_summary(result)
    print(f"\n[KO timing diagnostic] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
