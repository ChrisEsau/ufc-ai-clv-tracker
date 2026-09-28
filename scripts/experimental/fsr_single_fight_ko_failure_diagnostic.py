"""Reusable single-fight diagnostic for KO/TKO Monte Carlo direction failures.

Run one historical matchup through the current shadow finish engine at a higher
path count and print the information needed to diagnose WHY the MC favors one
fighter over the other.

Current engine under inspection
-------------------------------
- strong KD-collapse candidate
- locked age adjustment (KD resistance + damage durability only)
- provisional between-round recovery
- 3-round horizon by default

The script changes no FSR values or simulator constants.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v2_round_recovery as recovery
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_PATHS = 250
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/single_fight_diagnostics")

RELEVANT_FSR_TRAITS = [
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "recovery_ability",
    "distance_striking_pressure",
    "distance_striking_precision",
    "distance_striking_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
    "wrestling_entry",
    "wrestling_conversion",
    "td_defense",
    "control_imposition",
    "control_resistance",
    "reversal_ability",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose one historical KO/TKO MC matchup")
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--bout-id", type=str)
    selector.add_argument("--red", type=str, help="Red fighter name substring or fighter id")
    p.add_argument("--blue", type=str, help="Blue fighter name substring or fighter id; required with --red")
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def _text_match(value: object, query: str) -> bool:
    return query.strip().lower() in str(value).strip().lower()


def _fighter_name(profile: pd.Series) -> str:
    return base._display_name(profile)


def _resolve_bout(
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    bout_id: str | None,
    red_query: str | None,
    blue_query: str | None,
) -> tuple[pd.Series, tuple[pd.Series, pd.Series]]:
    if bout_id is not None:
        match = cohort[cohort["bout_id"].astype(str).eq(str(bout_id))]
        if len(match) != 1:
            raise ValueError(f"Expected exactly one cohort bout for {bout_id!r}; found {len(match)}")
        bout = match.iloc[0]
        return bout, pairs[str(bout["bout_id"])]

    if red_query is None or blue_query is None:
        raise ValueError("--blue is required when using --red")

    matches: list[tuple[int, str, str]] = []
    for idx, bout in cohort.iterrows():
        red, blue = pairs[str(bout["bout_id"])]
        red_values = [str(bout["r_id"]), _fighter_name(red)]
        blue_values = [str(bout["b_id"]), _fighter_name(blue)]
        direct = any(_text_match(v, red_query) for v in red_values) and any(
            _text_match(v, blue_query) for v in blue_values
        )
        reverse = any(_text_match(v, blue_query) for v in red_values) and any(
            _text_match(v, red_query) for v in blue_values
        )
        if direct or reverse:
            matches.append((idx, _fighter_name(red), _fighter_name(blue)))

    if len(matches) != 1:
        preview = ", ".join(f"{r} vs {b}" for _, r, b in matches[:12])
        raise ValueError(
            f"Expected one historical matchup for {red_query!r} vs {blue_query!r}; "
            f"found {len(matches)}. Matches: {preview}"
        )
    bout = cohort.loc[matches[0][0]]
    return bout, pairs[str(bout["bout_id"])]


def _master_row(bout_id: str) -> pd.Series | None:
    raw = pd.read_parquet(population.modern.MASTER_PATH).copy()
    raw["fight_id"] = raw["fight_id"].astype(str)
    rows = raw[raw["fight_id"].eq(str(bout_id))]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _corner_age(bout: pd.Series, corner: str) -> float | None:
    col = f"{corner}_age"
    return float(bout[col]) if col in bout.index and pd.notna(bout[col]) else None


def _effective_profile(profile: pd.Series, age: float | None) -> pd.Series:
    return ko.apply_locked_age_adjustment(profile, age)


def _expected_attempts_per_30s(profile: pd.Series, phase: str) -> float:
    pressure = base._value(profile, base.PHASE_PRESSURE[phase])
    return float(base.STRIKE_ATTEMPTS_PER_30S_BASE[phase] * np.exp((pressure - 50.0) / 12.0))


def _expected_accuracy(attacker: pd.Series, defender: pd.Series, phase: str) -> float:
    precision = base._value(attacker, base.PHASE_PRECISION[phase])
    defense = base._value(defender, base.PHASE_DEFENSE[phase])
    return float(
        base._sigmoid(base._logit(base.STRIKE_ACCURACY_BASE[phase]) + (precision - defense) / base.RATING_SCALE)
    )


def _tail_probability(profile: pd.Series) -> float:
    power = base._value(profile, "striking_power")
    return float(
        damage._sigmoid(
            damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
            + (power - 50.0) / damage.POWER_TAIL_RATING_SCALE
        )
    )


def _initial_kd_probability(profile: pd.Series, strike_damage: float) -> float:
    durability = base._value(profile, "damage_durability")
    resistance = base._value(profile, "knockdown_resistance")
    capacity = damage.reservoir_capacity_from_durability(durability)
    shock = float(strike_damage) / capacity
    logit_p = damage.KD_BASE_LOGIT + damage.KD_SHOCK_COEFFICIENT * shock + (
        50.0 - resistance
    ) / damage.KD_RESISTANCE_SCALE
    return float(np.clip(damage._sigmoid(logit_p), 0.0, 0.95))


def _profile_diagnostic_row(
    label: str,
    raw: pd.Series,
    effective: pd.Series,
    opponent_effective: pd.Series,
    age: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"fighter": label, "age": age}
    for trait in RELEVANT_FSR_TRAITS:
        row[trait] = base._value(raw, trait)
    row["effective_knockdown_resistance"] = base._value(effective, "knockdown_resistance")
    row["effective_damage_durability"] = base._value(effective, "damage_durability")
    row["reservoir_capacity"] = damage.reservoir_capacity_from_durability(
        row["effective_damage_durability"]
    )
    row["round_recovery_fraction"] = recovery.round_recovery_fraction(
        base._value(effective, "recovery_ability")
    )
    row["power_tail_probability"] = _tail_probability(effective)
    for phase in ("DISTANCE", "CLINCH", "GROUND"):
        key = phase.lower()
        row[f"expected_{key}_attempts_per_30s"] = _expected_attempts_per_30s(effective, phase)
        row[f"expected_{key}_accuracy"] = _expected_accuracy(effective, opponent_effective, phase)
        row[f"expected_{key}_landed_per_30s"] = (
            row[f"expected_{key}_attempts_per_30s"] * row[f"expected_{key}_accuracy"]
        )
    for strike_damage in (2.0, 5.0, 8.0, 12.0):
        row[f"initial_p_kd_if_damage_{strike_damage:g}"] = _initial_kd_probability(
            effective, strike_damage
        )
    return row


class DiagnosticRecoverySim(recovery.StaticFSRMCKOTKOV2RoundRecovery):
    """Recovery simulator that snapshots cumulative stats/reservoir at round ends."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.round_end_snapshots: list[dict[str, Any]] = []

    def _snapshot(self, round_no: int, *, before_recovery: bool) -> None:
        for fighter_index, stats in enumerate(self.stats):
            assert isinstance(stats, damage.DamageFighterStats)
            state = self.damage_state[fighter_index]
            self.round_end_snapshots.append(
                {
                    "round": round_no,
                    "fighter": fighter_index,
                    "before_recovery": int(before_recovery),
                    "sig_att": stats.sig_att,
                    "sig_landed": stats.sig_landed,
                    "td_att": stats.td_att,
                    "td_landed": stats.td_landed,
                    "control_seconds": stats.control_seconds,
                    "clinch_control_seconds": stats.clinch_control_seconds,
                    "ground_control_seconds": stats.ground_control_seconds,
                    "damage_dealt": stats.damage_dealt,
                    "damage_absorbed": stats.damage_absorbed,
                    "knockdowns_scored": stats.knockdowns_scored,
                    "knockdowns_absorbed": stats.knockdowns_absorbed,
                    "max_single_strike_damage": stats.max_single_strike_damage,
                    "reservoir_current": state.reservoir_current,
                    "reservoir_capacity": state.reservoir_capacity,
                    "reservoir_fraction": state.reservoir_fraction,
                    "distance_segments": stats.phase_segments["DISTANCE"],
                    "clinch_segments": stats.phase_segments["CLINCH"],
                    "ground_segments": stats.phase_segments["GROUND"],
                }
            )

    def run(self) -> ko.KOPath:
        events: list[dict[str, Any]] = []
        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None
            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller
                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1
                strike_notes = self._generate_striking(phase_start)
                if self.finish is not None:
                    self.finish.round = round_no
                    self.finish.segment = segment_no
                    self.finish.clock_start = self._clock_start(segment_no)
                    transition_note = "fight stopped"
                elif phase_start == "DISTANCE":
                    transition_note = self._distance_transition()
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition()
                else:
                    transition_note = self._ground_transition()
                events.append(
                    {
                        "round": round_no,
                        "segment": segment_no,
                        "clock_start": self._clock_start(segment_no),
                        "phase_start": phase_start,
                        "phase_end": self.phase,
                        "top_start": ground_controller_start,
                        "clinch_controller_start": clinch_controller_start,
                        "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                        "transition": transition_note,
                        "finish": self.finish is not None,
                    }
                )
                if self.finish is not None:
                    self._snapshot(round_no, before_recovery=True)
                    return ko.KOPath(events=events, stats=self.stats, finish=self.finish)
            self._snapshot(round_no, before_recovery=True)
            if round_no < self.rounds:
                self._apply_between_round_recovery(round_no)
                self._snapshot(round_no, before_recovery=False)
        return ko.KOPath(events=events, stats=self.stats, finish=None)


def _incremental_round_rows(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before = pd.DataFrame([r for r in snapshots if r["before_recovery"] == 1])
    if before.empty:
        return []
    cumulative_cols = [
        "sig_att", "sig_landed", "td_att", "td_landed", "control_seconds",
        "clinch_control_seconds", "ground_control_seconds", "damage_dealt", "damage_absorbed",
        "knockdowns_scored", "knockdowns_absorbed", "distance_segments", "clinch_segments", "ground_segments",
    ]
    rows: list[dict[str, Any]] = []
    for fighter in (0, 1):
        g = before[before["fighter"].eq(fighter)].sort_values("round").copy()
        prev = {col: 0.0 for col in cumulative_cols}
        for _, r in g.iterrows():
            out = {"round": int(r["round"]), "fighter": fighter}
            for col in cumulative_cols:
                out[col] = float(r[col]) - prev[col]
                prev[col] = float(r[col])
            out["max_single_strike_damage"] = float(r["max_single_strike_damage"])
            out["reservoir_fraction_end"] = float(r["reservoir_fraction"])
            rows.append(out)
    return rows


def _actual_master_stats(row: pd.Series | None) -> pd.DataFrame:
    if row is None:
        return pd.DataFrame()
    # Print any commonly useful canonical fight-level columns that actually exist.
    candidates = [
        "method", "finish_round", "finish_time", "time_format", "winner_id",
        "r_name", "b_name", "r_fighter", "b_fighter",
        "r_sig_str_landed", "b_sig_str_landed", "r_sig_str_attempted", "b_sig_str_attempted",
        "r_total_str_landed", "b_total_str_landed", "r_total_str_attempted", "b_total_str_attempted",
        "r_kd", "b_kd", "r_td_landed", "b_td_landed", "r_td_attempted", "b_td_attempted",
        "r_ctrl", "b_ctrl", "r_control_seconds", "b_control_seconds",
    ]
    present = [c for c in candidates if c in row.index]
    return pd.DataFrame([{"field": c, "value": row[c]} for c in present])


def _summary_stats(path_rows: pd.DataFrame, names: list[str], paths: int) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for fighter in (0, 1):
        g = path_rows[path_rows["fighter"].eq(fighter)]
        out.append(
            {
                "fighter": names[fighter],
                "P_KO_win": float((g["finish_winner"].eq(fighter)).mean()),
                "P_R1_KO_win": float((g["finish_winner"].eq(fighter) & g["finish_round"].eq(1)).mean()),
                "mean_sig_att": g["sig_att"].mean(),
                "mean_sig_landed": g["sig_landed"].mean(),
                "mean_accuracy": g["sig_landed"].sum() / max(g["sig_att"].sum(), 1),
                "mean_damage_dealt": g["damage_dealt"].mean(),
                "mean_KD_scored": g["knockdowns_scored"].mean(),
                "P_scores_KD": float(g["knockdowns_scored"].gt(0).mean()),
                "mean_max_strike_damage": g["max_single_strike_damage"].mean(),
                "mean_TD_att": g["td_att"].mean(),
                "mean_TD_landed": g["td_landed"].mean(),
                "mean_control_sec": g["control_seconds"].mean(),
                "mean_round_recovery": g["total_round_recovery"].mean(),
                "mean_final_reservoir_fraction": g["final_reservoir_fraction"].mean(),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    cohort, pairs = population._build_cohort()
    bout, (red_raw, blue_raw) = _resolve_bout(
        cohort,
        pairs,
        bout_id=args.bout_id,
        red_query=args.red,
        blue_query=args.blue,
    )
    bout_id = str(bout["bout_id"])
    red_age = _corner_age(bout, "r")
    blue_age = _corner_age(bout, "b")
    red_eff = _effective_profile(red_raw, red_age)
    blue_eff = _effective_profile(blue_raw, blue_age)
    names = [_fighter_name(red_raw), _fighter_name(blue_raw)]

    profile_rows = pd.DataFrame(
        [
            _profile_diagnostic_row(names[0], red_raw, red_eff, blue_eff, red_age),
            _profile_diagnostic_row(names[1], blue_raw, blue_eff, red_eff, blue_age),
        ]
    )

    rng = np.random.default_rng(args.seed)
    path_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    finish_rows: list[dict[str, Any]] = []

    print(f"Running {args.paths} paths: {names[0]} vs {names[1]} ({bout_id})", flush=True)
    for path_index in range(args.paths):
        sim = DiagnosticRecoverySim(
            red_raw,
            blue_raw,
            collapse=STRONG_COLLAPSE,
            rounds=args.rounds,
            seed=int(rng.integers(0, 2**31 - 1)),
            red_age=red_age,
            blue_age=blue_age,
        )
        result = sim.run()
        finish = result.finish
        for fighter in (0, 1):
            stats = sim.stats[fighter]
            assert isinstance(stats, damage.DamageFighterStats)
            path_rows.append(
                {
                    "path": path_index,
                    "fighter": fighter,
                    "fighter_name": names[fighter],
                    "finish_winner": finish.winner if finish is not None else np.nan,
                    "finish_round": finish.round if finish is not None else np.nan,
                    "sig_att": stats.sig_att,
                    "sig_landed": stats.sig_landed,
                    "td_att": stats.td_att,
                    "td_landed": stats.td_landed,
                    "control_seconds": stats.control_seconds,
                    "damage_dealt": stats.damage_dealt,
                    "damage_absorbed": stats.damage_absorbed,
                    "knockdowns_scored": stats.knockdowns_scored,
                    "knockdowns_absorbed": stats.knockdowns_absorbed,
                    "max_single_strike_damage": stats.max_single_strike_damage,
                    "total_round_recovery": sim.total_round_recovery[fighter],
                    "final_reservoir_fraction": sim.damage_state[fighter].reservoir_fraction,
                }
            )
        for row in _incremental_round_rows(sim.round_end_snapshots):
            row.update({"path": path_index, "fighter_name": names[int(row["fighter"])]})
            round_rows.append(row)
        if finish is not None:
            finish_rows.append(
                {
                    "path": path_index,
                    "winner": names[finish.winner],
                    "loser": names[finish.loser],
                    "round": finish.round,
                    "segment": finish.segment,
                    "clock_start": finish.clock_start,
                    "knockdown_on_strike": finish.knockdown_on_strike,
                    "recent_kd_before": finish.recent_kd_before,
                    "raw_strike_damage": finish.raw_strike_damage,
                    "effective_strike_damage": finish.effective_strike_damage,
                    "reservoir_before": finish.reservoir_before,
                    "reservoir_after": finish.reservoir_after,
                }
            )
        if (path_index + 1) % 50 == 0 or path_index + 1 == args.paths:
            print(f"[single fight diagnostic] paths {path_index + 1:,}/{args.paths:,}", flush=True)

    paths_df = pd.DataFrame(path_rows)
    rounds_df = pd.DataFrame(round_rows)
    finishes_df = pd.DataFrame(finish_rows)
    summary = _summary_stats(paths_df, names, args.paths)
    actual = _actual_master_stats(_master_row(bout_id))

    print("\n" + "=" * 130)
    print("SINGLE-FIGHT KO FAILURE DIAGNOSTIC")
    print("=" * 130)
    print(f"bout_id: {bout_id}")
    print(f"fight: {names[0]} vs {names[1]}")
    print(f"event_date: {bout['event_date']}")
    print(f"actual KO/TKO: {int(bout['actual_ko_tko'])}; actual finish round: {bout['actual_finish_round']}")
    print(f"paths: {args.paths}; horizon: {args.rounds} rounds")

    print("\nPRE-FIGHT FSR + DERIVED MC INPUTS")
    print(profile_rows.T.to_string(header=names, float_format=lambda x: f"{x:.4f}"))

    print("\nSIMULATED FULL-FIGHT SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    any_finish = len(finishes_df) / args.paths
    r1_finish = (finishes_df["round"].eq(1).sum() / args.paths) if not finishes_df.empty else 0.0
    print(f"\nP(any KO/TKO): {any_finish:.2%}")
    print(f"P(R1 KO/TKO): {r1_finish:.2%}")
    if not finishes_df.empty:
        print("\nSIMULATED FINISH DISTRIBUTION")
        print(
            finishes_df.groupby(["round", "winner"]).size().rename("finishes").reset_index().to_string(index=False)
        )
        print("\nFINISH-MECHANISM SUMMARY")
        print(f"finish on KD-causing strike: {finishes_df['knockdown_on_strike'].mean():.2%}")
        print(f"finish with recent KD already active: {finishes_df['recent_kd_before'].mean():.2%}")
        print(f"mean final effective strike damage: {finishes_df['effective_strike_damage'].mean():.3f}")
        print(f"mean reservoir immediately before finishing strike: {finishes_df['reservoir_before'].mean():.3f}")

    if not rounds_df.empty:
        agg = rounds_df.groupby(["round", "fighter_name"], as_index=False).agg(
            sig_att=("sig_att", "mean"),
            sig_landed=("sig_landed", "mean"),
            damage_dealt=("damage_dealt", "mean"),
            knockdowns_scored=("knockdowns_scored", "mean"),
            control_seconds=("control_seconds", "mean"),
            distance_segments=("distance_segments", "mean"),
            clinch_segments=("clinch_segments", "mean"),
            ground_segments=("ground_segments", "mean"),
            reservoir_fraction_end=("reservoir_fraction_end", "mean"),
        )
        print("\nROUND-BY-ROUND SIMULATED MEANS")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not actual.empty:
        print("\nAVAILABLE MASTER-FILE ACTUAL FIGHT FIELDS")
        print(actual.to_string(index=False))

    print("\nDIAGNOSTIC FLAGS")
    winner_id = str(bout.get("winner_id", ""))
    winner_name = names[0] if winner_id == str(bout["r_id"]) else names[1] if winner_id == str(bout["b_id"]) else "unknown"
    loser_name = names[1] if winner_name == names[0] else names[0] if winner_name == names[1] else "unknown"
    s = summary.set_index("fighter")
    if winner_name in s.index and loser_name in s.index:
        print(f"actual winner: {winner_name}; actual loser: {loser_name}")
        print(f"MC KO direction edge (winner-loser): {s.loc[winner_name, 'P_KO_win'] - s.loc[loser_name, 'P_KO_win']:+.3f}")
        print(f"R1 KO direction edge (winner-loser): {s.loc[winner_name, 'P_R1_KO_win'] - s.loc[loser_name, 'P_R1_KO_win']:+.3f}")
        print(f"sig-attempt edge: {s.loc[winner_name, 'mean_sig_att'] - s.loc[loser_name, 'mean_sig_att']:+.2f}")
        print(f"sig-landed edge: {s.loc[winner_name, 'mean_sig_landed'] - s.loc[loser_name, 'mean_sig_landed']:+.2f}")
        print(f"damage edge: {s.loc[winner_name, 'mean_damage_dealt'] - s.loc[loser_name, 'mean_damage_dealt']:+.2f}")
        print(f"KD edge: {s.loc[winner_name, 'mean_KD_scored'] - s.loc[loser_name, 'mean_KD_scored']:+.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / bout_id
    profile_rows.to_csv(f"{stem}_profiles.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    paths_df.to_csv(f"{stem}_paths.csv", index=False)
    rounds_df.to_csv(f"{stem}_rounds.csv", index=False)
    finishes_df.to_csv(f"{stem}_finishes.csv", index=False)
    print(f"\nWrote diagnostic CSVs with prefix: {stem}")
    print("No simulator constants or stored FSR values were changed.")


if __name__ == "__main__":
    main()
