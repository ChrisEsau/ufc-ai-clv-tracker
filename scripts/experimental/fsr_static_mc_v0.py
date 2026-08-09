"""Rudimentary static-state FSR Monte Carlo V0.

Purpose
-------
Generate inspectable 30-second MMA fight paths from FSR-26 profiles before
adding the dynamic state engine or a detailed finish model.

V0 deliberately includes:
- distance / clinch / ground phase state
- calibrated hierarchical transition priors
- takedown attempt vs takedown success split
- phase-specific striking attempts and landed strikes
- ground control, escapes, reversals, submission attempts
- deterministic seeding and verbose segment path output

V0 deliberately excludes:
- fatigue / adversity / recovery / score urgency modifiers
- damage-driven behavior changes
- KO/TKO/submission finishes
- judging / winner prediction

The goal is path plausibility, not predictive accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import exp, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)

SEGMENT_SECONDS = 30
SEGMENTS_PER_ROUND = 10
DEFAULT_ROUNDS = 3
RATING_SCALE = 12.0
MODIFIER_SCALE = 6.0

# Starting priors from hierarchical TD-attempt/success research V2.
DISTANCE_CLINCH_BASE = 0.04
DISTANCE_TD_ATTEMPT_BASE = 0.10
CLINCH_SEPARATE_BASE = 0.30
CLINCH_TD_ATTEMPT_BASE = 0.24
GROUND_EXIT_BASE = 0.40
TD_SUCCESS_LOGIT_OFFSET = -0.40

# V0 event-rate priors. These are intentionally simple placeholders used only
# to create plausible paths. They are NOT locked calibration constants.
STRIKE_ATTEMPTS_PER_SEGMENT_BASE = {
    "DISTANCE": 5.0,
    "CLINCH": 1.2,
    "GROUND": 1.6,
}
STRIKE_ACCURACY_BASE = {
    "DISTANCE": 0.40,
    "CLINCH": 0.68,
    "GROUND": 0.70,
}
SUB_ATTEMPT_BASE_PER_GROUND_SEGMENT = 0.045
REVERSAL_SHARE_OF_GROUND_EXITS = 0.18

PHASE_PRESSURE = {
    "DISTANCE": "distance_striking_pressure",
    "CLINCH": "clinch_striking_pressure",
    "GROUND": "ground_striking_pressure",
}
PHASE_PRECISION = {
    "DISTANCE": "distance_striking_precision",
    "CLINCH": "clinch_striking_precision",
    "GROUND": "ground_striking_precision",
}
PHASE_DEFENSE = {
    "DISTANCE": "distance_striking_defense",
    "CLINCH": "clinch_striking_defense",
    "GROUND": "ground_striking_defense",
}

REQUIRED_COLUMNS = {
    "fighter_id",
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
    "submission_pressure",
    "reversal_ability",
}


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -12.0, 12.0))
    return 1.0 / (1.0 + exp(-x))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return log(p / (1.0 - p))


def _modifier(delta: float, scale: float = MODIFIER_SCALE) -> float:
    return exp(float(np.clip(delta, -8.0, 8.0)) / scale)


def _prob(x: float, high: float = 0.95) -> float:
    return float(np.clip(x, 0.0, high))


def _value(profile: pd.Series, name: str, default: float = 50.0) -> float:
    value = profile.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def _display_name(profile: pd.Series) -> str:
    for key in ("fighter_name", "name", "fighter"):
        value = profile.get(key)
        if value is not None and not pd.isna(value):
            return str(value)
    return str(profile["fighter_id"])


def _style_preferences(profile: pd.Series) -> tuple[float, float, float]:
    d = _value(profile, "distance_striking_pressure")
    c = _value(profile, "clinch_striking_pressure")
    w = _value(profile, "wrestling_entry")
    ctrl = _value(profile, "control_imposition")

    distance_pref = d - 0.5 * c - 0.5 * w
    clinch_pref = c - 0.5 * d - 0.5 * w
    wrestling_pref = 0.75 * w + 0.25 * ctrl - 0.5 * d - 0.5 * c
    return distance_pref, clinch_pref, wrestling_pref


def _latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    date_col = next(
        (c for c in ("event_date", "fight_date", "date") if c in frame.columns),
        None,
    )
    if date_col is None:
        return frame.drop_duplicates("fighter_id", keep="last").copy()

    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values(["fighter_id", date_col])
    return work.drop_duplicates("fighter_id", keep="last").copy()


def load_profiles(path: Path = FSR_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"FSR-26 artifact missing required columns: {missing}")
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return _latest_rows(frame).reset_index(drop=True)


def find_profile(frame: pd.DataFrame, query: str) -> pd.Series:
    q = str(query).strip().lower()

    exact_id = frame[frame["fighter_id"].str.lower() == q]
    if len(exact_id) == 1:
        return exact_id.iloc[0]

    for name_col in ("fighter_name", "name", "fighter"):
        if name_col not in frame.columns:
            continue
        names = frame[name_col].astype(str)
        exact = frame[names.str.lower() == q]
        if len(exact) == 1:
            return exact.iloc[0]
        partial = frame[names.str.lower().str.contains(q, regex=False)]
        if len(partial) == 1:
            return partial.iloc[0]
        if len(partial) > 1:
            choices = ", ".join(partial[name_col].astype(str).head(10))
            raise ValueError(f"Ambiguous fighter query {query!r}: {choices}")

    raise ValueError(
        f"Could not resolve fighter {query!r}. Use fighter_id if the artifact "
        "does not contain a name column."
    )


@dataclass
class FighterStats:
    sig_att: int = 0
    sig_landed: int = 0
    td_att: int = 0
    td_landed: int = 0
    control_seconds: int = 0
    sub_att: int = 0
    reversals: int = 0
    phase_segments: dict[str, int] = field(
        default_factory=lambda: {"DISTANCE": 0, "CLINCH": 0, "GROUND": 0}
    )


@dataclass
class FightPath:
    events: list[dict[str, Any]]
    stats: list[FighterStats]


class StaticFSRMCV0:
    """Single-fight static path simulator with no dynamic-state engine."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        rounds: int = DEFAULT_ROUNDS,
        seed: int = 7,
    ) -> None:
        self.fighters = [red, blue]
        self.names = [_display_name(red), _display_name(blue)]
        self.rounds = int(rounds)
        self.rng = np.random.default_rng(seed)
        self.stats = [FighterStats(), FighterStats()]

        self.phase = "DISTANCE"
        self.ground_controller: int | None = None
        self.clinch_initiator: int | None = None

    def _other(self, i: int) -> int:
        return 1 - i

    def _initiative_actor(self, phase: str) -> int:
        if phase == "GROUND" and self.ground_controller is not None:
            # Controller has more offensive initiative, but bottom fighter can
            # still create escape/reversal/submission moments.
            if self.rng.random() < 0.68:
                return self.ground_controller
            return self._other(self.ground_controller)

        if phase == "CLINCH" and self.clinch_initiator is not None:
            if self.rng.random() < 0.58:
                return self.clinch_initiator

        return int(self.rng.integers(0, 2))

    def _td_success_prob(self, attacker: int) -> float:
        defender = self._other(attacker)
        edge = (
            _value(self.fighters[attacker], "wrestling_conversion")
            - _value(self.fighters[defender], "td_defense")
        ) / RATING_SCALE
        return _sigmoid(edge + TD_SUCCESS_LOGIT_OFFSET)

    def _td_attempt_hazard(self, attacker: int, phase: str) -> float:
        _, _, wrestling_pref = _style_preferences(self.fighters[attacker])
        base = (
            DISTANCE_TD_ATTEMPT_BASE
            if phase == "DISTANCE"
            else CLINCH_TD_ATTEMPT_BASE
        )
        return _prob(base * _modifier(wrestling_pref), high=0.70)

    def _distance_clinch_hazard(self, attacker: int) -> float:
        distance_pref, clinch_pref, _ = _style_preferences(self.fighters[attacker])
        return _prob(
            DISTANCE_CLINCH_BASE
            * _modifier(clinch_pref)
            * np.sqrt(_modifier(-distance_pref)),
            high=0.60,
        )

    def _clinch_separate_hazard(self, attacker: int) -> float:
        defender = self._other(attacker)
        _, clinch_pref, _ = _style_preferences(self.fighters[attacker])
        control_edge = (
            _value(self.fighters[attacker], "control_imposition")
            - _value(self.fighters[defender], "control_resistance")
        ) / RATING_SCALE
        return _prob(
            CLINCH_SEPARATE_BASE
            * _modifier(-clinch_pref)
            * exp(float(np.clip(-control_edge, -1.0, 1.0)) * 0.15),
            high=0.90,
        )

    def _ground_exit_hazard(self, controller: int) -> float:
        bottom = self._other(controller)
        escape_edge = (
            _value(self.fighters[bottom], "control_resistance")
            - _value(self.fighters[controller], "control_imposition")
        ) / RATING_SCALE
        reversal_edge = (
            _value(self.fighters[bottom], "reversal_ability")
            - _value(self.fighters[controller], "control_imposition")
        ) / RATING_SCALE
        modifier = exp(float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5)))
        return _prob(GROUND_EXIT_BASE * modifier, high=0.90)

    def _reversal_probability(self, bottom: int, controller: int) -> float:
        edge = (
            _value(self.fighters[bottom], "reversal_ability")
            - _value(self.fighters[controller], "control_imposition")
        ) / RATING_SCALE
        base_logit = _logit(REVERSAL_SHARE_OF_GROUND_EXITS)
        return _sigmoid(base_logit + 0.75 * edge)

    def _strike_attempts(self, fighter: int, phase: str) -> int:
        pressure = _value(self.fighters[fighter], PHASE_PRESSURE[phase])
        lam = STRIKE_ATTEMPTS_PER_SEGMENT_BASE[phase] * _modifier(pressure - 50.0, scale=12.0)
        return int(self.rng.poisson(max(lam, 0.0)))

    def _strike_accuracy(self, fighter: int, phase: str) -> float:
        opponent = self._other(fighter)
        precision = _value(self.fighters[fighter], PHASE_PRECISION[phase])
        defense = _value(self.fighters[opponent], PHASE_DEFENSE[phase])
        edge = (precision - defense) / RATING_SCALE
        return _sigmoid(_logit(STRIKE_ACCURACY_BASE[phase]) + edge)

    def _generate_striking(self, phase: str) -> list[str]:
        notes: list[str] = []
        for fighter in (0, 1):
            attempts = self._strike_attempts(fighter, phase)
            accuracy = self._strike_accuracy(fighter, phase)
            landed = int(self.rng.binomial(attempts, accuracy)) if attempts else 0
            self.stats[fighter].sig_att += attempts
            self.stats[fighter].sig_landed += landed
            if attempts:
                notes.append(f"{self.names[fighter]} {landed}/{attempts} sig")
        return notes

    def _maybe_submission_attempt(self, fighter: int) -> bool:
        pressure = _value(self.fighters[fighter], "submission_pressure")
        p = _prob(SUB_ATTEMPT_BASE_PER_GROUND_SEGMENT * _modifier(pressure - 50.0, scale=10.0), 0.35)
        if self.rng.random() < p:
            self.stats[fighter].sub_att += 1
            return True
        return False

    def _distance_transition(self, actor: int) -> str:
        # Entry events compete. Takedown attempt and clinch entry are separate
        # hazards; if both would occur, choose one proportional to hazards.
        p_clinch = self._distance_clinch_hazard(actor)
        p_td = self._td_attempt_hazard(actor, "DISTANCE")
        total = min(p_clinch + p_td, 0.90)
        if self.rng.random() >= total:
            return "stay distance"

        choose_td = self.rng.random() < (p_td / max(p_clinch + p_td, 1e-9))
        if choose_td:
            return self._attempt_takedown(actor, "DISTANCE")

        self.phase = "CLINCH"
        self.clinch_initiator = actor
        return f"{self.names[actor]} enters clinch"

    def _clinch_transition(self, actor: int) -> str:
        p_sep = self._clinch_separate_hazard(actor)
        p_td = self._td_attempt_hazard(actor, "CLINCH")
        total = min(p_sep + p_td, 0.92)
        if self.rng.random() >= total:
            return "clinch persists"

        choose_td = self.rng.random() < (p_td / max(p_sep + p_td, 1e-9))
        if choose_td:
            return self._attempt_takedown(actor, "CLINCH")

        self.phase = "DISTANCE"
        self.clinch_initiator = None
        return "fighters separate to distance"

    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        self.stats[attacker].td_att += 1
        success_p = self._td_success_prob(attacker)
        if self.rng.random() < success_p:
            self.stats[attacker].td_landed += 1
            self.phase = "GROUND"
            self.ground_controller = attacker
            self.clinch_initiator = None
            return (
                f"{self.names[attacker]} TD SUCCESS "
                f"({success_p:.2f}) from {source_phase.lower()}"
            )
        return (
            f"{self.names[attacker]} TD failed "
            f"({success_p:.2f}) from {source_phase.lower()}"
        )

    def _ground_transition(self) -> str:
        assert self.ground_controller is not None
        controller = self.ground_controller
        bottom = self._other(controller)

        # Controller receives this segment's coarse control credit.
        self.stats[controller].control_seconds += SEGMENT_SECONDS

        notes: list[str] = []
        if self._maybe_submission_attempt(controller):
            notes.append(f"{self.names[controller]} submission attempt")
        if self._maybe_submission_attempt(bottom):
            notes.append(f"{self.names[bottom]} submission attempt from bottom")

        p_exit = self._ground_exit_hazard(controller)
        if self.rng.random() >= p_exit:
            notes.append(f"ground control persists ({self.names[controller]} top)")
            return "; ".join(notes)

        p_rev = self._reversal_probability(bottom, controller)
        if self.rng.random() < p_rev:
            self.stats[bottom].reversals += 1
            self.ground_controller = bottom
            notes.append(f"{self.names[bottom]} REVERSAL")
            return "; ".join(notes)

        self.phase = "DISTANCE"
        self.ground_controller = None
        notes.append(f"{self.names[bottom]} escapes to distance")
        return "; ".join(notes)

    def run(self) -> FightPath:
        events: list[dict[str, Any]] = []
        for round_no in range(1, self.rounds + 1):
            # UFC rounds restart standing.
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, SEGMENTS_PER_ROUND + 1):
                phase_start = self.phase
                for s in self.stats:
                    s.phase_segments[phase_start] += 1

                actor = self._initiative_actor(phase_start)
                strike_notes = self._generate_striking(phase_start)

                if phase_start == "DISTANCE":
                    transition_note = self._distance_transition(actor)
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition(actor)
                else:
                    transition_note = self._ground_transition()

                events.append(
                    {
                        "round": round_no,
                        "segment": segment_no,
                        "clock_start": f"{5 - ((segment_no - 1) * 30) // 60}:{30 if segment_no % 2 == 0 else 0:02d}",
                        "phase_start": phase_start,
                        "phase_end": self.phase,
                        "initiative": self.names[actor],
                        "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                        "transition": transition_note,
                    }
                )

        return FightPath(events=events, stats=self.stats)


def print_path(path: FightPath, names: list[str]) -> None:
    print("\n" + "=" * 110)
    print("FSR STATIC MC V0 — PLAUSIBILITY PATH")
    print("=" * 110)
    for event in path.events:
        print(
            f"R{event['round']} S{event['segment']:02d} "
            f"[{event['phase_start']:8s} -> {event['phase_end']:8s}] "
            f"{event['striking']} | {event['transition']}"
        )

    print("\nSUMMARY")
    print("-" * 110)
    for i, name in enumerate(names):
        s = path.stats[i]
        total_segments = sum(s.phase_segments.values())
        td_pct = s.td_landed / s.td_att if s.td_att else 0.0
        print(
            f"{name}: sig {s.sig_landed}/{s.sig_att}, "
            f"TD {s.td_landed}/{s.td_att} ({td_pct:.1%}), "
            f"control {s.control_seconds}s, sub att {s.sub_att}, rev {s.reversals}"
        )
        if total_segments:
            shares = ", ".join(
                f"{phase.lower()} {s.phase_segments[phase] / total_segments:.1%}"
                for phase in ("DISTANCE", "CLINCH", "GROUND")
            )
            print(f"  path phase occupancy: {shares}")

    print("\nV0 NOTE: finishes, judging, fatigue, damage state, adversity, recovery, and urgency are disabled.")


def _default_matchup(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if len(frame) < 2:
        raise ValueError("Need at least two fighter profiles")
    return frame.iloc[0], frame.iloc[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a static FSR Monte Carlo V0 path")
    parser.add_argument("--red", help="fighter name or fighter_id")
    parser.add_argument("--blue", help="fighter name or fighter_id")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    print(f"[FSR MC V0] loading profiles from {args.fsr_path}", flush=True)
    profiles = load_profiles(args.fsr_path)
    print(f"[FSR MC V0] latest fighter profiles: {len(profiles):,}", flush=True)

    if args.red and args.blue:
        red = find_profile(profiles, args.red)
        blue = find_profile(profiles, args.blue)
    elif args.red or args.blue:
        raise SystemExit("Provide both --red and --blue, or neither.")
    else:
        red, blue = _default_matchup(profiles)
        print(
            "[FSR MC V0] no matchup supplied; using first two latest fighter profiles. "
            "Use --red/--blue for a named matchup.",
            flush=True,
        )

    names = [_display_name(red), _display_name(blue)]
    print(f"[FSR MC V0] matchup: {names[0]} vs {names[1]} | seed={args.seed}", flush=True)

    sim = StaticFSRMCV0(red, blue, rounds=args.rounds, seed=args.seed)
    path = sim.run()
    print_path(path, names)


if __name__ == "__main__":
    main()
