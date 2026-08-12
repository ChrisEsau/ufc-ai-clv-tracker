"""Rudimentary static-state FSR Monte Carlo V0.

Purpose
-------
Generate inspectable 10-second MMA fight paths from FSR-26 profiles before
adding the dynamic state engine or a detailed finish model.

V0 deliberately includes:
- distance / clinch / ground phase state
- calibrated hierarchical transition priors rescaled from 30s to 10s
- competing red/blue transition hazards rather than one chosen actor
- takedown attempt vs takedown success split
- explicit ground top/bottom ownership
- explicit clinch ownership: clinch initiator is controller until phase change
- separate clinch and ground control accounting
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

SEGMENT_SECONDS = 10
SEGMENTS_PER_ROUND = 30
DEFAULT_ROUNDS = 3
RATING_SCALE = 12.0
MODIFIER_SCALE = 6.0
CALIBRATION_INTERVAL_SECONDS = 30

# Locked 30-second research priors; converted to equivalent 10-second hazards.
# Control-persistence priors are provisional V0 values selected from the
# decision-only calibration finalists. They should be revalidated after the
# dynamic fatigue/damage state engine is enabled.
DISTANCE_CLINCH_BASE_30S = 0.04
DISTANCE_TD_ATTEMPT_BASE_30S = 0.10
CLINCH_SEPARATE_BASE_30S = 0.25
CLINCH_TD_ATTEMPT_BASE_30S = 0.24
GROUND_EXIT_BASE_30S = 0.20
TD_SUCCESS_LOGIT_OFFSET = -0.40


def _rescale_interval_prob(p: float, from_seconds: int, to_seconds: int) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    if from_seconds <= 0 or to_seconds <= 0:
        raise ValueError("interval lengths must be positive")
    return 1.0 - (1.0 - p) ** (to_seconds / from_seconds)


DISTANCE_CLINCH_BASE = _rescale_interval_prob(
    DISTANCE_CLINCH_BASE_30S, CALIBRATION_INTERVAL_SECONDS, SEGMENT_SECONDS
)
DISTANCE_TD_ATTEMPT_BASE = _rescale_interval_prob(
    DISTANCE_TD_ATTEMPT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, SEGMENT_SECONDS
)
CLINCH_SEPARATE_BASE = _rescale_interval_prob(
    CLINCH_SEPARATE_BASE_30S, CALIBRATION_INTERVAL_SECONDS, SEGMENT_SECONDS
)
CLINCH_TD_ATTEMPT_BASE = _rescale_interval_prob(
    CLINCH_TD_ATTEMPT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, SEGMENT_SECONDS
)
GROUND_EXIT_BASE = _rescale_interval_prob(
    GROUND_EXIT_BASE_30S, CALIBRATION_INTERVAL_SECONDS, SEGMENT_SECONDS
)

STRIKE_ATTEMPTS_PER_30S_BASE = {
    "DISTANCE": 5.0,
    "CLINCH": 1.2,
    "GROUND": 1.6,
}
STRIKE_ATTEMPTS_PER_SEGMENT_BASE = {
    phase: rate * (SEGMENT_SECONDS / CALIBRATION_INTERVAL_SECONDS)
    for phase, rate in STRIKE_ATTEMPTS_PER_30S_BASE.items()
}
STRIKE_ACCURACY_BASE = {
    "DISTANCE": 0.40,
    "CLINCH": 0.68,
    "GROUND": 0.70,
}
SUB_ATTEMPT_BASE_PER_30S_GROUND = 0.045
SUB_ATTEMPT_BASE_PER_GROUND_SEGMENT = _rescale_interval_prob(
    SUB_ATTEMPT_BASE_PER_30S_GROUND,
    CALIBRATION_INTERVAL_SECONDS,
    SEGMENT_SECONDS,
)
REVERSAL_SHARE_OF_GROUND_EXITS = 0.18
BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER = 0.20
BOTTOM_SUBMISSION_RATE_MULTIPLIER = 0.55

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
    raise ValueError(f"Could not resolve fighter {query!r}. Use fighter_id if needed.")


@dataclass
class FighterStats:
    sig_att: int = 0
    sig_landed: int = 0
    td_att: int = 0
    td_landed: int = 0
    control_seconds: int = 0
    clinch_control_seconds: int = 0
    ground_control_seconds: int = 0
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
        self.clinch_controller: int | None = None
        self.clinch_initiator: int | None = None

    def _other(self, i: int) -> int:
        return 1 - i

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
        # No artificial wrestling-preference clip and no arbitrary TD-attempt
        # ceiling. Keep only the mathematical probability bound required by the
        # competing-hazard sampler.
        raw_probability = base * exp(float(wrestling_pref) / MODIFIER_SCALE)
        return float(np.clip(raw_probability, 0.0, 1.0 - 1e-12))

    def _distance_clinch_hazard(self, attacker: int) -> float:
        distance_pref, clinch_pref, _ = _style_preferences(self.fighters[attacker])
        return _prob(
            DISTANCE_CLINCH_BASE
            * _modifier(clinch_pref)
            * np.sqrt(_modifier(-distance_pref)),
            high=0.60,
        )

    def _clinch_separate_hazard(self, controller: int) -> float:
        opponent = self._other(controller)
        _, clinch_pref, _ = _style_preferences(self.fighters[controller])
        control_edge = (
            _value(self.fighters[controller], "control_imposition")
            - _value(self.fighters[opponent], "control_resistance")
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
        modifier = exp(
            float(np.clip(0.60 * escape_edge + 0.40 * reversal_edge, -1.5, 1.5))
        )
        return _prob(GROUND_EXIT_BASE * modifier, high=0.90)

    def _reversal_probability(self, bottom: int, controller: int) -> float:
        edge = (
            _value(self.fighters[bottom], "reversal_ability")
            - _value(self.fighters[controller], "control_imposition")
        ) / RATING_SCALE
        return _sigmoid(_logit(REVERSAL_SHARE_OF_GROUND_EXITS) + 0.75 * edge)

    @staticmethod
    def _event_rate_from_probability(p: float) -> float:
        p = float(np.clip(p, 0.0, 1.0 - 1e-12))
        return -log(1.0 - p)

    def _sample_competing_event(
        self,
        events: list[tuple[str, float, int | None]],
    ) -> tuple[str, int | None] | None:
        """Sample at most one event from independent per-segment hazards."""
        positive: list[tuple[str, float, int | None]] = []
        for name, probability, actor in events:
            rate = self._event_rate_from_probability(probability)
            if rate > 0.0:
                positive.append((name, rate, actor))
        if not positive:
            return None
        total_rate = sum(rate for _, rate, _ in positive)
        if self.rng.random() >= 1.0 - exp(-total_rate):
            return None
        draw = self.rng.random() * total_rate
        running = 0.0
        for name, rate, actor in positive:
            running += rate
            if draw <= running:
                return name, actor
        name, _, actor = positive[-1]
        return name, actor

    def _strike_attempts(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> int:
        pressure = _value(self.fighters[fighter], PHASE_PRESSURE[phase])
        lam = (
            STRIKE_ATTEMPTS_PER_SEGMENT_BASE[phase]
            * _modifier(pressure - 50.0, scale=12.0)
            * rate_multiplier
        )
        return int(self.rng.poisson(max(lam, 0.0)))

    def _strike_accuracy(self, fighter: int, phase: str) -> float:
        opponent = self._other(fighter)
        precision = _value(self.fighters[fighter], PHASE_PRECISION[phase])
        defense = _value(self.fighters[opponent], PHASE_DEFENSE[phase])
        return _sigmoid(
            _logit(STRIKE_ACCURACY_BASE[phase])
            + (precision - defense) / RATING_SCALE
        )

    def _generate_strikes_for_fighter(
        self,
        fighter: int,
        phase: str,
        *,
        rate_multiplier: float = 1.0,
    ) -> str | None:
        attempts = self._strike_attempts(
            fighter,
            phase,
            rate_multiplier=rate_multiplier,
        )
        accuracy = self._strike_accuracy(fighter, phase)
        landed = int(self.rng.binomial(attempts, accuracy)) if attempts else 0
        self.stats[fighter].sig_att += attempts
        self.stats[fighter].sig_landed += landed
        if not attempts:
            return None
        return f"{self.names[fighter]} {landed}/{attempts} sig"

    def _generate_striking(self, phase: str) -> list[str]:
        notes: list[str] = []
        if phase == "GROUND" and self.ground_controller is not None:
            controller = self.ground_controller
            bottom = self._other(controller)
            top_note = self._generate_strikes_for_fighter(controller, "GROUND")
            if top_note:
                notes.append(f"{top_note} (top)")
            bottom_note = self._generate_strikes_for_fighter(
                bottom,
                "GROUND",
                rate_multiplier=BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER,
            )
            if bottom_note:
                notes.append(f"{bottom_note} (bottom)")
            return notes

        for fighter in (0, 1):
            note = self._generate_strikes_for_fighter(fighter, phase)
            if note:
                notes.append(note)
        return notes

    def _maybe_submission_attempt(
        self,
        fighter: int,
        *,
        rate_multiplier: float = 1.0,
    ) -> bool:
        pressure = _value(self.fighters[fighter], "submission_pressure")
        base = SUB_ATTEMPT_BASE_PER_GROUND_SEGMENT * rate_multiplier
        p = _prob(base * _modifier(pressure - 50.0, scale=10.0), 0.35)
        if self.rng.random() < p:
            self.stats[fighter].sub_att += 1
            return True
        return False

    def _distance_transition(self) -> str:
        event = self._sample_competing_event(
            [
                ("clinch", self._distance_clinch_hazard(0), 0),
                ("clinch", self._distance_clinch_hazard(1), 1),
                ("td", self._td_attempt_hazard(0, "DISTANCE"), 0),
                ("td", self._td_attempt_hazard(1, "DISTANCE"), 1),
            ]
        )
        if event is None:
            return "stay distance"

        kind, actor = event
        assert actor is not None
        if kind == "td":
            return self._attempt_takedown(actor, "DISTANCE")

        # V0 clinch ownership rule: whoever initiates the clinch owns it until
        # separation or a successful takedown changes the phase.
        self.phase = "CLINCH"
        self.clinch_initiator = actor
        self.clinch_controller = actor
        return f"{self.names[actor]} enters clinch -> CONTROLS"

    def _clinch_transition(self) -> str:
        assert self.clinch_controller is not None
        controller = self.clinch_controller

        # A segment beginning in the clinch is credited to the current controller.
        self.stats[controller].clinch_control_seconds += SEGMENT_SECONDS
        self.stats[controller].control_seconds += SEGMENT_SECONDS

        # The controller's imposition/resistance matchup determines how readily
        # the clinch breaks. Either fighter may still attempt a takedown.
        p_sep = self._clinch_separate_hazard(controller)
        event = self._sample_competing_event(
            [
                ("separate", p_sep, None),
                ("td", self._td_attempt_hazard(0, "CLINCH"), 0),
                ("td", self._td_attempt_hazard(1, "CLINCH"), 1),
            ]
        )
        if event is None:
            return f"clinch persists ({self.names[controller]} controlling)"

        kind, actor = event
        if kind == "td":
            assert actor is not None
            return self._attempt_takedown(actor, "CLINCH")

        self.phase = "DISTANCE"
        self.clinch_initiator = None
        self.clinch_controller = None
        return "fighters separate to distance"

    def _attempt_takedown(self, attacker: int, source_phase: str) -> str:
        self.stats[attacker].td_att += 1
        success_p = self._td_success_prob(attacker)
        if self.rng.random() < success_p:
            self.stats[attacker].td_landed += 1
            self.phase = "GROUND"
            self.ground_controller = attacker
            self.clinch_initiator = None
            self.clinch_controller = None
            return (
                f"{self.names[attacker]} TD SUCCESS ({success_p:.2f}) "
                f"from {source_phase.lower()} -> TOP"
            )
        # Failed TD from the clinch does not change clinch ownership.
        return (
            f"{self.names[attacker]} TD failed ({success_p:.2f}) "
            f"from {source_phase.lower()}"
        )

    def _ground_transition(self) -> str:
        assert self.ground_controller is not None
        controller = self.ground_controller
        bottom = self._other(controller)

        self.stats[controller].ground_control_seconds += SEGMENT_SECONDS
        self.stats[controller].control_seconds += SEGMENT_SECONDS

        notes: list[str] = []
        if self._maybe_submission_attempt(controller):
            notes.append(f"{self.names[controller]} submission attempt from top")
        if self._maybe_submission_attempt(
            bottom,
            rate_multiplier=BOTTOM_SUBMISSION_RATE_MULTIPLIER,
        ):
            notes.append(f"{self.names[bottom]} submission attempt from bottom")

        p_exit = self._ground_exit_hazard(controller)
        if self.rng.random() >= p_exit:
            notes.append(f"ground control persists ({self.names[controller]} top)")
            return "; ".join(notes)

        p_rev = self._reversal_probability(bottom, controller)
        if self.rng.random() < p_rev:
            self.stats[bottom].reversals += 1
            self.ground_controller = bottom
            notes.append(
                f"{self.names[bottom]} REVERSAL -> {self.names[bottom]} top"
            )
            return "; ".join(notes)

        self.phase = "DISTANCE"
        self.ground_controller = None
        notes.append(f"{self.names[bottom]} escapes to distance")
        return "; ".join(notes)

    @staticmethod
    def _clock_start(segment_no: int) -> str:
        elapsed = (segment_no - 1) * SEGMENT_SECONDS
        remaining = 5 * 60 - elapsed
        minutes, seconds = divmod(remaining, 60)
        return f"{minutes}:{seconds:02d}"

    def run(self) -> FightPath:
        events: list[dict[str, Any]] = []
        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, SEGMENTS_PER_ROUND + 1):
                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller

                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1

                strike_notes = self._generate_striking(phase_start)
                if phase_start == "DISTANCE":
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
                        "top_start": (
                            self.names[ground_controller_start]
                            if ground_controller_start is not None
                            else None
                        ),
                        "top_end": (
                            self.names[self.ground_controller]
                            if self.ground_controller is not None
                            else None
                        ),
                        "clinch_controller_start": (
                            self.names[clinch_controller_start]
                            if clinch_controller_start is not None
                            else None
                        ),
                        "clinch_controller_end": (
                            self.names[self.clinch_controller]
                            if self.clinch_controller is not None
                            else None
                        ),
                        "striking": (
                            "; ".join(strike_notes)
                            if strike_notes
                            else "no sig attempts"
                        ),
                        "transition": transition_note,
                    }
                )

        return FightPath(events=events, stats=self.stats)


def print_path(path: FightPath, names: list[str]) -> None:
    print("\n" + "=" * 110)
    print("FSR STATIC MC V0 — 10-SECOND PLAUSIBILITY PATH")
    print("=" * 110)
    for event in path.events:
        ownership = ""
        if event["top_start"]:
            ownership = f" | top={event['top_start']}"
        elif event["clinch_controller_start"]:
            ownership = f" | clinch_ctrl={event['clinch_controller_start']}"
        print(
            f"R{event['round']} {event['clock_start']} S{event['segment']:02d} "
            f"[{event['phase_start']:8s} -> {event['phase_end']:8s}] "
            f"{event['striking']} | {event['transition']}{ownership}"
        )

    print("\nSUMMARY")
    print("-" * 110)
    for i, name in enumerate(names):
        stats = path.stats[i]
        total_segments = sum(stats.phase_segments.values())
        td_pct = stats.td_landed / stats.td_att if stats.td_att else 0.0
        print(
            f"{name}: sig {stats.sig_landed}/{stats.sig_att}, "
            f"TD {stats.td_landed}/{stats.td_att} ({td_pct:.1%}), "
            f"control {stats.control_seconds}s "
            f"(clinch {stats.clinch_control_seconds}s + "
            f"ground {stats.ground_control_seconds}s), "
            f"sub att {stats.sub_att}, rev {stats.reversals}"
        )
        if total_segments:
            shares = ", ".join(
                f"{phase.lower()} {stats.phase_segments[phase] / total_segments:.1%}"
                for phase in ("DISTANCE", "CLINCH", "GROUND")
            )
            print(f"  path phase occupancy: {shares}")

    print(
        "\nV0 NOTE: finishes, judging, fatigue, damage state, adversity, "
        "recovery, and urgency are disabled."
    )


def _default_matchup(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if len(frame) < 2:
        raise ValueError("Need at least two fighter profiles")
    return frame.iloc[0], frame.iloc[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a static FSR Monte Carlo V0 path"
    )
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
            "[FSR MC V0] no matchup supplied; using first two latest fighter "
            "profiles. Use --red/--blue for a named matchup.",
            flush=True,
        )

    names = [_display_name(red), _display_name(blue)]
    print(
        f"[FSR MC V0] matchup: {names[0]} vs {names[1]} | seed={args.seed}",
        flush=True,
    )
    path = StaticFSRMCV0(red, blue, rounds=args.rounds, seed=args.seed).run()
    print_path(path, names)


if __name__ == "__main__":
    main()
