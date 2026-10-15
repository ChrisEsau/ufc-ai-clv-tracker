"""Authoritative clock, boundary lifecycle, and state-delta application."""

import math
from dataclasses import dataclass

from .config import FightConfig
from .contracts import FightContext, RateProvider, TimeAdvanceModel
from .events import ConsequenceEvent, FightFinished, PrimaryEvent, RoundEnded, RoundStarted
from .rng import RNGManager, RNGStream
from .scheduler import ExponentialScheduler
from .sinks import EventSink, NullEventSink, StateSnapshot
from .state import FightState, Phase, StateDelta


@dataclass(frozen=True)
class SimulationResult:
    state: StateSnapshot
    sink_result: object


class SimulationEngine:
    def __init__(
        self,
        config: FightConfig,
        rate_provider: RateProvider,
        time_advance_model: TimeAdvanceModel,
        rng_manager: RNGManager,
        sink: EventSink | None = None,
        scheduler: ExponentialScheduler | None = None,
        round_recovery_model=None,
        physiology_model=None,
        finish_model=None,
        submission_finish_model=None,
        judging_model=None,
    ) -> None:
        self.config = config
        self.rate_provider = rate_provider
        self.time_advance_model = time_advance_model
        self.rng_manager = rng_manager
        self.sink = sink or NullEventSink()
        self.scheduler = scheduler or ExponentialScheduler()
        self.round_recovery_model = round_recovery_model
        self.physiology_model = physiology_model
        self.finish_model = finish_model
        self.submission_finish_model = submission_finish_model
        self.judging_model = judging_model

    def _context(self, state: FightState) -> FightContext:
        return FightContext(
            self.config,
            state.fight_time_seconds,
            self.config.round_number_at(state.fight_time_seconds),
        )

    @staticmethod
    def _apply_delta(state: FightState, delta: StateDelta) -> None:
        if delta.phase is not None:
            state.phase = delta.phase
        if delta.set_ground_controller:
            state.ground_controller = delta.ground_controller
        if delta.set_clinch_controller:
            state.clinch_controller = delta.clinch_controller
        if delta.finished is not None:
            state.finished = delta.finished
        if delta.finish_reason is not None:
            state.finish_reason = delta.finish_reason
        if delta.winner is not None:
            state.winner = delta.winner
        if delta.finish_method is not None:
            state.finish_method = delta.finish_method
        if delta.red_stamina is not None:
            state.red_stamina = delta.red_stamina
        if delta.blue_stamina is not None:
            state.blue_stamina = delta.blue_stamina
        for name in ("red_cumulative_trauma", "blue_cumulative_trauma", "red_acute_vulnerability", "blue_acute_vulnerability"):
            value = getattr(delta, name)
            if value is not None:
                setattr(state, name, value)
        if delta.action_availability is not None:
            state.action_availability = delta.action_availability

    def _notify_event(self, event, state: FightState, before: StateSnapshot) -> None:
        after = StateSnapshot.from_state(state)
        self.sink.on_event(event, before, after)
        if self.judging_model is not None:
            self.judging_model.on_event(event, before, after)

    def _advance(self, state: FightState, dt_seconds: float) -> None:
        if dt_seconds < 0 or not math.isfinite(dt_seconds):
            raise ValueError("actual elapsed time must be finite and non-negative")
        before = StateSnapshot.from_state(state)
        delta = self.time_advance_model.advance(state, self._context(state), dt_seconds)
        self._apply_delta(state, delta)
        state.fight_time_seconds += dt_seconds
        self.sink.on_time_advance(dt_seconds, before, StateSnapshot.from_state(state))
        if self.judging_model is not None:
            self.judging_model.on_time_advance(dt_seconds, before, StateSnapshot.from_state(state))

    def run(
        self,
        state: FightState | None = None,
        stop_at_seconds: float | None = None,
    ) -> SimulationResult:
        state = state or FightState()

        run_horizon = self.config.fight_duration_seconds

        if stop_at_seconds is not None:
            stop_at_seconds = float(stop_at_seconds)
            if not math.isfinite(stop_at_seconds):
                raise ValueError("stop_at_seconds must be finite")
            if stop_at_seconds < state.fight_time_seconds:
                raise ValueError(
                    "stop_at_seconds cannot precede the initial fight clock"
                )
            run_horizon = min(
                stop_at_seconds,
                self.config.fight_duration_seconds,
            )

        if state.fight_time_seconds < 0 or state.fight_time_seconds > self.config.fight_duration_seconds:
            raise ValueError("initial fight clock is outside the configured horizon")
        if state.fight_time_seconds == 0 and not state.finished:
            before = StateSnapshot.from_state(state)
            self._apply_delta(
                state,
                StateDelta(
                    phase=Phase.STANDING,
                    set_ground_controller=True,
                    set_clinch_controller=True,
                ),
            )
            self._notify_event(RoundStarted(0.0, 1), state, before)

        scheduler_rng = self.rng_manager.stream(RNGStream.SCHEDULER)
        while not state.finished and state.fight_time_seconds < run_horizon:
            context = self._context(state)
            candidates = self.rate_provider.candidates(state, context)
            sampled_dt, candidate = self.scheduler.sample(candidates, scheduler_rng)

            scheduled_boundary = self.config.next_boundary_after(
                state.fight_time_seconds
            )
            boundary = min(
                scheduled_boundary,
                run_horizon,
            )
            to_boundary = max(0.0, boundary - state.fight_time_seconds)
            boundary_first = candidate is None or sampled_dt >= to_boundary
            actual_dt = to_boundary if boundary_first else sampled_dt
            self._advance(state, actual_dt)

            if boundary_first:
                # A diagnostic horizon may fall inside a scheduled round.
                # Stop there without pretending a round ended.
                if boundary < scheduled_boundary:
                    state.finished = True
                    state.finish_reason = "diagnostic_horizon"
                    before = StateSnapshot.from_state(state)
                    self._notify_event(
                        FightFinished(
                            boundary,
                            state.finish_reason,
                        ),
                        state,
                        before,
                    )
                    break

                round_number = self.config.round_number_at(max(0.0, boundary - 1e-12))
                snapshot = StateSnapshot.from_state(state)
                self._notify_event(RoundEnded(boundary, round_number), state, snapshot)
                if self.judging_model is not None:
                    card = self.judging_model.score_round(
                        round_number, self.rng_manager.stream(RNGStream.JUDGING)
                    )
                    self._notify_event(
                        ConsequenceEvent(boundary, "RoundScore", card), state, snapshot
                    )
                if (
                    boundary >= run_horizon
                    and run_horizon < self.config.fight_duration_seconds
                ):
                    state.finished = True
                    state.finish_reason = "diagnostic_horizon"
                    before = StateSnapshot.from_state(state)
                    self._notify_event(
                        FightFinished(
                            boundary,
                            state.finish_reason,
                        ),
                        state,
                        before,
                    )
                    break

                if boundary >= self.config.fight_duration_seconds:
                    if self.judging_model is not None:
                        self._apply_delta(state, self.judging_model.decision_delta())
                    else:
                        state.finished = True
                        state.finish_reason = "scheduled_horizon"
                    before = StateSnapshot.from_state(state)
                    self._notify_event(
                        FightFinished(boundary, state.finish_reason), state, before
                    )
                    break
                before = StateSnapshot.from_state(state)
                recovery_delta = (
                    self.round_recovery_model.recovery_delta(state)
                    if self.round_recovery_model is not None
                    else StateDelta()
                )
                self._apply_delta(
                    state,
                    StateDelta(
                        phase=Phase.STANDING,
                        set_ground_controller=True,
                        set_clinch_controller=True,
                        red_stamina=recovery_delta.red_stamina,
                        blue_stamina=recovery_delta.blue_stamina,
                    ),
                )
                self._notify_event(
                    RoundStarted(boundary, round_number + 1), state, before
                )
                continue

            before = StateSnapshot.from_state(state)
            resolution_rng = self.rng_manager.stream(candidate.rng_stream)
            resolution = candidate.resolve(state, self._context(state), resolution_rng)
            self._apply_delta(state, resolution.delta)
            primary = PrimaryEvent(
                state.fight_time_seconds, candidate.candidate_id, resolution.payload
            )
            self._notify_event(primary, state, before)
            if self.submission_finish_model is not None:
                submission_delta, submission_event = self.submission_finish_model.resolve(
                    state,
                    resolution.payload,
                    state.fight_time_seconds,
                    self.rng_manager.stream(RNGStream.SUBMISSION),
                    pre_action_state=before,
                )
                submission_before = StateSnapshot.from_state(state)
                self._apply_delta(state, submission_delta)
                if submission_event is not None:
                    self._notify_event(submission_event, state, submission_before)
            physiology_events = ()
            if self.physiology_model is not None:
                physiology_delta, physiology_events = self.physiology_model.resolve(
                    state, resolution.payload, state.fight_time_seconds,
                    self.rng_manager.stream(RNGStream.DAMAGE),
                    self.rng_manager.stream(RNGStream.KNOCKDOWN_FINISH),
                )
                physiology_before = StateSnapshot.from_state(state)
                self._apply_delta(state, physiology_delta)
                for event in physiology_events:
                    self._notify_event(event, state, physiology_before)
                    if self.finish_model is not None:
                        finish_delta, finish_event = self.finish_model.resolve(
                            state, event.payload, state.fight_time_seconds,
                            self.rng_manager.stream(RNGStream.KNOCKDOWN_FINISH),
                        )
                        finish_before = StateSnapshot.from_state(state)
                        self._apply_delta(state, finish_delta)
                        self._notify_event(finish_event, state, finish_before)
            for consequence in resolution.consequence_events:
                if consequence.timestamp_seconds != state.fight_time_seconds:
                    raise ValueError("consequence events must use the current timestamp")
                snapshot = StateSnapshot.from_state(state)
                self._notify_event(consequence, state, snapshot)
            if state.finished:
                snapshot = StateSnapshot.from_state(state)
                self._notify_event(
                    FightFinished(
                        state.fight_time_seconds,
                        state.finish_reason or "explicit_finish",
                    ),
                    state,
                    snapshot,
                )
        return SimulationResult(StateSnapshot.from_state(state), self.sink.finalize())
