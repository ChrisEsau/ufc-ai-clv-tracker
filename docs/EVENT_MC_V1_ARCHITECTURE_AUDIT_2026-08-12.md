# EVENT MC V1 Architecture Audit and Implementation Path

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Status: Architecture audit / design basis only. No simulator implementation changes are authorized by this document.

## Operating locks

- Do not change the current simulator.
- Keep FSR-32 hooked up for now.
- Preserve the corrected `wrestling_entry` ontology:
  - `wrestling_entry` = how often a fighter shoots.
  - `wrestling_conversion` = whether the shot succeeds.
  - `td_defense` = opponent prevention.
  - `control_imposition` = what happens after control is established.
- The new simulator should be isolated under `pipeline/simulation/event_mc_v1/`.
- Use composition rather than the current inheritance architecture.
- The physics clock should be continuous/event-driven; 10-second intervals may remain only for reporting if useful.
- Do not retune KO, SUB, TD, stamina, judging, or other calibration constants while building the initial kernel.
- Stored FSR profiles remain immutable pre-fight fighter identity. Dynamic fight state is separate.

---

# 1. Exact current inheritance/call chain

For the historical diagnostic currently being used, the effective chain is:

```text
run_single_historical_age_power_diagnostic.py
    │
    │ constructs
    ▼
StaticFSRMCFullFightV1
    fsr_static_mc_ko_sub_decision_v1.py
    │
    ▼
StaticFSRMCKOSUBV1
    fsr_static_mc_ko_sub_v1.py
    │
    ▼
RecoveryAuditSim
    fsr_mature_2020plus_r3_recovery_compare_curve16_exp2_200.py
    │
    ▼
AuditSim
    fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200.py
    │
    ▼
StaticFSRMCKOTKOV33GlobalRecovery
    fsr_static_mc_ko_tko_v3_3_global_recovery.py
    │
    ▼
StaticFSRMCKOTKOV32PhaseStamina
    fsr_static_mc_ko_tko_v3_2_phase_stamina.py
    │
    ▼
StaticFSRMCKOTKOV31RollingFSR
    fsr_static_mc_ko_tko_v3_1_rolling_fsr.py
    │
    ▼
StaticFSRMCKOTKOV3Stamina
    fsr_static_mc_ko_tko_v3_stamina.py
    │
    ▼
StaticFSRMCKOTKOV2RoundRecovery
    fsr_static_mc_ko_tko_v2_round_recovery.py
    │
    ▼
StaticFSRMCKOTKOV2KDCollapse
    fsr_static_mc_ko_tko_v2_kd_collapse_sweep.py
    │
    ▼
StaticFSRMCKOTKOV2
    fsr_static_mc_ko_tko_v2.py
    │
    ▼
StaticFSRMCDamageV1Ground017
    fsr_static_mc_damage_v1_ground017.py
    │
    ▼
StaticFSRMCDamageV1
    fsr_static_mc_damage_v1.py
    │
    ▼
StaticFSRMCV0
    fsr_static_mc_v0.py
```

That is 13 simulator classes deep from the current full-fight class to V0.

More importantly, the hierarchy is not passive inheritance. Many levels override the same core methods:

```text
run()
_generate_striking()
_generate_strikes_for_fighter()
_attempt_takedown()
_ground_transition()
_clinch_transition()
_maybe_submission_attempt()
_apply_landed_strikes()
_draw_strike_damage()
_knockdown_probability()
_apply_between_round_recovery()
_spend_stamina()
```

Behavior is therefore determined by the interaction of overrides scattered across many experimental modules.

The inheritance chain itself has effectively become part of the simulator physics.

---

# 2. What is reusable

A large amount of the actual mathematics and domain logic is worth keeping.

## Action-rate formulas

These are clean concepts:

```text
distance striking pressure -> strike attempt rate
clinch striking pressure   -> clinch strike attempt rate
ground striking pressure   -> ground strike attempt rate

wrestling_entry -> TD attempt rate

clinch preference -> clinch entry rate
```

The existing V0 already has a useful conversion between interval probability and an event rate:

```python
rate = -log(1 - p)
```

That is the mathematical bridge needed for the continuous-time scheduler.

## Strike resolution

The conceptual split is good:

```text
pressure -> attempt frequency

precision
vs opponent defense
-> landing probability
```

This should move into independent `action_rates` and `strikes` components.

## Takedown resolution

The current success model already honors the corrected ontology:

```text
wrestling_entry
    -> whether I shoot

wrestling_conversion
vs
td_defense
    -> whether I finish the shot
```

That separation should be preserved exactly.

## Ground-control matchup

The current relationships among:

```text
control_imposition
control_resistance
reversal_ability
```

are reusable as starting formulas.

The `0.17` ground-exit candidate is isolated enough that it can be preserved as a configuration value initially.

## Damage reservoir

The basic model is conceptually clean:

```text
damage_durability
    -> reservoir capacity

landed strike
    -> stochastic severity

striking_power
    -> heavy-damage tail

damage
    -> reservoir depletion
```

The reservoir formula and severity distributions can become a `DamageModel` with little conceptual change.

## Knockdowns / KO

Likewise:

```text
strike shock
+ knockdown resistance
+ reservoir condition
+ recent KD
-> P(knockdown)
```

and:

```text
reservoir <= 0
-> KO/TKO
```

can be retained initially.

The later KD-collapse calibration can also exist as a replaceable KO/damage extension rather than inheritance.

## Stamina

The reservoir itself is reusable:

```text
capacity
current
fraction
```

as are:

```text
action cost
depletion-resistance modifier
fatigue-performance penalty
between-round recovery
```

What should not be carried forward is the technique of implementing stamina by overriding every action method.

## Submission finish

The current conceptual lock should be retained:

```text
submission_pressure
    -> attempt rate

submission_conversion
vs submission_resistance
    -> finish probability
```

with the current neutral candidate:

```text
P(SUB | attempt) = 0.34
```

No submission retuning during kernel work.

## Judging

The current simple 10-9 scoring model is acceptable as `JudgingModelV1`.

Preserve its formulas initially; do not treat them as permanent truth.

---

# 3. What is too coupled to reuse directly

These pieces should be redesigned rather than copied.

## The `run()` methods

There are multiple almost-identical simulator loops across the inheritance chain.

That should disappear entirely.

There should be exactly one engine loop.

## Shared mutable fields

Current plug-in-like behavior reaches directly into fields such as:

```text
self.fighters
self.base_fighters
self.stats
self.phase
self.ground_controller
self.clinch_controller
self.clinch_initiator
self.damage_state
self.stamina_state
self.finish
self.pending_stamina_costs
```

No component in the new engine should depend on another component's private attributes.

## Stats as communication

Submission code currently detects an attempt by comparing:

```text
stats.sub_att before
parent ground transition
stats.sub_att after
```

That is fragile.

An explicit `SubmissionAttempt` event should replace that side-channel entirely.

## String parsing as mechanics

Current stamina code identifies escape/reversal behavior through strings such as:

```python
if "REVERSAL" in note:
    ...
elif "escapes to distance" in note:
    ...
```

That cannot survive into the new architecture.

Typed events should drive state and stamina consequences.

## 10-second state timers

For example:

```text
recent_knockdown_segments = 3
```

is tied directly to the old timestep.

In an event engine that should become something like:

```text
recent_knockdown_until = fight_time + 30.0 sec
```

Same meaning, no dependence on simulation tick size.

## Control accounting

Currently entering a clinch or ground segment implicitly awards exactly ten seconds of control.

Continuous time should instead award:

```text
actual duration between state transitions
```

This is one of the largest gains from the redesign.

---

# 4. Recommended package structure

Lock the new implementation under:

```text
pipeline/simulation/event_mc_v1/
```

Recommended structure:

```text
pipeline/simulation/event_mc_v1/
│
├── __init__.py
├── contracts.py
├── config.py
├── events.py
├── state.py
├── scheduler.py
├── engine.py
├── registry.py
│
├── components/
│   ├── __init__.py
│   ├── profiles.py
│   ├── age.py
│   ├── action_rates.py
│   ├── phase.py
│   ├── strikes.py
│   ├── takedowns.py
│   ├── submissions.py
│   ├── stamina.py
│   ├── damage.py
│   ├── ko.py
│   ├── recovery.py
│   ├── judging.py
│   └── stats.py
│
├── diagnostics/
│   ├── single_path_trace.py
│   ├── event_rate_audit.py
│   ├── matchup_replay.py
│   └── cohort_replay.py
│
└── README.md
```

Tests should follow normal repo convention outside the package:

```text
tests/simulation/event_mc_v1/
```

This preserves isolation from the current simulator.

---

# 5. Core data separation

Enforce three distinct concepts.

## `FighterProfile`

Immutable pre-fight FSR data.

Conceptually:

```text
fighter_id
fighter_name
age
FSR-32 traits
```

No simulation component ever mutates this.

## `EffectiveFighterProfile`

Output of the age/prefight transformation.

Initially the approved physical rule:

```text
age <= 30: 0

age > 30:
    -2 rating points/year
```

on:

```text
striking_power
knockdown_resistance
damage_durability
```

clamped `[10,90]`.

The older YAML age layer should not be imported into the new engine.

## `FightState`

Only actual path state.

This distinction is important because FSR means who the fighter is entering the fight, while `FightState` means what has happened to this particular Monte Carlo path.

---

# 6. Exact proposed `FightState`

Recommended shared state:

```text
FightState

clock:
    elapsed_fight_seconds: float
    elapsed_round_seconds: float
    round_number: int
    scheduled_rounds: int
    round_duration_seconds: float

phase:
    phase: DISTANCE | CLINCH | GROUND
    clinch_controller: FighterSide | None
    ground_controller: FighterSide | None
    phase_started_at: float

fighter dynamic state:
    red: FighterDynamicState
    blue: FighterDynamicState

fight status:
    started: bool
    finished: bool
    finish: FightFinish | None

round state:
    current_round_events: list[Event]
    completed_rounds: list[RoundRecord]

event history:
    events: list[Event]

sequence:
    next_event_id: int
```

And:

```text
FighterDynamicState

stamina_capacity: float
stamina_current: float

damage_capacity: float
damage_current: float

recent_knockdown_until: float | None

knockdowns_scored: int
knockdowns_absorbed: int

sig_attempted: int
sig_landed: int

td_attempted: int
td_landed: int

submission_attempts: int
reversals: int

clinch_control_seconds: float
ground_control_seconds: float
total_control_seconds: float
```

Important restriction:

Do not put effective FSR ratings into `FightState`.

Those should be calculated through component interfaces from:

```text
immutable effective profile
+
dynamic state
```

That prevents the simulator from repeatedly mutating fighter identity.

---

# 7. Exact event taxonomy

Make primary and derived events explicit.

## Lifecycle

```text
FightStarted
RoundStarted
RoundEnded
FightFinish
DecisionRendered
```

## Striking

```text
StrikeAttempt
StrikeLanded
StrikeMissed
DamageApplied
Knockdown
```

## Wrestling

```text
TakedownAttempt
TakedownLanded
TakedownFailed
```

## Clinch

```text
ClinchEntered
ClinchSeparated
```

## Ground

```text
GroundPositionEstablished
GroundEscape
StandUp
Reversal
MatReturn
```

## Submission

```text
SubmissionAttempt
SubmissionDefended
SubmissionFinish
```

## Physiology

```text
StaminaCostApplied
RecoveryApplied
```

Do not make `StrikeLanded`, `DamageApplied`, etc. scheduler candidates.

The scheduler chooses primary actions:

```text
StrikeAttempt
TakedownAttempt
ClinchEntered
ClinchSeparated
SubmissionAttempt
GroundEscape
Reversal
...
```

The resolver then emits consequence events synchronously:

```text
StrikeAttempt
    ↓
StrikeLanded
    ↓
DamageApplied
    ↓
Knockdown
    ↓
FightFinish
```

Those consequence events occur at the same simulation timestamp.

This prevents nonsensical waiting periods between a punch landing and its damage being applied.

---

# 8. Component contracts

These are the conceptual interfaces to lock.

## Scheduler

```text
Scheduler

next_event(
    state,
    candidates,
    rng
) -> ScheduledEvent | RoundBoundary
```

Responsibilities only:

```text
sum rates
sample exponential dt
select event proportional to rate
respect round/fight boundary
```

No UFC-specific formulas.

## Event-rate provider

```text
EventRateProvider

rates(
    state,
    profiles
) -> tuple[EventRate, ...]
```

Example:

```text
EventRate(
    event_type=TAKEDOWN_ATTEMPT,
    actor=RED,
    rate_per_second=0.037
)
```

This is where `wrestling_entry` belongs.

## Phase model

```text
PhaseModel

valid_primary_events(state) -> set[EventType]

apply_phase_event(
    event,
    state
) -> StateDelta
```

Owns:

```text
DISTANCE
CLINCH
GROUND
ownership
valid transitions
```

It does not decide TD success.

## Strike resolver

```text
StrikeResolver

resolve_attempt(
    actor,
    state,
    profiles,
    rng
) -> StrikeResolution
```

Uses:

```text
precision
opponent defense
phase
```

Not damage.

## Takedown resolver

```text
TakedownResolver

resolve_attempt(
    actor,
    source_phase,
    state,
    profiles,
    rng
) -> TakedownResolution
```

Uses:

```text
wrestling_conversion
vs td_defense
```

Not `control_imposition`.

This is where the corrected ontology is protected.

## Submission resolver

Two conceptual stages:

```text
attempt rate:
submission_pressure

finish resolution:
submission_conversion
vs submission_resistance
```

Interface:

```text
SubmissionResolver

resolve_attempt(...) ->
    SubmissionDefended | SubmissionFinish
```

## Stamina model

```text
StaminaModel

cost_for(event, state, profiles) -> StaminaCost

effective_modifiers(
    fighter,
    state,
    profile
) -> DynamicModifiers
```

It may modify future rates/effectiveness.

It may not mutate action-rate formulas directly.

## Damage model

```text
DamageModel

resolve_damage(
    strike_resolution,
    attacker,
    defender,
    state,
    profiles,
    rng
) -> DamageResult
```

Owns:

```text
strike severity
reservoir depletion amount
```

## KO model

```text
KOModel

evaluate(
    damage_result,
    state,
    profiles,
    rng
) -> KOResolution
```

Initially preserves current:

```text
KD probability
KD collapse
reservoir-exhaustion finish
recent-KD interaction
```

Because it is separate, alternatives can later be tested without rewriting the engine.

## Recovery model

```text
RecoveryModel

between_rounds(
    state,
    profiles
) -> tuple[RecoveryResult, ...]
```

It returns deltas.

It does not directly reach into stamina or damage implementations.

## Judging model

```text
JudgingModel

score_round(
    round_record,
    profiles
) -> RoundScore

score_fight(
    round_scores
) -> Decision
```

Use the current simple 10-9 baseline first.

## Age model

```text
AgeModel

apply_prefight(
    profile,
    fight_date,
    age
) -> EffectiveFighterProfile
```

Runs before `FightState` exists.

## Stats/audit model

```text
StatsCollector

on_event(event, state_before, state_after)

finalize(path) -> FightStatistics
```

Statistics should be an observer of events.

They should never be used as hidden communication between components.

That is a major architectural rule.

---

# 9. Component dependency rules

These should be treated as architectural invariants.

## Rule 1

Components receive:

```text
FightState
profiles
their configuration
explicit RNG where needed
```

Never another component's internal fields.

## Rule 2

Only `engine.py` composes components.

For example:

```text
engine
 ├─ scheduler
 ├─ rates
 ├─ phase
 ├─ strikes
 ├─ takedowns
 ├─ submissions
 ├─ stamina
 ├─ damage
 ├─ ko
 ├─ recovery
 ├─ judging
 └─ stats
```

Not:

```text
submission -> reaches inside phase
damage -> reaches inside stamina
stamina -> calls private takedown method
```

## Rule 3

Components should preferably return results/deltas/events, not mutate global state arbitrarily.

## Rule 4

FSR semantics stay literal:

```text
pressure / initiation
    -> event frequency

precision / conversion
    -> success

defense / resistance
    -> prevention

control
    -> persistence after position is established

dynamic state
    -> modifies the above
```

This is a central design principle.

## Rule 5

All stochasticity uses one explicit path RNG.

No component creates its own hidden RNG.

## Rule 6

The engine owns time.

A stamina model cannot advance time.

A phase model cannot advance time.

A resolver cannot advance time.

## Rule 7

Primary events consume simulated time.

Derived consequence events do not.

---

# 10. Continuous-time engine loop

The kernel should ultimately look conceptually like:

```text
initialize profiles
apply prefight age model
initialize FightState
emit FightStarted
emit RoundStarted

while fight not finished:

    determine valid primary events

    calculate λ for every valid event

    scheduler samples:
        dt ~ Exp(sum λ)

    if next event crosses round boundary:
        advance exactly to round boundary
        accrue phase/control time
        emit RoundEnded

        if final scheduled round:
            judging
            finish
        else:
            recovery
            next round
        continue

    advance time by dt

    accrue continuous phase/control time

    select event i:
        P(i) = λ_i / Σλ

    emit primary event

    resolve consequences

    apply state deltas

    emit consequence events

    update stats

    recalculate all rates
```

This directly addresses the temporal problem that motivated the rebuild.

---

# 11. Important wrestling design decision

The event engine should not treat:

```text
TakedownAttempt
```

as a phase-transition hazard.

It should be an action rate.

That distinction matters.

At distance:

```text
Rosas TD attempt
→ fail
→ still distance
→ rates immediately recalculate
→ Rosas can shoot again 2.7 seconds later
```

Or:

```text
TD attempt
→ succeeds
→ GroundPositionEstablished
→ ground rates replace distance rates
```

This naturally permits:

```text
shot
stuffed
reshot
```

without inventing a special reshot bonus.

That is exactly the kind of sequence the current fixed 10-second segments cannot represent cleanly.

---

# 12. Phase implementation plan

## Phase 0 — Audit

This document is the current Phase 0 output.

Required before completion:

- inheritance chain documented
- reusable/coupled logic identified
- contracts approved
- state approved
- event taxonomy approved
- current baseline replay frozen

No simulator implementation should begin until the architecture has been reviewed.

## Phase 1 — Kernel only

Implement no real UFC model.

Synthetic events only.

Test:

```text
Exponential waiting-time mean
competing-event proportions
same seed -> identical path
different seeds -> divergence
chronological event ordering
round boundary exactly respected
no negative dt
no event past round end
no event after finish
zero rates handled
single-event rate handled
```

Gate:

**Mathematical scheduler tests pass before adding UFC mechanics.**

## Phase 2 — Distance only

Add:

```text
strike attempts
TD attempts
clinch entries

strike hit/miss
TD success/failure
```

No KO/SUB initially.

Validate:

```text
sig attempts/min
TD attempts/15
TD success %
clinch entries
high-pressure > low-pressure
high wrestling_entry > low wrestling_entry
wrestling_conversion affects success, not attempts
control_imposition does NOT affect TD attempt rate
```

The last two should be explicit regression tests protecting the corrected wrestling ontology.

## Phase 3 — Clinch and ground

Add:

```text
clinch striking
clinch TD
separation
ground striking
submission attempts
escape
reversal
stand-up
mat return later if justified
```

Validate:

```text
phase time
control time
TD count
repeated TD sequences
submission attempts
top/bottom behavior
```

This is where the new timing architecture should be tested for whether it fixes high-entry wrestling volume.

## Phase 4 — Dynamic state

Port existing locked models individually:

```text
stamina_v1
damage_v1
ko_v1
submission_finish_v1
recovery_v1
```

No retuning initially.

Validate each component independently.

## Phase 5 — Judging

Use the current simple 10-9 baseline.

Validate deterministic round aggregation.

## Phase 6 — Historical replay

Run both engines over the same cohort.

Track:

```text
winner Brier
winner accuracy
KO rate
SUB rate
DEC rate
finish round
fight duration
sig attempts
sig landed
TD attempts
TD landed
TD %
control
phase time
submission attempts
knockdowns
```

Also stratify by:

```text
high wrestling_entry
low wrestling_entry
high control
3-round vs 5-round
age
experience
```

## Phase 7 — Ablations

Only then:

```text
damage_v1 vs v2
ko_v1 vs v2
stamina_v1 vs v2
age_none vs age_physical_v1
judging_simple vs future version
```

One component at a time.

---

# 13. Main architectural conclusion

The rebuild is justified.

The current simulator's individual formulas are not fundamentally the problem. Many are useful and should survive.

The structural problem is that:

```text
time
phase
actions
damage
stamina
KO
submission
recovery
judging
```

have gradually become intertwined through inherited methods and shared mutable state.

The new simulator should therefore reuse calibrated ideas, not reuse the inheritance architecture.

The design basis is:

> **Continuous-time event physics + composition-based components + immutable FSR profiles + explicit dynamic FightState.**

That gives `wrestling_entry` the intended meaning: **how often the fighter chooses to shoot when the current fight state permits it.**

---

# 14. Immediate next task

Before writing the first event kernel file:

1. Freeze a small current-simulator baseline dataset for regression comparison.
2. Preserve the current simulator exactly as-is.
3. Keep FSR-32 as the active fighter-profile source.
4. Keep the corrected `wrestling_entry` ontology unchanged.
5. Then begin Phase 1 with the generic scheduler/kernel only.
6. Do not add UFC mechanics until scheduler mathematical tests pass.
7. Do not retune any KO/SUB/TD/stamina/judging calibration while building the initial kernel.

This document is intended to be reviewed critically by another chat or engineer before implementation begins.