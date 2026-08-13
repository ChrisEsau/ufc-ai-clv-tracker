# EVENT MC V1 Architecture Audit and Implementation Path

Date: 2026-08-12

Architecture Revision: **v0.2**

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Status: Architecture audit / design basis only. No simulator implementation changes are authorized by this document.

## Revision History

| Revision | Date | Summary | Status |
|---|---|---|---|
| v0.1 | 2026-08-12 | Initial architecture audit and proposed implementation path | Superseded |
| v0.2 | 2026-08-12 | Critical architecture review incorporated: corrected inheritance wording, explicit current wrestling-consumer semantic blend, per-second rate units, continuous state advancement, reduced FightState, authoritative clock, generic scheduler, dynamic modifier pipeline, damage/KD/KO ownership, judging RNG, RNG stream strategy, cooldown/duration extension point, round-reset invariant, conditional ground-exit mapping, optional MatReturn, trace modes, Phase 2A/2B split, and concrete baseline freeze | Current |

**Revision-control rule:** every future architecture change must increment `Architecture Revision` and append a short revision-history entry describing what changed and why. Architecture changes must not be silently folded into the document.

## Operating Locks

- Do not change the current simulator during architecture work.
- Keep FSR-32 hooked up for now.
- Preserve the corrected `wrestling_entry` rating ontology:
  - `wrestling_entry` = intrinsic takedown initiation frequency.
  - `wrestling_conversion` = probability/ability to complete the shot.
  - `td_defense` = opponent prevention of the shot.
  - `control_imposition` = what happens after control is established.
- The new simulator remains isolated under `pipeline/simulation/event_mc_v1/`.
- Use composition rather than the current inheritance architecture.
- The physics clock is continuous/event-driven. Fixed 10-second intervals may remain only as optional reporting/aggregation bins.
- Do not retune KO, SUB, TD, stamina, judging, age, or other calibration constants while building the initial kernel and parity layers.
- Stored FSR profiles remain immutable pre-fight fighter identity. Dynamic fight state is separate.
- Do not silently combine a temporal-architecture change with a semantic/calibration change. Migration stages must make the source of observed differences attributable.

---

# 1. Current Inheritance / Call Chain

For the historical diagnostic currently being used, the effective simulator chain is:

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

There are **14 simulator classes connected by 13 inheritance edges / parent hops** from `StaticFSRMCFullFightV1` through `StaticFSRMCV0`.

The hierarchy is not passive inheritance. Many levels override the same core methods:

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

Behavior is therefore determined by interacting overrides distributed across many experimental modules. The inheritance chain itself has effectively become part of the simulator physics because fight timing, striking, takedowns, ground behavior, stamina, recovery, damage, KO, submissions, and judging are all partially defined at different inheritance levels.

The new simulator should reuse useful formulas and calibrated ideas, but **not reuse this inheritance architecture**.

---

# 2. Wrestling Ontology: Rating Corrected, Current Consumer Still Blended

The corrected FSR ontology is locked:

```text
wrestling_entry
    = intrinsic takedown initiation frequency

wrestling_conversion
    = probability/ability to complete the shot

td_defense
    = opponent prevention of the shot

control_imposition
    = what happens after control is established
```

However, this must be distinguished from how the **current simulator consumer** behaves.

Current V0 still computes a blended wrestling preference:

```text
wrestling_pref =
      0.75 * wrestling_entry
    + 0.25 * control_imposition
    - 0.50 * distance_striking_pressure
    - 0.50 * clinch_striking_pressure
```

and uses `wrestling_pref` in takedown-attempt hazard.

Therefore two statements are simultaneously true:

1. **The rating ontology is corrected.** `wrestling_entry` itself is intended to mean intrinsic initiation frequency.
2. **The existing simulator consumer is still semantically blended.** Its TD-attempt generation mixes `wrestling_entry`, `control_imposition`, and striking-pressure terms.

This distinction must not be hidden during migration.

The future event-driven model should intentionally separate intrinsic tendency from contextual/tactical modification:

```text
Intrinsic TD propensity:
    wrestling_entry
        ↓
base TD attempt rate

Context / tactical modifiers:
    phase
    stamina
    fight situation
    possibly opponent behavior
        ↓
context multiplier

Final TD attempt rate:
    base rate × context multiplier
```

`control_imposition` must **not** define intrinsic TD initiation in the new ontology-correct action-rate model.

To avoid conflating clock changes with semantic changes, Phase 2 is split into:

- **Phase 2A: temporal/mechanical parity** — reproduce the current blended consumer as faithfully as practical in continuous time.
- **Phase 2B: ontology-correct action-rate model** — deliberately replace the blended intrinsic TD consumer so `wrestling_entry` controls base initiation.

This semantic correction is intentional and must be documented as such when implemented.

---

# 3. What Is Reusable

A large amount of the current mathematics and domain logic is worth preserving as first-pass components.

## 3.1 Strike-rate and accuracy concepts

The conceptual split is useful:

```text
pressure -> attempt frequency

precision
vs opponent defense
-> landing probability
```

This should move into independent action-rate and strike-resolution components.

## 3.2 Takedown success

The current success matchup is conceptually aligned with the corrected ontology:

```text
wrestling_conversion
vs
td_defense
    -> whether the shot succeeds
```

That success/conversion relationship should be preserved initially.

## 3.3 Ground-control matchup

The relationships among:

```text
control_imposition
control_resistance
reversal_ability
```

are reusable as starting formulas.

The selected `GROUND_EXIT_BASE_30S = 0.17` shadow candidate may be preserved for parity, subject to exact conversion into per-second event rates.

## 3.4 Damage reservoir

The basic model remains conceptually useful:

```text
damage_durability
    -> reservoir capacity

landed strike
    -> stochastic primary severity

striking_power
    -> heavy-damage tail

primary damage
    -> reservoir depletion
```

## 3.5 Knockdown / finish concepts

The current KD relationship is also reusable as a first candidate:

```text
strike shock
+ knockdown resistance
+ reservoir condition
+ recent KD
-> P(knockdown)
```

Reservoir exhaustion can remain the initial KO/TKO stopping rule while the component boundaries are cleaned up.

KD-collapse trauma is preserved as a distinct configurable consequence rather than being implicitly embedded inside an all-purpose KO component.

## 3.6 Stamina

Reusable concepts include:

```text
capacity
current
fraction
action cost
depletion-resistance modifier
fatigue-performance penalty
between-round recovery
continuous positional cost per second
```

The current formulas can be carried forward as calibration candidates. The current technique of implementing stamina by overriding many unrelated action methods should not be carried forward.

## 3.7 Submission

Preserve the current conceptual split:

```text
submission_pressure
    -> attempt rate

submission_conversion
vs submission_resistance
    -> finish probability
```

Preserve the current neutral candidate during migration:

```text
P(SUB finish | attempt) = 0.34
```

No submission retuning during kernel work.

## 3.8 Judging

The current simple 10-9 model remains acceptable as `JudgingModelV1` for parity work, including seeded RNG tie-breaking where the current scorer requires it.

Use the term **reproducible judging**, not deterministic judging, while stochastic exact-tie resolution remains part of the contract.

---

# 4. What Is Too Coupled To Reuse Directly

## 4.1 Multiple `run()` loops

There are several near-duplicate simulation loops across the inheritance chain. The new simulator must have exactly **one engine loop**.

## 4.2 Shared mutable implementation fields

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

New components must communicate through explicit public contracts, typed state, typed events, modifiers, and returned deltas/results rather than another component's private attributes.

## 4.3 Statistics as hidden communication

Current submission logic detects attempts by observing before/after changes in `stats.sub_att`. That side-channel must be replaced with a typed `SubmissionAttempt` event.

## 4.4 String parsing as mechanics

Current stamina logic can infer reversals/escapes by parsing note strings. Typed events must replace all string-driven mechanics.

## 4.5 Fixed-step timers

State such as:

```text
recent_knockdown_segments = 3
```

is timestep-dependent. In the event engine, use time-based state such as:

```text
recent_knockdown_until = fight_time_seconds + 30.0
```

## 4.6 Fixed-step control accounting

The current engine can implicitly award 10 seconds of control because a segment began in a position. Continuous time must instead award the actual elapsed duration.

If a fighter controls for 6.27 seconds before the next event, exactly 6.27 seconds of control and sustained positional stamina cost should accrue.

---

# 5. Recommended Package Structure

Keep the implementation isolated under:

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
├── rng.py
├── modifiers.py
├── sinks.py
├── registry.py
│
├── components/
│   ├── __init__.py
│   ├── profiles.py
│   ├── age.py
│   ├── action_rates.py
│   ├── phase.py
│   ├── continuous_state.py
│   ├── strikes.py
│   ├── takedowns.py
│   ├── submissions.py
│   ├── stamina.py
│   ├── damage.py
│   ├── knockdowns.py
│   ├── knockdown_consequences.py
│   ├── finishes.py
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

The exact physical file count may be simpler in V1, but the logical contracts and ownership boundaries should remain distinct.

Tests should follow normal repository convention outside the package:

```text
tests/simulation/event_mc_v1/
```

---

# 6. Core Data Separation

Enforce separate concepts for identity, prefight transformation, dynamic physics, and accumulated observations.

## 6.1 `FighterProfile`

Immutable stored pre-fight FSR data.

Conceptually:

```text
fighter_id
fighter_name
FSR-32 traits
```

No simulation component mutates this object.

## 6.2 `EffectiveFighterProfile`

Immutable fight-night profile after prefight transformations such as age.

For the current shadow candidate, preserve the approved physical age rule without retuning:

```text
age <= 30:
    no adjustment

age > 30:
    -2 rating points/year after 30
```

applied to the currently approved physical traits:

```text
striking_power
knockdown_resistance
damage_durability
```

with existing rating bounds preserved.

The old YAML age layer is not imported into the new engine unless a later architecture revision explicitly changes this decision.

## 6.3 `FightState`

`FightState` represents only physical/path state required to determine future simulation behavior.

It should not double as a statistics object.

## 6.4 `FightLedger` / `StatsAccumulator`

A separate ledger or accumulator observes typed events and produces fight statistics, round records, judging inputs, and diagnostics.

Examples of ledger-owned data:

```text
sig_attempted
sig_landed
td_attempted
td_landed
submission_attempts
reversals
knockdowns_scored
knockdowns_absorbed
clinch_control_seconds
ground_control_seconds
total_control_seconds
phase occupancy
```

These values should not live in `FighterDynamicState` unless a future physical model explicitly requires a corresponding state variable.

If future strategy/urgency models need score or cumulative context, expose that intentionally through a public `FightContext` derived from the ledger. Do not let components inspect private stats fields as hidden communication.

**Architecture invariant:** stats are observers of events, never hidden communication between components.

---

# 7. Authoritative Fight State and Clock

Use one authoritative simulation clock:

```text
FightState

fight_time_seconds: float

phase:
    phase: DISTANCE | CLINCH | GROUND
    clinch_controller: FighterSide | None
    ground_controller: FighterSide | None
    phase_started_at: float

fighters:
    red: FighterDynamicState
    blue: FighterDynamicState

fight status:
    started: bool
    finished: bool
    finish: FightFinish | None

transient scheduling state, if needed:
    busy_until / cooldown_until information
```

`FighterDynamicState` should contain physical/tactical state needed for future physics, for example:

```text
stamina_capacity: float
stamina_current: float

damage_capacity: float
damage_current: float

recent_knockdown_until: float | None

future transient state only when it affects future rates/resolution
```

Do **not** maintain independently mutable values for:

```text
round_number
round_elapsed
round_remaining
```

Derive them from `fight_time_seconds` and immutable fight configuration.

Fight schedule configuration owns values such as:

```text
scheduled_rounds
round_duration_seconds
between_round_break_seconds if modeled
hard fight boundaries
```

This avoids clock drift and creates one source of truth for temporal ordering.

Only the engine advances `fight_time_seconds`.

---

# 8. Event Taxonomy

Primary events are scheduler candidates. Consequence events occur synchronously at the same simulation timestamp and do not independently consume simulation time.

## 8.1 Lifecycle

```text
FightStarted
RoundStarted
RoundEnded
FightFinish
DecisionRendered
```

## 8.2 Striking

```text
StrikeAttempt
StrikeLanded
StrikeMissed
DamageApplied
Knockdown
KnockdownConsequenceApplied
```

## 8.3 Wrestling

```text
TakedownAttempt
TakedownLanded
TakedownFailed
```

## 8.4 Clinch

```text
ClinchEntered
ClinchSeparated
```

## 8.5 Ground

Initial V1 events:

```text
GroundPositionEstablished
GroundEscape
StandUp
Reversal
```

`MatReturn` is **future / optional** and should not be required in initial V1. It implies a richer standing-control / cage-wrestling ontology than the current three-phase state can represent cleanly.

## 8.6 Submission

```text
SubmissionAttempt
SubmissionDefended
SubmissionFinish
```

## 8.7 Physiology / state

```text
StaminaCostApplied
ContinuousStaminaCostApplied
RecoveryApplied
```

Example synchronous consequence chain:

```text
StrikeAttempt
    ↓
StrikeLanded
    ↓
DamageApplied
    ↓
Knockdown
    ↓
KnockdownConsequenceApplied
    ↓
FightFinish
```

All consequences above share the same simulation timestamp unless a later architecture explicitly introduces event duration.

---

# 9. Continuous-Time Rate Units and Migration Rules

## 9.1 Hard unit rule

**ALL scheduler candidate event rates are expressed in `events per second`.**

No component may hand the scheduler a 10-second or 30-second probability disguised as a rate.

## 9.2 Probability-to-rate conversion

For a probability `p_interval` defined over an interval of `interval_seconds`:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

The current expression:

```text
rate = -ln(1 - p)
```

is the integrated hazard over the interval represented by `p`; it is not automatically an events-per-second rate unless the interval is one second.

## 9.3 Initial parity migration

For initial parity work:

1. Preserve the current final interval probabilities after all current trait/context transformations.
2. Convert those final probabilities exactly into per-second rates using the interval they represent.
3. Feed only per-second rates to the scheduler.

This isolates the effect of removing the fixed timestep.

## 9.4 Later native rate-space model

After parity is established, a cleaner model may be expressed directly in hazard space:

```text
base hazard per second
× trait multiplier
× context multiplier
= final hazard per second
```

Do not silently change both temporal architecture and rate calibration/semantics at the same time.

---

# 10. Scheduler Contract

The scheduler is generic and UFC-agnostic.

Conceptual interface:

```text
Scheduler.sample(
    candidates,
    rng
) -> (dt, event | None)
```

Each candidate includes a non-negative rate in events per second.

Scheduler responsibilities:

```text
sum rates
sample exponential waiting time
select event proportional to rate
return dt and selected event
```

The scheduler does **not** know about rounds, judges, UFC phases, fight endings, or hard boundaries.

For total rate `Λ`:

```text
dt ~ Exponential(Λ)
P(event_i selected) = lambda_i / Λ
```

For zero total rate:

```text
dt = infinity
event = None
```

The engine owns current time and the next hard boundary. The engine compares sampled `dt` with that boundary.

**Invariant:** only the engine advances time.

---

# 11. Continuous State Advancement

Discrete events are not the only source of state change. Some quantities accumulate continuously while time passes in a phase.

Add a dedicated contract such as:

```text
ContinuousStateModel / TimeAdvanceModel

advance(
    state,
    dt,
    profiles,
    modifiers
) -> StateDelta + optional emitted events
```

Initial responsibilities may include:

```text
clinch control time accrual
ground control time accrual
clinch-controller stamina cost per second
clinch-resistance stamina cost per second
ground-controller stamina cost per second
bottom-resistance stamina cost per second
future continuous physiological effects
```

The engine order is:

```text
sample dt
    ↓
compare with next hard boundary
    ↓
advance continuous state for actual elapsed dt
    ↓
advance authoritative clock by dt
    ↓
resolve scheduled event if boundary was not reached first
```

Example:

```text
Ground control persists for 6.27 seconds
    -> 6.27 seconds ground control credited
    -> 6.27 seconds top-control stamina cost
    -> 6.27 seconds bottom-resistance stamina cost
```

No 10-second rounding is permitted in the physics engine.

---

# 12. Dynamic Modifier Pipeline

Action-rate and resolution components must not call `StaminaModel` or other modifier-producing components internally.

Introduce a public modifier pipeline:

```text
immutable EffectiveFighterProfile
+
FightState
+
ModifierProviders
        ↓
DynamicModifiers snapshot
        ↓
ActionRateModel
StrikeResolver
TakedownResolver
DamageModel
KnockdownModel
SubmissionResolver
etc.
```

Conceptual structures:

```text
ModifierProvider
    provide(profile, state, context) -> FighterModifiers

DynamicModifierPipeline
    snapshot(profiles, state, context) -> DynamicModifiers
```

Example snapshot:

```text
DynamicModifiers
    red:
        output_multiplier
        power_multiplier
        defense_multiplier
        wrestling_multiplier
        accuracy_multiplier
    blue:
        ...
```

Not every field must be active in V1. Initially the currently locked rolling-fatigue candidate may only change effective striking power. The architecture should nevertheless permit later `stamina_v2` or adversity models to modify output, wrestling, defense, or other dimensions without rewriting `ActionRateModel`.

Components consume the immutable snapshot for the current event calculation/resolution cycle.

This is required for true plug-and-play composition.

---

# 13. Component Contracts and Ownership

## 13.1 Action rate model

```text
ActionRateModel

rates(
    state,
    profiles,
    modifiers,
    context
) -> tuple[EventRate, ...]
```

Every returned `EventRate.rate_per_second` is in events/second.

In Phase 2A, the model may reproduce current blended TD-attempt behavior for parity.

In Phase 2B, the TD initiation model becomes ontology-correct:

```text
wrestling_entry -> intrinsic base TD rate
context/modifiers -> multiplicative contextual adjustment
```

## 13.2 Phase model

```text
PhaseModel

valid_primary_events(state) -> set[EventType]
apply_phase_event(event, state) -> StateDelta
```

Owns phase and positional-ownership transitions only.

It does not determine TD success, strike damage, submission finish probability, or stamina formulas.

## 13.3 Strike resolver

```text
StrikeResolver

resolve_attempt(
    actor,
    state,
    profiles,
    modifiers,
    rng
) -> StrikeResolution
```

Uses strike precision, opponent defense, phase, and applicable modifiers.

It returns hit/miss and strike metadata; it does not own reservoir mutation.

## 13.4 Takedown resolver

```text
TakedownResolver

resolve_attempt(
    actor,
    source_phase,
    state,
    profiles,
    modifiers,
    rng
) -> TakedownResolution
```

Success uses:

```text
wrestling_conversion
vs td_defense
```

`control_imposition` does not decide whether the shot is initiated or whether the shot is technically completed unless a later architecture revision explicitly documents a new semantic role.

## 13.5 Submission resolver

Attempt frequency remains an action-rate concern derived from `submission_pressure`.

Finish resolution uses:

```text
submission_conversion
vs submission_resistance
```

Conceptual interface:

```text
SubmissionResolver.resolve_attempt(...) -> SubmissionDefended | SubmissionFinish
```

## 13.6 Stamina model

```text
StaminaModel

cost_for(event, state, profiles) -> StaminaCost
continuous_cost_rates(state, profiles) -> StaminaRates
between_round_recovery(...) -> StaminaRecovery
```

It owns stamina state mutation through explicit returned deltas/events. It does not rewrite action-rate formulas internally.

Its performance effects are exposed through `ModifierProvider` / `DynamicModifierPipeline`.

## 13.7 Damage / KD / KO ownership

Use the following causal ownership model:

```text
StrikeResolver
    -> landed / missed

DamageModel
    -> primary strike damage
    -> primary reservoir delta

KnockdownModel
    -> knockdown yes / no

KnockdownConsequenceModel
    -> configured collapse trauma / additional reservoir delta

FinishModel / KOModel
    -> determine stoppage from resulting state
```

Ownership rules:

- `DamageModel` owns primary strike-induced reservoir depletion.
- `KnockdownModel` owns KD probability and KD occurrence, not primary damage.
- `KnockdownConsequenceModel` owns any additional KD-collapse reservoir depletion when that candidate is enabled.
- `FinishModel/KOModel` evaluates the post-consequence state and decides whether a KO/TKO stoppage occurs.
- A KO model must not silently rewrite damage mechanics.
- Each state mutation type has one explicit owner.

The initial implementation may combine some of these contracts into fewer files, but their logical ownership must remain explicit.

## 13.8 Recovery model

```text
RecoveryModel

between_rounds(
    state,
    profiles
) -> RecoveryDelta(s)
```

Recovery returns explicit deltas rather than reaching into another component's private state.

## 13.9 Judging model

Preserve current seeded tie-breaking for parity:

```text
JudgingModel

score_round(
    round_record,
    profiles,
    rng
) -> RoundScore

score_fight(
    round_scores,
    rng
) -> Decision
```

Use **reproducible judging** terminology while RNG tie-breaking remains present.

## 13.10 Stats / ledger

```text
StatsAccumulator / FightLedger

on_event(event, state_before, state_after)
finalize() -> FightStatistics
```

The ledger is an event observer and public context source. It does not mutate physics state.

---

# 14. Round and Hard-Boundary Rules

The engine owns hard boundaries.

At each scheduling cycle:

```text
sample (dt, event)
calculate time_to_next_hard_boundary

if dt >= boundary_delta or event is None:
    advance continuous state to boundary
    advance clock exactly to boundary
    process boundary
else:
    advance continuous state by dt
    advance clock by dt
    resolve event
```

## 14.1 Locked round-start invariant

Every MMA round starts standing at distance:

```text
RoundStarted
    -> phase = DISTANCE
    -> ground_controller = None
    -> clinch_controller = None
    -> all previous positional ownership cleared
```

This preserves current round-start behavior and is locked unless a future architecture revision explicitly changes the ruleset.

## 14.2 Round-end behavior

At a round boundary:

```text
advance continuous state exactly to boundary
emit RoundEnded
freeze round ledger/record
apply between-round recovery if another round remains
emit next RoundStarted
reset position to DISTANCE
```

The exact wall-clock treatment of the one-minute corner break does not need to be simulated as active fight time in V1 unless future mechanics require it; recovery can remain a boundary operation.

---

# 15. Ground Exit Conditional Semantics

The current ground logic is conditional:

```text
sample whether ground exit occurs
    ↓
if exit occurs:
    sample reversal conditional on exit
        ↓
        reversal
        OR
        escape to distance
```

If the event-driven engine exposes `Reversal` and `GroundEscape` as separate competing primary events, preserve the total exit hazard exactly during parity work.

Let:

```text
lambda_ground_exit = converted current total ground-exit hazard
p_reversal_given_exit = current reversal conditional probability
```

Then:

```text
lambda_reversal =
    lambda_ground_exit * p_reversal_given_exit

lambda_escape =
    lambda_ground_exit * (1 - p_reversal_given_exit)
```

Therefore:

```text
lambda_reversal + lambda_escape
    = lambda_ground_exit
```

This is an explicit Phase 3 parity rule.

Do not assign each outcome the full ground-exit hazard; that would increase total exit frequency.

---

# 16. RNG Architecture

## 16.1 Root seed invariant

Every Monte Carlo path has one explicit root path seed.

Components may not instantiate hidden independent RNGs.

## 16.2 Named deterministic streams

Before or during Phase 1, resolve and implement a centrally owned RNG manager that can reproducibly spawn named streams from the root seed, for example:

```text
scheduler_rng
strike_resolution_rng
takedown_rng
submission_rng
damage_rng
knockdown_rng
ko_rng
judging_rng
```

Potential additional streams can be added deliberately when components require them.

Reason: component A/B tests should not cause unrelated downstream random draws to shift merely because one implementation consumes a different number of random values.

For example, if `damage_v2` draws two extra variates compared with `damage_v1`, that should not automatically alter every later takedown, submission, or judging draw.

The RNG manager remains centrally owned by the engine/path context. Named streams are deterministic children of the root path seed.

This is a design decision that must be finalized in Phase 1, not deferred until late ablation work.

---

# 17. Future Action Duration / Cooldown Extension Point

The first kernel uses continuous-time exponential waiting for primary events.

Exponential hazards are memoryless. Without additional constraints, physically implausible sequences could occur, for example:

```text
TD attempt
0.08 sec later TD attempt
0.15 sec later TD attempt
```

Do not add arbitrary cooldown rules in Phase 1 solely to make paths look better. However, the state/contracts must leave room for future semi-Markov or duration-aware behavior.

Possible future state/contracts include:

```text
cooldown_until
busy_until
minimum_action_interval
event duration
phase dwell duration
semi-Markov dwell distributions
```

Candidate generation should be able to suppress or modify rates when a fighter/action is temporarily unavailable.

The architecture must not assume all events are forever memoryless, even though the initial scheduler kernel is exponential.

---

# 18. Event Trace and Performance Modes

Full chronological traces are essential for diagnostics but too expensive to require for large cohort simulations.

Design event sinks / trace modes from the start.

Possible interface:

```text
EventSink.on_event(event, state_before, state_after)
EventSink.on_time_advance(dt, state_before, state_after)
```

Possible implementations:

```text
NullEventSink
StatsEventSink
FullTraceEventSink
```

or trace modes:

```text
TraceMode.NONE
TraceMode.SUMMARY
TraceMode.FULL
```

Expected behavior:

- `NONE`: retain only the minimum path result needed by the caller.
- `SUMMARY`: accumulate statistics/round data without retaining every event object.
- `FULL`: retain chronological event trace for diagnostics and path inspection.

Large Monte Carlo runs must not be required to hold every event object in memory.

---

# 19. Continuous-Time Engine Loop

Conceptual engine loop:

```text
load immutable FighterProfiles
apply prefight AgeModel -> EffectiveFighterProfiles
initialize FightState with fight_time_seconds = 0
initialize FightLedger / EventSink
initialize root RNG manager
emit FightStarted
emit RoundStarted
set phase = DISTANCE and clear positional ownership

while fight not finished:

    derive current round / remaining time from fight_time_seconds + config
    derive public FightContext from ledger if needed
    build DynamicModifiers snapshot
    determine valid primary events
    calculate per-second event rates

    scheduler.sample(candidates, scheduler_rng)
        -> dt, event

    find next hard boundary

    if event is None or sampled dt reaches/crosses boundary:
        elapsed = time_to_boundary
        ContinuousStateModel.advance(state, elapsed, profiles, modifiers)
        engine advances fight_time_seconds exactly to boundary
        process RoundEnded / recovery / next RoundStarted / final judging
        continue

    ContinuousStateModel.advance(state, dt, profiles, modifiers)
    engine advances fight_time_seconds += dt

    resolve selected primary event
    emit primary event
    apply explicit state delta(s)
    resolve and emit synchronous consequence events
    update ledger/event sinks

    if finish occurs:
        emit FightFinish
        stop
```

Key invariants:

- All scheduler rates are events/second.
- Only the engine advances the authoritative clock.
- Continuous state is advanced before resolving the event occurring after elapsed `dt`.
- Consequence events at the same timestamp do not consume additional waiting time.
- Round boundaries outrank events sampled beyond them.
- Round start resets position to distance.
- Stats/ledger do not drive components through hidden fields.

---

# 20. Baseline Freeze Before Implementation

Before implementing the new kernel, freeze a concrete comparison package from the current simulator.

The baseline is **not** an exact-output target for the new simulator. It exists so every change can be detected, explained, and attributed.

## 20.1 Deterministic single-path fixtures

Include fixed seeds for:

- Rob Font vs Raul Rosas Jr.
- Several striker-heavy archetype matchups.
- Several wrestler-heavy archetype matchups.
- Several grappler/submission-heavy matchups.
- At least one high-control matchup.
- At least one low-action matchup if available.
- At least one 5-round configuration if supported by the current comparison path.

Each fixture should record enough detail to reconstruct current behavior:

```text
seed
fighter profiles / IDs
ages
scheduled rounds
winner
method
finish round/time if any
sig attempts / landed
TD attempts / landed
TD success rate
control
phase occupancy
submission attempts
knockdowns
```

Where full current path traces are practical, retain them as diagnostic fixtures.

## 20.2 Aggregate historical baseline

Freeze the current aggregate historical cohort metrics used for model comparison, including where available:

```text
KO rate
SUB rate
DEC rate
finish-round distribution
fight duration
significant attempts
significant landed
TD attempts
TD landed
TD success rate
control
phase occupancy
submission attempts
knockdowns
winner probability
winner Brier score
winner accuracy
```

## 20.3 Attribution purpose

The event engine does not need exact path parity because changing from batched fixed intervals to continuous time legitimately changes event ordering and path outcomes.

The baseline exists to answer:

```text
What changed?
When did it change?
Was the difference caused by:
    temporal architecture,
    ontology correction,
    component port,
    or later calibration?
```

No Phase 1 implementation should begin until these baseline fixtures are defined and stored in a reproducible location.

---

# 21. Phase 0–7 Implementation Plan

## Phase 0 — Architecture audit and baseline freeze

Phase 0 is **not complete** until this document explicitly resolves and records:

- exact scheduler rate units;
- current blended TD consumer vs corrected wrestling ontology;
- probability-to-per-second-rate migration rule;
- continuous state advancement;
- one authoritative clock;
- generic scheduler / engine boundary ownership;
- reduced physical `FightState` and separate ledger;
- dynamic modifier pipeline;
- damage/KD/KD-consequence/finish mutation ownership;
- judging RNG behavior;
- round-start phase reset;
- conditional ground-exit rate mapping;
- event trace/performance modes;
- root seed and named RNG-stream strategy;
- future cooldown/action-duration extension point;
- `MatReturn` marked optional/future;
- concrete baseline fixtures and aggregate comparison set.

Revision v0.2 documents these architecture requirements.

Remaining Phase 0 operational task before code: **create/freeze the concrete current-simulator baseline comparison fixtures described in Section 20.**

No simulator mechanics are changed during Phase 0.

## Phase 1 — Generic scheduler/kernel only

Implement no real UFC mechanics yet.

Build and test:

```text
per-second EventRate contract
generic exponential scheduler
zero-rate -> infinity / None behavior
hard-boundary handling in engine
authoritative fight clock
continuous-state advancement hook
root RNG + deterministic named stream manager
trace/event-sink modes
synthetic typed events
```

Mathematical tests:

```text
Exponential waiting-time mean
competing-event proportions
same root seed -> identical path/streams
different seeds -> divergence
named stream independence from unrelated component draw counts
chronological ordering
exact hard-boundary handling
no negative dt
no event after finish
zero-rate handling
single-event handling
```

Gate: scheduler/kernel mathematical and reproducibility tests pass before UFC mechanics are added.

## Phase 2A — Distance temporal/mechanical parity

Port current distance mechanics as faithfully as practical into continuous time.

Include:

```text
strike attempts
TD attempts using current blended TD consumer
clinch entries
strike hit/miss
TD success/failure
```

Migration rule:

- preserve current final interval probabilities;
- convert those probabilities exactly to per-second rates;
- do not change wrestling semantics yet.

Goal: isolate what changes because the 10-second timestep is removed.

Validate:

```text
sig attempts/min
TD attempts/15
TD success %
clinch entries
rate-conversion parity at matched state/profile inputs
```

## Phase 2B — Ontology-correct action-rate model

Deliberately replace blended intrinsic TD attempt generation.

Target model:

```text
wrestling_entry
    -> intrinsic base TD attempt rate

phase / stamina / fight context / approved tactical terms
    -> context multiplier

final TD rate
    = base rate × context multiplier
```

Explicit regression expectations:

- increasing `wrestling_entry` increases intrinsic TD attempt rate;
- changing `wrestling_conversion` changes success, not intrinsic attempt frequency;
- changing `td_defense` changes opponent TD success prevention, not intrinsic initiation;
- changing `control_imposition` does not directly define intrinsic TD attempt rate;
- contextual modifiers are visible and separately auditable.

Goal: isolate what changes because TD-rate semantics are corrected.

## Phase 3 — Clinch and ground

Add:

```text
clinch striking
clinch TD attempts
separation
ground striking
submission attempts
ground escape
reversal
stand-up
continuous control accrual
continuous positional stamina costs
```

Do not require `MatReturn` in initial V1.

Preserve the conditional ground-exit mapping:

```text
lambda_reversal + lambda_escape = lambda_ground_exit
```

Validate:

```text
phase time
control time
TD counts
repeated TD sequences
submission attempts
top/bottom behavior
ground-exit total hazard parity
conditional reversal share
continuous-time control/stamina accrual
```

This is the first phase where the new clock can be meaningfully evaluated for repeated wrestling sequences such as shot -> stuff -> reshot.

## Phase 4 — Dynamic state and finishes

Port current locked models as independent components without retuning:

```text
stamina_v1
dynamic modifier provider
damage_v1
knockdown_v1
kd_consequence_v1
ko_finish_v1
submission_finish_v1
recovery_v1
age_physical_v1
```

Validate component ownership and independent behavior.

No component may secretly mutate another component's state domain.

## Phase 5 — Judging

Port the current simple 10-9 model with explicit RNG tie-breaking.

Validate reproducible round aggregation and decision outcomes for fixed seeds.

## Phase 6 — Historical replay

Run old and new engines over the same comparison cohort.

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

Stratify at minimum by:

```text
high wrestling_entry
low wrestling_entry
high control
3-round vs 5-round
age
experience where available
```

Differences should be attributed to migration stage rather than immediately tuned away.

## Phase 7 — Component ablations

Only after the full baseline architecture is operational:

```text
damage_v1 vs damage_v2
knockdown/ko_v1 vs alternatives
stamina_v1 vs stamina_v2
age_none vs age_physical_v1
judging_simple vs future judging model
other single-component alternatives
```

Use named deterministic RNG streams to make component A/B comparisons more stable and interpretable.

Change one component family at a time.

---

# 22. Architecture Invariants

The following are hard design invariants for `event_mc_v1` unless changed by a later architecture revision:

1. **Composition over inheritance.** Simulator physics are assembled from components rather than deep subclass chains.
2. **One authoritative clock.** `fight_time_seconds` is the single mutable fight-time source of truth.
3. **Engine owns time.** Components cannot advance the simulation clock.
4. **Rates are events/second.** Interval probabilities must be explicitly converted before scheduling.
5. **Continuous state advances for actual elapsed dt.** Control and sustained costs are not rounded to legacy segments.
6. **Immutable pre-fight identity.** Stored FSR and fight-night effective profiles are not mutated by path physics.
7. **Physical FightState is separate from stats.** The ledger observes events and may expose explicit context, but stats are never hidden component communication.
8. **Correct wrestling ontology is preserved.** `wrestling_entry` means intrinsic initiation; Phase 2A temporarily mirrors the legacy blended consumer only for migration attribution.
9. **Explicit modifier pipeline.** Dynamic physiology/tactics modify components through auditable modifier snapshots rather than private cross-component calls.
10. **One owner per mutation type.** Primary damage, KD occurrence, KD consequence trauma, and finish determination have explicit separate ownership.
11. **Typed events replace strings and side channels.** No mechanics depend on note parsing or before/after stats inspection.
12. **Synchronous consequences do not consume time.** Derived events share the primary event timestamp unless future duration semantics explicitly say otherwise.
13. **Round starts at distance.** Positional ownership is cleared at every `RoundStarted`.
14. **Ground-exit total hazard is preserved during parity.** Reversal and escape rates partition the existing total exit hazard.
15. **Root-seeded RNG is centrally owned.** No hidden component RNGs; named streams are preferred for ablation stability.
16. **Kernel remains extensible beyond memoryless hazards.** Cooldowns, busy periods, durations, and semi-Markov dwell models must remain possible later.
17. **Trace retention is configurable.** Large cohorts are not forced to retain full event histories.
18. **No simultaneous migration and retuning.** Temporal, semantic, and calibration changes are staged and attributable.

---

# 23. Immediate Next Task

No simulator implementation should begin yet.

The immediate next task is to complete the remaining operational part of Phase 0:

**Define and freeze the concrete current-simulator baseline comparison package described in Section 20.**

That baseline should include fixed single-path seeds, Font vs Rosas, representative style/archetype matchups, and the current aggregate historical metrics needed to attribute differences once the event-driven kernel is introduced.

After that baseline is frozen and reviewed, Phase 1 may begin with the generic scheduler/kernel only.

---

# 24. Architecture Questions Still To Resolve

Revision v0.2 resolves the major structural issues identified in the critical review. A small number of implementation-level architecture choices remain intentionally open and should be finalized before or during the named phase, without retuning simulator physics:

1. **Exact RNG stream implementation API (Phase 1).** Use a deterministic child-stream mechanism from one root seed; the exact Python abstraction (`SeedSequence`, stream registry, or equivalent) should be chosen during kernel implementation and tested for reproducibility/independence.
2. **Exact event-sink interface shape (Phase 1).** The architecture requires NONE/SUMMARY/FULL-equivalent behavior, but whether this is represented by enums, sink classes, or both can be decided during kernel implementation.
3. **Exact typed-state immutability/mutation mechanism (Phase 1).** Components should return explicit deltas/results; whether the engine applies frozen dataclass replacements or controlled mutable state can be decided based on performance while preserving ownership boundaries.
4. **Exact public `FightContext` contents (later phase).** Do not add score urgency or tactical context until a component actually needs it. When needed, derive it explicitly from the ledger rather than exposing private accumulator state.
5. **Action cooldown/duration semantics (future).** The extension point is required now, but actual refractory rules should not be invented during Phase 1 without evidence.

None of these unresolved items blocks the architecture direction. They are implementation-interface choices rather than unresolved semantic/calibration decisions.

---

# 25. Final Architecture Direction

Preserve the overall design:

```text
continuous-time event physics
+
composition-based components
+
immutable pre-fight / fight-night profiles
+
explicit physical dynamic FightState
+
separate event-driven FightLedger
+
typed primary and consequence events
+
continuous state advancement over elapsed dt
+
explicit dynamic modifier snapshots
+
clear damage/KD/finish ownership
+
reproducible root-seeded RNG streams
+
swappable age / stamina / damage / KD / KO / submission / judging components
```

The major migration principle is now explicit:

```text
first isolate temporal/mechanical change
then deliberately correct TD-initiation semantics
then port dynamic components without retuning
then validate historically
then run controlled ablations
```

The current simulator remains untouched and serves as the comparison baseline while `event_mc_v1` is developed in isolation.
