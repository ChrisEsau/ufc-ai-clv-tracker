# EVENT MC V1 Architecture Audit and Implementation Path

Date: 2026-08-12

Architecture Revision: **v0.3**

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Status: **Architecture Phase 0 closed. EVENT MC V1 simulator implementation has not started.**

## Revision History

| Revision | Date | Summary | Status |
|---|---|---|---|
| v0.1 | 2026-08-12 | Initial architecture audit and implementation path | Superseded |
| v0.2 | 2026-08-12 | Critical architecture review incorporated: precise inheritance count; corrected rating ontology vs blended current TD consumer; per-second rate units; continuous state advancement; smaller FightState; one clock; generic scheduler; dynamic modifier pipeline; damage/KD/KO ownership; judging RNG; named RNG-stream strategy; cooldown/duration extension point; round reset; conditional ground-exit mapping; optional MatReturn; trace modes; Phase 2A/2B split; concrete baseline requirements | Superseded |
| v0.3 | 2026-08-12 | Phase 0 closure: implementation-facing interfaces resolved, engine-only state mutation locked, event-sink behavior locked, named RNG streams fixed, baseline fixtures/seeds/metrics frozen, and Codex pre-implementation baseline gate documented | **Current** |

**Revision-control rule:** every future architecture change must increment `Architecture Revision` and append a revision-history entry describing what changed and why. Do not silently alter architecture decisions.

Companion Phase 0 closure document:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Codex baseline-materialization gate:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

---

# 1. Operating Locks

- Do not change the current simulator during EVENT MC V1 buildout; preserve it as a frozen comparison baseline.
- Keep FSR-32 connected initially.
- Preserve the corrected FSR rating ontology:
  - `wrestling_entry` = intrinsic takedown initiation frequency.
  - `wrestling_conversion` = shot completion ability/probability.
  - `td_defense` = opponent prevention of the shot.
  - `control_imposition` = post-position control ability.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, or other calibration constants during kernel/parity work.
- Build the new simulator only under `pipeline/simulation/event_mc_v1/`.
- Use composition, typed events, explicit state, and a continuous event clock.
- Fixed 10-second intervals may remain only for optional reporting/aggregation.
- Stored FSR profiles remain immutable pre-fight identity.
- Do not combine a temporal migration with an intentional semantic/calibration change unless a phase explicitly says to do so.

---

# 2. Current Simulator Architecture Audit

The effective historical diagnostic chain is:

```text
run_single_historical_age_power_diagnostic.py
    -> StaticFSRMCFullFightV1
    -> StaticFSRMCKOSUBV1
    -> RecoveryAuditSim
    -> AuditSim
    -> StaticFSRMCKOTKOV33GlobalRecovery
    -> StaticFSRMCKOTKOV32PhaseStamina
    -> StaticFSRMCKOTKOV31RollingFSR
    -> StaticFSRMCKOTKOV3Stamina
    -> StaticFSRMCKOTKOV2RoundRecovery
    -> StaticFSRMCKOTKOV2KDCollapse
    -> StaticFSRMCKOTKOV2
    -> StaticFSRMCDamageV1Ground017
    -> StaticFSRMCDamageV1
    -> StaticFSRMCV0
```

That is **14 simulator classes connected by 13 inheritance edges**.

The important problem is not merely depth. Behavior is distributed across overrides of:

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

The inheritance chain has effectively become part of simulator physics.

Reusable mathematics should be ported; the inheritance architecture should not.

---

# 3. Wrestling Ontology: Rating Correct, Current Consumer Still Blended

The rating definitions are locked:

```text
wrestling_entry      = intrinsic TD initiation frequency
wrestling_conversion = shot completion ability
td_defense            = opponent shot prevention
control_imposition    = control after position is established
```

However, current V0 does **not** consume `wrestling_entry` purely. It derives a blended preference approximately as:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

and uses that blended preference in TD-attempt hazard.

Therefore distinguish two truths:

1. the FSR rating ontology is corrected;
2. the current simulator consumer remains semantically blended.

EVENT MC V1 corrects this intentionally, but only after temporal parity is isolated.

Future ontology-correct rate model:

```text
wrestling_entry
    -> intrinsic base TD attempt rate

phase / stamina / fight situation / opponent context
    -> context multiplier

final TD attempt rate = base rate * context multiplier
```

`control_imposition` must not define intrinsic TD initiation.

---

# 4. Rate Units and Temporal Migration

All scheduler rates are **events per second**.

If the current simulator provides a probability `p_interval` over `interval_seconds`, parity conversion is:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

The old expression `-ln(1-p)` is integrated hazard over the interval represented by `p`; it is not automatically a per-second rate.

Initial parity rule:

- preserve current final interval probabilities;
- convert them exactly to per-second rates.

Later semantic/model cleanup may operate directly in rate space:

```text
base hazard * trait/context multiplier
```

Do not change both timing and calibration in the same experiment.

---

# 5. Target Package Structure

```text
pipeline/simulation/event_mc_v1/
├── __init__.py
├── contracts.py
├── config.py
├── events.py
├── state.py
├── scheduler.py
├── rng.py
├── engine.py
├── registry.py
├── sinks.py
├── components/
│   ├── profiles.py
│   ├── age.py
│   ├── modifiers.py
│   ├── action_rates.py
│   ├── phase.py
│   ├── strikes.py
│   ├── takedowns.py
│   ├── submissions.py
│   ├── stamina.py
│   ├── damage.py
│   ├── knockdowns.py
│   ├── finishes.py
│   ├── recovery.py
│   └── judging.py
└── diagnostics/
    ├── single_path_trace.py
    ├── event_rate_audit.py
    ├── matchup_replay.py
    └── cohort_replay.py
```

Tests:

`tests/simulation/event_mc_v1/`

Logical ownership is more important than exact file count.

---

# 6. Data Separation

## `FighterProfile`

Immutable stored FSR-32 identity.

## `EffectiveFighterProfile`

Immutable fight-night profile after prefight transformations such as the currently approved physical age rule.

Initial age parity remains:

```text
age <= 30: no physical reduction
age > 30: -2 rating points/year after 30
```

on the currently approved physical traits:

```text
striking_power
knockdown_resistance
damage_durability
```

Do not retune this during kernel work.

## `FightState`

Contains only physical/path state needed to determine future physics.

## `FightLedger` / `StatsAccumulator`

Observes typed events and owns accumulated statistics/judging inputs.

Stats never serve as hidden communication between components.

If future strategy needs accumulated score/context, expose an intentional read-only `FightContext` derived from state + ledger.

---

# 7. Authoritative State and Clock

One mutable time source:

```text
FightState.fight_time_seconds
```

Derive round number, round elapsed, and round remaining from fight configuration.

Fight configuration owns:

```text
scheduled_rounds
round_duration_seconds
hard boundaries
```

Core physical/path state includes:

```text
phase: DISTANCE | CLINCH | GROUND
clinch_controller
ground_controller
phase_started_at
red/blue stamina state
red/blue damage reservoir state
recent-KD transient state
finish state
optional transient action-availability state
```

Do not store accumulated sig/TD/sub/control totals in `FighterDynamicState` unless a future physical model truly requires an equivalent state variable.

Only the engine advances time.

---

# 8. Engine-Only State Mutation

The engine is the sole authoritative mutator of `FightState`.

Components return typed immutable results/deltas.

Conceptually:

```text
component input:
    state snapshot
    profiles
    DynamicModifiers
    explicit named RNG stream when stochastic

component output:
    typed result
    StateDelta
    consequence events
```

The engine applies deltas in a defined order.

This rule is intended to make component swaps and ablations inspectable and to prevent hidden mutation coupling.

---

# 9. Continuous State Advancement

Between discrete events, continuous state accrues over exact elapsed `dt`.

Required logical contract:

```text
TimeAdvanceModel.advance(
    state,
    dt,
    profiles,
    modifiers
) -> StateDelta / emitted events
```

At minimum this supports exact:

- clinch control time;
- ground control time;
- clinch-controller stamina cost/sec;
- clinch-resistance stamina cost/sec;
- ground-controller stamina cost/sec;
- bottom-resistance stamina cost/sec.

Engine order:

```text
sample dt/event
    -> advance continuous state for dt
    -> advance fight_time_seconds
    -> resolve scheduled event
```

If ground control lasts 6.27 seconds, exactly 6.27 seconds of control and sustained cost accrue.

---

# 10. Scheduler

The scheduler is UFC-agnostic:

```text
scheduler.sample(candidates, scheduler_rng) -> (dt, event | None)
```

It only:

- sums candidate rates;
- samples exponential waiting time;
- selects event proportional to rate.

It knows nothing about rounds.

For zero total rate:

```text
dt = infinity
event = None
```

The engine compares sampled `dt` against the next hard boundary and advances to that boundary when needed.

Only primary events consume simulation time. Consequence events happen synchronously at the same timestamp.

---

# 11. Dynamic Modifier Pipeline

Flow:

```text
EffectiveFighterProfile
+
FightState
+
registered ModifierProviders
    -> DynamicModifiers snapshot
```

Rate/resolution components consume the snapshot rather than calling stamina internals.

Conceptually the snapshot may expose:

```text
red/blue:
    output_multiplier
    power_multiplier
    defense_multiplier
    wrestling_multiplier
    other explicit future modifiers
```

Initial parity should expose only effects required by the locked current candidate.

The interface must allow future stamina versions to alter more systems without rewriting action-rate models.

---

# 12. Event Taxonomy

Lifecycle:

```text
FightStarted
RoundStarted
RoundEnded
FightFinish
DecisionRendered
```

Striking:

```text
StrikeAttempt
StrikeLanded
StrikeMissed
DamageApplied
Knockdown
KnockdownConsequenceApplied
```

Wrestling:

```text
TakedownAttempt
TakedownLanded
TakedownFailed
```

Clinch:

```text
ClinchEntered
ClinchSeparated
```

Ground:

```text
GroundPositionEstablished
GroundEscape
StandUp
Reversal
```

Submission:

```text
SubmissionAttempt
SubmissionDefended
SubmissionFinish
```

Physiology:

```text
StaminaCostApplied
RecoveryApplied
```

`MatReturn` is future/optional and requires richer standing-control/cage-wrestling state.

Primary scheduler candidates are actions/transitions. Derived outcomes such as `StrikeLanded`, `DamageApplied`, and `Knockdown` occur synchronously after the primary action.

---

# 13. Component Ownership

## ActionRateModel

Produces primary event rates in events/sec.

## PhaseModel

Owns valid phases/ownership/transitions; does not decide TD success.

## StrikeResolver

Owns attempt hit/miss from precision vs defense; not damage.

## TakedownResolver

Owns TD success from `wrestling_conversion` vs `td_defense`; not intrinsic entry and not control persistence.

## SubmissionResolver

Attempt rate comes from submission pressure/context. Finish resolution uses submission conversion vs resistance.

## StaminaModel / ModifierProvider

Owns stamina costs/state and produces explicit modifiers; does not rewrite unrelated model internals.

## DamageModel

Owns primary strike damage and primary reservoir delta.

## KnockdownModel

Owns knockdown yes/no.

## KnockdownConsequenceModel

Owns optional KD-collapse trauma/additional reservoir delta.

## FinishModel / KOModel

Determines stoppage from resulting state; may not secretly rewrite primary damage physics.

## RecoveryModel

Returns between-round deltas; does not directly reach into component internals.

## JudgingModel

Consumes round ledger/record and explicit judging RNG. Exact current tie behavior is reproducible, not fully deterministic.

---

# 14. RNG Architecture

Use one root path seed and centrally owned named deterministic streams.

Initial names are fixed:

```text
scheduler
strike_resolution
takedown_resolution
submission_resolution
damage
knockdown_finish
judging
```

No component creates a hidden RNG.

The manager should derive streams reproducibly from the root seed and stable stream name mapping, preferably with `numpy.random.SeedSequence` or equivalent.

Reason: swapping `damage_v1` for `damage_v2` should not automatically shift all later unrelated takedown/submission/judging random draws merely because one damage model consumes a different draw count.

---

# 15. Event Sinks / Trace Modes

Required logical behavior:

```text
NullEventSink      -> retain no trace
StatsEventSink     -> aggregate ledger only
FullTraceEventSink -> retain chronological typed events
```

Large cohorts must not be required to hold every event object in memory.

Trace retention is an observer concern, not simulator physics.

---

# 16. Future Duration / Cooldown Support

Phase 1 uses exponential memoryless waits unless a test reveals a hard blocker.

Contracts must still permit future:

```text
busy_until
cooldown_until[action_family]
minimum action interval
event duration
semi-Markov dwell distributions
```

Reserve an explicit transient `ActionAvailabilityState` or equivalent in path state. It is inactive by default during parity work.

Do not implement arbitrary cooldown constants in Phase 1.

---

# 17. Round Invariants

Every `RoundStarted` resets position to:

```text
phase = DISTANCE
ground_controller = None
clinch_controller = None
all other positional ownership cleared
```

This is locked unless a future architecture revision changes rules explicitly.

---

# 18. Ground Exit Parity

Current ground logic is conditional:

```text
sample exit
    -> if exit, sample reversal conditional on exit
```

If EVENT MC V1 represents reversal and escape as competing continuous events, preserve total exit hazard:

```text
lambda_reversal = lambda_ground_exit * P(reversal | exit)
lambda_escape   = lambda_ground_exit * (1 - P(reversal | exit))
```

Therefore:

```text
lambda_reversal + lambda_escape = lambda_ground_exit
```

This is an explicit Phase 3 parity test.

---

# 19. Baseline Freeze

Architecture-review snapshot:

`7b98ac629dacc094342ba7f6668ffc77aed3b246`

FSR-32 contract:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Baseline artifact contract, fixtures, seeds, known comparison observations, and output paths are fully specified in:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

The baseline exists for attribution, not for forcing exact EVENT MC V1 output parity.

Required anchor is Rob Font vs Raul Rosas Jr. plus representative power, high-volume striking, submission/grappling, and sustained wrestling/control fixtures.

Required aggregate metrics include winner calibration, methods, finish timing, striking, TDs, control, phase occupancy, submissions, and knockdowns.

---

# 20. Implementation Phases

## Phase 0 — architecture and baseline contract

**Closed at v0.3.**

The execution gate immediately before Phase 1 is current-simulator baseline materialization using:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

That task may run/add diagnostic orchestration but may not implement EVENT MC V1 mechanics or change the old simulator.

## Phase 1 — generic event kernel

No UFC calibration/model port yet.

Build/test:

- one authoritative clock;
- generic per-second-rate scheduler;
- hard-boundary handling in engine;
- zero-rate behavior;
- typed primary/consequence events;
- engine-only delta application;
- continuous `advance(dt)` contract;
- named RNG manager;
- event sinks;
- round lifecycle/reset;
- cooldown/duration extension state without active arbitrary cooldown physics.

Mathematical/reproducibility tests must pass before Phase 2.

## Phase 2A — distance temporal/mechanical parity

Port current distance behavior faithfully enough to isolate the effect of removing 10-second bins.

Include:

- strike attempts;
- hit/miss;
- TD attempts;
- TD success/failure;
- clinch entry.

Preserve current final interval probabilities and convert exactly to per-second rates.

Do not yet correct the blended TD-entry semantics.

## Phase 2B — ontology-correct TD initiation

Deliberately replace intrinsic TD initiation with `wrestling_entry`-driven base rate plus explicit context multipliers.

Measure this semantic correction separately from temporal migration.

## Phase 3 — clinch and ground

Add clinch persistence/separation, clinch TDs, ground persistence, ground striking, submissions, escape/reversal, exact control accrual, and sustained positional stamina costs.

Preserve conditional total ground-exit hazard.

## Phase 4 — dynamic state components

Port current locked concepts as swappable components without retuning:

- stamina;
- dynamic modifiers;
- damage reservoir;
- knockdowns;
- KD consequence/collapse;
- KO/TKO finish;
- submission finish;
- recovery;
- age prefight transform.

## Phase 5 — judging

Port current simple 10-9/no-draw baseline with explicit judging RNG tie resolution.

## Phase 6 — historical replay

Compare old vs new on frozen fixtures/cohorts across winner metrics, methods, finish timing, offense, wrestling, control, phase occupancy, submissions, and knockdowns.

Stratify especially by high/low wrestling entry, control style, scheduled rounds, age, and experience where available.

## Phase 7 — controlled ablations

Only after historical replay:

```text
stamina_v1 vs v2
damage_v1 vs v2
KO/finish alternatives
age alternatives
judging alternatives
action-rate alternatives
```

One component at a time.

---

# 21. Codex Execution Strategy

Codex receives one phase/gate per prompt.

Every prompt must contain:

1. exact repo/branch/source docs;
2. scope and non-goals;
3. files allowed to change;
4. architecture contracts/invariants;
5. tests required;
6. commands to run;
7. required report back;
8. explicit stop condition before the next phase.

Codex must not be asked to build the entire simulator in one task.

The first Codex task is baseline materialization, not EVENT MC V1 implementation.

---

# 22. Phase 0 Closure Statement

The architecture questions required to start implementation are resolved.

Implementation-level details intentionally deferred without changing semantics include:

- exact Python names/shapes of immutable delta/result classes;
- exact helper API around stable named RNG derivation;
- exact serialized full-trace schema beyond typed event requirements;
- future expanded `FightContext` contents;
- actual cooldown/refractory constants;
- richer cage-wrestling/mat-return positional ontology.

Those details may be chosen during the relevant implementation phase only if they preserve the contracts above. Any architecture-level deviation requires revision `v0.4` or later with revision-history documentation.

**Architecture Phase 0 is closed at v0.3.**