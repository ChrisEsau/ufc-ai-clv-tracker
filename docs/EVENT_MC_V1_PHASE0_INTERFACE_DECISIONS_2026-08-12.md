# EVENT MC V1 Phase 0 Interface Decisions

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Status: **Locked Phase 0 architecture decisions for Codex implementation. Documentation only.**

Architecture parent: `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Baseline contract: `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

## Purpose

Revision v0.2 deliberately left a small set of implementation-interface choices open. This document closes those choices so Codex does not invent architecture while implementing Phase 1.

These decisions do not retune simulator physics and do not authorize changes to the current simulator.

## 1. RNG manager — LOCKED

Use one integer `root_seed` per Monte Carlo path.

The engine owns an `RNGManager` that exposes deterministic named NumPy generators.

Do **not** derive streams from Python's randomized `hash()` and do **not** make stream identity depend on spawn/call order.

Use a stable integer ID table, conceptually:

```text
SCHEDULER = 10
STRIKE_RESOLUTION = 20
TAKEDOWN = 30
SUBMISSION = 40
DAMAGE = 50
KNOCKDOWN = 60
KO_FINISH = 70
JUDGING = 80
RECOVERY = 90
```

A named stream should be constructed deterministically from the pair:

```text
(root_seed, stable_stream_id)
```

Recommended NumPy mechanism:

```python
np.random.default_rng(np.random.SeedSequence([root_seed, stable_stream_id]))
```

Equivalent deterministic mechanisms are acceptable only if tests prove:

- same root seed + stream ID => same sequence;
- unrelated stream draw counts do not alter this stream;
- adding a new stream does not change existing stream sequences;
- no component creates a hidden RNG.

The stable stream ID table is part of the public reproducibility contract once Phase 1 ships.

## 2. Event sink / trace architecture — LOCKED

Use an `EventSink` protocol/class contract rather than embedding trace storage in `FightState`.

Conceptual interface:

```text
EventSink
    on_event(event, state_before, state_after) -> None
    on_time_advance(dt, state_before, state_after) -> None
    finalize() -> sink-specific result
```

Required implementations:

```text
NullEventSink
StatsEventSink
FullTraceEventSink
CompositeEventSink
```

Required user-facing trace modes:

```text
TraceMode.NONE
TraceMode.SUMMARY
TraceMode.FULL
```

Recommended mapping:

- `NONE`: null/minimal sink needed only for final path outcome.
- `SUMMARY`: `StatsEventSink` without chronological event retention.
- `FULL`: `CompositeEventSink(StatsEventSink, FullTraceEventSink)`.

A cohort simulation must not be forced to keep all event objects in memory.

## 3. FightState mutation model — LOCKED

Use an engine-owned mutable `FightState` for runtime performance and simplicity.

Components do **not** mutate it directly.

Components return frozen/immutable typed results and/or `StateDelta` objects. The engine applies those deltas in a single controlled mutation path.

Conceptual rule:

```text
component reads state
    -> returns typed result/delta/events
engine validates delta
    -> engine applies delta to FightState
```

Benefits:

- one mutation authority;
- component ownership is auditable;
- no deep-copy requirement on every event;
- easier invariant checks;
- practical performance for large Monte Carlo cohorts.

`ContinuousStateModel.advance(...)` follows the same pattern: return a delta plus optional emitted bookkeeping events; engine applies it before advancing the clock to the scheduled event timestamp/boundary.

## 4. FightContext — LOCKED MINIMUM FOR EARLY PHASES

Do not expose the entire ledger to action components.

For Phase 1 and Phase 2A, `FightContext` contains only derived schedule/time information needed by generic contracts, conceptually:

```text
fight_time_seconds
round_number
round_elapsed_seconds
round_remaining_seconds
scheduled_rounds
```

These values are derived from authoritative fight time + immutable schedule config.

No score urgency, cumulative TD counts, damage statistics, or judging state is exposed in early phases.

When a later approved strategy/urgency model actually requires cumulative fight context, extend `FightContext` deliberately and revise the architecture document. The ledger remains the source, but components consume only explicitly published context fields.

## 5. Future cooldown/duration extension — LOCKED SHAPE, INACTIVE PHYSICS

Do not invent cooldown constants in Phase 1.

However, state/candidate contracts must support action availability.

Reserve a typed concept such as:

```text
ActionAvailability
    actor
    action_type
    available_at_fight_time
```

or an equivalent engine-owned availability table.

Initial V1 behavior:

- all supported actions are immediately available unless phase rules prohibit them;
- no minimum action interval is imposed;
- scheduler remains exponential/memoryless.

Future models may set `available_at_fight_time` or introduce event duration / `busy_until` without replacing the scheduler contract.

The action-rate candidate builder must be able to exclude temporarily unavailable candidates before scheduling.

## 6. Event timestamps and ordering — LOCKED

Every primary event has one authoritative `fight_time_seconds` timestamp.

Synchronous consequence events use the same timestamp and carry deterministic sequence numbers for ordering.

Ordering at one timestamp is causal, for example:

```text
StrikeAttempt
StrikeLanded
DamageApplied
Knockdown
KnockdownConsequenceApplied
FightFinish
```

No consequence in that chain independently advances time.

The engine owns a monotonic `event_sequence` counter. The counter is ordering metadata, not physics state.

## 7. Hard-boundary tie rule — LOCKED

If the scheduler samples an event whose timestamp is exactly at or beyond the next round/fight hard boundary, the hard boundary wins.

Operational comparison:

```text
if dt >= time_to_boundary:
    advance continuous state to boundary
    advance clock to boundary
    process boundary
    discard/resample the scheduled event in the new state
```

This avoids events occurring after the legal round clock expires.

Use a small numerical tolerance only for floating-point comparison, never to create extra fight time.

## 8. Continuous-state application order — LOCKED

For an event sampled after elapsed `dt`:

```text
1. derive modifiers/context at interval start as required by the current continuous model
2. compute/apply continuous state delta over actual dt
3. advance authoritative clock by dt
4. refresh modifiers/context for the event timestamp
5. resolve primary event
6. apply primary-event delta
7. resolve/apply synchronous consequences in causal order
8. notify event sinks
```

If a future continuous differential model requires within-interval changing modifiers, that is a later architecture revision. Initial V1 sustained costs are piecewise constant between discrete events/boundaries.

## 9. Stats and control accounting — LOCKED

`FightLedger` / `StatsEventSink` owns accumulated statistical totals.

Control time is driven by `on_time_advance(dt, ...)` using the physical phase/controller state that existed during the elapsed interval.

Therefore a 6.27-second ground interval produces exactly 6.27 seconds of ground/total control in the ledger.

Physical stamina costs over that interval are owned by `ContinuousStateModel`/`StaminaModel`, not by the ledger.

The ledger observes the result; it does not cause physical state mutation.

## 10. Configuration ownership — LOCKED

Immutable fight configuration is separate from `FightState`.

Conceptually:

```text
FightConfig
    scheduled_rounds
    round_duration_seconds
    ruleset/version
    trace_mode
    component configuration references
```

Round breaks are boundary operations in initial V1 rather than active fight-time intervals.

The fight clock measures active fight time only unless a future revision explicitly introduces wall-clock/corner-time simulation.

## 11. Component registration — LOCKED

Use explicit composition at engine construction, not service-location magic and not subclass discovery.

Conceptually:

```text
EventMCComponents(
    action_rates=...,
    phase=...,
    continuous_state=...,
    strikes=...,
    takedowns=...,
    submissions=...,
    stamina=...,
    damage=...,
    knockdowns=...,
    knockdown_consequences=...,
    finishes=...,
    recovery=...,
    judging=...,
    modifiers=...,
)
```

A registry/config helper may build this bundle, but the final engine receives explicit component instances.

This makes an ablation visible as a component substitution rather than an inheritance change.

## 12. Phase 1 scope guard — LOCKED

Phase 1 contains generic infrastructure and synthetic test components only.

It may implement:

- contracts;
- event types/base metadata;
- `FightState` and `FightConfig` skeletons;
- state deltas;
- generic scheduler;
- engine hard-boundary loop;
- RNG manager;
- event sinks;
- continuous-state hook;
- synthetic event-rate providers/resolvers for tests.

It must **not** port real UFC strike, TD, ground, KO, SUB, stamina, age, or judging formulas yet.

That separation is required so scheduler mathematics can be validated independently of UFC calibration.

## 13. Phase 0 closure definition

Architecture design is closed for Phase 0 when all of the following are true:

- architecture audit revision contains the critical-review requirements;
- this interface-decision document is committed;
- baseline-freeze contract is committed;
- no unresolved semantic/calibration decisions remain for Phase 1;
- current simulator remains untouched.

Operational Phase 0 is closed only after the baseline capture contract has been executed and its manifest/summaries reviewed.

Because this ChatGPT environment has repository read/write access but no repository Python execution environment, the numeric baseline capture must be the **first Codex preflight task**. It is not simulator implementation and must complete before Phase 1 code begins.

## 14. No remaining Phase 1 architecture discretion for Codex

Codex may choose ordinary Python implementation details (module-private helper names, formatting, straightforward data-structure details) but must not independently change:

- scheduler units;
- clock ownership;
- RNG stream identity strategy;
- state/ledger separation;
- engine-only mutation authority;
- continuous-state timing order;
- event-sink modes;
- round-start reset;
- wrestling ontology/migration staging;
- damage/KD/finish ownership;
- Phase 1 scope.

Any need to change one of these requires an architecture revision before implementation proceeds.
