# EVENT MC V1 — Phase 0 Closure

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Architecture source of truth: `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Status: **Phase 0 architecture closed; no EVENT MC V1 simulator implementation has started.**

## Purpose

This document closes the architecture/audit stage and freezes the implementation-facing decisions that Codex must follow when EVENT MC V1 work begins. It does not authorize changes to the current simulator and it does not retune any simulator constants.

## Locked operating constraints

- Keep the current simulator untouched as the comparison baseline.
- Keep FSR-32 connected for the initial EVENT MC V1 implementation.
- Preserve the corrected rating ontology:
  - `wrestling_entry` = intrinsic takedown initiation frequency.
  - `wrestling_conversion` = ability/probability to complete a shot.
  - `td_defense` = opponent prevention of the shot.
  - `control_imposition` = what happens after control is established.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, or other calibration constants during kernel/parity work.
- Do not modify production simulation code as part of Phase 0.
- EVENT MC V1 remains isolated under `pipeline/simulation/event_mc_v1/`.

## Frozen repository/data references

The Phase 0 architecture review was completed against branch head:

`7b98ac629dacc094342ba7f6668ffc77aed3b246`

This commit changes documentation only relative to the prior architecture-audit commit, so it is safe to use as the code snapshot identifier for the existing simulator comparison baseline.

FSR-32 simulator-facing artifact contract:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

FSR-32 builder:

`scripts/experimental/build_fsr_32_database.py`

Historical aligned-cohort source:

`scripts/experimental/fsr_32_historical_cohort.py`

Current historical single-bout diagnostic entry point:

`scripts/experimental/run_single_historical_age_power_diagnostic.py`

Current full-fight simulator entry class:

`StaticFSRMCFullFightV1`

## Final Phase 0 implementation-interface decisions

### 1. Engine owns all state mutation

Components do not freely mutate shared global state.

The engine is the sole authoritative mutator of `FightState`.

Components return typed immutable result/delta objects. The engine applies those deltas in a defined order and emits consequence events.

This prevents hidden cross-component mutation and makes ablations inspectable.

### 2. One authoritative clock

`FightState.fight_time_seconds` is the single mutable simulation clock.

Round number, round elapsed, and round remaining are derived from immutable fight configuration.

Only the engine advances time.

### 3. Continuous advancement contract

Before a scheduled discrete event is resolved, the engine advances all continuous state through elapsed `dt`.

Conceptual order:

```text
sample dt/event
    -> advance continuous state for dt
    -> advance fight_time_seconds
    -> resolve scheduled event
    -> apply event consequences
    -> emit events / update sinks
    -> recalculate rates
```

Continuous state includes exact control-time accrual and sustained per-second stamina costs in clinch/ground positions.

### 4. Scheduler contract

The scheduler is UFC-agnostic.

```text
scheduler.sample(candidates, rng) -> (dt, event | None)
```

All rates are **events per second**.

For current interval probability `p_interval` over `interval_seconds`:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

For zero total rate:

```text
dt = infinity
event = None
```

The engine, not the scheduler, owns round/fight hard boundaries.

### 5. Named deterministic RNG streams

Use one root path seed.

The root seed deterministically spawns centrally owned named streams. Components never instantiate hidden RNGs.

Initial stream names are locked as:

```text
scheduler
strike_resolution
takedown_resolution
submission_resolution
damage
knockdown_finish
judging
```

The implementation should use a deterministic fixed-name mapping from the root seed so swapping one component does not unnecessarily shift unrelated random draws in other systems.

The exact Python mechanism may use `numpy.random.SeedSequence` or an equivalently reproducible central implementation, but stream names and ownership must remain explicit and testable.

### 6. Dynamic modifier pipeline

No component should call stamina internals directly.

Flow:

```text
EffectiveFighterProfile
+
FightState
+
registered ModifierProviders
    -> DynamicModifiers snapshot
```

Resolvers/rate models consume the snapshot.

Initial parity candidates may expose only currently locked effects, but the interface must permit later output/power/defense/wrestling modifiers without rewriting action-rate components.

### 7. Event sinks / trace modes

Event retention is separated from simulator physics.

Required logical sinks:

```text
NullEventSink      # no retained trace
StatsEventSink     # aggregate ledger/statistics only
FullTraceEventSink # complete chronological event trace
```

Equivalent names are acceptable, but the behavior is locked.

Large cohort runs must not be forced to retain every event object in memory.

### 8. FightState vs ledger

`FightState` contains only values required for future physics:

- `fight_time_seconds`
- phase and positional ownership
- stamina state
- damage reservoir state
- recent-KD transient state
- finish state
- explicit transient scheduling/availability state when needed

Accumulated observations live in a separate ledger/stats accumulator:

- significant attempts/landed
- TD attempts/landed
- submissions
- reversals
- knockdowns
- control totals
- phase occupancy
- judging inputs

If future tactical strategy needs accumulated score/context, the engine exposes a read-only `FightContext` derived from state + ledger. Components may not inspect private stats fields.

### 9. Future cooldown/duration extension point

Phase 1 uses memoryless exponential scheduling unless tests prove a hard blocker.

However, V1 contracts must support future non-memoryless mechanics without redesigning the engine.

Reserve an explicit `ActionAvailabilityState` / equivalent transient state capable of representing:

```text
busy_until
cooldown_until[action_family]
```

Default state is empty/inactive during initial parity work.

Future semi-Markov dwell distributions or event durations are allowed as architecture extensions, not Phase 1 requirements.

### 10. Damage/KD/KO mutation ownership

The causal ownership is locked:

```text
StrikeResolver
    -> landed/missed

DamageModel
    -> primary strike damage
    -> primary reservoir delta

KnockdownModel
    -> knockdown yes/no

KnockdownConsequenceModel
    -> optional KD-collapse trauma / additional reservoir delta

FinishModel / KOModel
    -> stoppage decision from resulting state
```

Only one component owns each mutation type. A KO implementation may not silently rewrite damage physics.

The physical file layout may combine small V1 classes, but logical ownership remains distinct.

### 11. Reproducible judging

Current simple judging parity retains seeded RNG tie-breaking for exact ties.

Judging therefore receives explicit judging RNG access and is described as **reproducible**, not fully deterministic.

### 12. Round-start position reset

Every new round starts:

```text
phase = DISTANCE
ground_controller = None
clinch_controller = None
other positional ownership cleared
```

This is a locked invariant for V1 parity.

### 13. Ground-exit parity

The current ground sequence is conditional:

```text
ground exit occurs?
    -> if yes, reversal or escape?
```

When split into competing continuous-time primary events:

```text
lambda_reversal = lambda_ground_exit * P(reversal | exit)
lambda_escape   = lambda_ground_exit * (1 - P(reversal | exit))
```

Therefore:

```text
lambda_reversal + lambda_escape = lambda_ground_exit
```

This must be tested explicitly in Phase 3.

### 14. MatReturn remains future/optional

`MatReturn` is not a required initial V1 event because current state only models `DISTANCE`, `CLINCH`, and `GROUND`.

It requires a richer standing-control/cage-wrestling positional ontology before it becomes a first-class event.

## Wrestling migration rule

The corrected FSR rating ontology is already locked, but the current V0 consumer is still blended.

Current V0 effectively derives:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

and uses that blended preference in TD-attempt hazard.

Therefore migration is intentionally split:

### Phase 2A — temporal/mechanical parity

Port the current final distance/clinch event probabilities as faithfully as possible into per-second rates.

Goal: measure the effect of removing fixed 10-second time bins without simultaneously changing TD semantics.

### Phase 2B — ontology-correct TD initiation

Then intentionally replace intrinsic TD initiation with:

```text
wrestling_entry
    -> base intrinsic TD attempt rate

phase / stamina / fight context / opponent context
    -> context multiplier

final TD rate = base rate * context multiplier
```

`control_imposition` must not define intrinsic TD initiation.

This semantic correction must be measured separately from the temporal correction.

# Comparison baseline contract

The old simulator does not need to be matched bit-for-bit by EVENT MC V1. The baseline exists to identify and attribute differences.

## Baseline reproducibility metadata

Every baseline artifact must record:

```text
repository
branch
commit_sha
FSR artifact path
simulator entry class
age rule
paths
seed/root seed
cohort definition
fight identifiers
metric definitions
```

## Deterministic path seeds

Single-path trace fixtures use these root seeds for each selected matchup:

```text
7
17
20260811
```

These traces exist to expose causal path differences, not population accuracy.

## Matchup-level Monte Carlo configuration

For stable comparison matchups:

```text
paths = 1000
root seed = 20260811
```

Required anchor matchup:

```text
Rob Font vs Raul Rosas Jr.
event date: 2026-03-07
bout_id: bed89a91da9d04c1
actual: Raul Rosas Jr. by decision
```

Latest recorded FSR-32 diagnostic anchor from the current research run:

```text
Font win probability: 59.3%
Rosas win probability: 40.7%
Font TD attempts/path: 0.68
Rosas TD attempts/path: 4.49
Font TD landed/path: 0.29
Rosas TD landed/path: 2.37
Font control seconds/path: 32.04
Rosas control seconds/path: 284.15
Font significant attempts/path: 127.61
Rosas significant attempts/path: 60.81
```

These numbers are comparison evidence, not calibration targets.

Additional deterministic matchup fixtures should include the following known historical styles/events where profiles resolve in the aligned FSR-32 cohort:

```text
Derrick Lewis vs Chris Daukaus — 2021-12-18
Max Holloway vs Calvin Kattar — 2021-01-16
Charles Oliveira vs Dustin Poirier — 2021-12-11
Merab Dvalishvili vs Petr Yan — 2023-03-11
```

If one does not resolve in the mature aligned cohort, the baseline-materialization task must replace it with a documented same-archetype matchup selected from the cohort, not silently drop the fixture category.

Fixture intent:

```text
Font/Rosas           high-entry wrestling stress case
Lewis/Daukaus        high-power / KO stress case
Holloway/Kattar      high-volume striking stress case
Oliveira/Poirier     submission/grappling stress case
Dvalishvili/Yan      sustained wrestling/control stress case
```

## Aggregate historical cohort baseline

Use the existing mature 2020+ aligned FSR-32 historical cohort.

For inexpensive repeated parity checks, lock a deterministic first-200-bout slice with:

```text
paths per bout = 10
root seed = 20260810
```

For full method validation, preserve the existing mature 2020+ submission audit cohort contract:

```text
1,565 fights
10 paths/fight
historical SUB rate = 16.23%
current simulated SUB rate = 16.49%
neutral P(SUB | attempt) = 34%
historical submission attempts/fight = 0.5655
simulated attempts/path = 0.4994
historical >=1 submission-attempt rate = 35.02%
simulated >=1 submission-attempt rate = 35.08%
```

These are frozen comparison observations, not targets to force EVENT MC V1 toward.

## Required baseline metrics

For every matchup/cohort baseline where applicable, capture:

```text
winner probabilities
winner Brier / accuracy when actual outcomes are available
KO/TKO rate
SUB rate
DEC rate
finish round / fight duration
significant attempts
significant landed
TD attempts
TD landed
TD success rate
clinch control seconds
ground control seconds
total control seconds
DISTANCE / CLINCH / GROUND occupancy
submission attempts
knockdowns
```

## Baseline artifact location

Materialized comparison outputs should be written under:

`data/experimental/event_mc_v1_baseline/`

Recommended files:

```text
manifest.json
single_path_traces.jsonl
matchup_summary.csv
cohort_200_summary.csv
full_method_baseline.csv
```

If repository policy excludes generated data files from Git, commit the manifest and summary metadata and document the generated-data location/checksum instead. Do not weaken the reproducibility contract.

# Codex handoff strategy

EVENT MC V1 will be implemented by Codex using one tightly scoped prompt per phase/gate rather than one large open-ended prompt.

Every Codex prompt will include:

1. exact repo/branch and source-of-truth docs;
2. scope and explicit non-goals;
3. files allowed to create/change;
4. contracts/invariants that must be preserved;
5. tests to add before/with implementation;
6. commands Codex must run;
7. required output/report back;
8. a stop condition prohibiting work on the next phase.

Codex should never be asked to "build the whole simulator" in one task.

The first Codex task is a **Phase 0 baseline-materialization preflight**, not V1 simulator implementation. Its exact prompt is stored separately in:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

After that preflight is verified, Codex receives the Phase 1 kernel prompt.

# Phase 0 closure criteria

Architecture Phase 0 is considered closed when all of the following are true:

- inheritance/coupling audit is documented;
- per-second rate units are locked;
- one authoritative clock is locked;
- continuous state advancement is locked;
- FightState/ledger separation is locked;
- engine-only state mutation is locked;
- dynamic modifier pipeline is locked;
- damage/KD/KO ownership is locked;
- named RNG strategy is locked;
- event-sink/trace strategy is locked;
- round-start reset is locked;
- ground-exit conditional mapping is locked;
- duration/cooldown extension point is reserved;
- current blended wrestling consumer is distinguished from corrected ontology;
- Phase 2A/2B split is locked;
- comparison baseline contract/fixtures/seeds/metrics are explicitly specified;
- Codex baseline-materialization preflight is specified.

**Architecture Phase 0 is closed by this document.**

The baseline materialization prompt is the required execution gate immediately before Phase 1 coding. It may generate baseline data using the existing simulator, but it may not modify existing simulator behavior or implement EVENT MC V1 code.

# Unresolved questions after Phase 0

No unresolved question is allowed to change the fundamental architecture during Phase 1 without a new architecture revision.

The following are intentionally deferred implementation details, not unresolved architecture semantics:

- exact Python class names for immutable delta/result objects;
- exact `SeedSequence` helper API used to derive named streams;
- exact serialization format of full trace events beyond the required typed fields;
- exact future contents of `FightContext` beyond read-only state+ledger derivation;
- actual cooldown/refractory values, which are not part of initial parity work;
- future richer positional ontology required for mat returns/cage wrestling.

Any change to the architecture-level decisions above requires incrementing the architecture revision and documenting why.