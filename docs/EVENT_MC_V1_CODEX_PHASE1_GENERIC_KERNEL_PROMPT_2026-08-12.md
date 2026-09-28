# Codex Prompt — EVENT MC V1 Phase 1 Generic Continuous-Time Kernel

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Architecture revision: **v0.3**

Status: **Phase 1 implementation authorized by user with the Phase 0 numerical baseline explicitly deferred because the frozen FSR-32 parquet is unavailable in the Codex environment. The Phase 0 baseline gate is NOT a PASS.**

## Read first — mandatory source-of-truth order

Before touching code, read the current versions of:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
7. this prompt

The architecture audit v0.3 remains canonical. The Phase 0 baseline freeze remains the future numerical comparison contract; it has merely been **deferred**, not waived or replaced.

## Explicit process exception for this task

The frozen Phase 0 operational baseline could not be materialized because this Codex environment does not contain:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

The user has explicitly chosen to continue implementation and provide the FSR artifact later.

Therefore:

- do **not** claim Phase 0 operational baseline PASS;
- do **not** rebuild FSR-32 or its upstream chain;
- do **not** use missing baseline data as permission to tune or reinterpret mechanics;
- Phase 1 is authorized only because it is a **generic kernel** that requires no UFC/FSR mechanics;
- before Phase 2A historical/mechanical parity work, the deferred FSR-32 baseline should be revisited unless the user explicitly changes that plan again.

---

# Phase 1 Objective

Implement the minimal, generic, deterministic, continuous-time event-simulation kernel under:

`pipeline/simulation/event_mc_v1/`

This phase is infrastructure only.

**Do not implement real UFC fight mechanics in Phase 1.**

The kernel must be capable of later supporting UFC simulation, but every Phase 1 behavior should be testable using synthetic events/rates/components.

---

# Absolute Non-Goals

Do **not** in this task:

- port striking formulas;
- port takedown formulas;
- implement wrestling-entry semantics;
- implement clinch or ground mechanics;
- implement submissions;
- implement stamina physiology;
- implement damage reservoirs;
- implement knockdowns;
- implement KO/TKO logic;
- implement between-round physical recovery;
- implement age adjustment;
- implement judging/scoring logic beyond reserving the named RNG stream;
- load FSR-32;
- rebuild FSR data;
- modify the existing simulator;
- modify any current simulator calibration constants;
- modify existing FSR builders/traits;
- create a new simulator inheritance stack;
- create tactical urgency or score-aware context;
- add arbitrary action cooldown constants;
- add `MatReturn` mechanics;
- tune anything against historical outcomes.

If you find yourself needing a real fighter trait, UFC probability, or historical calibration constant, stop: it is outside Phase 1.

---

# Required Package Scope

Create the Phase 1 package under:

```text
pipeline/simulation/event_mc_v1/
```

A reasonable initial structure is:

```text
pipeline/simulation/event_mc_v1/
├── __init__.py
├── contracts.py
├── config.py
├── events.py
├── state.py
├── rng.py
├── scheduler.py
├── sinks.py
├── engine.py
└── registry.py           # only if it materially simplifies component/synthetic test wiring
```

Tests belong under:

```text
tests/simulation/event_mc_v1/
```

Do not create empty files merely to match a diagram. Prefer a smaller coherent implementation if a proposed file has no Phase 1 responsibility yet.

Logical ownership boundaries below are mandatory even if exact file placement differs slightly.

---

# Locked Phase 1 Interfaces and Semantics

## 1. One authoritative mutable fight clock

The sole mutable fight-time source is:

```text
FightState.fight_time_seconds
```

Round number, elapsed round time, round remaining time, and fight remaining time are derived from immutable configuration plus `fight_time_seconds`.

Only the engine may advance the authoritative clock.

Do not maintain independent mutable round clocks/timers.

## 2. Fight configuration

Create a small immutable fight/kernel configuration sufficient for generic timing, at minimum conceptually including:

```text
scheduled_rounds
round_duration_seconds
```

Default UFC-like values are acceptable as generic defaults (`3`, `300.0`) only as timing defaults; do not add UFC action mechanics.

Expose helpers/properties for hard boundaries rather than duplicating arithmetic throughout the engine.

The kernel must support arbitrary positive round duration and round count in tests.

## 3. Generic phase/state shell

Phase 1 needs only the positional shell required to enforce the frozen round-start reset contract.

A minimal typed phase enum may include:

```text
DISTANCE
CLINCH
GROUND
```

`FightState` should contain only future-relevant physical/kernel state, not accumulated stats.

At minimum it should support:

```text
fight_time_seconds
phase
ground_controller
clinch_controller
finished / terminal indication
finish metadata only if needed generically
future action-availability transient state
```

Do not put strike/TD/submission counters in `FightState`.

## 4. Engine-owned state mutation

Components/synthetic resolvers must not directly mutate authoritative `FightState`.

Use typed immutable result/delta objects.

Conceptual shape:

```text
component sees state/context
-> returns immutable StateDelta/result
-> engine applies delta
-> engine emits resulting event(s)
```

`FightState` itself may be mutable internally for performance, but only the engine applies authoritative changes.

Tests must demonstrate that state application occurs through the engine/delta boundary.

## 5. Event model

Create typed generic event contracts sufficient for Phase 1 synthetic tests and lifecycle events.

At minimum distinguish:

- a scheduled primary event that consumes stochastic waiting time;
- engine/lifecycle consequence events that occur at the current timestamp without sampling another wait;
- round lifecycle events such as `RoundStarted` / `RoundEnded`;
- terminal/fight-finished indication if needed by the generic kernel.

Do not define the full UFC event taxonomy yet unless zero-behavior placeholders are truly necessary. Synthetic test event types are preferred.

Event timestamps must be monotonic and derive from the authoritative clock.

## 6. Event rates — unit lock

All scheduler rates are:

```text
events per second
```

Create a typed `EventRate`-style contract carrying at least:

```text
event/candidate identity
rate_per_second
```

Negative rates are invalid.

Zero-rate candidates must not be selected.

## 7. Exact probability-to-rate helper

Provide and test the exact conversion:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

Requirements:

- `interval_seconds > 0`;
- `0 <= p_interval < 1` for finite hazard conversion;
- `p_interval = 0` -> `0.0`;
- invalid inputs fail clearly;
- do not confuse integrated interval hazard `-ln(1-p)` with per-second rate.

Keep this helper generic and independent from UFC constants.

## 8. Generic exponential scheduler

The scheduler interface should remain essentially:

```text
scheduler.sample(candidates, rng) -> (dt, event_or_candidate | None)
```

Scheduler responsibilities only:

1. inspect non-negative candidate rates;
2. sum positive rates;
3. if total rate is zero, return `(inf, None)`;
4. sample exponential wait with mean `1 / total_rate`;
5. select one positive candidate proportional to `rate_i / total_rate`.

Scheduler must know nothing about:

- UFC rounds;
- round boundaries;
- fighters;
- phases;
- stamina;
- damage;
- judging;
- FSR.

Hard boundaries belong to the engine.

## 9. RNG manager — exact stable stream lock

Use one integer root path seed.

No hidden component RNGs.

Do not use Python's randomized `hash()` for stream identities.

Do not make stream identity depend on call order or spawn order.

Use the fixed stable stream ID table:

```text
SCHEDULER = 10
STRIKE_RESOLUTION = 20
TAKEDOWN = 30
SUBMISSION = 40
DAMAGE = 50
KNOCKDOWN_FINISH = 60
JUDGING = 70
```

Preferred derivation:

```python
SeedSequence([root_seed, stable_stream_id])
default_rng(...)
```

Expose named stream access centrally through an `RNGManager`-style owner.

Phase 1 will actively use the scheduler stream. Other streams may simply be obtainable/tested now so future component draw counts cannot perturb unrelated streams.

Required behavior:

- same root seed + same named stream -> same sequence;
- different root seeds -> divergent sequences;
- draws from one named stream do not advance another named stream;
- requesting streams in different orders produces the same per-stream sequences;
- no component instantiates its own RNG.

## 10. Continuous-state advancement hook

The engine must support generic continuous advancement over exact elapsed `dt` before the scheduled event resolves.

Conceptual engine ordering:

```text
sample candidate dt
-> compare with next hard boundary
-> choose actual elapsed dt = min(sampled dt, time to boundary)
-> advance continuous state over exact elapsed dt
-> advance authoritative clock by exact elapsed dt
-> if boundary came first: process boundary lifecycle/reset
-> else: resolve scheduled event
-> apply engine-owned result/delta/consequence events
-> sinks/ledger observers receive chronological notifications
-> next loop recomputes candidates/rates
```

No 10-second rounding.

No segment counters.

No fixed control-duration accounting.

The Phase 1 continuous-state component can be a protocol/interface plus synthetic test implementation. Do not implement stamina/control physiology yet.

## 11. Hard round/fight boundaries

Engine owns all hard boundaries.

A stochastic event sampled beyond the next hard boundary must **not** occur before that boundary.

At a round boundary:

1. continuous state advances exactly to the boundary;
2. authoritative clock reaches the exact boundary;
3. round-end lifecycle is processed;
4. if another round remains, round-start lifecycle occurs at the same fight timestamp;
5. round-start reset is applied;
6. candidate rates are recomputed from the reset state before another stochastic wait is sampled.

At `RoundStarted`, enforce the locked reset:

```text
phase = DISTANCE
ground_controller = None
clinch_controller = None
other positional ownership cleared if any exists
```

At the final scheduled fight boundary, terminate without sampling/resolving events past the fight horizon.

## 12. Event sinks / trace behavior

Implement logical equivalents of:

```text
NullEventSink
StatsEventSink
FullTraceEventSink
```

A Protocol is preferred with semantics conceptually like:

```text
on_time_advance(dt, state_before, state_after)
on_event(event, state_before, state_after)
finalize()
```

A composite sink is acceptable if useful.

Requirements:

- null sink retains no history;
- summary/stats sink accumulates only compact observations needed by synthetic tests;
- full-trace sink retains chronological detail;
- engine physics must not depend on which sink is attached;
- sink mode must not change RNG consumption or simulation result;
- future large cohorts must be able to run without retaining full traces.

Do not use sink/stat values as hidden communication back into the engine.

## 13. Action availability extension point

Reserve explicit transient architecture for future semi-Markov/cooldown behavior.

Conceptual shape:

```text
ActionAvailabilityState
busy_until
cooldown_until[action_family]
```

Requirements for Phase 1:

- default state is inactive/empty;
- it does not alter event rates unless a synthetic test explicitly wires it;
- do not invent UFC cooldown durations;
- do not add arbitrary refractory constants.

This is an extension point only.

## 14. Minimal context contract

If the engine/components need a `FightContext`, keep it minimal and read-only.

It may expose only values derivable from immutable config and authoritative state for Phase 1.

Do not add:

- score urgency;
- likely winner;
- tactical desperation;
- hidden accumulated statistics;
- strategy state.

---

# Required Phase 1 Tests

Add focused tests under `tests/simulation/event_mc_v1/`.

The test suite must cover at least all of the following.

## Scheduler mathematics

### Exponential waiting-time mean

For a fixed total rate `lambda`, use enough deterministic samples to verify empirical mean is reasonably close to:

```text
1 / lambda
```

Use a statistically sensible tolerance that is stable in CI. Do not use an excessively expensive sample count.

### Competing-event proportions

For synthetic rates such as `1, 2, 7`, verify selection frequencies are approximately:

```text
0.1, 0.2, 0.7
```

with deterministic RNG and stable tolerances.

### Single positive candidate

A single positive candidate is always selected and wait distribution follows its rate.

### Zero total rate

All-zero candidates return:

```text
(inf, None)
```

without error or RNG-dependent fake event.

### Invalid rates

Negative / NaN / otherwise invalid rates fail clearly rather than silently participating.

## Probability-to-rate conversion

Verify:

- exact known examples;
- zero probability;
- interval scaling;
- invalid intervals;
- invalid probability domain.

## RNG reproducibility

Verify:

- same root seed reproduces scheduler sequence;
- different root seed diverges;
- each fixed named stream reproduces independently;
- drawing heavily from `DAMAGE` does not alter `SCHEDULER` sequence;
- requesting streams in different orders does not change their sequences.

## Engine chronology and boundaries

With synthetic rate providers/events, verify:

- event timestamps are monotonic;
- no negative `dt`;
- exact continuous advancement occurs before event resolution;
- a sampled event beyond a round boundary is suppressed by the boundary;
- exact boundary timestamp is reached without overshoot;
- next round starts at the same boundary timestamp;
- round-start phase is `DISTANCE`;
- positional controllers are cleared on round start;
- candidates/rates after boundary see the reset state;
- no primary event occurs after final fight horizon;
- engine stops immediately after an explicit synthetic finish event;
- no events occur after finish.

## Sink invariance

Using identical root seed/config/components:

- null sink, stats sink, and full-trace sink yield identical physical terminal state/result;
- sink choice does not alter RNG sequence/result;
- full trace is chronological;
- null sink does not retain trace payload.

## Engine-owned mutation

Include at least one synthetic component that returns a delta and verify the engine applies it. Avoid designing public component APIs that expect direct mutation of `FightState`.

## Continuous advancement exactness

Use a synthetic continuous-state advancer such as an accumulator with known slope and prove that it receives exact elapsed `dt`, including truncated `dt` at hard boundaries.

## Scheduler UFC-agnostic invariant

Tests/import structure should make clear `scheduler.py` has no dependency on UFC phase/round/fighter mechanics. A simple dependency/import test or code structure assertion is acceptable if useful; architectural separation should primarily be obvious from implementation.

---

# Expected Public API Quality

Prefer:

- typed dataclasses/enums/protocols;
- frozen/immutable result objects where appropriate;
- explicit units in field names (`*_seconds`, `*_per_second`);
- small pure helpers;
- deterministic behavior;
- no global mutable RNG;
- no import-time side effects;
- no dependency on experimental simulator modules.

Avoid premature abstractions. Phase 1 should be small enough to understand end-to-end.

---

# Existing Code Protection

Before finalizing, inspect the diff and explicitly confirm:

- no existing simulator module changed;
- no FSR builder/trait file changed;
- no existing calibration constant changed;
- no production simulation path changed;
- no historical artifacts were generated/rebuilt;
- no Phase 2 UFC mechanics were introduced.

The new package must live beside the current simulator, not replace it.

---

# Test Execution

Run the new Phase 1 tests directly.

Also run a reasonable existing simulation regression subset available in the environment. If known pre-existing tests fail (for example the previously observed age-contract failures), distinguish them explicitly from new regressions.

If import-path setup is needed, use the repository-prescribed method from `AGENTS.md` rather than changing package behavior solely to satisfy the local shell.

Do not “fix” unrelated pre-existing tests in this Phase 1 task.

---

# Commit / PR Rules

Follow `AGENTS.md` repository policy.

If the Phase 1 implementation and tests are clean:

- commit the Phase 1 changes on `feature/fsr-32-stamina-shadow`;
- push the branch if repository policy/environment allows;
- create the required PR targeting `dev` if `AGENTS.md` requires that workflow.

Do not merge the PR.

Do not begin Phase 2A in the same task.

---

# Required Codex Report Back

Return a concise but complete implementation report containing:

1. branch and starting commit SHA;
2. final commit SHA;
3. files created/changed;
4. package/API overview;
5. exact scheduler semantics implemented;
6. exact RNG stream implementation and stable ID mapping;
7. authoritative clock/boundary lifecycle implementation;
8. state mutation/delta ownership implementation;
9. event sink implementation;
10. action-availability extension-point implementation;
11. tests added, including test counts;
12. exact test commands/results;
13. any pre-existing failures encountered, clearly separated from new failures;
14. git diff/stat summary;
15. confirmation that no current simulator/FSR/calibration code changed;
16. confirmation that no real UFC mechanics were implemented;
17. PR number/link if created;
18. any architectural ambiguity encountered that should be resolved before Phase 2A.

End with exactly one of:

```text
PHASE 1 GENERIC KERNEL GATE: PASS
```

or

```text
PHASE 1 GENERIC KERNEL GATE: FAIL
```

If FAIL, list exact blockers.

## Stop condition

Stop after Phase 1 generic kernel implementation, tests, commit/report/PR work.

**Do not begin Phase 2A.**