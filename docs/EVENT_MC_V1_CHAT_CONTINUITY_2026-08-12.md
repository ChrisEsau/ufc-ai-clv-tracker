# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions so the assistant can recover exactly what it was doing, what Codex was asked to do, what is locked, and what the next review step is if conversation context is lost.

This file is **not** the architecture source of truth. It is the assistant's running continuity checkpoint and must point back to the canonical architecture / closure / Codex documents.

## Update rule

After every new Codex prompt is given to the user, update this file before moving on.

Each update must record:

- date/time if known;
- current architecture revision;
- current implementation phase / gate;
- exact Codex prompt file path;
- purpose of that Codex task;
- hard non-goals / locks relevant to the task;
- what Codex is expected to return;
- what the assistant must review next;
- whether the next phase is authorized yet;
- any new decisions or observed results.

Do not silently overwrite history. Append a new checkpoint entry and update the `Current State` section.

---

# Current State

Architecture revision: **v0.3**

Architecture status: **Phase 0 architecture is closed. The Phase 0 numerical/operational baseline did NOT pass because the frozen FSR-32 parquet is unavailable in Codex's isolated environment. The user explicitly chose to defer that artifact-dependent baseline and move forward with infrastructure implementation. This is a process exception, not a baseline PASS and not an architecture redesign.**

Implementation status: **Phase 1 generic continuous-time kernel is now authorized. No real UFC mechanics are authorized in Phase 1.**

Current Codex task prompt:

`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

Canonical architecture document:

`docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Phase 0 closure record:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Locked implementation-interface decisions:

`docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`

Canonical future numerical/reproducibility baseline contract:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Deferred frozen artifact:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Important: the baseline freeze remains valid and should be materialized when the exact FSR-32 artifact is later made available. Do not replace it silently with a rebuilt artifact.

---

# Core Project Goal

Build a new continuous-time, event-driven, composition-based UFC Monte Carlo simulator under:

`pipeline/simulation/event_mc_v1/`

The current inheritance-based simulator remains untouched as a frozen comparison baseline.

The design objective is to simulate actual fight flow with continuous event timing while keeping major systems independently swappable/testable:

- action rates;
- phase transitions;
- striking;
- takedowns;
- submissions;
- stamina;
- damage;
- knockdowns;
- KO/TKO finishes;
- recovery;
- age transforms;
- judging;
- statistics / event sinks.

---

# Hard Locks

- Do not modify the current simulator while EVENT MC V1 is built.
- Keep FSR-32 connected initially when UFC mechanics are later introduced.
- Do not rebuild the entire FSR database unless explicitly approved.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, or other calibration constants during kernel/parity work.
- Use composition, not a new inheritance chain.
- One authoritative fight clock.
- All scheduler event rates are expressed in **events per second**.
- Only the engine advances time.
- Continuous state advances over exact elapsed `dt` before the scheduled event resolves.
- Stats / ledger are observers, never hidden communication between components.
- Components do not create hidden RNGs.
- Engine owns authoritative state mutation through typed results/deltas.
- Round start resets phase to `DISTANCE` and clears positional ownership.
- `MatReturn` is future/optional and not required in initial V1.
- Do not combine temporal migration, ontology correction, and calibration changes in one step.

---

# Locked Wrestling Ontology

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

Important migration fact:

The current V0 simulator still consumes a blended wrestling preference approximately as:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Therefore:

- Phase 2A will reproduce the current blended consumer in continuous time for temporal/mechanical parity.
- Phase 2B will deliberately implement the ontology-correct TD initiation model where `wrestling_entry` defines intrinsic base TD attempt rate and context applies separately.

Never hide this semantic correction inside the temporal migration.

---

# Phase 0 Baseline Status — Deferred, Not Passed

Codex successfully repaired its Git checkout and verified the exact feature branch lineage.

The numerical baseline could not run because the required ignored/generated artifact was absent from Codex's isolated filesystem:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Codex then performed the approved artifact-recovery search and returned:

`FSR-32 ARTIFACT RECOVERY: NOT FOUND`

The search covered the accessible `/workspace`, `/mnt`, `/tmp` roots and attempted the requested Codespaces-style roots, but Codex is not actually running inside the user's development Codespace. No exact FSR-32 candidate was found, no FSR artifact was rebuilt, and no baseline outputs were fabricated.

The user stated the FSR can be supplied later and explicitly chose to move on.

Decision:

- Phase 0 architecture remains closed at v0.3.
- Phase 0 operational/numerical baseline remains **DEFERRED / NOT PASSED**.
- Do not call it PASS in later reports.
- Do not rebuild the FSR chain to manufacture a replacement baseline unless explicitly approved.
- Phase 1 generic infrastructure may proceed because it requires no UFC/FSR mechanics.
- Revisit the frozen baseline before Phase 2A parity work unless the user explicitly changes that plan again.

Known unrelated test debt observed during Phase 0 checks:

```text
34 passed
2 pre-existing age-contract failures
```

Those failures were not created by EVENT MC V1 work and are not part of Phase 1 unless they become a direct regression conflict.

---

# Current Codex Task — Phase 1 Generic Continuous-Time Kernel

Codex must read and execute:

`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

Phase 1 objective:

Implement the smallest coherent generic continuous-time event kernel under:

`pipeline/simulation/event_mc_v1/`

Phase 1 includes only infrastructure:

- typed generic events/contracts;
- immutable timing configuration;
- authoritative `FightState.fight_time_seconds` clock;
- generic exponential competing-risk scheduler;
- events-per-second rate contract;
- exact probability-to-rate conversion helper;
- stable named RNG streams from one root path seed;
- engine-owned hard round/fight boundaries;
- exact continuous `advance(dt)` hook before event resolution;
- engine-owned typed state-delta application;
- round lifecycle and locked round-start positional reset;
- null/stats/full-trace event sinks with physics invariance;
- inactive action-availability/cooldown extension point;
- synthetic tests for mathematical, temporal, reproducibility, mutation-ownership, and sink invariants.

Absolute Phase 1 non-goals:

- no striking formulas;
- no takedown formulas;
- no wrestling-entry implementation;
- no clinch/ground mechanics;
- no submissions;
- no stamina;
- no damage/KD/KO;
- no recovery;
- no age transforms;
- no judging mechanics;
- no FSR loading;
- no historical calibration/tuning;
- no changes to current simulator/FSR/calibration code.

Stop after Phase 1 tests/report/commit/PR work. Do not begin Phase 2A.

---

# Assistant Review Required When Codex Returns

Do **not** automatically proceed because Codex says PASS.

Independently review:

- files and diff;
- whether current simulator or FSR code changed;
- package dependency direction;
- scheduler UFC-agnostic behavior;
- rate units and exact probability-to-rate math;
- stable named RNG stream derivation and independence;
- one authoritative clock and exact boundary handling;
- continuous advancement ordering;
- engine-owned mutation/delta boundary;
- round-start positional reset;
- sink invariance and RNG invariance;
- action-availability extension point remaining inactive/no arbitrary constants;
- test quality and results;
- any new failure vs pre-existing age-contract failures;
- PR/commit state;
- confirmation no Phase 2 UFC mechanics slipped in.

If Phase 1 is clean, mark `PHASE 1 GENERIC KERNEL GATE: PASS` after independent review.

Before authorizing Phase 2A, revisit the deferred FSR-32 numerical baseline unless the user explicitly approves another exception.

---

# Planned Phase 2A After Phase 1 + Baseline Revisit

Phase 2A is distance temporal/mechanical parity only:

- strike attempts;
- strike hit/miss;
- TD attempts using the **legacy blended consumer**;
- TD success/failure;
- clinch entry;
- exact final interval probability -> per-second rate conversion.

Do not perform ontology correction in Phase 2A.

Phase 2B later changes only TD initiation semantics so `wrestling_entry` becomes intrinsic base TD attempt rate and context becomes a separate multiplier.

---

# Checkpoint History

## Checkpoint 001 — 2026-08-12 22:33 America/Chicago

Phase 0 operational baseline prompt issued. Phase 1 unauthorized.

## Checkpoint 002 — 2026-08-12 22:37 America/Chicago

Codex execution plan approved with exact-remote-branch guardrail. Phase 1 unauthorized.

## Checkpoint 003 — 2026-08-12 22:40 America/Chicago

Codex stopped because no Git remote existed. Remote-unblock prompt issued. Phase 1 unauthorized.

## Checkpoint 004 — 2026-08-12 22:49 America/Chicago

Codex repaired the Git environment and verified the feature branch, but Phase 0 baseline failed because the required frozen FSR-32 parquet was absent. Artifact-recovery prompt issued. Rebuilding the FSR chain was deliberately not authorized.

## Checkpoint 005 — 2026-08-12 22:59 America/Chicago

Codex artifact recovery returned:

`FSR-32 ARTIFACT RECOVERY: NOT FOUND`

The user then explicitly directed: **move on; the FSR can be given to Codex later.**

Process decision:

- baseline remains deferred and must never be described as passed;
- architecture v0.3 is unchanged;
- no controlled FSR reconstruction is authorized;
- Phase 1 generic kernel is authorized despite the deferred numerical baseline because it contains no UFC/FSR mechanics;
- the deferred numerical baseline should be revisited before Phase 2A unless the user changes that plan again.

New Codex prompt issued:

`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

Expected Codex return:

- full Phase 1 implementation report;
- exact branch/start/final SHAs;
- files changed;
- scheduler/RNG/clock/boundary/mutation/sink design summary;
- new tests and exact results;
- pre-existing failures separated from new regressions;
- confirmation no existing simulator/FSR/calibration code changed;
- confirmation no real UFC mechanics were implemented;
- PR info if created;
- final `PHASE 1 GENERIC KERNEL GATE: PASS` or `FAIL`.

Next assistant action:

Independently review the Phase 1 implementation and tests. Do not authorize Phase 2A automatically. Revisit the deferred FSR-32 baseline first unless explicitly directed otherwise.

Phase 1 authorized now: **YES**.

Phase 2A authorized now: **NO**.
