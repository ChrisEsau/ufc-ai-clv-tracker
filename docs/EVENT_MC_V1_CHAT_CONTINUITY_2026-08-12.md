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

Architecture status: **Phase 0 architecture closed. EVENT MC V1 simulator implementation has not started.**

Operational status: **Codex execution plan for Phase 0 baseline materialization has been reviewed and approved. Codex is authorized to execute the frozen observational baseline task only. Phase 1 is NOT yet authorized.**

Canonical architecture document:

`docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Phase 0 closure record:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Locked implementation-interface decisions:

`docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`

Canonical numerical/reproducibility baseline contract:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Current Codex task prompt:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

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
- Keep FSR-32 connected initially.
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

# Phase 0 Closure Summary

Phase 0 architecture is closed at revision v0.3.

Resolved architecture decisions include:

- 14 current simulator classes connected by 13 inheritance edges;
- continuous-time event scheduler;
- events/sec rate units;
- exact interval-probability to per-second-rate conversion;
- one authoritative `fight_time_seconds` clock;
- engine-owned hard boundaries;
- exact continuous state advancement over `dt`;
- smaller physical `FightState` and separate `FightLedger` / stats accumulator;
- explicit dynamic modifier pipeline;
- explicit damage -> knockdown -> knockdown consequence -> finish ownership;
- reproducible judging with explicit RNG tie-break stream;
- one root path seed with centrally owned deterministic named RNG streams;
- event sinks / NONE-SUMMARY-FULL equivalent trace behavior;
- future cooldown / duration extension point;
- locked round-start reset;
- conditional ground-exit hazard partitioning;
- Phase 2A / 2B split;
- concrete baseline fixtures, seeds, metrics, output contract.

---

# Current Codex Task — Phase 0 Baseline Materialization

Codex must read and execute:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

This task is a **pre-implementation gate**.

Codex is allowed to add observational baseline/diagnostic orchestration if necessary, but must not implement EVENT MC V1 or alter current simulator mechanics/constants.

The canonical baseline contract is:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Required comparison coverage includes:

- Rob Font vs Raul Rosas Jr. anchor matchup;
- representative power / KO matchup;
- high-volume striking matchup;
- grappling / submission matchup;
- sustained wrestling / control matchup;
- deterministic single-path fixtures;
- 1000-path matchup summaries;
- deterministic mature historical audit cohort;
- winner/method/striking/TD/control/phase/submission/KD metrics;
- exact FSR-32 artifact checksum and run metadata.

The baseline is a **ruler, not a calibration target**.

---

# What Codex Must Return From Current Task

The assistant should expect Codex to report at minimum:

1. files created / changed;
2. exact commit SHA actually run;
3. exact commands run;
4. tests and results;
5. fixture resolution or deterministic replacements;
6. matchup baseline summaries;
7. cohort baseline summary;
8. manifest / checksum locations;
9. any mismatch against prior Font/Rosas or submission anchors and the identified reason;
10. confirmation that no EVENT MC V1 implementation was started and no current simulator mechanics/constants were changed;
11. explicit baseline gate PASS or FAIL.

---

# Assistant Review Required When Codex Returns

Do **not** automatically proceed to Phase 1 merely because Codex says PASS.

The assistant must independently review:

- git diff / changed files;
- whether any current simulator file was modified;
- whether FSR-32 remained the input source;
- manifest metadata and checksums;
- fixture set / any replacements;
- deterministic seed contract;
- path counts and historical cohort definition;
- output metrics and sanity against recorded Font/Rosas / submission evidence;
- test results;
- whether generated baseline artifacts are reproducible and located where the contract requires.

If the baseline task is clean, mark the operational Phase 0 gate PASS.

Only then prepare the next Codex prompt for **Phase 1 — generic continuous-time kernel**.

---

# Planned Phase 1 Scope After Baseline PASS

Phase 1 is generic kernel work only. No real UFC mechanics.

Expected Phase 1 implementation areas:

- package skeleton under `pipeline/simulation/event_mc_v1/`;
- typed base contracts/events/state;
- authoritative clock;
- generic exponential scheduler;
- events-per-second `EventRate` contract;
- zero-rate -> `(infinity, None)` behavior;
- hard-boundary handling in engine;
- continuous-state advancement hook;
- centrally owned root-seed RNG manager with named deterministic streams;
- event sinks / trace modes;
- state-delta application;
- synthetic round lifecycle and round-start distance reset;
- extension point for inactive `busy_until` / `cooldown_until` state.

Phase 1 mathematical/reproducibility tests must cover:

- exponential waiting-time mean;
- competing-event selection proportions;
- same root seed reproducibility;
- different seed divergence;
- named RNG-stream isolation;
- chronological event ordering;
- exact hard-boundary behavior;
- zero total rate handling;
- single-candidate behavior;
- no negative `dt`;
- no event past fight finish;
- round-start position reset;
- continuous `advance(dt)` ordering.

Stop after kernel tests pass. Do not port UFC action-rate formulas in the same Codex task.

---

# Checkpoint History

## Checkpoint 001 — 2026-08-12 22:33 America/Chicago

Current phase: Phase 0 operational baseline gate.

Architecture revision: v0.3.

Codex prompt issued:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

Purpose:

Materialize a deterministic, checksummed baseline from the untouched current simulator before any EVENT MC V1 code begins.

Next assistant action:

Review Codex's report and repository changes against the Phase 0 baseline contract. If and only if the gate is clean, prepare a tightly scoped Phase 1 kernel prompt and append Checkpoint 002 to this file.

Phase 1 authorized now: **NO**.

## Checkpoint 002 — 2026-08-12 22:37 America/Chicago

Current phase: Phase 0 operational baseline execution.

Codex returned a read-only implementation plan and requested approval before making changes.

Plan reviewed against canonical prompt and approved with one branch-resolution guardrail:

- local absence of `feature/fsr-32-stamina-shadow` is acceptable only if Codex fetches/resolves the exact configured remote branch;
- Codex must check out that exact branch and record the resulting commit SHA;
- if the remote branch itself does not exist or resolves to an unexpected lineage, Codex must stop and report rather than creating a substitute branch from another base.

Approved execution counts remain exactly:

```text
3 deterministic trace seeds per resolved fixture: 7, 17, 20260811
5 resolved matchup summaries × 1000 paths, root seed 20260811
first 200 eligible mature aligned bouts × 10 paths, root seed 20260810
full 1565-fight method/submission baseline where supported observationally
```

Codex may add observational baseline orchestration only if needed. No EVENT MC V1 simulator implementation is authorized.

Expected next return: full Phase 0 execution report ending in explicit PASS or FAIL.

Next assistant action: independently review Codex's actual branch/commit, diff, manifest/checksums, fixtures, outputs, tests, and gate result before authorizing Phase 1.

Phase 1 authorized now: **NO**.
