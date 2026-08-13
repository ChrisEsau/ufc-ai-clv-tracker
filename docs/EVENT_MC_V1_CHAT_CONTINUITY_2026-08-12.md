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

Operational status: **Codex hit an environmental blocker because its local checkout had no configured Git remote. A dedicated remote-unblock prompt has been issued. Codex is authorized to add the known `origin`, fetch and verify the exact `feature/fsr-32-stamina-shadow` branch, re-read current source-of-truth docs, and then resume the already-approved Phase 0 baseline materialization task. Phase 1 is NOT yet authorized.**

Canonical architecture document:

`docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Phase 0 closure record:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Locked implementation-interface decisions:

`docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`

Canonical numerical/reproducibility baseline contract:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Current Codex task prompt:

`docs/EVENT_MC_V1_CODEX_PHASE0_REMOTE_UNBLOCK_2026-08-12.md`

After branch verification, Codex must resume:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

Verified GitHub branch head at unblock-prompt issuance:

`6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6`

Known repository remote URL:

`https://github.com/ChrisEsau/ufc-ai-clv-tracker.git`

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

# Current Codex Task — Phase 0 Remote Unblock + Baseline Materialization

Codex must first read and execute:

`docs/EVENT_MC_V1_CODEX_PHASE0_REMOTE_UNBLOCK_2026-08-12.md`

This task exists only to repair the Git environment enough to verify the exact requested feature branch.

Known remote:

`https://github.com/ChrisEsau/ufc-ai-clv-tracker.git`

Verified feature branch at issuance:

`feature/fsr-32-stamina-shadow`

Verified head at issuance:

`6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6`

Codex reported its local environment before the unblock prompt as:

```text
local branch: work
local SHA: eebb8134fd2f7cb4eb68ffcb93464ab74883633f
configured remotes: none
working tree: clean
Python: 3.14.4
```

Codex stopped correctly under the earlier branch guardrail.

After successful remote configuration/fetch/lineage verification, Codex must re-read the current Phase 0 docs from the fetched branch and then resume:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

without asking for another plan approval unless a new scope/architecture conflict appears.

This remains a **pre-implementation gate**. No EVENT MC V1 code is authorized.

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

1. successful remote configuration and exact fetched branch SHA/lineage verification;
2. files created / changed;
3. exact commit SHA actually run;
4. exact commands run;
5. tests and results;
6. fixture resolution or deterministic replacements;
7. matchup baseline summaries;
8. cohort baseline summary;
9. manifest / checksum locations;
10. any mismatch against prior Font/Rosas or submission anchors and the identified reason;
11. confirmation that no EVENT MC V1 implementation was started and no current simulator mechanics/constants were changed;
12. explicit baseline gate PASS or FAIL.

If remote/authentication/lineage verification fails again, Codex should stop before materialization and report the blocker.

---

# Assistant Review Required When Codex Returns

Do **not** automatically proceed to Phase 1 merely because Codex says PASS.

The assistant must independently review:

- exact fetched branch and lineage;
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

## Checkpoint 003 — 2026-08-12 22:40 America/Chicago

Current phase: Phase 0 environmental unblock before operational baseline capture.

Codex returned:

`PHASE 0 OPERATIONAL BASELINE GATE: FAIL`

Reason: local checkout had **no configured Git remotes**, so the exact `feature/fsr-32-stamina-shadow` remote branch could not be fetched/verified. Codex made no repository changes and did not run baseline simulations.

Independent GitHub verification by the assistant confirmed:

```text
repository: ChrisEsau/ufc-ai-clv-tracker
remote URL: https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
branch: feature/fsr-32-stamina-shadow
verified head: 6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6
local SHA reported by Codex: eebb8134fd2f7cb4eb68ffcb93464ab74883633f
```

The local SHA is the parent of the verified remote head.

New Codex prompt issued:

`docs/EVENT_MC_V1_CODEX_PHASE0_REMOTE_UNBLOCK_2026-08-12.md`

Purpose:

- add/verify the known `origin` remote;
- fetch the exact feature branch;
- verify head/lineage;
- switch to the exact feature branch;
- re-read current Phase 0 source-of-truth docs;
- resume the already-approved baseline-materialization prompt without another plan-approval round if verification succeeds.

Hard non-goal: still no EVENT MC V1 implementation and no simulator/FSR/calibration changes.

Expected next assistant action:

Review Codex's remote verification and full baseline result if it proceeds. If remote verification fails again, diagnose only that blocker. Phase 1 remains unauthorized.

Phase 1 authorized now: **NO**.
