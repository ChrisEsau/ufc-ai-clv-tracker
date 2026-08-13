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

Operational status: **Codex successfully repaired the Git checkout and verified the exact feature branch, but the Phase 0 operational baseline gate failed because the frozen generated FSR-32 parquet is absent from the isolated checkout. The current task is exact artifact recovery only. Rebuilding the FSR chain is NOT authorized. Phase 1 is NOT authorized.**

Canonical architecture document:

`docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Phase 0 closure record:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Locked implementation-interface decisions:

`docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`

Canonical numerical/reproducibility baseline contract:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Current Codex task prompt:

`docs/EVENT_MC_V1_CODEX_PHASE0_FSR32_ARTIFACT_RECOVERY_2026-08-12.md`

If exact artifact recovery succeeds, Codex must resume:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

Required frozen artifact:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Verified branch from latest Codex report:

`feature/fsr-32-stamina-shadow @ 937ce6e98e15143ddf8b676346a9f0362126b809`

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

# Current Codex Task — FSR-32 Artifact Recovery

Codex must read and execute:

`docs/EVENT_MC_V1_CODEX_PHASE0_FSR32_ARTIFACT_RECOVERY_2026-08-12.md`

Purpose:

Recover an exact pre-existing copy (or byte-identical backup) of the frozen FSR-32 parquet used by the current simulator research baseline.

Search local/mounted workspaces, sibling worktrees, persistent mounts, backups, caches, and any already-downloaded run artifacts visible to the environment.

For every candidate, record path, size, mtime, SHA-256, parquet shape/schema, and date coverage. If a credible candidate is found, copy it without transformation into the expected ignored path and verify source/destination byte identity.

Do **not** rebuild FSR-32 or any upstream FSR generation in this task.

If no exact artifact is found, stop and report `FSR-32 ARTIFACT RECOVERY: NOT FOUND`; the user/assistant will then decide whether to recover the artifact from another machine/Codespace or approve a controlled reconstruction with a new baseline identity.

The two age-contract test failures found by Codex are pre-existing branch test debt and are not part of this recovery task.

---

# What Codex Must Return From Current Task

If found:

- `FSR-32 ARTIFACT RECOVERY: FOUND`;
- source absolute path;
- destination path;
- SHA-256;
- row/column counts and date coverage;
- byte-identical source/destination confirmation;
- then the complete resumed Phase 0 baseline PASS/FAIL report.

If not found:

- `FSR-32 ARTIFACT RECOVERY: NOT FOUND`;
- all roots searched;
- all partial/older candidate artifacts found;
- no rebuild performed.

---

# Assistant Review Required When Codex Returns

If recovery succeeds, independently review artifact identity metadata and the resumed Phase 0 baseline outputs before authorizing Phase 1.

If recovery fails, do not authorize a blind rebuild. Determine whether the frozen parquet can be copied/uploaded from the original development Codespace/local environment. Only if exact recovery is impossible should a separate controlled-reconstruction decision be considered, with a newly identified baseline artifact and documented loss of byte-for-byte historical identity.

Phase 1 remains unauthorized.

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

Stop after kernel tests pass. Do not port UFC action-rate formulas in the same Codex task.

---

# Checkpoint History

## Checkpoint 001 — 2026-08-12 22:33 America/Chicago

Phase 0 operational baseline prompt issued. Phase 1 unauthorized.

## Checkpoint 002 — 2026-08-12 22:37 America/Chicago

Codex execution plan approved with exact-remote-branch guardrail. Phase 1 unauthorized.

## Checkpoint 003 — 2026-08-12 22:40 America/Chicago

Codex stopped because no Git remote existed. Remote-unblock prompt issued. Phase 1 unauthorized.

## Checkpoint 004 — 2026-08-12 22:49 America/Chicago

Current phase: Phase 0 artifact recovery before operational baseline capture.

Codex successfully repaired the checkout and verified:

```text
remote: https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
branch: feature/fsr-32-stamina-shadow
SHA: 937ce6e98e15143ddf8b676346a9f0362126b809
lineage ancestor 6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6: PASS
working tree: clean
Python: 3.14.4
```

Phase 0 baseline execution then failed because the required ignored/generated artifact was absent:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Its immediate FSR-28 upstream artifact and older generated snapshot chain were also absent. No baseline simulations or artifacts were fabricated. No current simulator/FSR/calibration changes were made.

Existing tests observed independently of any changes:

```text
34 passed
2 pre-existing age-contract failures
```

Decision: **do not rebuild the FSR chain yet** because that would no longer verify the exact frozen input used by prior research and there is no historical checksum to prove equivalence.

New Codex prompt issued:

`docs/EVENT_MC_V1_CODEX_PHASE0_FSR32_ARTIFACT_RECOVERY_2026-08-12.md`

Purpose: search visible local/mounted storage for the exact existing FSR-32 parquet or byte-identical backup, validate candidates, and resume baseline materialization automatically if found.

Expected next assistant action: review artifact-recovery report. If FOUND, verify identity and resumed baseline. If NOT FOUND, determine whether the original development Codespace/local environment can supply the frozen parquet before considering controlled reconstruction.

Phase 1 authorized now: **NO**.
