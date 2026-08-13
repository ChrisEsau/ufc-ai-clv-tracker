# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions. This file is not the architecture source of truth; canonical architecture and Phase 0 contracts are referenced below.

## Update rule
After every new Codex prompt, update this file. Preserve checkpoint history, current gate state, prompt path, hard locks, expected Codex return, and next assistant review.

---

# Current State

Architecture revision: **v0.3**
Architecture status: **Phase 0 architecture closed.**

Phase 0 operational baseline: **PASS.**

The exact frozen FSR-32 artifact was recovered from the temporary GitHub Release asset and verified byte-for-byte. Frozen SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Revalidated baseline scope:
- five frozen fixtures with no substitutions/omissions;
- 15 deterministic traces (5 fixtures x seeds 7, 17, 20260811);
- five 1,000-path matchup summaries using root seed 20260811;
- stable first-200-bout cohort with 10 paths/bout and root seed 20260810;
- full 1,565-fight method/submission anchor with neutral submission probability 34%;
- all recorded output SHA-256 values matched manifest entries.

Revalidated compact cohort results:
- winner accuracy: 59.0%;
- winner Brier: 0.27745;
- KO/TKO: 25.0%;
- SUB: 17.1%;
- DEC: 57.9%.

Safety confirmation from Phase 0 PASS:
- FSR-32 downloaded, not rebuilt;
- parquet copied without transformation and not committed;
- current simulator untouched;
- FSR builders/ratings/ontology/maturity rules untouched;
- no calibration retuning;
- no EVENT MC V1 implementation during Phase 0;
- Phase 2A/2B not started.

Implementation status: **The user has now explicitly authorized Phase 1 generic continuous-time kernel implementation. Phase 1 has not yet been reviewed or passed. Phase 2A/2B remain NOT AUTHORIZED.**

Current Codex execution prompt:
`docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`

Detailed Phase 1 implementation specification:
`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

Important: the prepared Phase 1 specification contains stale wording saying Phase 0 baseline was deferred. The 2026-08-13 execution prompt overrides only those status statements. Phase 0 is now PASS; all kernel contracts/non-goals/tests in the prepared specification remain in force.

Canonical docs:
- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_FSR32_FROZEN_ARTIFACT_IDENTITY_2026-08-12.md`

---

# Hard Locks

- Do not modify the current inheritance-based simulator.
- Keep FSR-32 as initial profile source when UFC mechanics are introduced.
- Do not rebuild FSR-32 or upstream FSR artifacts.
- Do not alter or commit the frozen parquet.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, KD, or other calibration constants during kernel/parity work.
- Use composition, not a new inheritance chain.
- One authoritative fight clock: `FightState.fight_time_seconds`.
- Scheduler rates are events/second.
- Only engine advances time; continuous state advances exact elapsed `dt` before event resolution.
- Stats/ledger are observers, not hidden communication.
- Components do not create hidden RNGs.
- Engine owns authoritative state mutation through typed results/deltas.
- Round start resets phase to `DISTANCE` and clears positional ownership.
- `MatReturn` remains future/optional.
- Do not combine timing migration, wrestling ontology correction, and calibration changes.
- Phase 1 is generic infrastructure only; no real UFC mechanics.
- Phase 2A and Phase 2B remain unauthorized.

---

# Locked Wrestling Ontology

```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention
control_imposition   = persistence/behavior after control is established
```

Current V0 blended consumer remains approximately:
```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Migration split:
- Phase 2A: temporal/mechanical parity using the legacy blended consumer.
- Phase 2B: ontology correction so `wrestling_entry` directly drives intrinsic TD initiation.

---

# Current Phase 1 Scope

Phase 1 is generic kernel infrastructure only:
- typed generic events/contracts;
- immutable timing configuration;
- authoritative `FightState.fight_time_seconds` clock;
- generic exponential competing-risk scheduler;
- all rates expressed in events/second;
- exact interval-probability -> per-second rate conversion;
- centrally owned root-seed RNG manager with stable named streams;
- engine-owned hard round/fight boundaries;
- exact continuous `advance(dt)` hook before event resolution;
- engine-owned typed state-delta application;
- round lifecycle and locked round-start positional reset;
- null/stats/full-trace event sinks with physics/RNG invariance;
- inactive action-availability/cooldown extension point;
- synthetic mathematical, temporal, reproducibility, mutation-ownership, and sink tests.

Absolute Phase 1 non-goals:
- no real striking formulas;
- no takedown formulas;
- no wrestling-entry implementation;
- no clinch/ground mechanics;
- no submissions;
- no stamina;
- no damage/KD/KO;
- no recovery;
- no age transforms;
- no judging mechanics beyond reserving its RNG stream;
- no FSR loading/rebuilding;
- no historical tuning;
- no Phase 2A/2B implementation.

---

# Assistant Review Required When Phase 1 Returns

Do not automatically accept Codex PASS. Independently review:
- exact repo/branch/start/final SHAs;
- files and diff;
- current simulator/FSR/calibration untouched;
- scheduler remains UFC-agnostic;
- rate units and probability-to-rate math;
- stable named RNG stream derivation and independence;
- one authoritative clock and exact boundary handling;
- continuous advancement ordering;
- engine-owned mutation/delta boundary;
- round-start positional reset;
- sink invariance and RNG invariance;
- action-availability extension point remains inactive/no arbitrary constants;
- test quality/results;
- no real UFC mechanics or Phase 2 work slipped in;
- PR/commit state.

Only after independent review may ChatGPT mark:
`PHASE 1 GENERIC KERNEL GATE: PASS`

Phase 2A remains unauthorized until that review is complete.

---

# Checkpoint History

## 001 — 2026-08-12 22:33 America/Chicago
Phase 0 operational baseline prompt issued. Phase 1 unauthorized.

## 002 — 2026-08-12 22:37
Codex baseline execution plan approved with exact-remote-branch guardrail.

## 003 — 2026-08-12 22:40
Codex stopped because isolated checkout had no Git remote. Remote-unblock prompt issued.

## 004 — 2026-08-12 22:49
Codex repaired Git environment and verified feature branch, but baseline failed because frozen FSR-32 parquet was absent. Artifact recovery issued; no rebuild authorized.

## 005 — 2026-08-12 22:59
User said to move on and supply FSR later. A Phase 1 generic-kernel prompt was prepared in the repo, but the user did NOT launch it.

## 006 — 2026-08-12 23:08
User found exact frozen parquet in real Codespace and established SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## 007 — 2026-08-12 23:18
User published exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`. Release-ingest prompt issued.

## 008 — 2026-08-12 23:21
A Codex run started in a generic `work` checkout with no remote/auth and could not access governing docs/release. It changed nothing. Bootstrap prompt issued.

## 009 — 2026-08-12 23:24
User clarified Phase 1 had never been launched. Continuity corrected: current work remained Phase 0 only.

## 010 — 2026-08-13 near 00:00 America/Chicago
Phase 0 operational baseline returned PASS after exact artifact recovery and idempotent revalidation.

Verified:
- exact FSR-32 release asset downloaded and destination SHA matched frozen SHA;
- byte identity confirmed with `cmp`;
- artifact remained gitignored/uncommitted;
- five fixtures, 15 traces, five 1,000-path summaries, 200x10 cohort, and full 1,565-fight method/submission anchor revalidated;
- all manifest output checksums matched;
- compact cohort: accuracy 59.0%, Brier 0.27745, KO/TKO 25.0%, SUB 17.1%, DEC 57.9%;
- no simulator/FSR/calibration changes;
- no EVENT MC V1 implementation performed.

Final gate:
`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

## 011 — 2026-08-13 00:03 America/Chicago
User said **Proceed**, explicitly authorizing Phase 1 implementation.

New authoritative execution prompt:
`docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`

This wrapper corrects stale deferred-baseline wording in the prepared Phase 1 spec and authorizes execution of the generic kernel only.

Expected Codex return: complete Phase 1 implementation/test report ending in `PHASE 1 GENERIC KERNEL GATE: PASS` or `FAIL`.

Next assistant action: independently review Phase 1 code/tests before any Phase 2 authorization.

Phase 1 execution authorized: **YES**.
Phase 1 reviewed/passed: **NO**.
Phase 2A authorized: **NO**.
Phase 2B authorized: **NO**.
