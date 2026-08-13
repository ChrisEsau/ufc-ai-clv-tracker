# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions. This file is not the architecture source of truth; canonical architecture and phase contracts are referenced below.

## Update rule
After every new Codex prompt, update this file. Preserve checkpoint history, current gate state, prompt path, hard locks, expected Codex return, and next assistant review.

---

# Current State

Architecture revision: **v0.3**
Architecture status: **Phase 0 architecture closed.**

Phase 0 operational baseline: **PASS.**
Phase 1 generic continuous-time kernel: **PASS after independent ChatGPT review.**
Phase 2A distance temporal/mechanical parity: **AUTHORIZED and current task; not yet passed.**
Phase 2B: **NOT AUTHORIZED.**

Frozen FSR-32 path:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Frozen Phase 0 compact baseline:
- winner accuracy 59.0%;
- winner Brier 0.27745;
- KO/TKO 25.0%;
- SUB 17.1%;
- DEC 57.9%;
- 5 frozen fixtures;
- 15 deterministic traces;
- five 1,000-path matchup summaries;
- first 200 eligible bouts x 10 paths;
- full 1,565-fight method/submission anchor;
- recorded artifact checksums revalidated.

Phase 1 implementation commit:
`1debecab69a141bf2f81179f3436af569733b750`

Phase 1 verified:
- one authoritative `FightState.fight_time_seconds` clock;
- immutable timing config and derived clock views;
- UFC-agnostic competing-risk scheduler;
- rates in events/second;
- exact probability-to-rate conversion;
- stable named RNG streams via `SeedSequence([root_seed, stable_stream_id])`;
- hard-boundary truncation and round reset;
- continuous advancement before event resolution;
- engine-owned immutable state deltas;
- null/stats/full-trace observer sinks;
- inactive action-availability extension point;
- 24 Phase 1 tests + 30 existing simulator regression tests passed.

Known non-blocking future seam:
Primary resolution currently returns one state delta plus consequence notifications. Later damage -> KD -> finish chains will likely need sequential consequence modules that can each return/apply their own deltas. Do not retrofit this prematurely.

Development standard locked by user:
**Working + predictive + modular + easy to iterate. Do not optimize for perfection or exhaustive defensive hardening. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Damage/KO external review workstream may occur in parallel. Do not hold up EVENT MC progress for it. Only preserve replaceable subsystem seams so a later damage/KD/KO design can be swapped in cleanly.

Current governing Phase 2A prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`

Current Phase 2A retry/bootstrap prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2A_RETRY_2026-08-13.md`

The first Phase 2A Codex attempt stopped safely at stale commit `1debecab69a141bf2f81179f3436af569733b750` because it had not fetched the later documentation commit containing the governing prompt. No Phase 2A implementation occurred. The prompt is now verified present on the remote feature branch. Retry requires fetching/pulling the latest feature branch before execution.

Canonical docs:
- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2A_RETRY_2026-08-13.md`

---

# Hard Locks

- Do not modify the current inheritance-based simulator.
- Keep FSR-32 as initial profile source.
- Do not rebuild/rewrite the frozen FSR-32 parquet.
- Do not alter FSR ratings/builders/ontology/maturity/leakage rules.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, KD, or other calibration constants during parity work.
- Use composition, not a new inheritance chain.
- One authoritative fight clock.
- Scheduler rates remain events/second.
- Only engine advances time; continuous state advances exact elapsed `dt` before event resolution.
- Stats/ledger/sinks are observers, not hidden communication.
- Components do not create hidden RNGs.
- Engine owns authoritative state mutation.
- Round start resets phase to DISTANCE and clears positional ownership.
- `MatReturn` remains future/optional.
- Do not combine timing migration, wrestling ontology correction, and calibration changes.
- Phase 2B remains unauthorized.
- Keep future damage/KD/KO mechanics replaceable behind clean modular interfaces; do not block current progress waiting for their redesign.

---

# Wrestling Ontology / Migration Split

Correct ontology:
```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention
control_imposition   = persistence after advantageous position
```

Legacy Phase 2A consumer approximately:
```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Migration split:
- Phase 2A: preserve legacy blended consumer/current formulas; change timing architecture only.
- Phase 2B: deliberately correct TD initiation semantics so `wrestling_entry` drives intrinsic base TD attempts and context applies separately.

---

# Current Phase 2A Scope

DISTANCE primary action families only:
1. red strike attempt;
2. blue strike attempt;
3. red takedown attempt;
4. blue takedown attempt;
5. red clinch-entry attempt;
6. blue clinch-entry attempt.

Port current effective logic for strike attempt frequency/hit-miss, TD attempt frequency using legacy blended consumer, TD landed/failed using current conversion vs defense logic, clinch-entry frequency/success, and phase transitions after successful TD/clinch entry.

No stamina, damage, KD, KO/TKO, submissions, recovery, age, judging, ground internals, clinch internals, tactical urgency, MatReturn, or ontology correction.

Migration rule:
```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```
Preserve final legacy interval probability first, then convert exactly. Do not reinterpret probabilities directly as rates or invent new bases.

Validation classifications:
1. exact formula mismatch = bug;
2. unit/conversion mismatch = bug;
3. continuous competing-hazard effect = expected architecture difference;
4. discrete multi-attempt-per-segment vs one-at-a-time events = expected/modeling decision;
5. missing downstream phase mechanics = out of scope;
6. wrestling semantic correction = defer to Phase 2B.

Required real-fixture diagnostics include Font/Rosas, Lewis/Daukaus, Holloway/Kattar, and Merab/Yan where inputs resolve cleanly. Font/Rosas is non-replaceable for wrestling stress audit.

---

# Assistant Review Required When Phase 2A Returns

Independently review actual commit/diff. Verify:
- legacy formulas/functions were traced rather than approximated from memory;
- strike attempt/landing mappings preserve current mechanics;
- TD attempt mapping uses legacy blended consumer, not Phase 2B semantics;
- TD success remains conversion vs defense;
- clinch entry maps current effective logic;
- interval probabilities use correct interval units and exact hazard conversion;
- Phase 1 scheduler/clock/RNG invariants remain intact;
- current simulator/FSR/calibration untouched;
- no downstream physiology/finish mechanics slipped in;
- formula-level parity tests meaningful;
- matched-state MC differences classified rather than tuned away;
- real-fixture diagnostics interpretable, especially Font/Rosas;
- implementation remains small/modular and not over-engineered.

Only after independent review may ChatGPT mark:
`PHASE 2A DISTANCE TEMPORAL PARITY GATE: PASS`.

Do not authorize Phase 2B automatically.

---

# Checkpoint History

## 001 — 2026-08-12 22:33 America/Chicago
Phase 0 operational baseline prompt issued.

## 002 — 2026-08-12 22:37
Codex baseline execution plan approved with exact-remote-branch guardrail.

## 003 — 2026-08-12 22:40
Codex stopped because isolated checkout had no Git remote. Remote-unblock prompt issued.

## 004 — 2026-08-12 22:49
Codex repaired Git environment but baseline failed because frozen FSR-32 parquet was absent. Artifact recovery issued; no rebuild authorized.

## 005 — 2026-08-12 22:59
User said to move on and supply FSR later. Phase 1 prompt prepared but not launched.

## 006 — 2026-08-12 23:08
User found exact frozen parquet and established SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## 007 — 2026-08-12 23:18
User published exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`.

## 008 — 2026-08-12 23:21
Codex generic workspace lacked remote/auth; changed nothing. Bootstrap prompt issued.

## 009 — 2026-08-12 23:24
User clarified Phase 1 had never launched. Continuity corrected.

## 010 — 2026-08-13 near 00:00 America/Chicago
Phase 0 returned PASS after exact artifact recovery/revalidation.
Final gate: `PHASE 0 OPERATIONAL BASELINE GATE: PASS`.

## 011 — 2026-08-13 00:03
User explicitly authorized Phase 1.

## 012 — 2026-08-13 about 00:15
Codex returned Phase 1 commit `1debecab69a141bf2f81179f3436af569733b750`; ChatGPT independently reviewed and accepted.
Final reviewed gate: `PHASE 1 GENERIC KERNEL GATE: PASS`.

## 013 — 2026-08-13 00:18
User explicitly authorized Phase 2A. Governing prompt created:
`docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`.

## 014 — 2026-08-13 00:27 America/Chicago
First Phase 2A Codex attempt stopped safely before implementation because its checkout remained at Phase 1 commit `1debecab69a141bf2f81179f3436af569733b750` and therefore did not contain the later governing prompt. No files/code/artifacts changed. This was a stale-checkout/document-sync issue, not a Phase 2A logic failure.

Verified afterward that the governing prompt is present on remote `feature/fsr-32-stamina-shadow`.

New retry/bootstrap prompt added on the feature branch:
`docs/EVENT_MC_V1_CODEX_PHASE2A_RETRY_2026-08-13.md`

Retry prompt commit:
`38fba6af0a77a7b93183f2fcdd7fec406e387fcf`

Required next Codex action: fetch/pull latest feature branch, verify the governing prompt exists, then execute Phase 2A exactly. No Phase 2B or downstream mechanics.

Phase 0: **PASS**.
Phase 1: **PASS**.
Phase 2A authorized: **YES**.
Phase 2A reviewed/passed: **NO**.
Phase 2B authorized: **NO**.
