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

Frozen FSR-32:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Revalidated frozen baseline:
- 5 frozen fixtures;
- 15 deterministic traces (seeds 7, 17, 20260811);
- 5 matchup summaries at 1,000 paths, root seed 20260811;
- first 200 eligible bouts x 10 paths, root seed 20260810;
- full 1,565-fight method/submission anchor;
- all recorded output SHA-256 values matched manifest entries;
- compact cohort: winner accuracy 59.0%, Brier 0.27745, KO/TKO 25.0%, SUB 17.1%, DEC 57.9%.

Phase 1 generic continuous-time kernel: **PASS after independent ChatGPT review.**

Phase 1 implementation commit:
`1debecab69a141bf2f81179f3436af569733b750`

Phase 1 verified:
- one authoritative `FightState.fight_time_seconds` clock;
- immutable timing config and derived clock views;
- generic UFC-agnostic exponential competing-risk scheduler;
- all rates in events/second;
- exact probability -> rate conversion;
- stable named RNG streams via `SeedSequence([root_seed, stable_stream_id])`;
- exact hard-boundary truncation and round reset;
- continuous advancement before event resolution;
- engine-owned immutable state deltas;
- null/stats/full-trace observer sinks;
- inactive action-availability extension point;
- 24 Phase 1 tests passed;
- 30 existing simulator regression tests passed;
- no current simulator, FSR, ontology, or calibration changes.

Known non-blocking future seam from Phase 1 review:
Primary resolution currently returns one state delta plus consequence notifications. Later damage -> KD -> finish chains will likely need sequential consequence modules that can each return/apply their own deltas. Do not retrofit this prematurely; it is not a Phase 2A blocker.

Development standard locked by user:
**Do not optimize for perfection or exhaustive defensive hardening. Build a working, measurable, modular simulator that is easy to iterate. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Phase 2A distance temporal/mechanical parity: **AUTHORIZED and current task.**

Current Codex prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`

Phase 2B: **NOT AUTHORIZED.**

Canonical docs:
- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`

---

# Hard Locks

- Do not modify the current inheritance-based simulator.
- Keep FSR-32 as initial profile source.
- Do not rebuild or rewrite the frozen FSR-32 parquet.
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

---

# Locked Wrestling Ontology and Migration Split

Correct ontology:
```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention
control_imposition   = persistence after advantageous position
```

Legacy current-simulator consumer used for Phase 2A parity approximately:
```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Migration split:
- **Phase 2A:** preserve the legacy blended consumer and current effective formulas; change timing architecture only.
- **Phase 2B:** deliberately correct TD initiation semantics so `wrestling_entry` drives intrinsic base TD attempt rate and context applies separately.

Do not hide Phase 2B semantics inside Phase 2A.

---

# Current Phase 2A Scope

Goal: place the first real UFC mechanics on the continuous-time engine while preserving current semantics/calibration so differences are attributable to temporal architecture.

Primary DISTANCE action families only:
1. red strike attempt;
2. blue strike attempt;
3. red takedown attempt;
4. blue takedown attempt;
5. red clinch-entry attempt;
6. blue clinch-entry attempt.

Port current effective logic for:
- strike attempt frequency;
- strike hit/miss;
- TD attempt frequency using legacy blended wrestling consumer;
- TD landed/failed using current wrestling_conversion vs td_defense logic;
- clinch-entry frequency/success;
- phase transition after successful TD/clinch entry.

No stamina, damage, KD, KO/TKO, submissions, recovery, age, judging, ground internals, clinch internals, tactical urgency, MatReturn, or ontology correction.

Migration rule:
For any current final interval probability, preserve that final probability then convert exactly:
```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```
Do not reinterpret interval probability directly as a rate and do not invent new bases.

Phase 2A validation must distinguish:
1. exact formula mismatch = bug;
2. unit/conversion mismatch = bug;
3. continuous competing-hazard effect = expected architecture difference;
4. discrete multi-attempt-per-segment vs one-at-a-time events = expected/modeling decision;
5. missing downstream phase mechanics = out of scope;
6. wrestling semantic correction = defer to Phase 2B.

Required real-fixture diagnostics include Font/Rosas, Lewis/Daukaus, Holloway/Kattar, and Merab/Yan where FSR inputs resolve cleanly. Font/Rosas is non-replaceable for the wrestling stress audit.

---

# Assistant Review Required When Phase 2A Returns

Independently review the actual commit/diff, not only Codex's PASS line.

Verify:
- exact current legacy source formulas/functions traced rather than approximated from memory;
- strike attempt and landing mappings preserve current mechanics;
- TD attempt mapping uses the legacy blended consumer, not corrected Phase 2B semantics;
- TD success remains conversion vs defense;
- clinch entry maps current effective logic;
- every legacy interval probability is converted with correct interval units;
- scheduler/clock/RNG Phase 1 invariants remain intact;
- current simulator/FSR/calibration remain untouched;
- no downstream physiology/finish mechanics slipped in;
- formula-level parity tests are meaningful;
- matched-state Monte Carlo comparisons classify differences rather than tuning them away;
- frozen real-fixture diagnostics are interpretable, especially Font/Rosas;
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
User said to move on and supply FSR later. Phase 1 prompt was prepared but not launched.

## 006 — 2026-08-12 23:08
User found exact frozen parquet in real Codespace and established SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## 007 — 2026-08-12 23:18
User published exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`.

## 008 — 2026-08-12 23:21
A Codex run started in a generic `work` checkout with no remote/auth and changed nothing. Bootstrap prompt issued.

## 009 — 2026-08-12 23:24
User clarified Phase 1 had never been launched. Continuity corrected.

## 010 — 2026-08-13 near 00:00 America/Chicago
Phase 0 operational baseline returned PASS after exact artifact recovery and idempotent revalidation.

Final gate:
`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

## 011 — 2026-08-13 00:03 America/Chicago
User explicitly authorized Phase 1. Execution prompt issued.

## 012 — 2026-08-13 about 00:15 America/Chicago
Codex returned Phase 1 implementation commit `1debecab69a141bf2f81179f3436af569733b750` with 24 Phase 1 tests passing and 30 existing simulator regression tests passing. ChatGPT independently inspected the commit, engine/state/sinks/tests and accepted the gate.

Final reviewed gate:
`PHASE 1 GENERIC KERNEL GATE: PASS`

Non-blocking future seam recorded: later sequential damage/KD/finish consequences may need consequence-owned deltas; do not retrofit until needed.

User reiterated development philosophy: working, flexible, measurable, iterative architecture over perfection; predictive moneyline/prop performance is the ultimate target.

## 013 — 2026-08-13 00:18 America/Chicago
User said **Proceed**, explicitly authorizing Phase 2A distance temporal/mechanical parity.

New Codex prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`

Prompt commit:
`9e2d560c767de90594ecaefca5b19ac6234e74fd`

Purpose:
- port only DISTANCE strike attempts/hit-miss, legacy-blended TD attempts/success-failure, and clinch entry;
- preserve current effective formulas/semantics and convert final legacy interval probabilities exactly to per-second hazards;
- compare old/new distributionally without tuning;
- use frozen FSR-32 for real fixture diagnostics;
- keep Phase 2B wrestling correction and all downstream mechanics out of scope.

Expected Codex return ends in:
`PHASE 2A DISTANCE TEMPORAL PARITY GATE: PASS` or `FAIL`.

Next assistant action: independently review Phase 2A code, formula tracing, parity diagnostics, and fixture results before any Phase 2B authorization.

Phase 0: **PASS**.
Phase 1: **PASS**.
Phase 2A execution authorized: **YES**.
Phase 2A reviewed/passed: **NO**.
Phase 2B authorized: **NO**.
