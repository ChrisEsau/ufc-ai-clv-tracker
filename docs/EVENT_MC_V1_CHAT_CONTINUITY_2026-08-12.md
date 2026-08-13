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
Operational status: **PHASE 0 OPERATIONAL BASELINE GATE: PASS.**

Implementation status: **Phase 1 generic continuous-time kernel has NOT started.** A prepared Phase 1 prompt exists in the repo, but the user has not instructed Codex to execute it. Phase 2A/2B remain unauthorized.

The frozen FSR-32 artifact was recovered from the temporary GitHub Release, verified byte-for-byte, placed at the expected ignored path, and used only to revalidate the existing frozen baseline. No EVENT MC V1 implementation work occurred during this validation.

Exact frozen FSR-32 identity:
```text
Original Codespace path:
/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet

Expected / verified SHA-256:
621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a

Release tag:
event-mc-v1-fsr32-handoff

Release asset:
fsr_32_prefight_snapshots.parquet

Release size:
3.20 MiB
```

Canonical docs:
- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_FSR32_FROZEN_ARTIFACT_IDENTITY_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md` — prepared only; NOT executed

---

# Phase 0 Operational Baseline — PASS

Codex revalidated the existing baseline with the exact frozen FSR-32 artifact.

Validated baseline structure:
- five frozen fixtures resolved with no substitutions or omissions;
- fifteen deterministic single-path traces: five fixtures × seeds `7`, `17`, `20260811`;
- five 1,000-path matchup summaries using root seed `20260811`;
- stable first-200-bout cohort with 10 paths/bout and root seed `20260810`;
- all recorded output SHA-256 values recomputed and matched manifest entries;
- full 1,565-fight method/submission anchor remains materialized with unchanged neutral `P(SUB | attempt) = 34%`.

Revalidated compact cohort results:
```text
Winner accuracy: 59.0%
Winner Brier score: 0.27745
KO/TKO rate: 25.0%
Submission rate: 17.1%
Decision rate: 57.9%
```

The matchup artifact retains winner, method, striking, takedown, control, phase, submission, reversal, knockdown, and Font/Rosas legacy wrestling-consumer metrics.

Safety confirmation from Codex:
- FSR-32 downloaded, not rebuilt;
- parquet copied without rewriting, normalization, recompression, or reserialization;
- parquet not committed;
- no simulator code modified;
- no FSR builder, rating, ontology, or maturity rule modified;
- no calibration constant retuned;
- no EVENT MC V1 implementation performed;
- Phase 2A and Phase 2B not started;
- verification produced no tracked changes;
- existing work remains in PR #64.

Validated checks included repo fetch/checkout, release download, source/destination SHA-256, byte comparison with `cmp`, git-ignore verification, baseline fixture/trace/cohort/path metadata checks, output checksum validation, and a final clean/synchronized git status.

Final gate:
`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

---

# Hard Locks

- Do not modify the current inheritance-based simulator while EVENT MC V1 is built.
- Keep FSR-32 as initial profile source when UFC mechanics are introduced.
- Do not rebuild FSR-32 or upstream FSR artifacts unless explicitly approved.
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
- Phase 1 must not start until the user explicitly authorizes it.
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

Migration split remains locked:
- Phase 2A: temporal/mechanical parity using the legacy blended consumer.
- Phase 2B: ontology correction so `wrestling_entry` directly drives intrinsic TD initiation.

---

# Phase 1 Status

Prepared prompt:
`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

It has **not been executed**.

When the user explicitly authorizes Phase 1, its scope is generic kernel infrastructure only:
- typed generic events/contracts;
- immutable timing configuration;
- authoritative fight clock;
- exponential competing-risk scheduler;
- rates in events/second;
- probability-to-rate conversion;
- stable named RNG streams;
- engine-owned hard boundaries;
- exact continuous `advance(dt)` ordering;
- typed state-delta application;
- round lifecycle/reset shell;
- null/stats/full-trace sinks;
- inactive cooldown/action-availability extension point;
- synthetic mathematical, temporal, reproducibility, mutation-ownership, and sink tests.

No real UFC mechanics in Phase 1.

---

# Assistant Review / Next Decision

Phase 0 is now accepted as operationally materialized and reproducible based on the Codex validation report and exact frozen-artifact identity.

Next step is **not automatic**. The user decides when to explicitly authorize Phase 1.

When Phase 1 is eventually returned, independently review:
- files/diff;
- current simulator and FSR code untouched;
- scheduler UFC-agnostic behavior;
- units and probability-to-rate math;
- named RNG derivation/independence;
- single authoritative clock and hard-boundary handling;
- exact `advance(dt)` ordering;
- engine-owned mutation;
- round-start reset;
- sink/RNG invariance;
- inactive cooldown extension point;
- tests/regressions;
- confirmation no real UFC mechanics slipped in.

Phase 2A remains unauthorized until Phase 1 is implemented and independently reviewed.

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
User said to move on and supply FSR later. A Phase 1 generic-kernel prompt was prepared, but the user did NOT launch it.

## 006 — 2026-08-12 23:08
User found exact frozen parquet in real Codespace and established SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## 007 — 2026-08-12 23:18
User published exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`. Release-ingest prompt issued.

## 008 — 2026-08-12 23:21
A Codex run started in a generic `work` checkout with no remote/auth and no governing docs. It changed nothing. Bootstrap recovery prompt issued.

## 009 — 2026-08-12 23:24
User clarified Phase 1 had never been launched. Continuity corrected: Phase 1 not started and not authorized for execution until explicit user approval.

## 010 — 2026-08-13 00:01 America/Chicago
Codex successfully restored the correct repository/branch context, downloaded the frozen FSR-32 release asset, verified the exact SHA-256, copied it byte-for-byte into the ignored expected path, and revalidated all frozen Phase 0 baseline artifacts/checksums.

Revalidated compact cohort:
```text
accuracy 59.0%
Brier 0.27745
KO/TKO 25.0%
SUB 17.1%
DEC 57.9%
```

No FSR rebuild, simulator modification, ontology/maturity change, calibration retune, or EVENT MC V1 implementation occurred.

Final result:
`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

Phase 1 implementation started: **NO**.
Phase 1 execution authorized now: **NO**.
Phase 2A authorized: **NO**.
