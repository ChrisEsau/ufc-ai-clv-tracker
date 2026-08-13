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

Implementation status: **Phase 1 generic continuous-time kernel has NOT started. A Phase 1 prompt document exists in the repo, but the user has not instructed Codex to execute it. Phase 2A/2B are not authorized.**

Phase 0 operational baseline: **NOT YET PASSED.** The exact frozen FSR-32 artifact exists and has been published as a temporary GitHub Release asset. The current task is still to restore Codex repo access, ingest that exact artifact, and complete the frozen Phase 0 baseline before any Phase 1 implementation begins.

Latest Codex bootstrap prompt:
`docs/EVENT_MC_V1_CODEX_BOOTSTRAP_RELEASE_BASELINE_2026-08-12.md`

This prompt must first restore the correct remote/branch, then read and execute:
`docs/EVENT_MC_V1_CODEX_FSR32_RELEASE_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

Exact repository:
`ChrisEsau/ufc-ai-clv-tracker`

Important typo correction from failed Codex diagnostic:
`ChrisEsau` is correct; `ChrisEsasu` is wrong.

Exact frozen FSR-32 identity:
```text
Original Codespace path:
/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet

Expected SHA-256:
621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a

Release tag:
event-mc-v1-fsr32-handoff

Release asset:
fsr_32_prefight_snapshots.parquet

GitHub release-reported size:
3.20 MiB
```

Canonical docs:
- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_FSR32_FROZEN_ARTIFACT_IDENTITY_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md` (prepared only; NOT executed)

---

# Hard Locks

- Do not modify the current inheritance-based simulator.
- Keep FSR-32 as initial profile source when UFC mechanics are introduced.
- Do not rebuild FSR-32 or upstream FSR artifacts now that the exact frozen artifact is known.
- Do not alter or commit the parquet.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, KD, or other calibration constants during baseline/kernel/parity work.
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
- Phase 1 implementation is not currently authorized for execution.
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

# Phase 1 Status

A detailed Phase 1 generic-kernel prompt has been prepared at:
`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

However, the user has **not** told Codex to execute it. Therefore:

- no Phase 1 implementation should be assumed to exist;
- no Phase 1 tests should be assumed to have run;
- no Phase 1 commit/PR should be assumed to exist;
- the next Codex work remains Phase 0 artifact ingestion/baseline recovery only.

When the user later explicitly authorizes Phase 1, its scope remains generic kernel infrastructure only: typed events/contracts, immutable timing config, authoritative clock, exponential scheduler, rates/sec, probability-to-rate conversion, named RNG streams, hard boundaries, exact `advance(dt)` ordering, typed state deltas, round lifecycle/reset shell, sinks, inactive cooldown extension point, and synthetic tests. No real UFC mechanics.

---

# Phase 0 Frozen Baseline Recovery

Original Phase 0 execution failed only because Codex lacked the generated FSR-32 parquet. The user located the exact file in the real Codespace and established SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

Because direct Codex attachment was unavailable, the user installed `gh` in the real Codespace and published the exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`.

The first release-ingest Codex attempt failed before ingestion because it started in a fresh generic workspace:
- branch `work`;
- clean tree;
- no `origin` remote;
- governing EVENT_MC docs absent locally;
- no GitHub token/`gh` session;
- private-release API request returned 404;
- one diagnostic request also misspelled owner as `ChrisEsasu`;
- no files changed and no baseline run started.

This is an environment/bootstrap failure, not an FSR or baseline failure.

New bootstrap prompt:
`docs/EVENT_MC_V1_CODEX_BOOTSTRAP_RELEASE_BASELINE_2026-08-12.md`

It instructs Codex to:
1. inspect the generic workspace without discarding changes;
2. add exact remote `https://github.com/ChrisEsau/ufc-ai-clv-tracker.git` if `origin` is absent;
3. fetch authenticated repo state using only provisioned credentials and never expose secrets;
4. check out `feature/fsr-32-stamina-shadow` from `origin`;
5. verify/read the governing docs;
6. download the private release asset using an authenticated mechanism;
7. verify SHA-256 before use;
8. execute the existing release-ingest prompt and frozen Phase 0 baseline;
9. stop with an explicit auth-blocked result if repository or release authentication truly cannot be restored.

---

# Assistant Review Requirements

When Codex returns the bootstrap/baseline result, independently verify:
- correct repo owner `ChrisEsau` and target branch;
- exact remote/commit state;
- downloaded and destination SHA both equal frozen SHA;
- parquet unmodified/uncommitted;
- exact fixtures/seeds/path counts/cohort ordering;
- manifest/checksums;
- Font/Rosas anchors;
- compact 200x10 cohort;
- full method/submission baseline;
- no tuning/current-simulator/FSR changes.

Only after the Phase 0 baseline is reviewed should the user decide whether to explicitly start Phase 1.

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
User said to move on and supply FSR later. A Phase 1 generic-kernel prompt was prepared in the repo, but the user did NOT launch it. Earlier continuity incorrectly described Phase 1 as authorized/in progress; this is corrected below.

## 006 — 2026-08-12 23:08
User found exact frozen parquet in real Codespace and established SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## 007 — 2026-08-12 23:18
User published exact parquet as GitHub Release asset `event-mc-v1-fsr32-handoff/fsr_32_prefight_snapshots.parquet`. Release-ingest prompt issued.

## 008 — 2026-08-12 23:21
Latest Codex run reported it was in a new generic `work` checkout with no remote/auth and no governing docs. It did not modify anything. One API diagnostic used misspelled owner `ChrisEsasu`.

New prompt issued:
`docs/EVENT_MC_V1_CODEX_BOOTSTRAP_RELEASE_BASELINE_2026-08-12.md`

## 009 — 2026-08-12 23:24 America/Chicago
User clarified: **they never told Codex to start Phase 1.**

Correction:
- Phase 1 implementation has NOT started.
- The Phase 1 prompt file exists only as a prepared future task.
- Current Codex work remains exclusively Phase 0 bootstrap + exact FSR-32 ingestion + frozen operational baseline.
- Phase 1 must not start until the user explicitly authorizes it after Phase 0 review.

Expected Codex outcome now: restore correct remote/branch and complete release ingestion + frozen Phase 0 baseline, or return a precise auth-blocked gate without changing simulator/FSR/calibration state.

Phase 1 implementation started: **NO**.
Phase 1 execution authorized now: **NO**.
Phase 2A authorized: **NO**.
