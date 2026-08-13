# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions so the assistant can recover what was decided, what Codex was asked to do, what remains locked, and what must be reviewed next.

This file is **not** the architecture source of truth. Canonical architecture and Phase 0 contracts remain the documents referenced below.

## Update rule

After every new Codex prompt is given to the user, update this file. Preserve checkpoint history and update `Current State`.

---

# Current State

Architecture revision: **v0.3**

Architecture status: **Phase 0 architecture closed.**

Implementation status: **Phase 1 generic continuous-time kernel is authorized. No Phase 2A/2B UFC mechanics are authorized yet.**

Phase 0 operational baseline status: **DEFERRED / NOT YET PASSED**, but the exact frozen FSR-32 artifact has now been identified in the user's real Codespace and is ready to be supplied to Codex.

Exact frozen FSR-32 identity:

```text
Codespace source path:
/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet

SHA-256:
621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a

reported size:
~3.3M
```

Canonical artifact identity note:

`docs/EVENT_MC_V1_FSR32_FROZEN_ARTIFACT_IDENTITY_2026-08-12.md`

Current artifact-ingest / baseline-resume Codex prompt:

`docs/EVENT_MC_V1_CODEX_FSR32_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

Current Phase 1 Codex prompt:

`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

Canonical architecture document:

`docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`

Phase 0 closure record:

`docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Locked interface decisions:

`docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`

Frozen baseline contract:

`docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Baseline execution prompt:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

---

# Hard Locks

- Do not modify the current inheritance-based simulator while EVENT MC V1 is built.
- Keep FSR-32 connected initially when UFC mechanics are introduced.
- Do not rebuild FSR-32 or the upstream FSR chain now that the exact frozen artifact is known.
- Do not retune KO, SUB, TD, stamina, judging, age, damage, recovery, or other calibration constants during kernel/parity work.
- Use composition, not a new inheritance chain.
- One authoritative fight clock: `FightState.fight_time_seconds`.
- Scheduler rates are events/second.
- Only engine advances time.
- Continuous state advances exact elapsed `dt` before event resolution.
- Stats/ledger are observers, not hidden communication.
- Components do not create hidden RNGs.
- Engine owns authoritative state mutation through typed results/deltas.
- Round start resets phase to `DISTANCE` and clears positional ownership.
- `MatReturn` remains future/optional.
- Do not combine temporal migration, wrestling ontology correction, and calibration changes in one step.

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

- Phase 2A: temporal/mechanical parity using current blended consumer.
- Phase 2B: ontology correction so `wrestling_entry` directly drives intrinsic TD initiation.

---

# Phase 1 Scope

Phase 1 is generic kernel infrastructure only:

- typed generic events/contracts;
- immutable timing config;
- authoritative clock;
- exponential competing-risk scheduler;
- event rates in events/second;
- exact probability-to-rate conversion;
- stable named RNG streams from one root path seed;
- engine-owned hard round/fight boundaries;
- exact continuous `advance(dt)` ordering;
- typed state-delta application;
- round lifecycle/reset shell;
- null/stats/full-trace sinks;
- inactive cooldown/action-availability extension point;
- synthetic math/temporal/reproducibility tests.

No real UFC mechanics in Phase 1.

---

# Deferred Phase 0 Baseline — Now Recoverable

The original Phase 0 numerical gate failed because Codex's isolated `/workspace/...` environment did not contain the generated FSR-32 parquet and is not the user's actual GitHub Codespace.

Codex searched its accessible filesystem and correctly returned:

`FSR-32 ARTIFACT RECOVERY: NOT FOUND`

The user then located the actual frozen parquet in the real Codespace:

`/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Local identity check returned:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

The user now wants to give the artifact to Codex immediately.

New approved Codex prompt:

`docs/EVENT_MC_V1_CODEX_FSR32_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

Codex must verify the supplied attachment against the exact SHA before copying it into the ignored expected path. It must not rewrite/rebuild the parquet. If checksum matches, it may resume the frozen Phase 0 baseline using the untouched current simulator.

Phase 1 work and Phase 0 baseline measurement must remain logically separate; the old baseline continues to use `StaticFSRMCFullFightV1` and the frozen FSR-32 artifact.

---

# Assistant Review Requirements

When Codex returns Phase 1 work, independently review:

- diff/files changed;
- current simulator/FSR untouched;
- scheduler is UFC-agnostic;
- units and probability-to-rate math;
- RNG stream derivation/independence;
- one clock / exact hard boundaries;
- continuous `advance(dt)` ordering;
- engine-owned mutation;
- round-start reset;
- sink invariance and RNG invariance;
- cooldown extension remains inactive;
- tests and regressions;
- no UFC mechanics slipped into Phase 1.

When Codex returns the resumed Phase 0 baseline, independently review:

- supplied source SHA and destination SHA both equal `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`;
- artifact was copied byte-for-byte and not committed;
- exact fixtures/seeds/path counts/cohort ordering;
- manifest/checksums;
- Font/Rosas values and prior anchors;
- 200×10 cohort summary;
- full method/submission baseline;
- no tuning/current-simulator/FSR changes.

Do not authorize Phase 2A until both the Phase 1 kernel and the recovered numerical baseline have been reviewed, unless the user explicitly changes that requirement.

---

# Checkpoint History

## Checkpoint 001 — 2026-08-12 22:33 America/Chicago
Phase 0 operational baseline prompt issued. Phase 1 unauthorized.

## Checkpoint 002 — 2026-08-12 22:37 America/Chicago
Codex baseline execution plan approved with exact-remote-branch guardrail.

## Checkpoint 003 — 2026-08-12 22:40 America/Chicago
Codex stopped because isolated checkout had no Git remote. Remote-unblock prompt issued.

## Checkpoint 004 — 2026-08-12 22:49 America/Chicago
Codex repaired Git environment and verified feature branch, but Phase 0 baseline failed because frozen FSR-32 parquet was absent. Artifact-recovery prompt issued. No rebuild authorized.

## Checkpoint 005 — 2026-08-12 22:59 America/Chicago
Codex artifact recovery returned NOT FOUND. User explicitly authorized moving forward with Phase 1 generic kernel while leaving Phase 0 numerical baseline deferred. Phase 2A remained unauthorized.

## Checkpoint 006 — 2026-08-12 23:08 America/Chicago
User located the exact frozen FSR-32 parquet in the real GitHub Codespace.

Identity:

```text
source path: /workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet
SHA-256: 621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a
reported size: 3.3M
```

User requested that the artifact be given to Codex now.

New Codex prompt issued:

`docs/EVENT_MC_V1_CODEX_FSR32_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

Purpose:

- accept the user-supplied parquet attachment;
- verify exact SHA-256 before use;
- copy byte-for-byte to the expected ignored FSR-32 path;
- inspect metadata without rewriting;
- resume the deferred frozen Phase 0 operational baseline;
- keep Phase 1 implementation logically separate;
- do not rebuild FSR or begin Phase 2.

Expected next assistant action: review whichever Codex result returns first (Phase 1 kernel and/or resumed Phase 0 baseline) independently. Phase 2A remains unauthorized.

Phase 1 authorized now: **YES**.

Phase 2A authorized now: **NO**.
