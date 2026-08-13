# Codex Execution Prompt — EVENT MC V1 Phase 1 Generic Kernel

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`
Architecture revision: **v0.3**

## Status

**Phase 0 architecture: CLOSED.**

**Phase 0 operational baseline: PASS.**

The exact frozen FSR-32 artifact was recovered and verified byte-for-byte with SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

The frozen baseline artifacts were revalidated, including all five fixtures, 15 deterministic traces, five 1,000-path matchup summaries, the stable 200-bout x 10-path cohort, and the full 1,565-fight method/submission anchor. Recorded output checksums matched their manifest entries. No simulator, FSR, ontology, maturity rule, or calibration code was changed.

The user has now explicitly authorized **Phase 1 implementation**.

Phase 2A and Phase 2B remain **NOT AUTHORIZED**.

## Governing documents — read before coding

Read the current versions in this order:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
7. `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`
8. this execution prompt

## Important override to the prepared Phase 1 prompt

The detailed Phase 1 specification in:

`docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`

remains the implementation contract for scheduler, RNG, clock, state mutation, hard boundaries, sinks, action-availability extension point, package scope, tests, and non-goals.

However, its earlier statements saying the Phase 0 operational baseline is deferred/not passed are now obsolete.

This execution prompt overrides those status statements only.

The correct state is:

- Phase 0 operational baseline: **PASS**;
- Phase 1 implementation: **AUTHORIZED NOW**;
- Phase 2A: **NOT AUTHORIZED**;
- Phase 2B: **NOT AUTHORIZED**.

Do not otherwise weaken or broaden the prepared Phase 1 specification.

## Task

Implement exactly Phase 1: the minimal generic deterministic continuous-time event-simulation kernel under:

`pipeline/simulation/event_mc_v1/`

with focused tests under:

`tests/simulation/event_mc_v1/`

Follow the prepared Phase 1 prompt exactly for all required contracts and tests.

This phase is infrastructure only. Use synthetic events/components/rates in tests.

## Absolute non-goals

Do NOT implement or port:

- real UFC striking mechanics;
- takedown mechanics;
- wrestling-entry semantics;
- clinch/ground mechanics;
- submissions;
- stamina physiology;
- damage reservoirs;
- knockdowns or KO/TKO;
- recovery;
- age adjustment;
- judging/scoring mechanics beyond reserving the named RNG stream;
- FSR loading or rebuilding;
- historical calibration or tuning;
- Phase 2A temporal parity;
- Phase 2B wrestling ontology correction.

Do not modify the existing simulator or current FSR/calibration code.

## Repository guardrail

Before coding:

- verify the exact repository and branch;
- fetch `origin` if needed;
- ensure the checkout is `feature/fsr-32-stamina-shadow`;
- inspect the working tree and preserve unrelated user changes;
- do not silently work from a generic `work` branch.

If the environment starts without the correct remote/branch, restore it using the exact repository:

`https://github.com/ChrisEsau/ufc-ai-clv-tracker.git`

before proceeding.

## Required completion report

Return a concise but complete report containing:

1. exact repository, branch, start SHA, final SHA;
2. files added/changed;
3. implemented Phase 1 contracts/components;
4. scheduler mathematics and units;
5. RNG stream design and independence;
6. clock/boundary/continuous-advance ordering;
7. engine-owned mutation/delta behavior;
8. sink behavior/invariance;
9. action-availability extension behavior;
10. exact tests run and results;
11. any pre-existing unrelated failures separated from new regressions;
12. explicit confirmation that existing simulator, FSR, ontology, and calibration code were untouched;
13. explicit confirmation that no real UFC mechanics or Phase 2 work were introduced;
14. commit/PR details if created.

End with exactly one of:

`PHASE 1 GENERIC KERNEL GATE: PASS`

or

`PHASE 1 GENERIC KERNEL GATE: FAIL`

If FAIL, list exact blockers.

Stop after Phase 1. Do not begin Phase 2A.