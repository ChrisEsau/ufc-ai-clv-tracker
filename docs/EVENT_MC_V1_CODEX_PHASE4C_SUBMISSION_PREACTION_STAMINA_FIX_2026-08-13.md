# EVENT MC V1 — CODEX PHASE 4C SUBMISSION PRE-ACTION STAMINA FIX

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Purpose
Complete Phase 4C by correcting one action-ordering inconsistency without changing any calibration values or submission formula coefficients.

## Independent review finding
`PhaseCandidate.resolve()` computes the submission attempt stamina cost in its returned `StateDelta`. The engine applies that primary delta before `SubmissionFinishModel.resolve()` is called. `SubmissionFinishModel.probability()` then reads attacker/defender stamina from the already-updated `FightState`.

Therefore the current submission attempt's own stamina cost can reduce its own conversion probability.

This conflicts with the established Phase 4A timing rule:

> the current action uses pre-action dynamic/physiological state; the current action cost affects subsequent events.

## Required correction
Submission conversion for the currently selected `submission_attempt` must use the PRE-ACTION stamina state.

The submission attempt stamina cost must still be applied exactly once and must affect all subsequent events.

Do this with the smallest clean change. Preferred approaches include passing an immutable pre-action snapshot/context to the submission finish resolver or explicitly passing pre-action red/blue stamina values. Do not add another mutable state object or second clock.

## Hard requirements

- Do not change the Phase 4C formula coefficients.
- Do not change submission-attempt generation or rate.
- Do not change action stamina costs.
- Do not change RNG streams or RNG draw order except where unavoidable to preserve the same logical sampling point.
- Do not change KO/TKO mechanics.
- Do not change KD calibration.
- Do not change FSR-32 or rebuild FSR.
- Do not change judging, age, tactical urgency, phase mechanics, or weight-class calibration.
- Engine remains the sole `FightState` mutator.
- Finishing submission attempt and its `ActionOutcome` remain accounted exactly once before exactly one `FightFinished`.
- No future primary event or time advance after a terminal SUB.

## Tests
Add focused tests proving:

1. Submission conversion receives pre-action attacker and defender stamina.
2. Changing only the current attempt cost does NOT change that same attempt's `P(SUB)` when pre-action state and RNG seed are identical.
3. The attempt cost still changes post-attempt stamina and therefore can affect a later attempt.
4. Existing trait monotonicity / position / stochastic tests remain passing.
5. Existing terminal accounting tests remain passing for both SUB and KO/TKO.
6. Lewis/Daukaus deterministic KO/TKO sanity remains unchanged.
7. Oliveira/Poirier deterministic seed-22 submission remains structurally valid; if exact P(SUB) or terminal result changes solely because this corrects pre-vs-post action stamina semantics, report the before/after value explicitly rather than tuning to preserve it.

## Validation
Run:

```bash
python -m pytest -q tests/simulation/event_mc_v1 tests/experimental/test_fsr_static_mc_v0.py
```

Run the deterministic traces:

```bash
python -m pipeline.simulation.event_mc_v1.single_fight \
  --fight-id b22eab3aa1522f40 --paths 1 --trace --seed 22
```

```bash
python -m pipeline.simulation.event_mc_v1.single_fight \
  --fight-id 4b7ec02b39fc6f70 --paths 1 --trace --seed 20260813
```

Verify frozen FSR-32 checksum remains:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Stop condition
Stop after the sequencing correction, tests, deterministic trace comparison, checksum, commit/push, and report.

Expected final line:

`PHASE 4C SUBMISSION PRE-ACTION STAMINA FIX GATE: PASS`

or FAIL.
