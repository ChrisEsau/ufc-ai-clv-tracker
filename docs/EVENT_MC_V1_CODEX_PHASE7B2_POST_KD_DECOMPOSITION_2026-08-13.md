# EVENT MC V1 — Phase 7B2 Post-KD Decomposition

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Status
AUTHORIZED FOR MEASUREMENT ONLY.

Phase 7B promoted only:
`defaults.knockdown.midpoint_impact_ratio: 8.0 -> 36.0`

Do not change any simulator mechanic or config in this phase.

## Objective
Rerun the Phase 7A decomposition in the corrected KD environment before authorizing KO/TKO conversion calibration.

The old Phase 7A KO-conversion diagnostics were measured under the overactive KD environment and are now stale because fight censoring, KD classification, acute vulnerability, and finish-check exposure changed.

## Required cohort
Use the same mature-fighter chronological framework used in Phases 6/7A.

At minimum rerun:
- 100 eligible fights
- 10 paths/fight
- start year 2020
- seed 20260813

If runtime permits, also provide train/holdout summaries using:
- train: 2020-2024
- holdout: 2025+

No parameter search is authorized.

## Required measurements
Report current post-KD-calibration values for:

### Striking exposure
- attempts/15min
- landed/15min
- landing rate

### Knockdowns
- KD/100 landed
- KD/15min
- zero-KD path share
- multi-KD path share
- KD round distribution
- KD phase distribution

### KO/TKO conversion
- total landed-strike finish checks
- finish checks/path
- finish checks/15min
- P(finish | KD strike)
- P(finish | non-KD landed strike)
- share of KO/TKO finishing strikes that were not KDs
- share of KO/TKO paths with zero prior KDs

### Outcomes and timing
- KO/TKO share
- SUB share
- DEC share
- non-decision finish-round distribution
- mean non-decision finish time

### Impact
Repeat compact impact distributions for:
- all landed
- non-KD
- KD
- fight-ending

Use count, mean, median, p75, p90, p95, p99, max.

### Trauma
Repeat the existing compact trauma/check bins. Clearly state they are descriptive and censored, not causal.

## Before/after table
Compare the new midpoint-36 environment against Phase 7A midpoint-8 anchors:

Phase 7A anchors:
- attempts/15min 194.30
- landed/15min 83.32
- KD/100 landed 3.718
- KD/15min 3.098
- P(finish | KD) 46.53%
- P(finish | non-KD landed) 1.322%
- non-KD finishing-strike share 42.38%
- KO/TKO paths with zero prior KDs 67.81%
- finish checks/path 27.111
- KO/TKO population share 81.4%

Phase 7B search subsets showed KO/TKO remained roughly 73-76% after midpoint 36; the purpose here is to establish the exact decomposition under the current committed config.

## Probability / mechanics invariants
This phase must not modify:
- config/event_mc_v1.yaml
- impact generation
- knockdown parameters
- KO/TKO parameters
- action or phase rates
- stamina
- submissions
- judging
- RNG ownership/order
- FSR-32
- weight-class overrides
- age
- tactical urgency

The frozen FSR-32 checksum must remain:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Validation
Run:
1. relevant focused tests;
2. full EVENT MC test suite;
3. compileall;
4. `git diff --check`;
5. checksum;
6. clean-tree check.

## Return
Report:
- exact cohort and command;
- post-KD decomposition;
- before/after table versus Phase 7A;
- whether excess KO/TKO is now dominated by non-KD repeated checks, KD-strike conversion, or both;
- no calibration recommendation beyond evidence;
- exact files changed (diagnostics/tests/docs only if changes are necessary);
- commit/push status.

Expected final line:

`PHASE 7B2 POST-KD DECOMPOSITION GATE: PASS`

or FAIL.
