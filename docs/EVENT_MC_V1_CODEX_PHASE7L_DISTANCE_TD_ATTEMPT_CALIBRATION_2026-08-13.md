# EVENT MC V1 — Phase 7L DISTANCE Takedown Attempt Calibration

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

## Objective

Calibrate **only** the global DISTANCE takedown-attempt base so EVENT MC better reproduces historical UFCStats total takedown-attempt exposure.

Phase 7K established the structural diagnosis:

- Train historical TD attempts/15 = **6.169**; EVENT MC = **5.070**.
- Holdout historical TD attempts/15 = **7.282**; EVENT MC = **5.036**.
- Train historical TD success = **34.87%**; EVENT MC = **39.65%**.
- Holdout historical TD success = **30.23%**; EVENT MC = **40.19%**.
- Completed TD exposure is already comparatively close because elevated success conversion partially compensates for deficient attempts.
- About three quarters of simulator TD attempts arise from DISTANCE entries.

Therefore Phase 7L changes **attempt generation only**. Do not fix the success offset in this phase.

## Current locks

Keep all current committed values fixed except the one authorized candidate parameter.

- `defaults.distance.td_attempt_base_30s = 0.10` — **only authorized search/promote parameter**
- `defaults.clinch.td_attempt_base_30s = 0.24` — frozen
- `defaults.distance.td_success_logit_offset = -0.40` — frozen
- corrected `wrestling_entry` ontology — frozen

Strike clocks:
- distance = 6.0 / 30s
- clinch = 3.6 / 30s
- ground = 1.6 / 30s

Strike accuracies:
- distance = 0.40
- clinch = 0.68
- ground = 0.70

Submission:
- attempt base = 0.045
- bottom multiplier = 1.0
- conversion intercept = -0.60
- top/bottom conversion bonuses = 0.0 / 0.0

KD midpoint = 36
finish midpoint = 36

Frozen FSR-32 SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

If the parquet is missing, restore only the immutable GitHub Release asset from tag `event-mc-v1-fsr32-handoff`, verify the exact SHA before use, copy byte-for-byte to the ignored active path, verify again, and never rebuild/rewrite/commit it.

## Historical source and cohorts

Use the same authoritative Phase 7K source and exact semantics:

`data/fight_details/ufc_round_stats.parquet`

Historical TD fields:
- `td_attempted`
- `td_landed`

Aggregate both fighters across all observed rounds and retain zero-action fights. Use corrected total-elapsed fight exposure.

Use the exact established cohorts:

TRAIN:
- 100 fights
- 2020-01-18 through 2020-07-25

HOLDOUT:
- 50 fights
- 2025-01-11 through 2025-03-22

Seed: `20260813`
Use the established common deterministic seed scheme.

Historical anchors from Phase 7K must reproduce before candidate evaluation:

TRAIN:
- attempts/fight = 5.190
- attempts/15 = 6.169
- completed/fight = 1.810
- completed/15 = 2.151
- success = 34.87%
- fights with >=1 attempt = 85%
- zero-attempt fights = 15%
- multi-attempt fights = 72%

HOLDOUT:
- attempts/fight = 6.220
- attempts/15 = 7.282
- completed/fight = 1.880
- completed/15 = 2.201
- success = 30.23%
- fights with >=1 attempt = 94%
- zero-attempt fights = 6%
- multi-attempt fights = 84%

If these historical anchors do not reproduce, STOP and FAIL the gate rather than calibrating against a changed denominator.

## Candidate search

Search only `defaults.distance.td_attempt_base_30s` in memory first.

Coarse grid:

`0.10, 0.12, 0.14, 0.16, 0.18`

Run coarse candidates at **3 paths/fight** on both train and holdout using common seeds.

Do not assume proportionality between the interval probability and realized attempts; measure the simulator output.

After reviewing the coarse region, choose up to three adjacent finalists around the best temporal compromise. You may include intermediate values such as `0.13`, `0.15`, or `0.17` if supported by the coarse results.

Run finalists at **10 paths/fight** on both train and holdout with common seeds.

## Primary calibration targets

For every candidate and split report:

- total TD attempts/path
- total TD attempts/15 simulated minutes
- paths with >=1 TD attempt
- zero-attempt path share
- multi-attempt path share
- median TD attempts/path
- 25th/75th percentile attempts/path if practical

The primary decision metrics are:

1. TD attempts/15
2. TD attempts/path versus historical attempts/fight

Activity-share/distribution metrics are secondary guardrails.

Evaluate train and holdout **separately**. Do not collapse them into a single opaque combined score.

A promotion is allowed only if the same neighborhood materially improves attempt exposure on both splits versus 0.10 and represents a defensible temporal compromise. Do not force either split to an exact target at the expense of the other.

If train and holdout do not support the same neighborhood, make **no promotion** and report the temporal disagreement.

## Required decomposition and downstream guardrails

For every finalist report simulator-side:

### TD entry source
- DISTANCE attempts/path
- DISTANCE attempts/15
- DISTANCE attempt share
- CLINCH attempts/path
- CLINCH attempts/15
- CLINCH attempt share

The CLINCH clock must remain unchanged. Any realized CLINCH change should arise only from changed phase residence/censoring.

### TD completion/conversion guardrails
Report but **do not optimize** in this phase:

- completed TDs/path
- completed TDs/15
- TD success percentage
- paths with >=1 completion
- DISTANCE completions/path and success rate
- CLINCH completions/path and success rate

It is expected that completed TDs may overshoot historical values as attempt exposure rises because Phase 7K already showed the shared success conversion is too high. That is not a reason to secretly retune success here. Quantify it for the next phase.

### Phase residence / grappling
- distance seconds/path if available
- clinch seconds/path
- ground seconds/path
- submission attempts/path
- submission attempts/15
- P(SUB | attempt)

### Striking
- modeled significant-comparator attempts/15
- landed/15
- landing percentage
- distance/clinch/ground attempt shares

### Outcomes / finish guardrails
- KO/TKO %
- SUB %
- DEC %
- KD/path
- KD/15
- mean fight duration
- mean non-decision finish time

Do not compensate for downstream movement in Phase 7L.

## Hard freeze

Do NOT change:

- CLINCH TD attempt base
- TD success offset
- any TD success formula
- `wrestling_entry` ontology
- any FSR calculation
- clinch legacy blend
- strike clocks or accuracies
- submission parameters
- stamina
- damage
- KD
- KO finish
- phase transition mechanics
- ground exit/reversal
- judging
- RNG
- age
- urgency
- weight-class overrides
- round-specific rates

Only `defaults.distance.td_attempt_base_30s` may be promoted, and only after the finalist evidence supports it.

## Decision requirements

Explicitly answer:

1. Which DISTANCE TD base best matches total historical attempt exposure on TRAIN?
2. Which best matches HOLDOUT?
3. Is there a common supported neighborhood?
4. Does the candidate improve both attempts/fight and attempts/15 versus baseline?
5. What happens to zero-/multi-attempt path shares?
6. How much do completed TDs and success conversion overshoot once attempt exposure is corrected?
7. Does increased wrestling materially distort strike exposure, phase residence, SUB, KO/DEC, KD, or timing?
8. Should one DISTANCE TD base be promoted?

If yes, change exactly that one YAML value and rerun the promoted 10-path result to confirm it.

If no, leave YAML unchanged.

## Testing

Add focused tests for:

- candidate injection changes only DISTANCE TD attempt base
- CLINCH TD base remains 0.24
- success offset remains -0.40
- corrected `wrestling_entry` path remains active
- historical Phase 7K TD anchors reproduce
- source totals still equal DISTANCE + CLINCH
- all strike/submission/KD/finish locks remain unchanged

Run:

`python -m pytest -q tests/simulation/event_mc_v1 tests/experimental/test_fsr_static_mc_v0.py`

`python -m compileall pipeline scrapers tabs utils tests/simulation/event_mc_v1`

`sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

`git diff --check`

`git diff -- config/event_mc_v1.yaml`

If a value is promoted, the YAML diff must contain **only** the DISTANCE TD attempt-base change. If no promotion occurs, the YAML diff must be empty.

## Continuity

Update:

`docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`

Record:

- Phase 7K PASS and its structural conclusion
- Phase 7L coarse and finalist values
- train/holdout attempt targets and candidate results
- selected/promoted DISTANCE TD base or no-promotion decision
- completion/success consequences
- key phase/strike/submission/outcome guardrails
- all still-frozen TD and global calibration values
- next recommended phase

## Expected final line

`PHASE 7L DISTANCE TAKEDOWN ATTEMPT CALIBRATION GATE: PASS`

or FAIL.
