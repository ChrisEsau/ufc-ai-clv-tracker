# EVENT MC V1 — Phase 7N Global Coupled Re-Audit

## Execution
Begin immediately when this prompt is received. Do not ask for confirmation or approval to start. Execute the authorized diagnostics, tests, documentation, commit, and push without waiting for another user message. Stop only if technically blocked or if continuing would exceed the explicit scope below.

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

## Objective

Perform a **measurement-only global coupled re-audit** of the current committed EVENT MC after Phase 7M. No calibration or simulator-mechanics changes are authorized.

The purpose is to establish the current fully coupled global state after strike-clock, TD-attempt, and TD-success calibration and identify the single most important remaining global mismatch before any round-specific R1/R2/R3 calibration.

## Current committed calibration — freeze all

- DISTANCE strike attempts/30s = 6.0
- CLINCH strike attempts/30s = 3.6
- GROUND strike attempts/30s = 1.6
- DISTANCE/CLINCH/GROUND accuracy = 0.40 / 0.68 / 0.70
- DISTANCE TD attempt base/30s = 0.16
- CLINCH TD attempt base/30s = 0.24
- shared TD success offset = -0.85
- submission attempt base/30s = 0.045
- submission bottom multiplier = 1.0
- submission conversion intercept = -0.60
- submission top/bottom bonuses = 0.0 / 0.0
- KD midpoint = 36
- finish midpoint = 36

Corrected `wrestling_entry` ontology remains frozen.

Frozen FSR-32 SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

If missing, restore only the immutable release asset from tag `event-mc-v1-fsr32-handoff`, verify the exact SHA before and after byte-for-byte copy, and never rebuild/rewrite/recompress/commit it.

## Cohorts and simulator

Use the established chronological cohorts:

TRAIN:
- 100 fights
- 2020-01-18 through 2020-07-25

HOLDOUT:
- 50 fights
- 2025-01-11 through 2025-03-22

Use:
- current committed EVENT MC unchanged
- 10 paths/fight
- seed `20260813`
- established deterministic common-seed scheme
- corrected total-elapsed historical fight exposure

Reuse the authoritative historical sources and exact semantics established in Phases 7I–7M. Do not invent comparators.

## Fail-closed provenance checks

Before simulation, reproduce the established historical anchors. At minimum verify:

TRAIN TD:
- attempts/fight 5.190
- attempts/15 6.169105605156109
- completed/fight 1.810
- completed/15 2.1514607216440385
- success 34.8747591522158%

HOLDOUT TD:
- attempts/fight 6.220
- attempts/15 7.2816670568953406
- completed/fight 1.880
- completed/15 2.200889721376727
- success 30.22508038585209%

Also verify the historical significant-strike anchors used by Phase 7I/7J and the corrected elapsed-time semantics. If any anchor or source meaning changes, STOP and FAIL rather than silently recomputing a new target.

## Required full historical-vs-MC report

Print a readable comparison for both TRAIN and HOLDOUT with columns:

- Metric
- Historical
- EVENT MC
- Absolute Difference
- Relative Difference %
- Percentage-point Difference where applicable

Save the same comparison in a machine-readable Phase 7N JSON/report.

Include every historically comparable global metric currently supported by the Phase 7 measurement stack. At minimum:

### Fight / timing
- historical fights
- simulated paths
- mean fight duration
- mean non-decision finish time

### Outcomes
- KO/TKO %
- SUB %
- DEC %

### Significant strikes — established primary comparator
- attempts/fight or path
- attempts/15
- landed/fight or path
- landed/15
- landing %

### Significant-strike phases
For DISTANCE, CLINCH, and GROUND where supported:
- attempts/15
- landed/15
- accuracy
- attempt share
- landed share if supported

Do not fabricate phase-specific total-strike fields.

### Takedowns
- attempts/fight or path
- attempts/15
- completed/fight or path
- completed/15
- success %
- fight/path share with >=1 attempt
- fight/path share with >=1 completion
- zero-attempt share
- multi-attempt share
- attempt quartiles
- completion quartiles where supported

Also print MC-only DISTANCE vs CLINCH TD decomposition:
- attempts/path and /15
- completions/path and /15
- success %
- attempt share
- completion share

Do not invent historical DISTANCE-vs-CLINCH TD entry classification.

### Knockdowns
- KD/fight or path
- KD/15
- KD/100 comparable landed significant strikes
- zero-KD share where available
- multi-KD share where available

Retain the semantic caveat that MC landed-strike events and historical UFCStats significant strikes are closest comparators, not definition-identical.

### Submissions
- submission attempts/fight or path
- submission attempts/15
- fight/path share with >=1 attempt where available
- P(SUB | attempt) where semantically supported
- historical SUB outcome %
- MC SUB outcome %

If historical attempt-level conversion is not available/comparable, explicitly label it `historical comparator unavailable`.

### Phase residence / control
Print MC-only:
- DISTANCE seconds/path
- CLINCH seconds/path
- GROUND seconds/path

Print historical comparison only if the authoritative source actually supports a trustworthy same-definition denominator. Otherwise explicitly write `historical comparator unavailable`.

### Current calibration state
Print the complete frozen calibration values listed above.

## Additional diagnostic summaries

For each major metric calculate the signed relative error where mathematically meaningful.

Classify each historically comparable metric into:
- CLOSE: absolute relative error <= 5%
- MODERATE: >5% and <=10%
- MATERIAL: >10% and <=20%
- LARGE: >20%

For percentage/share metrics also inspect percentage-point error so tiny denominators do not mislead classification.

Produce a compact summary table of all metrics classified MATERIAL or LARGE for TRAIN and HOLDOUT.

## Required coupled diagnosis

Explicitly assess these families independently:

1. strike attempt generation
2. strike landing / accuracy
3. strike phase composition
4. TD attempt generation
5. TD completion / success conversion
6. KD generation
7. submission attempt generation
8. submission conversion / SUB outcomes
9. KO/TKO outcomes
10. decision outcomes
11. fight timing
12. phase residence / control, where comparison is supported

For every family state:
- train status
- holdout status
- whether both splits tell the same story
- likely upstream/downstream coupling responsible for the residual
- whether it needs more global calibration before round-specific validation

Do not infer causation beyond what the measurements/mechanics support.

## Most important output: next-step decision

Rank the remaining global mismatches by priority, considering:

1. magnitude of historical-vs-MC error
2. consistency across train and holdout
3. whether the mismatch is upstream and can distort many downstream metrics
4. whether the historical comparator is definitionally strong
5. whether fixing it risks undoing already-good global outcomes

Then answer exactly:

- `GLOBAL ENVIRONMENT READY FOR ROUND-SPECIFIC VALIDATION: YES` or `NO`

If YES:
- explain why remaining global errors are acceptable/deferred
- recommend the first R1/R2/R3 validation family

If NO:
- name exactly ONE next global subsystem to investigate/calibrate
- identify the one parameter family or mechanism to examine first
- do not calibrate it in Phase 7N

Do not recommend simultaneous multi-parameter tuning.

## Important context to examine

Known post-7M values include approximately:

TRAIN:
- significant attempts/15 230.71 vs historical 238.17
- TD attempts/15 7.186 vs historical 6.169
- TD completed/15 2.179 vs historical 2.151
- TD success 30.33% vs historical 34.87%
- KO/TKO 24.2% vs historical 25%
- SUB 16.4% vs historical 17%
- DEC 59.4% vs historical 58%
- KD/15 0.399 vs historical about 0.440

HOLDOUT:
- significant attempts/15 228.83 vs historical 256.59
- TD attempts/15 7.093 vs historical 7.282
- TD completed/15 2.272 vs historical 2.201
- TD success 32.03% vs historical 30.23%
- KO/TKO 26.8% vs historical 28%
- SUB 20.4% vs historical 18%
- DEC 52.8% vs historical 54%
- KD/15 0.326, historical target to be printed from authoritative data

Recompute from source/current simulator; do not merely echo these approximate values.

## Hard freeze

No YAML changes.
No simulator mechanics changes.
No FSR changes.
No RNG changes.
No strike-clock changes.
No strike-accuracy changes.
No TD-attempt changes.
No TD-success changes.
No submission changes.
No stamina changes.
No damage/trauma/KD/KO changes.
No phase-transition changes.
No judging changes.
No age/urgency/weight-class overrides.
No round-specific calibration.

This phase is measurement and decision only.

## Testing

Add focused tests that prove:
- current global locks exactly match the committed Phase 7M state
- historical TD anchors fail closed
- historical significant-strike anchors/semantics remain unchanged
- no YAML mutation occurs
- all required report families are present
- unavailable historical comparators are explicitly labeled, not fabricated
- MATERIAL/LARGE mismatch classification is deterministic
- final readiness decision is present in report output

Run:

`python -m pytest -q tests/simulation/event_mc_v1 tests/experimental/test_fsr_static_mc_v0.py`

`python -m compileall pipeline scrapers tabs utils tests/simulation/event_mc_v1`

`sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

`git diff --check`

`git diff -- config/event_mc_v1.yaml`

The YAML diff must be empty.

## Continuity

Update `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md` with:
- Phase 7M PASS and promoted TD success offset -0.85
- Phase 7N complete train/holdout global summary
- MATERIAL/LARGE mismatch list
- coupled diagnosis
- readiness YES/NO
- exact next subsystem if NO, or first round-specific family if YES
- all frozen calibration values

## Delivery

Commit the measurement/test/documentation changes, push the branch, update PR #64 as appropriate, and leave the working tree clean. Do not commit generated JSON/CSV/parquet outputs unless an existing project convention explicitly requires a tracked diagnostic artifact.

Expected final line:

`PHASE 7N GLOBAL COUPLED RE-AUDIT GATE: PASS`

or FAIL.
