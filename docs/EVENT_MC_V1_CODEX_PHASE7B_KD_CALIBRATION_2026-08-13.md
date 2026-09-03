# EVENT MC V1 — PHASE 7B KD CALIBRATION

Date: 2026-08-13
Status: AUTHORIZED FOR IMPLEMENTATION
Repository: ChrisEsau/ufc-ai-clv-tracker
Branch: feature/fsr-32-stamina-shadow

## Objective
Calibrate the knockdown probability mapping only. Do not tune strike generation, landing, impact generation, KO/TKO finish conversion, submissions, stamina, judging, FSR, age, tactical urgency, or weight-class overrides.

Phase 7A established that strike exposure is not the main KD error:
- historical attempts/15min 169.50 vs simulated 194.30 (1.15x)
- historical landed/15min 93.18 vs simulated 83.32 (0.89x)
- historical KD/100 landed 0.280 vs simulated 3.718 (13.28x)
- historical KD/15min 0.261 vs simulated 3.098 (11.87x)

Current KD config:
- trauma_erosion_scale = 80.0
- resistance_scale = 32.0
- acute_erosion_scale = 1.0
- slope = 2.0
- midpoint_impact_ratio = 8.0
- acute_increment = 0.50
- acute_half_life_seconds = 30.0

## Hard calibration lock
For Phase 7B, allow changes only to:
- `knockdown.midpoint_impact_ratio`

Keep all other KD parameters fixed. In particular, do not change slope, resistance scaling, trauma erosion, acute vulnerability, impact generation, or finish conversion in this phase.

Rationale: midpoint is the cleanest global level parameter. We first test whether the existing impact distribution and KD shape can reproduce historical KD exposure by shifting the global threshold only. Do not add complexity unless this one-parameter calibration demonstrably cannot fit the targets.

## Calibration targets
Primary targets:
1. KD per 100 landed strikes
2. KD per 15 minutes of actual exposure

Secondary diagnostics, not primary optimization targets:
- zero-KD fight/path share
- multi-KD fight/path share
- KD round distribution
- KD phase distribution
- winner accuracy / Brier / log loss
- KO/TKO / SUB / DEC shares
- mean finish time and finish-round distribution

Do not optimize directly for KO/TKO rate in Phase 7B because finish conversion remains intentionally frozen. Report its movement as a downstream consequence only.

## Historical cohort and validation split
Use the leakage-safe frozen FSR-32 mature-fighter cohort already used by Phase 6/7A.

Create a deterministic temporal calibration split rather than fitting and evaluating on the same fights. Prefer:
- training/calibration: completed eligible fights from 2020 through 2024
- holdout: completed eligible fights from 2025 onward

If available data make that split impractical, use the nearest sensible chronological split and report exact dates/fight counts. Do not random-split fights across time.

Use all eligible fights in each split if runtime is practical. If coarse-search runtime is excessive, use a deterministic fixed training subset for the initial sweep, then rerun finalists on the full training and holdout cohorts.

## Search design
Build a reusable calibration runner that can supply an in-memory knockdown midpoint override without editing the committed YAML for every candidate.

Coarse candidate grid should include the current value and a broad enough range to reach the historical target. A reasonable starting grid is:
`8, 12, 16, 24, 32, 48, 64, 96, 128`

If the optimum is bracketed between candidates, run a narrow refinement around the best region. Do not expand into additional parameters during this phase.

Use common deterministic seeds across candidates so candidate differences are not dominated by Monte Carlo noise.

Use a practical low path count for coarse search, then substantially increase paths for the top candidates and final holdout confirmation. Report exact path counts.

## Candidate scoring
Rank candidates primarily by normalized error on both KD targets:
- KD/100 landed
- KD/15min

Do not choose a value that matches one target while materially worsening the other when a balanced candidate exists.

Also report historical vs simulated:
- zero-KD share
- multi-KD share
- KD by round
- KD by phase

Winner and method metrics are guardrails only at this stage.

## Promotion rule
Do not automatically commit a new YAML value merely because the search has a numeric winner.

Return the ranked candidate table and explicitly identify the recommended midpoint. If one candidate is clearly supported on both train and temporal holdout, you MAY update only `defaults.knockdown.midpoint_impact_ratio` to that value in `config/event_mc_v1.yaml`, but only after showing the before/after holdout metrics in the report.

If train and holdout disagree materially, or one midpoint cannot reproduce both exposure targets reasonably, STOP without changing YAML and report that Phase 7B needs a shape/impact follow-up.

No weight-class overrides.

## Required before/after reporting
For baseline and recommended candidate, report separately on train and holdout:
- attempts/15min
- landed/15min
- landing rate
- KD/100 landed
- KD/15min
- zero-KD share
- multi-KD share
- KD by round
- KD by phase
- KO/TKO / SUB / DEC shares
- mean non-decision finish time
- winner accuracy
- Brier
- log loss
- runtime

Confirm strike exposure remains effectively unchanged except for censoring effects caused by altered fight termination.

## Invariants
Do not change:
- impact distribution generator
- KO/TKO finish formula or coefficients
- submission mechanics
- action/phase rates
- stamina
- judging
- FSR-32
- RNG stream ownership
- age
- tactical urgency
- weight-class overrides

Frozen FSR-32 SHA-256 must remain:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Run focused tests, EVENT MC tests, compileall, diff-check, checksum, and clean-tree verification.

## Return
Report:
1. exact temporal split
2. candidate grid and path counts
3. ranked candidate results
4. train and holdout before/after metrics
5. whether a YAML midpoint was promoted
6. exact config diff if promoted
7. downstream KO/TKO movement without tuning it
8. all test/check results
9. FSR checksum
10. clean-tree status

Expected final line:

`PHASE 7B KD CALIBRATION GATE: PASS`

or FAIL.
