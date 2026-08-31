# EVENT MC V1 Phase 7C — Finish Midpoint Calibration

Date: 2026-08-13

Purpose: calibrate only the global finish threshold after KD midpoint 36 corrected knockdown exposure.

Current post-KD diagnostic:
- KD/100 landed 0.328
- KD/15min 0.276
- KO/TKO 73.1%
- SUB 3.4%
- DEC 23.5%
- mean non-decision finish 215.81s
- P(finish|KD) 58.82%
- P(finish|non-KD landed) 1.830%
- 90.42% of finishing strikes non-KD
- 96.44% of KO/TKO paths zero prior KD

ONLY parameter allowed to move:
`defaults.finish.midpoint_impact_ratio`
Current value 10.0.

Freeze every other parameter, including finish slope, knockdown bonus, all resistance/trauma terms, KD midpoint 36, impact generation, rates, stamina, submissions, judging, RNG, FSR, weight-class overrides, age, and urgency.

Use chronological mature-fighter cohorts:
- train: 2020-2024
- holdout: 2025+
Report counts and dates.

Use common seeds. Start coarse grid:
`10 16 24 32 48 64 96 128 192`
Refine around the best bracket.

Compute historical KO/TKO target separately for train and holdout. Do not hard-code the 25% Phase 6 sample value.

Primary target: simulated KO/TKO share versus historical KO/TKO share on train and holdout.
Secondary guardrail: historical versus simulated non-decision finish time and finish-round distribution.

For finalists report:
- KO/TKO, SUB, DEC
- P(finish|KD)
- P(finish|non-KD landed)
- non-KD finishing-strike share
- KO paths with zero prior KD
- finish checks/path and /15min
- KD/100 landed and KD/15min
- zero/multi-KD
- attempts/15min and landed/15min
- winner accuracy, Brier, log loss
- runtime

Do not optimize SUB, DEC, or predictive metrics directly. They are downstream diagnostics.

Promote one new finish midpoint only if train and holdout support the same region, KO incidence improves materially on both, finish timing is not pathological, and KD calibration remains approximately intact. Otherwise make no YAML change and report that the finish model needs another degree of freedom.

Tests must prove candidate override changes only finish midpoint, same-seed determinism, KD midpoint stays 36, and unrelated calibration values remain fixed.

If promoted, show exact one-line YAML diff.

Final line exactly:
`PHASE 7C KO/TKO MIDPOINT CALIBRATION GATE: PASS`
or
`PHASE 7C KO/TKO MIDPOINT CALIBRATION GATE: FAIL`
