# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 14:19 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6 population historical validation: PASS
- Phase 7A strike/impact/KD/KO decomposition: PASS
- Phase 7B KD calibration: PASS at `66b927f72c399304e466055902435ecf74e885d6`
- Phase 7B2 post-KD decomposition: PASS at `eba174cdb9a6276a7a061f9b4c973bbc1a463ad8`
- Phase 7C finish midpoint calibration: PASS at `659d7963954334f1fd330cd9f138550a42409ffa`
- Phase 7D post-finish submission decomposition: AUTHORIZED / current next phase
- Submission calibration changes: NOT YET AUTHORIZED
- Age, tactical urgency, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Phase 7B KD calibration final
Only promoted config change:
`defaults.knockdown.midpoint_impact_ratio: 8.0 -> 36.0`

Temporal result supported the same region on 2020-2024 train and 2025+ holdout. KD exposure moved from roughly 12x historical into the historical order of magnitude. Phase 7B gate: PASS.

## Phase 7B2 final
At KD midpoint 36 on the 100-fight x 10-path diagnostic:
- KD/100 landed 0.328
- KD/15min 0.276
- KO/TKO 73.1%
- P(finish|KD) 58.82%
- P(finish|non-KD) 1.830%
- non-KD finishing-strike share 90.42%
- KO paths with zero prior KDs 96.44%
Interpretation: remaining KO/TKO excess was dominated by repeated non-KD finish checks.

## Phase 7C finish midpoint calibration final
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE7C_FINISH_MIDPOINT_CALIBRATION_2026-08-13.md`
Prompt commit: `d8980863f641acac693261ce8c7e583beb78a8d5`
Implementation commit: `659d7963954334f1fd330cd9f138550a42409ffa`

Only promoted config change:
`defaults.finish.midpoint_impact_ratio: 10.0 -> 36.0`

KD midpoint remains 36.0. Finish slope, knockdown bonus, resistance terms, impact generation, rates, stamina, submissions, judging, RNG, FSR and overrides remain fixed.

Chronological refined subset results at finish midpoint 36:
- train historical KO/TKO 25.0%, simulated 23.0%
- holdout historical KO/TKO 28.0%, simulated 30.7%
- train SUB 6.0%, DEC 71.0%
- holdout SUB 2.0%, DEC 67.3%
- train mean non-decision finish 370.6s
- holdout mean non-decision finish 348.1s
- P(finish|KD): train 20.4%, holdout 24.3%
- P(finish|non-KD): train 0.21%, holdout 0.31%
- KD exposure remained in approximately calibrated range; longer exposure increased observed KD counts through reduced censoring.

Phase 7C conclusion: one global finish midpoint fixed aggregate KO/TKO incidence on both temporal splits without adding another finish degree of freedom. Finish timing remains early, but submission underproduction is now the larger unresolved method-distribution error, so do not add another KO parameter yet.

Phase 7C gate: `PHASE 7C KO/TKO MIDPOINT CALIBRATION GATE: PASS`.

## Phase 7D post-finish submission decomposition
Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE7D_POST_FINISH_SUBMISSION_DECOMPOSITION_2026-08-13.md`
Prompt commit: `bb67232d77294079df0a52395c6ab2345121d0cf`

Measurement only. Current committed environment is KD midpoint 36 and finish midpoint 36.

Purpose: old submission-attempt exposure measurements are stale because Phase 7C reduced KO/TKO censoring and greatly lengthened simulated paths. Remeasure submission attempts/path, attempts/15min, true path share with >=1 attempt, conversion, round/position/ground exposure where available, method mix, finish timing and predictive diagnostics before authorizing any submission calibration.

No config or mechanics changes are authorized in Phase 7D.

Expected return: `PHASE 7D POST-FINISH SUBMISSION DECOMPOSITION GATE: PASS` or FAIL.

Next assistant action: independently review Phase 7D and determine whether the remaining SUB deficit is attempt-exposure limited, conversion limited, or both. Then authorize one narrow submission subsystem calibration at a time.
