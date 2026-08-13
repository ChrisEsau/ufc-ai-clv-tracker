# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 13:52 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6 population historical validation: PASS
- Phase 7A strike/impact/KD/KO decomposition: PASS
- Phase 7B KD calibration: PASS at `66b927f72c399304e466055902435ecf74e885d6`
- Phase 7B2 post-KD decomposition: AUTHORIZED / current next phase
- KO/TKO conversion calibration: NOT YET AUTHORIZED
- Submission calibration: NOT YET AUTHORIZED
- Age, tactical urgency, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Phase 7A final anchors
- historical attempts/15min 169.50 vs simulated 194.30
- historical landed/15min 93.18 vs simulated 83.32
- historical KD/100 landed 0.280 vs simulated 3.718
- historical KD/15min 0.261 vs simulated 3.098
- simulated P(finish | KD) 46.53%
- simulated P(finish | non-KD landed) 1.322%
- non-KD finishing-strike share 42.38%
- KO/TKO paths with zero prior KDs 67.81%
- finish checks/path 27.111

Interpretation lock: strike volume is not the primary KD problem; KD probability conditional on landing was the strongest demonstrated error. Impact generation and KO conversion remained frozen for Phase 7B.

## Phase 7B KD calibration final
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE7B_KD_CALIBRATION_2026-08-13.md`
Prompt commit: `f63fe8390f9b32abcb1690dfda31b1411d8e9e1f`
Implementation commit: `66b927f72c399304e466055902435ecf74e885d6`

Only promoted config change:
`defaults.knockdown.midpoint_impact_ratio: 8.0 -> 36.0`

All other knockdown parameters and all impact/KO/stamina/submission/judging/action/phase/RNG/FSR settings remain unchanged.

Temporal cohort:
- train eligible fights 1,096, 2020-01-18 through 2024-12-14
- holdout eligible fights 327, 2025-01-11 through 2026-08-01

Refined midpoint 36 diagnostics:
Train subset: KD/100 landed 0.283; KD/15min 0.242; KO/TKO 73.2%; SUB 3.6%; DEC 23.2%.
Holdout subset: KD/100 landed 0.227; KD/15min 0.201; KO/TKO 76.4%; SUB 2.0%; DEC 21.6%.

Phase 7B conclusion: one global KD midpoint was enough to reduce KD exposure from roughly 12x historical to the historical order of magnitude on train and holdout, while KO/TKO remained grossly excessive. Phase 7B gate: PASS.

## Phase 7B2 post-KD decomposition
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE7B2_POST_KD_DECOMPOSITION_2026-08-13.md`
Prompt commit: `2663750f7bb51c311bbeaab3804d8f46037f2355`

Purpose: rerun Phase 7A measurements under committed KD midpoint 36 before any KO/TKO calibration. Old conversion diagnostics are stale because corrected KD frequency changes censoring, acute vulnerability and finish-check exposure.

Required measurements include current KD exposure, finish checks, P(finish|KD), P(finish|non-KD), non-KD finishing share, KO paths with zero prior KD, outcome/timing distributions, impact tails, and trauma bins. No config or mechanic changes are authorized.

Expected return: `PHASE 7B2 POST-KD DECOMPOSITION GATE: PASS` or FAIL.

Next assistant action: review Phase 7B2 and decide the narrow Phase 7C KO/TKO calibration parameterization. Do not start submission calibration until KO/TKO environment is corrected.
