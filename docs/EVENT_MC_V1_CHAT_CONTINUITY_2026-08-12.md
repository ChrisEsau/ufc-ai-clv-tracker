# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 14:00 America/Chicago
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
- Phase 7C finish midpoint calibration: AUTHORIZED / current next phase
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

## Phase 7B2 post-KD decomposition final
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE7B2_POST_KD_DECOMPOSITION_2026-08-13.md`
Prompt commit: `2663750f7bb51c311bbeaab3804d8f46037f2355`
Implementation commit: `eba174cdb9a6276a7a061f9b4c973bbc1a463ad8`

100-fight x 10-path post-KD anchors at midpoint 36:
- attempts/15min 195.48
- landed/15min 84.05
- KD/100 landed 0.328
- KD/15min 0.276
- zero-KD 88.6%; multi-KD 0.5%
- finish checks/path 36.234; finish checks/15min 84.050
- P(finish | KD) 58.82%
- P(finish | non-KD landed) 1.830%
- non-KD finishing-strike share 90.42%
- KO/TKO paths with zero prior KDs 96.44%
- KO/TKO 73.1%; SUB 3.4%; DEC 23.5%
- mean non-decision finish time 215.81s
- R1 share of non-decisions 74.12%

Interpretation lock:
- corrected KD exposure did not solve excessive KO/TKO;
- non-KD repeated finish checks are now the population-dominant KO/TKO channel;
- KD-strike conversion is high but rare and not population dominant;
- old midpoint-8 KO conversion measurements are obsolete due changed censoring.

Phase 7B2 gate: PASS.

## Phase 7C finish midpoint calibration
Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE7C_FINISH_MIDPOINT_CALIBRATION_2026-08-13.md`
Prompt commit: `d8980863f641acac693261ce8c7e583beb78a8d5`

Hard scope:
- ONLY `defaults.finish.midpoint_impact_ratio` may move from current 10.0;
- finish slope, KD logit bonus, durability/resistance/trauma/acute terms remain fixed;
- KD midpoint remains 36.0;
- impact generation, action/phase rates, stamina, submissions, judging, RNG, FSR, weight-class overrides, age and urgency remain fixed.

Calibration design:
- chronological mature-fighter train 2020-2024 and holdout 2025+;
- compute historical KO/TKO target separately per split;
- common seeds;
- coarse grid 10,16,24,32,48,64,96,128,192, then refine around best bracket;
- primary target KO/TKO share; finish timing is secondary guardrail;
- SUB/DEC and predictive metrics are downstream diagnostics only;
- promote one finish midpoint only if train and holdout support the same region, KO incidence improves on both, timing is not pathological, and KD calibration remains approximately intact;
- if one midpoint cannot work cleanly, do not force promotion; return evidence that the finish model needs another degree of freedom.

Expected return: `PHASE 7C KO/TKO MIDPOINT CALIBRATION GATE: PASS` or FAIL.

Next assistant action: independently review Phase 7C search, exact config diff if promoted, temporal holdout support, KD preservation, finish timing, and downstream SUB/DEC movement before authorizing submission calibration.
