# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 22:22 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6: PASS; exposure-normalized historical anchors corrected in Phase 7D1
- Phase 7A: PASS; historical per-time anchors corrected in Phase 7D1
- Phase 7B KD midpoint 36: committed but not independently reconfirmed after time correction
- Phase 7B2: PASS
- Phase 7C finish midpoint 36: PASS and revalidated after time correction
- Phase 7D submission decomposition: PASS measurement only; calibration deferred
- Phase 7D1 historical exposure-time correction: PASS at `af1e56fdfcdb9823fcbd099dd441ec44b9e37485`
- Phase 7D2 KD target reconciliation: PASS; no promotion
- Phase 7E bottom submission attempt neutralization: PASS
- Phase 7F submission conversion position neutralization: PASS
- Phase 7G submission attempt-rate calibration: diagnostic complete; no promotion
- Phase 7H submission conversion intercept calibration: PASS; intercept promoted to -0.60
- Phase 7I strike exposure definition + baseline audit: PASS
- Phase 7J global strike-attempt calibration: PASS; distance/clinch clocks promoted to 6.0/3.6
- Phase 7K global takedown decomposition audit: PASS; measurement only
- Phase 7L distance takedown-attempt calibration: PASS; DISTANCE TD base promoted 0.10 -> 0.16
- Phase 7M shared TD-success calibration: PASS; shared TD success offset promoted -0.40 -> -0.85
- Phase 7N global coupled re-audit: PASS; readiness NO; next global subsystem remains significant-strike attempt generation and phase composition
- Fresh 100-fight predictive replay: PASS measurement-only sidecar; winner discrimination materially failed
- Winner-discrimination / FSR-wiring audit: AUTHORIZED, measurement only
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Current committed calibration
- `defaults.distance.strike_attempts_per_30s = 6.0`
- `defaults.clinch.strike_attempts_per_30s = 3.6`
- `defaults.ground.strike_attempts_per_30s = 1.6`
- strike accuracies distance/clinch/ground = 0.40/0.68/0.70
- `defaults.distance.td_attempt_base_30s = 0.16`
- `defaults.clinch.td_attempt_base_30s = 0.24`
- `defaults.distance.td_success_logit_offset = -0.85`
- `defaults.knockdown.midpoint_impact_ratio = 36.0`
- `defaults.finish.midpoint_impact_ratio = 36.0`
- `defaults.submission_attempts.base_30s = 0.045`
- `defaults.submission_attempts.bottom_multiplier = 1.0`
- `defaults.submission_finish.top_position_bonus = 0.0`
- `defaults.submission_finish.bottom_position_bonus = 0.0`
- `defaults.submission_finish.intercept = -0.60`

## Corrected historical anchors from Phase 7D1
On the same 100-fight cohort:
- observed seconds/fight: 757.16
- strike attempts/15min: 285.681
- landed strikes/15min: 157.045
- KD/15min: 0.439801
- KD/100 landed: 0.280048 unchanged
- KD/fight: 0.370 unchanged
- submission attempts/15min: 0.7251
- submission attempts/fight: 0.610 unchanged
- mean non-decision finish time: 402.762s
- method shares unchanged: KO/TKO 25.0%, SUB 17.0%, DEC 58.0%

Phase 7D1 implementation commit: `af1e56fdfcdb9823fcbd099dd441ec44b9e37485`.
Authoritative `match_time_sec` is total elapsed fight time; legacy final-round clock is supported only with explicit semantics.

## Submission position lock
Top and bottom submission attempt generation and conversion are position-neutral unless later UFC-specific evidence supports an intrinsic positional coefficient. Current locks: bottom attempt multiplier 1.0; top/bottom conversion bonuses 0.0/0.0.

## Phase 7J global strike-attempt calibration
Phase 7J used UFCStats significant-strike attempts as the primary comparator and promoted distance `6.0` and clinch `3.6`. Historical significant attempts/15 were 238.17 train and 256.59 holdout; the promoted candidate produced 237.82 and 232.99 before later TD coupling. Ground strike rate remains 1.6; accuracies remain 0.40/0.68/0.70. Gate: PASS.

## Phase 7K global takedown decomposition audit
Historical train TD attempts/completions were 6.169/2.151 per 15 with 34.87% success; holdout 7.282/2.201 with 30.23% success. The pre-7L simulator had deficient attempt exposure and elevated success conversion. Gate: PASS.

## Phase 7L distance takedown-attempt calibration
Phase 7L promoted only `defaults.distance.td_attempt_base_30s` from 0.10 to 0.16. CLINCH stayed 0.24. At 0.16, train attempts/15 were 6.634 versus 6.169 historical and holdout 6.558 versus 7.282. Frozen high conversion caused completion overexposure, motivating Phase 7M. Gate: PASS.

## Phase 7M shared takedown-success calibration
Phase 7M promoted only shared `defaults.distance.td_success_logit_offset` from -0.40 to -0.85. At the promoted state train TD completion was 2.179/15 versus 2.151 historical and holdout 2.272 versus 2.201; success was 30.33% versus 34.87% train and 32.03% versus 30.23% holdout. Reduced conversion returned residence from GROUND to standing phases and recovered significant-strike exposure without changing strike clocks. Gate: PASS.

## Phase 7N global coupled re-audit result
Phase 7N was measurement-only and made no YAML, mechanics, FSR, or RNG change. Train mean duration was 747.46s versus 757.16 historical; KO/TKO/SUB/DEC 24.2/16.4/59.4% versus 25/17/58. Significant attempts/15 were 230.71 versus 238.17, landed/15 105.14 versus 114.79, accuracy 45.57% versus 48.20%. TD attempts/completions/15 were 7.186/2.179 versus 6.169/2.151. KD/15 was 0.399 versus 0.440. Submission attempts/15 were 0.513 versus 0.725.

Holdout mean duration was 706.71s versus 768.78 historical; nondecision finish time 355.74s versus 484.30. KO/TKO/SUB/DEC 26.8/20.4/52.8% versus 28/18/54. Significant attempts/15 were 228.83 versus 256.59, landed/15 109.64 versus 121.77, accuracy 47.91% versus 47.46%. TD attempts/completions/15 were 7.093/2.272 versus 7.282/2.201. KD/15 was 0.326 versus 0.468. Submission attempts/15 were 0.606 versus 0.562.

Readiness decision: `GLOBAL ENVIRONMENT READY FOR ROUND-SPECIFIC VALIDATION: NO`. Exactly one next global subsystem remains significant-strike attempt generation and phase composition; first follow-up should measure/sensitize DISTANCE-versus-CLINCH strike opportunity allocation before calibration.

## Fresh 100-fight predictive replay authorization
Prompt: `docs/EVENT_MC_V1_CODEX_FRESH_100_FIGHT_PREDICTIVE_REPLAY_2026-08-13.md`
Prompt commit: `07c0337d403afb91906cb25f2cbc519840ed1f5e`

This is a sidecar predictive validation and does not supersede the Phase 7N recommendation. Codex must begin immediately without asking for confirmation. Select the first 100 chronological eligible completed UFC fights strictly after 2025-03-22, excluding all established train/holdout fights. Eligibility requires both leakage-safe frozen FSR-32 prefight snapshots, a decisive supported result, method normalizable to KO_TKO/SUB/DEC, and current simulator format support. No cherry-picking.

Run the current post-7M simulator unchanged at 250 paths/fight with seed 20260813 and deterministic fight/path seed derivation. Actual winner/method fields may be attached only after simulator inputs/path generation for scoring.

Required fight-level output for all 100 fights includes actual winner/method, P(red), P(blue), predicted winner and correctness, marginal KO_TKO/SUB/DEC probabilities, predicted method and correctness, six joint winner-method path probabilities, top joint class/correctness, actual/simulated timing. Joint probabilities must come directly from path frequencies rather than multiplying marginals.

Required aggregate metrics include winner accuracy, Brier, log loss, confidence-bucket accuracy; method accuracy/log loss/Brier and class-specific performance; six-class joint accuracy/log loss/confusion; timing MAE; all winner misses and high-confidence misses; and fresh-cohort global historical-vs-MC sanity metrics for methods, timing, significant strikes and phase mix, TDs, KDs, and submission attempts. Print the full 100-row table in the final Codex response and save a machine-readable report. No calibration/YAML/mechanics/FSR/RNG changes are authorized.

Expected final line: `FRESH 100-FIGHT EVENT MC PREDICTIVE REPLAY: PASS` or FAIL.

## Fresh 100-fight predictive replay result
The predictive replay selected the first 100 chronologically ordered eligible fights strictly after 2025-03-22, spanning 2025-03-29 through 2025-08-16. Selection excluded 106 candidate fights for missing qualifying frozen FSR-32 state and two for unsupported/incomplete results before the cohort was complete; overlap with the established train and holdout cohorts was zero. The immutable FSR SHA remained `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

The current post-7M simulator ran 250 paths for each of the 100 fights with seed 20260813 and deterministic fight/path seed derivation. Actual results were attached only after simulation. Winner accuracy was 49/100 (49.0%), winner Brier score 0.3133, and binary log loss 0.8687. Actual red win rate was 54.0% versus mean predicted red probability 53.82%. Accuracy was 52.63% when red was predicted and 44.19% when blue was predicted. Confidence did not discriminate reliably: 55 of 100 predictions were at least 70% confident but only 52.73% were correct, while 28 were at least 80% and only 46.43% were correct. There were 26 incorrect predictions at or above 70% and 15 at or above 80%.

Method accuracy was 61/100 (61.0%), multiclass Brier score 0.5485, and method log loss 0.9516. Historical KO_TKO/SUB/DEC shares were 27/13/60%; mean MC probabilities were 25.78/16.88/57.34%. Recall was 29.63% KO_TKO, 15.38% SUB, and 85.0% DEC, showing that aggregate method shares were much better aligned than fight-level finish-class discrimination. The highest-probability six-class winner-method combination was correct 31/100 (31.0%); six-class log loss was 2.3853 and mean probability assigned to the actual joint result was 24.25%.

Historical mean duration was 758.97 seconds versus 724.30 simulated; per-fight expected-duration MAE was 295.35 seconds. Historical nondecision finish time was 382.43 seconds versus 352.55 simulated. On the fresh cohort, historical versus MC global values were: significant-strike attempts/15 244.87 versus 230.34, landed/15 114.48 versus 109.14, and accuracy 46.75% versus 47.38%; TD attempts/15 7.862 versus 7.762, completions/15 2.704 versus 2.308, and success 34.39% versus 29.74%; KD/15 0.534 versus 0.409; submission attempts/15 0.593 versus 0.575. MC-only DISTANCE/CLINCH/GROUND residence was 488.24/60.47/175.59 seconds per path.

No calibration, YAML, mechanics, FSR, RNG, or market-odds change was made. The result is a predictive validation finding only and does not authorize tuning. Gate: PASS.

## Winner discrimination / FSR wiring audit authorization
Prompt: `docs/EVENT_MC_V1_CODEX_WINNER_DISCRIMINATION_AUDIT_2026-08-13.md`
Prompt commit: `a208af3a770f619c4127775586f1d3b0a9fd10f0`

The fresh 100-fight replay materially failed winner discrimination despite comparatively healthy population method composition. The next authorized task is therefore a measurement-only root-cause audit, and it temporarily takes priority over further global strike calibration. No simulator, YAML, FSR, age, stamina, or RNG change is authorized.

The audit must prove target-fight prefight FSR snapshot identity/chronology, trace fight-night age adjustment end to end, establish whether enabled durability/KD-resistance age rules are actually applied at EVENT MC runtime or prebaked in FSR-32, map every consumed FSR trait to its runtime destination and mathematical direction, compare RFS MC V2 fatigue architecture against EVENT MC stamina, inspect stamina trait mapping and defaults, quantify nonlinear matchup-transform sensitivity, and decompose all wrong high-confidence (>=75%) fresh-cohort predictions against comparable correct high-confidence controls.

Required root-cause classifications are CONFIRMED DEFECT / STRONG EVIDENCE / POSSIBLE CONTRIBUTOR / NOT SUPPORTED for stale/wrong FSR selection, leakage, missing age wiring, inverted trait direction, missing/unused traits, wrong stamina mapping, insufficient stamina effects, over-amplified transforms, dominant wrong-way FSR families, judging/finish direction defects, and any newly discovered cause. The audit must recommend exactly one next action and make no promotion or tuning.

Expected final line: `EVENT MC V1 WINNER DISCRIMINATION AUDIT: PASS` or FAIL.
