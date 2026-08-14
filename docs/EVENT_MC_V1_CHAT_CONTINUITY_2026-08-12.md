# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 19:45 America/Chicago
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
- Phase 7M shared TD-success calibration: authorized/current next phase
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Current committed calibration
- `defaults.distance.strike_attempts_per_30s = 6.0`
- `defaults.clinch.strike_attempts_per_30s = 3.6`
- `defaults.ground.strike_attempts_per_30s = 1.6`
- strike accuracies distance/clinch/ground = 0.40/0.68/0.70
- `defaults.distance.td_attempt_base_30s = 0.16`
- `defaults.clinch.td_attempt_base_30s = 0.24`
- `defaults.distance.td_success_logit_offset = -0.40`
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

## Post-correction state
At committed 36/36 on the 100-fight x 10-path rerun before later submission changes:
- simulated KO/TKO 25.6%, SUB 5.7%, DEC 68.7%
- simulated KD/100 landed 0.438
- simulated KD/15min 0.383
- simulated submission attempts/path 0.380
- simulated submission attempts/15min 0.423
- simulated path share with >=1 attempt 27.4%
- simulated P(SUB|attempt) 15.0%
- simulated mean non-decision finish time 387.43s

Finish midpoint 36 remained supported after correction. KD midpoint 36 was retained after Phase 7D2 because corrected KD/15 and KD/100 landed disagreement was driven primarily by lower/non-definition-identical simulated landed-strike exposure.

## Submission position lock
Top and bottom submission attempt generation and conversion are position-neutral unless later UFC-specific evidence supports an intrinsic positional coefficient. Current locks: bottom attempt multiplier 1.0; top/bottom conversion bonuses 0.0/0.0.

## Phase 7D2 KD target reconciliation
Prompt: `docs/EVENT_MC_V1_CODEX_PHASE7D2_KD_TARGET_RECONCILIATION_2026-08-13.md`
Prompt commit: `720ab5ccbbda9001ad873959f2e44068bf9d639b`

Measurement only. Keep KD midpoint 36 and finish midpoint 36 committed. Compare in-memory KD midpoint candidates 32, 36, 40, 44, 48 on the same 100-fight x 10-path cohort and report separately:
- KD/fight or path
- KD/100 landed
- KD/15min
- zero/multi-KD shares
- landed/fight or path and landed/15min
- KO/TKO share
- mean fight duration

Do not rank with one combined objective and do not promote YAML. Determine whether corrected evidence supports a KD midpoint change or whether the conflict is mainly upstream strike exposure/comparability.

Expected return: `PHASE 7D2 KD TARGET RECONCILIATION GATE: PASS`.

Phase 7D2 result: the exact common-seed 100-fight x 10-path comparison was completed for in-memory KD midpoints 32, 36, 40, 44, and 48 with finish midpoint fixed at 36. No combined score or ranking was used. Historical landed exposure was 132.12/fight and 157.045/15min, versus roughly 78-80/path and 87.4-87.5/15min simulated across candidates. Midpoint 48 closely matched KD/100 landed (0.269 vs 0.280 historical) but materially undershot KD/path (0.215 vs 0.370) and KD/15min (0.235 vs 0.440). Midpoint 36 was closer on KD/path (0.344) and KD/15min (0.383) but high on KD/100 landed (0.438). This conflict is primarily attributable to lower/non-definition-identical simulated landed-strike exposure; corrected evidence does not justify a midpoint promotion. Committed KD and finish midpoints remain 36.

## Phase 7E bottom submission-attempt neutralization
Phase 7E changed only `defaults.submission_attempts.bottom_multiplier` from 0.55 to 1.00. The 100-fight x 10-path rerun increased attempts from 380 to 474 (0.380 to 0.474/path; 0.423 to 0.529/15min) and paths with an attempt from 27.4% to 32.3%. Top/bottom attempts were 224/250, with exposure-normalized rates of 1.122/1.252 per 15 positional ground minutes. SUB moved from 5.7% to 6.6%; KO/TKO remained 25.5% and DEC was 67.9%. Conversion remained frozen in that phase. Gate: PASS.

## Phase 7F submission conversion position neutralization
Phase 7F changed only `defaults.submission_finish.top_position_bonus` from 0.25 to 0.0; `bottom_position_bonus` remained 0.0, the conversion intercept remained -2.20, and `submission_attempts.bottom_multiplier` remained 1.0. The neutral 100-fight x 10-path baseline produced 477 attempts, 61 SUB finishes, and 12.79% conversion. Observed top/bottom conversion narrowed from 16.07%/12.00% to 13.27%/12.35% (gap 4.07pp -> 0.92pp). Attempts remained stable at 0.477/path and 0.531/15min; SUB moved from 6.6% to 6.1%, KO/TKO remained 25.5%, and DEC moved to 68.4%. Gate: PASS.

## Phase 7G global submission-attempt rate calibration
Phase 7G searched only `submission_attempts.base_30s` using common seeds. The six-point coarse grid used 3 paths/fight; finalists 0.045, 0.050, and 0.055 used 100 train fights (2020-01-18–2020-07-25), 50 holdout fights (2025-01-11–2025-03-22), and 10 paths/fight.

Train supported approximately 0.050–0.055: at 0.055, simulated exposure was 0.617 attempts/path, 0.691/15min, and 38.2% paths with an attempt versus historical 0.610, 0.725, and 37.0%. Holdout supported the existing 0.045 or lower: at 0.045, simulated exposure was already 0.554/path and 0.613/15min versus historical 0.480 and 0.562, while its 33.8% path share remained below historical 38.0%. Increasing the base improved holdout path share but worsened both count/rate overexposure. Because train and holdout did not support the same region across all three primary metrics, no value was promoted and `base_30s` remains 0.045. Gate disposition: no promotion.

## Phase 7H global submission-conversion intercept calibration
The frozen FSR-32 release asset was restored only after its SHA-256 matched `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`; the parquet remains uncommitted. The common-seed coarse grid used 3 paths/fight and the refined `-0.7,-0.6,-0.5,-0.4,-0.3,-0.2` region used 10 paths/fight on 100 train and 50 holdout fights. Both splits supported intercept -0.6: train historical/simulated SUB was 17.0%/17.3%, holdout 18.0%/18.2%; KO/TKO remained 23.8% and 25.4%. Only `submission_finish.intercept` was promoted from -2.20 to -0.60. Attempt base remains 0.045, bottom multiplier 1.0, position bonuses 0.0, and KD/finish midpoints 36. Gate: PASS.

## FSR-32 release recovery procedure
If the ignored local FSR-32 parquet is missing in a future Codex sandbox, do not search indefinitely and do not rebuild it. Recover it from GitHub Release tag `event-mc-v1-fsr32-handoff`, asset `fsr_32_prefight_snapshots.parquet`. Preferred command is `gh release download event-mc-v1-fsr32-handoff --repo ChrisEsau/ufc-ai-clv-tracker --pattern 'fsr_32_prefight_snapshots.parquet' --dir /tmp/event_mc_v1_fsr32_handoff`. Verify the downloaded SHA-256 equals exactly `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a` before use. Then copy byte-for-byte to `data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet` and verify the destination checksum again. Never rebuild, rewrite, recompress, or commit the parquet. The old release-ingest document's instruction to resume Phase 0 is historical; after recovery, resume whatever current phase is blocked.

## Phase 7I strike exposure definition + baseline audit
Phase 7I completed as measurement-only; no YAML or mechanics changed. The modeled strike is a meaningful offensive strike opportunity: every scheduled event is one attempt, every landed attempt enters impact physiology, and there is no low-impact/non-significant or target-location family. UFCStats significant strikes are therefore the recommended primary comparator for both attempt generation and landing probability; total strikes remain a secondary upper-bound diagnostic. Train modeled versus significant attempts/15 were 199.31/238.17 and landed/15 were 88.34/114.79; holdout values were 198.63/256.59 and 93.41/121.77. Overall landing accuracy was much closer: train 44.32%/48.20%, holdout 47.03%/47.46%. The remaining discrepancy is primarily attempt exposure and phase composition (too little clinch, too much distance), with a secondary train landing gap and a major field-definition mismatch if total strikes are used. Phase 7H remains PASS; intercept -0.60, position-neutral submission locks, and KD/finish midpoints 36 remain fixed. Gate: PASS.

## Phase 7J global strike-attempt calibration
Phase 7J used UFCStats significant-strike attempts as the primary comparator and searched only the global DISTANCE and CLINCH strike-attempt rates. The common-seed 3-path coarse grid crossed distance `5.0, 5.5, 6.0, 6.5` with clinch `1.2, 2.0, 2.8, 3.6` on 100 chronological train fights and 50 chronological holdout fights. The practical Pareto region was distance `6.0-6.25` and clinch `3.6`: it materially corrected global underexposure and clinch composition on both temporal splits without changing ground mechanics or accuracy.

The 10-path finalist comparison promoted distance `6.0` and clinch `3.6`. Versus historical significant attempts/15 of 238.17 train and 256.59 holdout, the promoted candidate produced 237.82 and 232.99; the committed baseline produced 199.31 and 198.63. Train distance/clinch/ground attempt shares moved from 91.97/2.75/5.28% to 89.57/6.24/4.19% versus historical 86.32/6.94/6.74%. Holdout moved from 92.38/2.21/5.41% to 90.18/5.33/4.49% versus historical 89.83/4.91/5.26%. The intermediate distance `6.25` candidate improved holdout global exposure but overran train and further reduced ground share, so `6.0/3.6` was preferred as the stable temporal compromise.

Only `defaults.distance.strike_attempts_per_30s` changed from 5.0 to 6.0 and `defaults.clinch.strike_attempts_per_30s` changed from 1.2 to 3.6. Ground strike rate remains 1.6; distance/clinch/ground accuracies remain 0.40/0.68/0.70. Submission attempt base remains 0.045, bottom multiplier 1.0, submission intercept -0.60, position bonuses 0.0/0.0, and KD/finish midpoints remain 36. Round-specific RFS validation and all downstream recalibration are deferred. Gate: PASS.

## Phase 7K global takedown decomposition audit
Phase 7K is measurement-only and passed with no YAML or mechanics change. The authoritative round source is `data/fight_details/ufc_round_stats.parquet`; exact fields are `td_attempted` and `td_landed`, aggregated across both `corner` rows and every observed `round`, while exposure uses the master fight's corrected total-elapsed `match_time_sec`.

Historical train (100 fights, 2020-01-18 through 2020-07-25) produced 519 attempts and 181 completions: 5.19/1.81 per fight, 6.169/2.151 per 15 observed minutes, and 34.87% success. Attempt/completion fight shares were 85%/72%; zero/multi-attempt shares were 15%/72%; attempt quartiles were 1/4/8 and completion quartiles 0/1/3. Historical holdout (50 fights, 2025-01-11 through 2025-03-22) produced 311/94: 6.22/1.88 per fight, 7.282/2.201 per 15 minutes, and 30.23% success. Attempt/completion shares were 94%/70%; zero/multi-attempt shares 6%/84%; quartiles were 2/5/8 attempts and 0/1/3 completions.

The unchanged 10-path EVENT MC baseline produced train 4.207 attempts and 1.668 completions/path, 5.070/2.010 per 15 minutes, 39.65% success, and 90.6%/77.3% paths with an attempt/completion. Holdout produced 4.120/1.656 per path, 5.036/2.024 per 15 minutes, 40.19% success, and 87.0%/73.8% path shares. Simulator entry decomposition was train 73.996% DISTANCE attempts (3.113/path) and 26.004% CLINCH attempts (1.094/path); completions were 73.441%/26.559%. Holdout was 75.777% DISTANCE attempts (3.122/path) and 24.223% CLINCH (0.998/path); completions were 74.034%/25.966%.

Both temporal splits tell the same story: total TD attempts are low (about 18% train and 31% holdout by per-time exposure), completions are comparatively aligned (about 7%/8% low), and success conversion is high (about +4.8/+9.9 percentage points). The completion alignment is therefore partly compensatory. UFCStats does not identify DISTANCE-versus-CLINCH TD entry, so the historical source cannot directly allocate the attempt deficit. On the simulator side most attempts are DISTANCE entries, making DISTANCE generation the largest numerical contributor, but no source-specific TD parameter should be promoted from this audit alone. Recommended sequence: diagnose/sensitize global attempt generation first—starting with the DISTANCE clock because it owns roughly three quarters of attempts—then remeasure before considering the shared success offset. Current TD bases 0.10/0.24 and success offset -0.40 remain unchanged. Gate: PASS.

## Phase 7L distance takedown-attempt calibration
Phase 7L reproduced the Phase 7K historical anchors exactly before candidate evaluation and searched only `defaults.distance.td_attempt_base_30s`. The 3-path coarse grid evaluated 0.10, 0.12, 0.14, 0.16, and 0.18 with common seeds on the 100-fight train and 50-fight temporal holdout cohorts. Both splits supported increasing the DISTANCE clock; the shared practical neighborhood was 0.14-0.18. Finalists 0.14, 0.16, and 0.18 were rerun at 10 paths/fight.

The promoted compromise is 0.16. It produced train 5.426 attempts/path and 6.634/15min versus historical 5.190 and 6.169, and holdout 5.150/path and 6.558/15min versus historical 6.220 and 7.282. Candidate 0.14 fit train closely but left holdout materially low; 0.18 fit holdout per-time exposure but overran train. At 0.16, DISTANCE/CLINCH attempts were 4.420/1.006 per train path and 4.294/0.856 per holdout path. Attempt shares were 81.46%/18.54% train and 83.38%/16.62% holdout.

As anticipated, frozen elevated success conversion caused completions to overshoot: train 2.165/path, 2.647/15min, 39.90% success versus historical 1.810, 2.151, 34.87%; holdout 2.110/path, 2.687/15min, 40.97% versus historical 1.880, 2.201, 30.23%. No compensation was made. Only the DISTANCE attempt base changed 0.10 to 0.16. CLINCH base remains 0.24, TD success offset remains -0.40, and wrestling ontology, FSR, strikes, submissions, phases, stamina, damage, KD/finish, judging, RNG, and overrides remain frozen. The next TD phase should revalidate and calibrate shared success conversion, not retune attempts simultaneously. Gate: PASS.

At the promoted 0.16 state, downstream strike exposure was approximately 221.25 significant-comparator attempts/15 train and 219.20 holdout versus historical 238.17 and 256.59. This interaction is not grounds to retune strike clocks yet because the currently excessive TD success rate sends too much residence to GROUND; lowering conversion may restore standing time and strike exposure.

## Phase 7M shared takedown-success calibration authorization
Phase 7M is authorized to search and, only if supported on both temporal splits, promote **one parameter only**: `defaults.distance.td_success_logit_offset`, current `-0.40`. DISTANCE TD attempt base 0.16 and CLINCH TD attempt base 0.24 are frozen.

Use train 100 fights (2020-01-18 through 2020-07-25), holdout 50 fights (2025-01-11 through 2025-03-22), seed 20260813, common deterministic seeds, and the authoritative round-level `td_attempted` / `td_landed` historical source with corrected elapsed exposure. Before calibration reproduce and fail-closed validate the exact Phase 7K historical TD anchors.

Coarse TD-success offset grid: `-0.40, -0.55, -0.70, -0.85, -1.00` at 3 paths/fight. Choose a small adjacent finalist set, allowing intermediate values around the supported region, then rerun finalists at 10 paths/fight. Primary targets are TD success %, completed TD/15, and completed TD/path/fight. TD attempts/15 and attempts/path are hard guardrails and must remain materially improved relative to the pre-7L state. Do not jointly retune TD attempts.

For every finalist decompose DISTANCE and CLINCH attempts, completions, success rates, shares, residence, submissions, strikes, outcomes, KD, and timing. Explicitly measure whether reduced TD conversion returns residence to standing phases and improves significant-strike exposure without changing strike clocks.

**User-required Phase 7M deliverable:** after the final promoted candidate (or final no-promotion state), print a complete historical-vs-EVENT-MC global metrics comparison for TRAIN and HOLDOUT separately. Use readable tables with historical, MC, absolute difference, relative difference %, and percentage-point difference for shares/rates where appropriate. Include every historically comparable global metric currently available from the Phase 7 measurement stack, at minimum: fight duration/timing; KO/SUB/DEC; significant strike attempts and landed per fight/path and /15; overall significant-strike accuracy; distance/clinch/ground significant-strike attempts/15, landed/15, accuracy, attempt shares and landed shares where available; TD attempts/completions per fight/path and /15, success %, activity shares, zero/multi-attempt shares, quartiles; KD/path or fight, KD/15 and KD/100 comparable landed; submission attempts/path/fight and /15, path/fight attempt share where available and P(SUB|attempt); phase residence/control where historically supported. Simulator-only residence diagnostics must still be printed and clearly labeled MC-only. Missing historical comparators must be labeled unavailable rather than fabricated. Also print the complete current calibration values. Save the same comparison into the diagnostic JSON/report, not only terminal output.

Promotion allowed only if the same practical TD-success-offset region materially improves success and completions on both temporal splits without destroying TD attempt exposure or causing major downstream instability. If promoted, YAML may change only `defaults.distance.td_success_logit_offset`; otherwise YAML remains unchanged. All strikes, TD attempt clocks, wrestling ontology, FSR, submission parameters, stamina, damage, KD/finish, phases, judging, RNG, age, urgency, WC overrides, and round-specific calibration are frozen.

Expected final line: `PHASE 7M SHARED TAKEDOWN SUCCESS CALIBRATION GATE: PASS` or FAIL.

### Phase 7M result
Phase 7M reproduced the fail-closed historical TD anchors exactly and searched only the shared `defaults.distance.td_success_logit_offset`. The 3-path coarse grid was `-0.40,-0.55,-0.70,-0.85,-1.00`; both temporal splits supported the `-0.70` to `-0.85` region. Ten-path finalists were `-0.70,-0.775,-0.85`. Offset `-0.85` was promoted because completed TD exposure was closest on both splits while attempt bases remained frozen: train completion was 2.179/15min versus 2.151 historical, and holdout was 2.272 versus 2.201. Success became 30.33% train versus 34.87% historical and 32.03% holdout versus 30.23%; `-0.775` was a more symmetric success compromise but left greater completion overexposure on both splits.

TD attempt exposure remained acceptable but rose through changed residence/censoring: train 7.186/15min versus 6.169 historical, holdout 7.093 versus 7.282. DISTANCE and CLINCH share the same offset and produced success of 30.0%/31.6% train and 31.7%/33.7% holdout. Reduced conversion returned train DISTANCE/CLINCH/GROUND residence from 466.7/62.0/207.4 seconds/path at the Phase 7L baseline to 504.7/72.5/170.3; holdout moved from 451.3/50.4/205.1 to 480.9/59.2/166.6. Significant-comparator attempts/15 recovered from 221.25 to 230.71 train and 219.20 to 228.83 holdout without a strike-clock change.

Submission attempts/path fell from 0.493 to 0.426 train and 0.574 to 0.476 holdout as less GROUND residence reduced opportunities; SUB shares moved 18.5% to 16.4% and 20.8% to 20.4%. KO/TKO/DEC ended at 24.2%/59.4% train and 26.8%/52.8% holdout. KD/path and KD/15 were 0.331/0.399 train and 0.256/0.326 holdout. Mean fight/nondecision time was 747.5/382.4 seconds train and 706.7/355.7 holdout.

The required final global comparison was printed and saved in the Phase 7M JSON for both splits. It contains timing, methods, significant strikes and phase composition, TDs, KDs, submissions, explicit unavailable historical phase-residence labels, semantic caveats, and the complete calibration state. Final locks: strike clocks 6.0/3.6/1.6; accuracies .40/.68/.70; TD attempt bases .16/.24; TD success offset -.85; submission attempt .045, multiplier 1.0, intercept -.60, bonuses 0/0; KD/finish midpoints 36/36. Recommended next phase: re-audit global exposure after the coupled strike/TD changes before any round-specific calibration. Gate: PASS.
