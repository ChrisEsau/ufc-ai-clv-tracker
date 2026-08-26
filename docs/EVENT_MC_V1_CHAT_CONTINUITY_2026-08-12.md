# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-14 04:12 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

Terminology note: Chris will refer to this file as the **chat md** going forward.

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

## 2026-08-14 night handoff — striking volume vs phase preference split

### Why this became the next priority
Fresh-100 winner discrimination materially failed while aggregate method/environment calibration remained much healthier. A focused 10-fight audit of wrong decision predictions then compared historical UFCStats components against EVENT MC decision-path averages. The strongest directional failure was significant-strike generation, while wrestling/control direction was much healthier:

- significant-strike attempt leader correct in only about 3/10 fights;
- significant-strike landed leader correct in only about 4/10;
- takedown-attempt and takedown-landed direction about 8/10;
- control-time direction about 9/10;
- grappling magnitude/persistence was still too weak even when direction was correct.

This separated the current diagnosis into at least three distinct systems: striking matchup direction, grappling magnitude/persistence, and uncalibrated judging. Do not collapse them into one tuning problem.

### Same 10 diagnostic fights
1. Raul Rosas Jr. vs Vince Morales
2. Christian Rodriguez vs Melquizael Costa
3. Rafa Garcia vs Vinc Pichel
4. Pat Sabatini vs Joanderson Brito
5. Loma Lookboonmee vs Istela Nunes
6. Yan Xiaonan vs Virna Jandiroba
7. Jim Miller vs Chase Hooper
8. Giga Chikadze vs David Onama
9. Ian Machado Garry vs Carlos Prates
10. Michel Pereira vs Abus Magomedov

### Key actual-vs-MC striking examples
- Rosas/Morales actual sig attempts/landed favored Rosas modestly, but EVENT MC generated a very large Morales volume edge.
- Miller/Hooper actual favored Hooper in attempts and landings, while EVENT MC flipped the striking edge toward Miller.
- Giga/Onama actual Onama landed more, while EVENT MC gave Giga a large striking edge.
- Garry/Prates actual Garry dominated volume and landings, while EVENT MC reversed both attempt and landed direction.
- Pereira/Abus actual Pereira attempted more while Abus landed slightly more; EVENT MC reversed the important volume relationship.

### FSR prefight values for the same 10
The frozen prefight rows showed that `distance_striking_pressure` often favored the fighter that EVENT MC then over-generated. Important examples:

- Rosas 46.7237 vs Morales 55.6049
- Rodriguez 46.9386 vs Costa 50.0992
- Garcia 50.4945 vs Pichel 47.5205
- Sabatini 42.9876 vs Brito 45.6007
- Loma 44.7043 vs Nunes 51.4956
- Yan 54.5340 vs Virna 49.0565
- Miller 48.3523 vs Hooper 47.3430
- Giga 54.3194 vs Onama 53.1367
- Garry 49.5819 vs Prates 52.4409
- Pereira 49.6113 vs Abus 52.1915

These values by themselves should not yet be called defective. They are persistent historical ratings, not direct matchup predictions. The critical question became whether EVENT MC is using them for the wrong semantic job.

### Canonical `distance_striking_pressure` definition — confirmed
Canonical builder:
`scripts/experimental/build_fsr_canonical_database.py`

The canonical builder imports:
`scripts/experimental/fsr_distance_striking_pressure_v1.py`

That module defines pressure as:

- 60% percentile of `distance_attempts_per_round`
- 40% percentile of `distance_attempt_share`

Source columns:

- `rfs_phase_base_fight_distance_attempts_per_round`
- `rfs_phase_base_fight_distance_attempt_share`
- `rfs_phase_interact_fight_distance_attempts` only for observation quality/update exposure

The persistent Elo-style update starts at 50, rating scale 12, initial K 7, decaying K with update count, and quality approximately `1 - exp(-distance_attempts / 10)`. Same-date updates are simultaneous and population pools are prior-date only.

Semantic conclusion: the trait is not pure combat-pressure/forward-pressure. It mixes two concepts:

1. distance striking volume/activity;
2. distance phase/style share.

The 40% distance-attempt-share component is explicitly phase/style composition, not absolute output.

### EVENT MC consumer — confirmed coupling defect/ontology problem
Relevant file:
`pipeline/simulation/event_mc_v1/components/formulas.py`

EVENT MC currently maps `distance_striking_pressure` directly into intrinsic DISTANCE strike-attempt intensity:

`expected_per_30s = 6.0 * exp(clip(distance_striking_pressure - 50.0, -8.0, 8.0) / 12.0)`

Then:

`rate_per_second = expected_per_30s / 30.0`

The opponent is not involved in this intrinsic distance-strike rate. Stamina/output modifiers are applied afterward.

Current shared config uses `modifier_clip = 8.0`, so the exponential modifier saturates outside ratings 42 to 58. Within that band, a 45-vs-55 difference still creates roughly a 2.3x rate ratio. For Rosas/Morales specifically, 46.72 vs 55.60 implies roughly 4.57 vs 9.57 intrinsic distance attempts per 30s, about a 2.1x Morales rate before fight dynamics. This closely explains the large simulated Morales strike-volume edge.

The same `distance_striking_pressure` is also consumed inside `style_preferences()`, together with clinch pressure, wrestling entry, and control. It therefore affects transition/phase behavior as well as strike attempt rate. This means one FSR trait currently does two fundamentally different jobs.

### Agreed ontology change — do not implement blindly yet
Chris and ChatGPT agreed that EVENT MC needs **separate phase preference and action volume** concepts.

Target conceptual structure:

**Phase preference / phase-seeking behavior**
- answers: where does this fighter try to make the fight happen?
- should influence transitions and phase residence;
- examples: distance preference, clinch preference, wrestling/ground-seeking preference;
- should not directly determine how quickly strikes occur once the fighter is in that phase.

**Action volume / activity intensity**
- answers: once in a phase, how frequently does this fighter act?
- examples: distance striking volume, clinch striking volume, ground striking volume, takedown-attempt volume, submission-attempt volume;
- should drive event rates conditional on being in the relevant phase.

**Efficiency / success traits** remain separate:
- striking precision vs opponent defense;
- wrestling conversion vs TD defense;
- submission conversion vs submission resistance;
- etc.

A fighter must be able to have high distance preference but low distance strike volume, or moderate distance preference and high volume. The current ontology cannot represent those separately.

### Important data limitation
UFCStats does not provide trustworthy phase-time denominators for distance/clinch/ground. Therefore `distance_attempts_per_round` is not a pure conditional attempts-per-minute-while-at-distance measure; it still confounds phase occupancy and action rate. Do not pretend otherwise or fabricate phase-time denominators.

### Exact next step for the next chat/session
Do **not** immediately tune `rating_scale`, change YAML, or rewrite the FSR. First perform an ontology/source audit across all three phases.

Inspect and document the underlying RFS definitions for:

- `distance_attempt_share`
- `distance_attempts_per_round`
- the equivalent clinch pressure/activity inputs
- the equivalent ground pressure/activity inputs
- any existing phase-control / phase-imposition / phase-mix / opponent-suppression features that may be better suited to pure phase preference

Specifically inspect:

- `pipeline/round_stats/build_round_fighter_phase_baseline.py`
- `pipeline/round_stats/build_round_fighter_phase_interaction.py`
- related RFS feature contracts/ontology files
- `scripts/experimental/fsr_clinch_striking_v1.py`
- `scripts/experimental/fsr_ground_striking_v1.py`
- `scripts/experimental/fsr_distance_striking_pressure_v1.py`
- Event MC `style_preferences()` and all phase transition consumers
- Event MC phase-specific strike-rate consumers

Then propose a consistent three-phase ontology that separates:

1. **phase preference / phase imposition**;
2. **conditional action intensity / volume**;
3. **accuracy/conversion/defense**.

The proposal must explicitly state what can be measured directly from existing UFCStats/RFS data, what is only a proxy, and where lack of phase-time exposure prevents a pure estimate.

After defining that ontology, run a measurement-only historical predictive calibration for the proposed volume signal(s): bucket or regress leakage-safe prefight signal against **future out-of-sample next-fight strike attempts**, so we learn the actual mapping from fighter rating/signal to future attempt volume instead of assuming the current exponential `exp((rating-50)/12)` transformation.

Only after those two steps should we authorize an Event MC implementation/A-B change.

### Current working hypothesis
The main striking failure may not be that the stored FSR rating is numerically bad. The stronger hypothesis is that a mixed historical activity/style rating is being interpreted by Event MC as both:

- a direct exponential physical strike-rate multiplier; and
- a phase-preference/transition input.

That coupling can amplify modest rating differences into large simulated volume gaps and can make the same historical evidence count twice. Treat this as the leading striking-generation root-cause hypothesis until the next ontology/source audit either confirms or rejects it.

### Do not lose these parallel unresolved items
- Judge weights are hand-picked and remain uncalibrated; judge calibration is still required, but it will not fix upstream generated-performance reversals by itself.
- Grappling direction looked comparatively strong in the 10-fight audit, but TD completions, control persistence, and submission attempts were often too weak in magnitude.
- Do not modify frozen FSR-32, rebuild the canonical artifact, or globally retune current calibration before completing the phase-preference/volume separation audit.
- Keep the simulator modular: phase selection and action intensity should remain independently replaceable/tunable.

## 2026-08-14 follow-up — global striking volume + phase preference diagnostic

### Artifact path clarification
The old `data/features/round_fighter_state_history.parquet` currently has only 93 columns and does not contain the enriched Phase Baseline / Phase Interaction observation columns required for this audit. The enriched historical artifact used for the measurement diagnostic is:

`data/simulation/rfs_mc_v2_shared_state/historical_fighter_state.parquet`

It contains the Phase Baseline and Phase Interaction evidence, including distance/clinch/ground attempts-per-round and attempt-share fields. A KD-resistance study parquet also contains them, but that study artifact was not used for the decomposition diagnostic.

### Three-phase ontology audit result
Distance, clinch, and ground `*_striking_pressure` are all built with the same structure:

- 60% percentile of phase significant-strike attempts per observed round;
- 40% percentile of that phase's share of all significant-strike attempts.

Therefore all three current pressure traits mix overall striking activity with phase/style allocation. The observed quantities are also algebraically related:

`phase_attempts_per_round = sig_attempts_per_round * phase_attempt_share`

This means the two pressure inputs are not independent pieces of evidence.

### Historical decomposition diagnostic
Using 13,354 to 13,390 fighter-fight observations from `historical_fighter_state.parquet`:

Phase share vs attempts per observed round Spearman correlations:
- DISTANCE: 0.4191
- CLINCH: 0.9113
- GROUND: 0.9719

OLS R2 of attempts-per-round explained by phase share alone:
- DISTANCE: 0.2211
- CLINCH: 0.5331
- GROUND: 0.6394

Adding simple access/context proxies barely changed these values:
- DISTANCE + TD attempts/round: 0.2212
- CLINCH + control/round: 0.5337
- GROUND + TD attempts/round + control/round: 0.6411

Residual activity persistence using prior EWM to next fight:
- DISTANCE: 0.2459
- CLINCH: 0.1927
- GROUND: 0.1446

The residual activity signals were strongly correlated across phases:
- distance vs clinch residual: 0.699
- distance vs ground residual: 0.508
- clinch vs ground residual: 0.612

Interpretation: the residual appears to contain a substantial common fighter-level striking-activity component rather than three cleanly independent phase-specific activity traits.

### Direct test of proposed decomposition
Proposed structure tested:

`predicted phase volume = prior global significant-strike attempts/round * prior phase attempt share`

Global significant-strike volume itself was persistent:
- prior EWM global sig attempts/round -> next-fight global sig attempts/round Spearman = 0.2982, n=11,204.

Phase-preference persistence:
- DISTANCE share: 0.2303
- CLINCH share: 0.2215
- GROUND share: 0.1444

Next-fight phase-volume prediction:

DISTANCE:
- prior raw phase volume: 0.3328
- global volume x phase share: 0.3326
- current FSR pressure: 0.2654

CLINCH:
- prior raw phase volume: 0.2110
- global volume x phase share: 0.2036
- current FSR pressure: 0.1583

GROUND:
- prior raw phase volume: 0.1235
- global volume x phase share: 0.1188
- current FSR pressure: 0.0962

Scaled MAE also modestly favored the proposed decomposition over current FSR pressure in all three phases.

### Interpretation / leading design
This is strong evidence that a cleaner ontology can separate activity from style without materially losing next-fight predictive information.

Leading design is now:

1. **Global striking volume / activity** — how busy the fighter generally is; drives strike-event clocks.
2. **Phase preference / phase allocation** — where the fighter tends to operate or where his offense is allocated; should influence phase transitions/residence or phase-specific allocation, not directly set strike cadence.
3. **Efficiency / defense** — precision, defense, wrestling conversion/TD defense, submission conversion/resistance remain separate.
4. **Dynamic state** — stamina, damage, recovery, etc. remain path-state modifiers.

Do not create three new independent `distance_striking_volume`, `clinch_striking_volume`, and `ground_striking_volume` ratings by simply residualizing the old data. The cross-phase residual correlations suggest starting with one global striking-volume trait, with phase-specific population baselines and separate phase preference. Phase-specific fighter pace modifiers can be added later only if replay proves independent predictive value.

### Same-10-fight implications
The proposed decomposition fixes some, but not all, wrong striking directions:

- Giga/Onama: current pressure slightly favors Giga (54.32 vs 53.14), while proposed prior distance volume favors Onama (46.25 vs 40.44), which is directionally more sensible relative to the actual fight.
- Garry/Prates: proposed prior distance volume still slightly favors Prates (31.04 vs 26.64); Garry's actual output surge was outside both fighters' historical expectations, so this miss is not solved by the ontology change alone.
- Rosas/Morales: proposed prior distance volume still strongly favors Morales (38.05 vs 9.20); Rosas' route must therefore come from phase change/wrestling/grappling suppression rather than pretending he is the historically busier distance striker.
- Loma/Nunes: proposed prior distance volume still favors Nunes (29.00 vs 15.66); Loma's winning route similarly depends more on phase change/wrestling than on baseline distance volume.

This reinforces that not every wrong prediction has the same root cause. Striking-volume ontology, phase-transition/wrestling suppression, grappling magnitude/persistence, and judge calibration remain distinct workstreams.

### Exact next step
Do not implement the new striking-volume trait in EVENT MC yet. Next define the **phase-preference side** carefully, especially because `ground_attempt_share` cannot be treated as pure ground-seeking preference. Determine what combination of strike phase shares, takedown-entry tendencies, control/imposition evidence, and opponent-adjusted phase interaction should represent:

- distance preference;
- clinch preference;
- wrestling / ground-seeking preference.

The phase-preference representation must have a clear simulator job: affect phase selection / transition hazards / residence, while global striking volume affects strike-event cadence. Preserve efficiency traits separately.

After phase-preference ontology is defined and measured, authorize a controlled Event MC A/B implementation rather than globally retuning the existing pressure transform.
