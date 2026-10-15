# EVENT MC V1 — Fresh 100-Fight Predictive Replay

Begin immediately when this prompt is received. Do not ask for confirmation or approval to start. Execute the authorized diagnostic, tests, documentation, commit, and push without waiting for another user message.

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Objective
Run the CURRENT committed EVENT MC V1, unchanged, on a fresh set of 100 historical fights that were not used in the established train or holdout calibration cohorts. Compare predicted winner and predicted method against actual historical winner and method fight-by-fight and in aggregate.

This is predictive validation only. Do not calibrate or change mechanics/YAML/FSR/RNG.

## Fresh cohort selection
Use the first 100 chronological completed UFC fights strictly AFTER `2025-03-22` that satisfy all of the following:
- both fighters have valid leakage-safe frozen FSR-32 prefight snapshots available as required by the simulator;
- fight has a decisive supported result (no draw, no NC, no overturned result);
- actual method can be normalized to `KO_TKO`, `SUB`, or `DEC`;
- fight format is supported by the current simulator;
- exclude every fight in the existing 100-fight train cohort and 50-fight holdout cohort;
- do not cherry-pick by outcome, favorite status, division, or simulator coverage beyond the stated eligibility requirements.

Print the actual selected date range and all 100 bout IDs before running simulation. Fail if exactly 100 eligible fights cannot be formed under this rule; do not silently substitute calibration-cohort fights.

## Simulator state
Use the active post-7M calibration exactly as committed:
- distance/clinch/ground strike clocks = `6.0 / 3.6 / 1.6`
- strike accuracies = `0.40 / 0.68 / 0.70`
- distance TD attempt base = `0.16`
- clinch TD attempt base = `0.24`
- shared TD success offset = `-0.85`
- submission attempt base = `0.045`
- submission bottom multiplier = `1.0`
- submission conversion intercept = `-0.60`
- submission top/bottom bonuses = `0.0 / 0.0`
- KD midpoint = `36`
- finish midpoint = `36`

Frozen FSR SHA-256 must equal exactly:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

If missing, restore only from release tag `event-mc-v1-fsr32-handoff`, verify the exact SHA before and after copying, and never rebuild/rewrite/recompress/commit the parquet.

## Monte Carlo
Run `250 paths/fight` for all 100 fights.
Seed: `20260813`.
Use deterministic per-fight/per-path seed derivation so the replay is exactly reproducible.

Do not use actual result fields anywhere in feature construction, FSR selection, simulator inputs, or path generation. Attach actual winner/method only AFTER simulation for scoring.

## Per-fight required output
Print one readable table with one row per fight and save the same rows machine-readably.

Columns, at minimum:
- event_date
- bout_id
- red_fighter
- blue_fighter
- actual_winner
- actual_method (`KO_TKO`, `SUB`, `DEC`)
- MC P(red win)
- MC P(blue win)
- predicted_winner
- predicted_winner_probability
- winner_correct
- MC P(KO_TKO)
- MC P(SUB)
- MC P(DEC)
- predicted_method
- predicted_method_probability
- method_correct
- MC P(red KO_TKO)
- MC P(red SUB)
- MC P(red DEC)
- MC P(blue KO_TKO)
- MC P(blue SUB)
- MC P(blue DEC)
- predicted_joint_winner_method
- predicted_joint_probability
- joint_correct
- actual_finish_round if available
- actual_elapsed_seconds
- simulated_mean_elapsed_seconds

Joint winner-method probabilities must come directly from path outcome frequencies, not by multiplying marginal winner and method probabilities unless independence is actually true by construction (it should not be assumed).

## Aggregate scoring
Report for the 100-fight fresh cohort:

### Winner
- top-pick winner accuracy
- number correct / 100
- binary Brier score
- binary log loss
- red actual win rate
- mean MC red win probability
- accuracy by confidence buckets: 50-55%, >55-60%, >60-70%, >70-80%, >80%
- count and accuracy for MC favorites >=60%, >=70%, >=80%

### Method
- top-pick method accuracy
- number correct / 100
- multiclass Brier score if practical
- multiclass log loss
- actual vs MC aggregate shares for KO_TKO / SUB / DEC
- per-class precision/recall or, at minimum, class-specific hit rate and predicted count

### Joint winner + method
- top joint-class accuracy across the six classes
- number correct / 100
- six-class log loss if practical
- confusion summary for the six joint classes

### Timing
- historical mean fight duration
- MC mean simulated duration
- MAE of per-fight expected duration versus actual elapsed time
- historical and MC mean nondecision finish time

## Error review
Print all winner misses with:
- fight
- actual winner
- MC predicted winner
- MC winner probability
- actual method
- top predicted method

Then separately print:
- all high-confidence winner misses at >=70%
- all high-confidence winner misses at >=80%
- all method misses where predicted method probability >=60%

Do not explain misses using age, layoff, matchup narrative, or outside knowledge in this task. Just identify them quantitatively for later review.

## Global cohort sanity metrics
Also print historical vs MC aggregate fight-environment metrics for THIS fresh 100-fight cohort using the same definitions from Phase 7N:
- KO_TKO / SUB / DEC shares
- mean duration and mean nondecision finish time
- significant-strike attempts/15, landed/15, accuracy, and D/C/G attempt shares
- TD attempts/15, completions/15, success
- KD/15 and KD/100 modeled comparable landed strikes with semantic caveat
- submission attempts/15

This lets us see whether predictive misses coincide with population-level drift on this untouched cohort.

## Output files
Write a JSON report containing:
- cohort-selection metadata and bout IDs
- current calibration state
- per-fight prediction rows
- aggregate winner/method/joint/timing metrics
- confidence-bucket metrics
- miss lists
- fresh-cohort global historical-vs-MC sanity metrics

A CSV of the 100 fight rows is also acceptable/useful, but do not commit generated data artifacts unless repository convention explicitly requires it. Prefer `/tmp` or ignored diagnostics paths.

## Hard freeze
No YAML change. No mechanics change. No FSR change. No new calibration. No round-specific tuning. No market odds. No external result lookup if repository historical data already supplies the actual outcome.

## Testing
Add focused tests for:
- fresh cohort starts strictly after 2025-03-22;
- no overlap with train/holdout cohorts;
- actual outcome fields are attached only after simulation input construction;
- winner probabilities sum to 1;
- method probabilities sum to 1;
- six joint winner-method probabilities sum to 1;
- marginal probabilities equal the sums of appropriate joint probabilities;
- deterministic seed reproducibility;
- active calibration locks unchanged;
- frozen FSR SHA exact.

Run the relevant EVENT MC tests plus compileall, FSR SHA check, `git diff --check`, and verify `git diff -- config/event_mc_v1.yaml` is empty.

## Continuity
Update `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md` with the exact fresh cohort rule/date range, aggregate winner/method/joint performance, major high-confidence misses, global sanity metrics, and the fact that no calibration was changed.

## Final response
The final Codex response MUST include the full 100-row prediction-vs-actual table, not only a summary.

End with:
`FRESH 100-FIGHT EVENT MC PREDICTIVE REPLAY: PASS`

or FAIL.