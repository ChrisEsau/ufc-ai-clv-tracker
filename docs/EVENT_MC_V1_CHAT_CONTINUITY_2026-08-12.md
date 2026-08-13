# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 18:06 America/Chicago
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
- Phase 7I strike exposure definition + baseline audit: PASS; UFCStats significant strikes are the primary comparator
- Phase 7J global strike-attempt generation + phase-composition calibration: current next phase
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Current committed calibration
- `defaults.distance.strike_attempts_per_30s = 5.0`
- `defaults.clinch.strike_attempts_per_30s = 1.2`
- `defaults.ground.strike_attempts_per_30s = 1.6`
- `defaults.distance.strike_accuracy = 0.40`
- `defaults.clinch.strike_accuracy = 0.68`
- `defaults.ground.strike_accuracy = 0.70`
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

## Phase 7D2 KD target reconciliation
Phase 7D2 compared KD midpoints 32, 36, 40, 44, and 48 with finish midpoint fixed at 36. Midpoint 48 matched KD/100 landed but materially undershot KD/path and KD/15 because modeled landed-strike exposure was much lower than historical total-strike exposure. No KD promotion. KD and finish midpoints remain 36. Gate: PASS.

## Phase 7E bottom submission-attempt neutralization
Changed only `defaults.submission_attempts.bottom_multiplier` from 0.55 to 1.00. Attempts rose from 0.380 to 0.474/path and 0.423 to 0.529/15min; paths with attempt rose from 27.4% to 32.3%. Gate: PASS.

## Phase 7F submission conversion position neutralization
Changed only `defaults.submission_finish.top_position_bonus` from 0.25 to 0.0. Observed top/bottom conversion gap narrowed from 4.07pp to 0.92pp. Gate: PASS.

## Phase 7G global submission-attempt rate calibration
Searched only `submission_attempts.base_30s`. Train favored approximately 0.050-0.055 while holdout favored current 0.045 or lower, so no value was promoted. `base_30s` remains 0.045.

## Phase 7H global submission-conversion intercept calibration
Restored frozen FSR-32 from release only after exact SHA verification. Common-seed coarse and finalist searches on 100 train and 50 holdout fights both supported intercept -0.60. Train historical/simulated SUB was 17.0%/17.3%; holdout 18.0%/18.2%. Only `submission_finish.intercept` was promoted from -2.20 to -0.60. KD/finish midpoints remain 36. Gate: PASS.

## FSR-32 release recovery procedure
If the ignored local FSR-32 parquet is missing in a future Codex sandbox, do not rebuild it. Recover it from GitHub Release tag `event-mc-v1-fsr32-handoff`, asset `fsr_32_prefight_snapshots.parquet`. Preferred command: `gh release download event-mc-v1-fsr32-handoff --repo ChrisEsau/ufc-ai-clv-tracker --pattern 'fsr_32_prefight_snapshots.parquet' --dir /tmp/event_mc_v1_fsr32_handoff`. Verify SHA-256 equals exactly `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`, copy byte-for-byte to `data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`, and verify destination SHA again. Never rebuild, rewrite, recompress, or commit the parquet. Resume the current blocked phase after recovery.

## Phase 7I strike exposure definition + baseline audit
Phase 7I completed measurement-only with no YAML or mechanics changes. One EVENT MC strike is a meaningful offensive strike opportunity; every landed modeled strike enters impact/trauma/KD and there is no separate low-impact/non-significant or target-location family. UFCStats significant strikes are therefore the closest primary comparator for both attempt generation and landing probability; UFCStats total strikes remain a secondary upper-bound diagnostic.

Historical significant-strike attempts/15 versus modeled attempts/15:
- Train: 238.170 historical vs 199.307 model (83.68% of target)
- Holdout: 256.591 historical vs 198.629 model (77.41% of target)

Historical significant-strike accuracy versus modeled accuracy:
- Train: 48.20% historical vs 44.32% model
- Holdout: 47.46% historical vs 47.03% model

Historical significant-strike attempt shares versus modeled shares:
- Train: distance 86.32% vs 91.97%; clinch 6.94% vs 2.75%; ground 6.74% vs 5.28%
- Holdout: distance 89.83% vs 92.38%; clinch 4.91% vs 2.21%; ground 5.26% vs 5.41%

Conclusion: the main remaining strike discrepancy is low attempt exposure plus deficient clinch composition; ground share is already close; landing probability is secondary and should remain frozen during attempt calibration. Gate: PASS.

## Phase 7J global strike-attempt generation + phase-composition calibration
Phase 7J is authorized to calibrate only strike attempt generation. Primary historical target is UFCStats significant-strike attempts/15 over the same Phase 7G/7H/7I train and holdout cohorts. Round-specific calibration is deferred until global event rates are close; round-specific RFS parquet stats will later be used for R1/R2/R3 validation across strikes, TDs, submissions, KD, control, and related activity.

Authorized candidate parameters only:
- `defaults.distance.strike_attempts_per_30s` (current 5.0)
- `defaults.clinch.strike_attempts_per_30s` (current 1.2)

`defaults.ground.strike_attempts_per_30s = 1.6` is frozen for this phase because modeled ground significant-strike share is already close to historical. All strike landing accuracies are frozen: distance 0.40, clinch 0.68, ground 0.70.

Primary promotion target: improve overall modeled strike attempts/15 toward historical significant-strike attempts/15 on both train (238.170) and holdout (256.591). Secondary composition guardrail: materially improve deficient clinch attempt share while avoiding a worse ground mismatch; distance share should fall toward historical as a consequence. Do not calibrate round-by-round shape yet.

Suggested coarse common-seed grid: distance `5.0, 5.5, 6.0, 6.5` crossed with clinch `1.2, 2.0, 2.8, 3.6`, 3 paths/fight, seed 20260813, 100 train and 50 holdout fights. Use current committed baseline as an explicit candidate. Select a small Pareto/finalist region and rerun at 10 paths/fight. Do not use a hidden weighted scalar objective; report train and holdout global attempt errors and phase-share errors separately. Promotion requires both temporal splits to support the same practical region, overall attempt exposure to improve materially on both, clinch share to improve, and KO/SUB/KD/timing guardrails to remain interpretable. If no candidate satisfies this, promote nothing.

Hard freeze during 7J: all strike accuracy, ground strike rate, phase transition/residence rates, damage/impact, KD/KO, submissions, stamina, TDs, judging, RNG, FSR, age, urgency, weight-class overrides. Extra strikes may naturally alter censoring and outcomes; those effects must be reported rather than compensated by tuning another subsystem.

After any 7J promotion, rerun current train/holdout guardrails and mark KD36/finish36/submission -0.60 for revalidation after strike exposure is stabilized. No round-specific calibration yet.

Expected return: `PHASE 7J GLOBAL STRIKE ATTEMPT GENERATION CALIBRATION GATE: PASS` or FAIL/no promotion.
