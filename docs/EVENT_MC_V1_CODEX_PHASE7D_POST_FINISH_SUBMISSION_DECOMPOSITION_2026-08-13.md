# EVENT MC V1 — Phase 7D Post-Finish Submission Decomposition

Measurement only. Do not change simulator mechanics or config.

Current committed calibration:
- knockdown.midpoint_impact_ratio = 36.0
- finish.midpoint_impact_ratio = 36.0

Purpose: remeasure submission exposure and method distribution after Phase 7C because reduced KO/TKO censoring lengthened paths substantially. Old submission-attempt measurements are now stale.

Use the same mature-fighter historical framework and actual elapsed exposure.

Run at minimum the same 100-fight x 10-path cohort from start-year 2020 with seed 20260813. Also report chronological train 2020-2024 and holdout 2025+ summary if practical without changing mechanics.

Report historical vs simulated:
- KO/TKO, SUB, DEC shares
- submission attempts per fight/path
- submission attempts per 15 elapsed minutes
- share of fights/paths with >=1 submission attempt
- submission finishes
- P(SUB | simulated submission attempt)
- attempt exposure by top/bottom position if available without new mechanics
- attempt exposure by round
- attempt exposure by phase/ground residence where available
- ground seconds per 15 minutes and control seconds if existing sinks support it
- mean non-decision finish time and finish-round distribution
- winner accuracy, Brier, log loss as diagnostics only

Preserve current KD and KO decomposition metrics as guardrails:
- KD/100 landed
- KD/15min
- KO/TKO share

Do not tune submission attempt rates or submission conversion yet. Do not change action rates, phase rates, stamina, judging, RNG, FSR, weight-class overrides, age, urgency, damage, KD, or finish parameters.

The output must identify whether the remaining SUB deficit is primarily attempt-exposure limited, conversion limited, or both.

Expected final line:
PHASE 7D POST-FINISH SUBMISSION DECOMPOSITION GATE: PASS
or FAIL.
