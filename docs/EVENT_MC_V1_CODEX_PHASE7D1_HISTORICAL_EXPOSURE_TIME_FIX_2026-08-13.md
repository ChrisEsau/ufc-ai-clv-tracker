# EVENT MC V1 — Phase 7D1 Historical Exposure-Time Correction

Measurement correction only. Do not calibrate or change simulator mechanics.

## Critical bug
The authoritative master pipeline stores `match_time_sec` as TOTAL ELAPSED FIGHT TIME. See `pipeline/common/fight_time.py` and `pipeline/data_maintenance/run_staged_derived_stats_transformer.py`.

EVENT MC diagnostics currently use `observed_duration_seconds(row) = (finish_round - 1) * 300 + match_time_sec`, which double-counts prior rounds when `match_time_sec` is already elapsed.

## Required work
1. Correct EVENT MC historical duration handling so authoritative master rows use elapsed `match_time_sec` exactly once.
2. Prefer the shared canonical fight-time helper / repair semantics rather than duplicating ambiguous logic.
3. Add tests including:
   - R1 elapsed 120 -> 120
   - R2 elapsed 420 -> 420, NOT 720
   - R3 elapsed 750 -> 750, NOT 1350
   - compatibility handling for any legacy final-round-only value must be explicit and tested, not guessed silently.
4. Search all EVENT MC diagnostics/calibration code for use of `observed_duration_seconds`, `match_time_sec`, or manual `(finish_round-1)*300` historical exposure arithmetic and correct the same semantic bug everywhere.
5. Do NOT change any simulation config or mechanics.

## Recompute affected historical anchors
Using the same 100-fight x 10-path cohort where applicable, report OLD vs CORRECTED for:
- historical observed seconds/fight
- historical strike attempts/15min
- historical landed strikes/15min
- historical KD/15min
- historical submission attempts/15min
- historical mean non-decision finish time

State explicitly which metrics are unaffected:
- method shares
- attempts/fight
- KD/fight
- KD/100 landed
- finish rounds
- simulator outputs for unchanged seeds/config

## Calibration revalidation
Current committed config remains:
- KD midpoint = 36
- finish midpoint = 36

Do not change these values in this phase.

After correcting historical exposure:
- rerun the current 100-fight x 10-path Phase 7D diagnostic;
- rerun enough Phase 7B/7C diagnostics to determine whether midpoint 36/36 remains supported under corrected historical targets;
- do NOT automatically recalibrate or promote a different value.

For KD, emphasize KD/100 landed because it is exposure-independent. Report corrected KD/15 historical comparison as a revalidation diagnostic.

For KO/TKO, method-share calibration is exposure-independent and should remain directly comparable. Recompute finish-time guardrails using corrected historical duration.

Submission calibration remains BLOCKED until this correction is accepted.

Expected final line:
PHASE 7D1 HISTORICAL EXPOSURE TIME FIX GATE: PASS
or FAIL.
