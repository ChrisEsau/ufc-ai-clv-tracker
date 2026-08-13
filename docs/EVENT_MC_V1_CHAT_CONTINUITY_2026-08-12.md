# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 14:48 America/Chicago
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
- Phase 7D2 KD target reconciliation: current next phase; measurement only
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Current committed calibration
- `defaults.knockdown.midpoint_impact_ratio = 36.0`
- `defaults.finish.midpoint_impact_ratio = 36.0`

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
At committed 36/36 on the 100-fight x 10-path rerun:
- simulated KO/TKO 25.6%, SUB 5.7%, DEC 68.7%
- simulated KD/100 landed 0.438
- simulated KD/15min 0.383
- simulated submission attempts/path 0.380
- simulated submission attempts/15min 0.423
- simulated path share with >=1 attempt 27.4%
- simulated P(SUB|attempt) 15.0%
- simulated mean non-decision finish time 387.43s

Finish midpoint 36 remains supported after correction. KD midpoint 36 is unresolved because corrected historical KD/15 and KD/100 landed now pull in different directions; a narrow 32/36/40 check favored 40 only under the prior combined objective.

## Submission position lock
For future submission conversion calibration, top and bottom submission attempts are to be treated 1:1 for now. Do not apply an intrinsic top-position conversion bonus unless UFC-specific evidence supports it. Current explicit top-position bonus must be neutralized before/within the first authorized submission-conversion calibration step, not silently retained.

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
