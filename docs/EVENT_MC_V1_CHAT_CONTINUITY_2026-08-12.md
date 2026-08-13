# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 14:33 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6: PASS; exposure-normalized historical metrics require recheck after duration semantic issue
- Phase 7A: PASS; historical per-time anchors require recheck
- Phase 7B KD midpoint 36: committed; revalidation requested, no value change authorized
- Phase 7B2: PASS
- Phase 7C finish midpoint 36: committed; timing guardrail revalidation requested, no value change authorized
- Phase 7D submission decomposition: measured; next calibration deferred pending historical-time correction
- Phase 7D1 historical exposure-time correction: current phase
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Current committed calibration
- `defaults.knockdown.midpoint_impact_ratio = 36.0`
- `defaults.finish.midpoint_impact_ratio = 36.0`

## Phase 7D pre-correction result
Implementation: `259662a2766bf469abfd13de08a19579757fb3c7`
100 fights x 10 paths:
- historical/sim KO_TKO 25.0% / 25.6%
- historical/sim SUB 17.0% / 5.7%
- historical/sim DEC 58.0% / 68.7%
- historical attempts/fight 0.610; simulated attempts/path 0.380
- historical fights with attempt 37.0%; simulated paths with attempt 27.4%
- simulated P(SUB|attempt) 15.0%

## Historical duration semantic issue
Repository review established that authoritative master `match_time_sec` is already TOTAL ELAPSED FIGHT TIME. `pipeline/common/fight_time.py` and the staged derived-stats transformer explicitly construct it that way.

EVENT MC `observed_duration_seconds()` added prior-round seconds a second time. This can overstate historical exposure for rounds 2+.

Metrics requiring correction/recheck:
- historical strike attempts/15min
- historical landed/15min
- historical KD/15min
- historical submission attempts/15min
- historical mean non-decision finish time
- any similar historical per-time metric using the same arithmetic

Metrics not affected by this specific issue:
- method shares
- attempts/fight
- KD/fight
- KD/100 landed
- finish-round labels
- simulator outputs under unchanged seeds/config

## Phase 7D1
Prompt: `docs/EVENT_MC_V1_CODEX_PHASE7D1_HISTORICAL_EXPOSURE_TIME_FIX_2026-08-13.md`
Prompt commit: `cb157a2b64b8937f04b6a57bf2b3d206ff693105`

Required: correct historical duration semantics, add tests, search all EVENT MC diagnostics for duplicated elapsed-time arithmetic, recompute affected anchors, rerun Phase 7D at unchanged 36/36, and revalidate Phase 7B/7C conclusions without changing calibration values.

Expected return: `PHASE 7D1 HISTORICAL EXPOSURE TIME FIX GATE: PASS` or FAIL.

Phase 7D1 result: the authoritative elapsed-time contract is now used throughout EVENT MC diagnostics, with legacy final-round clock compatibility available only through an explicit argument. On the same 100-fight cohort, old versus corrected historical values were: observed seconds/fight 1276.16 -> 757.16; strike attempts/15min 169.498 -> 285.681; landed/15min 93.176 -> 157.045; KD/15min 0.261 -> 0.440; submission attempts/15min 0.430 -> 0.725; and mean non-decision finish time 652.762s -> 402.762s. Method shares, attempts/fight, KD/fight, KD/100 landed, finish-round shares, and identical-seed simulator outputs were unchanged.

At fixed midpoint 36/36, finish midpoint 36 remains supported by exposure-independent KO/TKO shares and improved corrected timing guardrails. KD midpoint 36 is not independently reconfirmed: the corrected exposure target plus unchanged KD/100-landed evidence favored midpoint 40 over 36 among the narrow 32/36/40 revalidation grid. No calibration changed; KD and submission calibration remain blocked pending review.
