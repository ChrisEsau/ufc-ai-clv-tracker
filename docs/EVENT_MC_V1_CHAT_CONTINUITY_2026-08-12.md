# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 16:46 America/Chicago
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
- Phase 7D2 KD target reconciliation: PASS; no KD promotion
- Phase 7E bottom submission-attempt neutralization: PASS
- Phase 7F submission conversion position neutralization: PASS
- Phase 7G submission-attempt calibration: diagnostic complete; promotion gate FAIL / no promotion
- Phase 7H submission-conversion intercept calibration: harness implemented; execution blocked until frozen FSR-32 is restored from the GitHub Release asset
- Age, urgency, real weight-class tuning: not authorized

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## FSR-32 recovery / Codex sandbox access procedure

If a fresh Codex sandbox or checkout does not contain the ignored frozen FSR-32 parquet, **do not search indefinitely for a local copy and do not rebuild FSR-32**. The exact frozen artifact is intentionally available as a GitHub Release asset for recovery.

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Release tag: `event-mc-v1-fsr32-handoff`

Release asset: `fsr_32_prefight_snapshots.parquet`

Expected asset size: `3,364,786` bytes

Required SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Expected ignored destination:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Preferred recovery method when `gh` is available:

```bash
mkdir -p /tmp/event_mc_v1_fsr32_handoff

gh release download event-mc-v1-fsr32-handoff \
  --repo ChrisEsau/ufc-ai-clv-tracker \
  --pattern 'fsr_32_prefight_snapshots.parquet' \
  --dir /tmp/event_mc_v1_fsr32_handoff

sha256sum /tmp/event_mc_v1_fsr32_handoff/fsr_32_prefight_snapshots.parquet
```

The downloaded SHA must match the frozen SHA exactly before use. If it differs, STOP and report the observed checksum.

After checksum PASS:

```bash
mkdir -p data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow
cp /tmp/event_mc_v1_fsr32_handoff/fsr_32_prefight_snapshots.parquet \
  data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet

sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet
```

Verify the destination SHA also matches exactly. The copy must be byte-for-byte only: do not rewrite, rebuild, reserialize, recompress, normalize, or otherwise transform the parquet. Do not commit the parquet.

Historical governing release-ingest prompt:
`docs/EVENT_MC_V1_CODEX_FSR32_RELEASE_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

Important current-state rule: that historical prompt says to resume Phase 0 because it was written during Phase 0. **Do not resume Phase 0 now.** Reuse only its artifact-ingest/checksum procedure, then resume whatever current deferred phase requires FSR-32. As of this update, that is **Phase 7H submission-conversion intercept calibration**.

## Current committed calibration
- `defaults.knockdown.midpoint_impact_ratio = 36.0`
- `defaults.finish.midpoint_impact_ratio = 36.0`
- `defaults.submission_attempts.base_30s = 0.045`
- `defaults.submission_attempts.bottom_multiplier = 1.0`
- `defaults.submission_finish.top_position_bonus = 0.0`
- `defaults.submission_finish.bottom_position_bonus = 0.0`
- `defaults.submission_finish.intercept = -2.20` pending Phase 7H calibration

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

Phase 7D2 result: the exact common-seed 100-fight x 10-path comparison was completed for in-memory KD midpoints 32, 36, 40, 44, and 48 with finish midpoint fixed at 36. No combined score or ranking was used. Historical landed exposure was 132.12/fight and 157.045/15min, versus roughly 78-80/path and 87.4-87.5/15min simulated across candidates. Midpoint 48 closely matched KD/100 landed (0.269 vs 0.280 historical) but materially undershot KD/path (0.215 vs 0.370) and KD/15min (0.235 vs 0.440). Midpoint 36 was closer on KD/path (0.344) and KD/15min (0.383) but high on KD/100 landed (0.438). This conflict is primarily attributable to lower/non-definition-identical simulated landed-strike exposure; corrected evidence does not justify a midpoint promotion. Committed KD and finish midpoints remain 36. Submission conversion was untouched, and the future 1:1 top/bottom lock remains in force.

## Phase 7E bottom submission-attempt neutralization

Phase 7D2 passed; KD midpoint and finish midpoint remain 36. Phase 7E changes only `defaults.submission_attempts.bottom_multiplier` from 0.55 to 1.00. The 100-fight x 10-path rerun increased attempts from 380 to 474 (0.380 to 0.474/path; 0.423 to 0.529/15min) and paths with an attempt from 27.4% to 32.3%. Top/bottom attempts were 224/250, with exposure-normalized rates of 1.122/1.252 per 15 positional ground minutes. SUB moved from 5.7% to 6.6%; KO/TKO remained 25.5% and DEC was 67.9%. Conversion remains frozen: 66 finishes from 474 attempts (13.9%), with the existing `submission_finish.top_position_bonus = 0.25` temporarily unchanged until a dedicated conversion phase. The remaining deficit is now conversion-dominant, although global attempt exposure remains below historical 0.610/fight and 0.725/15min.

## Phase 7F submission conversion position neutralization

Phase 7D2 and Phase 7E remain PASS. Phase 7F changes only `defaults.submission_finish.top_position_bonus` from 0.25 to 0.0; `bottom_position_bonus` remains 0.0, the conversion intercept remains -2.20, and `submission_attempts.bottom_multiplier` remains 1.0. The neutral 100-fight x 10-path baseline produced 477 attempts, 61 SUB finishes, and 12.79% conversion. Observed top/bottom conversion narrowed from 16.07%/12.00% to 13.27%/12.35% (gap 4.07pp -> 0.92pp). Attempts remained stable at 0.477/path and 0.531/15min; SUB moved from 6.6% to 6.1%, KO/TKO remained 25.5%, and DEC moved to 68.4%. No global submission calibration is authorized yet. A simple uncensored exposure ratio implies roughly 35.6% conversion would be needed for 17% SUB at current attempt exposure, about 2.79x the neutral observed conversion; this is only a planning diagnostic, not a fitted intercept. KD midpoint and finish midpoint remain 36.

## Phase 7G global submission-attempt rate calibration

Phase 7F passed: attempt generation and conversion are position-neutral (`bottom_multiplier = 1.0`; top/bottom conversion bonuses = 0.0), while the conversion intercept remains -2.20 and KD/finish midpoints remain 36. Phase 7G searched only `submission_attempts.base_30s` using common seeds. The six-point coarse grid used 3 paths/fight; finalists 0.045, 0.050, and 0.055 used 100 train fights (2020-01-18–2020-07-25), 50 holdout fights (2025-01-11–2025-03-22), and 10 paths/fight.

Train supported approximately 0.050–0.055: at 0.055, simulated exposure was 0.617 attempts/path, 0.691/15min, and 38.2% paths with an attempt versus historical 0.610, 0.725, and 37.0%. Holdout supported the existing 0.045 or lower: at 0.045, simulated exposure was already 0.554/path and 0.613/15min versus historical 0.480 and 0.562, while its 33.8% path share remained below historical 38.0%. Increasing the base improved holdout path share but worsened both count/rate overexposure. Because train and holdout do not support the same region across all three primary metrics, no value was promoted and `base_30s` remains 0.045. Conversion calibration remains deferred pending acceptance of this temporal disagreement.

## Phase 7H global submission-conversion intercept calibration

Phase 7G diagnostic execution is complete and its promotion gate remains FAIL/no promotion. Attempt base remains 0.045, bottom multiplier remains 1.0, both position bonuses remain 0.0, and KD/finish midpoints remain 36. Phase 7H authorizes only `submission_finish.intercept` candidates evaluated end to end against split-specific historical SUB fight share.

The Phase 7H harness and isolation tests were implemented, but a fresh Codex checkout did not contain the ignored frozen FSR-32 parquet required to resolve either chronological cohort. Candidate execution therefore stopped before simulation; no intercept was promoted and the committed value remains -2.20. The exact frozen parquet is recoverable from the GitHub Release asset documented above. Phase 7H should be resumed only after release download, pre-use SHA verification, byte-for-byte copy to the ignored destination, and destination SHA verification.
