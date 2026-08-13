# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 12:28 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0: PASS
- Phase 1: PASS
- Phase 2A: PASS
- Phase 2B: PASS
- Phase 3: PASS
- Phase 4A: PASS
- Phase 4B1 impact/trauma/KD: PASS; calibration intentionally deferred
- Phase 4B1 config externalization: PASS
- Phase 4B2 KO/TKO: PASS
- Phase 4B2 single-fight runner + terminal-action accounting: PASS
- Phase 4C submission finishes: PASS after pre-action stamina correction
- Phase 5A deterministic judging: IMPLEMENTED, but final PASS withheld pending one narrow aggression-family filter correction
- Age, tactical urgency, population calibration, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`
SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Hard locks
- one fight clock; engine-only state mutation; named deterministic RNG streams; scheduler UFC-agnostic; sinks observer-only;
- FSR-32 read-only; no rebuild;
- current action uses pre-action state; its stamina cost affects later events;
- striking power enters physiology once; KD/KO do not multiply power/stamina again;
- no deterministic health-bar finish;
- global calibration + optional future partial weight-class override; committed overrides empty;
- terminal action/outcome counted exactly once before one lifecycle finish;
- judging: no draws, no 10-8/10-10, every round exactly one 10-9 winner, no three-judge noise in Phase 5A.

## Phase 4C final
Implementation `6de03e753633b9b527f76324a21708a61023317f` plus sequencing correction `dc33b61afec933883d6c4692a421bcf0b709a4a8`.
Submission conversion uses existing frozen `submission_conversion`/`submission_resistance`, keeps attempt generation separate, uses pre-action stamina, and is terminal through the shared lifecycle.
Final gate: `PHASE 4C SUBMISSION FINISH MECHANICS GATE: PASS`.

## Phase 5A deterministic judging
Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE5A_DETERMINISTIC_JUDGING_2026-08-13.md`
Prompt commit: `2c3fbef6b9f2f12475d5669c6aa3f06157479bc4`
Implementation commit: `5df6f82b60ce02d0bcc9c4358d0528d5eb78d7ad`

Implemented:
- round-local observer judging;
- primary effectiveness = effective striking + effective grappling;
- striking = impact + small landed support + KD emphasis;
- grappling = submission threat + successful TDs + successful reversals; passive control excluded from primary;
- aggression secondary; control tertiary;
- 10-9 only, no draws;
- scheduled horizon -> terminal DEC by round majority;
- runner round cards and DEC summaries;
- descriptive historical decision rate 46.61%; no calibration.

Independent review finding:
`DeterministicJudgingModel.on_event()` currently increments aggression for every `ActionAttempt`, so defensive/non-offensive `ground_escape`, `clinch_separation`, and `ground_reversal` receive aggression credit. This conflicts with the locked definition of effective aggression as offensive initiative.

Governing narrow fix prompt:
`docs/EVENT_MC_V1_CODEX_PHASE5A_AGGRESSION_FILTER_FIX_2026-08-13.md`
Prompt commit: `1c62967efe46e3f023cf269edd922530171da239`

Aggression-credit families only:
- strike
- takedown
- clinch_entry
- clinch_strike
- clinch_takedown
- ground_strike
- submission_attempt

No aggression credit for ground_escape, clinch_separation, or ground_reversal. Successful reversal keeps its existing effective-grappling credit.

Expected return:
`PHASE 5A AGGRESSION FILTER FIX GATE: PASS` or FAIL.

Next assistant action: independently review that exact correction. If clean, accept final `PHASE 5A DETERMINISTIC JUDGING GATE: PASS`.

## Checkpoint history
- 001-041: Phases 0 through 4C completed and independently accepted; terminal accounting and submission pre-action stamina corrections included.
- 042: user authorized Phase 5A deterministic judging with no draws and 10-9-only rounds.
- 043: Phase 5A implemented at `5df6f82b...`; core architecture/tests/diagnostics completed without calibration.
- 044: independent review found defensive/non-offensive action families receiving aggression credit; narrow fix prompt issued at `1c62967e...`.
