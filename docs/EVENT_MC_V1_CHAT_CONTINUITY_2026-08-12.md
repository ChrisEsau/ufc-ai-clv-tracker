# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 11:35 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. Preserve gate state, prompt path, hard locks, expected return, and checkpoint history.

## Current gate state
- Phase 0: PASS
- Phase 1: PASS
- Phase 2A: PASS
- Phase 2B: PASS
- Phase 3: PASS
- Phase 4A: PASS
- Phase 4B1 impact/trauma/KD: PASS
- Phase 4B1 KD calibration: OPEN / intentionally deferred by user
- Phase 4B1 config externalization: PASS
- Phase 4B2 KO/TKO finish mechanics: PASS after independent review
- Phase 4B2 single-fight runner: PASS
- Phase 4B2 terminal-action accounting correction: PASS / bookkeeping-only
- Phase 4C submission finish mechanics: IMPLEMENTED, independent review found one pre-action stamina sequencing fix required before PASS
- Judging, age, tactical urgency, population calibration, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`
SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Key implementation commits:
- Phase 1 `1debecab69a141bf2f81179f3436af569733b750`
- Phase 2A `5b7574c7689ffa2e55821a49fca47a2c1c937991`
- Phase 2B `004740e54618c134e08aa553164c381508811481`, `809389bdabe208e93536034bc795bbcf7e1ab038`
- Phase 3 `5a6c15af9c2315f23c231d0f34cd3cefddba4578`
- Phase 4A `8155bc45de5fa26fa6077dd870716234d54690c9`
- Phase 4B1 physiology `65a6f2d4e703af5c777f1943f134728b715d4c55`
- Config externalization completion `b8b2b870595c9cc62255b4d63b7c20de56e9550f`, `77661563c3a833a3e87b60e4b3ae6caecd648cb8`
- Phase 4B2 KO/TKO `3960bae022ac7d3129a593a102704d2b39a46b28`
- Single-fight runner completion `da622892e33753e3f4aa10b00c88963150630f41`
- Phase 4C implementation `6de03e753633b9b527f76324a21708a61023317f`

## Hard architecture locks
- one authoritative fight clock;
- engine-only state mutation through typed deltas;
- named deterministic RNG streams;
- scheduler UFC-agnostic;
- sinks/trace tools observer-only;
- FSR-32 read-only, never rebuilt;
- current action uses pre-action dynamic/physiological state; its stamina cost affects subsequent events;
- striking power enters physiology once through impact;
- stamina enters striking power once through pre-action DynamicModifiers;
- KD/KO do not multiply raw power/stamina again;
- cumulative trauma persists; acute vulnerability decays continuously;
- no deterministic health-bar exhaustion finish;
- one immutable effective calibration from `config/event_mc_v1.yaml` with optional future partial weight-class override;
- committed weight-class overrides remain empty;
- terminal action accounting preserves the already-resolved action/outcome exactly once before the single lifecycle finish while blocking all future primary events.

## Phase 4B2 reviewed result
`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: PASS`.

The finish model consumes existing impact/post-trauma physiology, derives current finish resistance from durability/KD resistance plus trauma/acute vulnerability, permits fresh one-shot finishes, and terminates with one `FightFinished` lifecycle event and no later primary events. Historical KO/TKO anchor is 32.79%; five-fixture mechanics rates were 70-95%, intentionally left uncalibrated.

## Codespaces single-fight runner
Runner completion commit: `da622892e33753e3f4aa10b00c88963150630f41`.
It resolves canonical historical fight IDs to frozen FSR-32 prefight profiles, supports deterministic seeds, prints chronological action/phase/stamina/physiology/KD/finish traces, and prints multi-path KO/TKO/SUB/horizon/KD/TD/phase/control summaries. Lewis/Daukaus ID `4b7ec02b39fc6f70` resolves correctly.

A manual trace exposed a lifecycle/accounting bug in which a finishing strike's own `ActionOutcome` was skipped after terminal state. This was corrected bookkeeping-only: the already-resolved finishing action is counted exactly once, winner/time/RNG physics remain unchanged, one `FightFinished` is emitted, and all future primary events/time advance remain blocked. Lewis/Daukaus seed `20260813` stays blue KO/TKO at `26.623215196672668`, now with four landed blue strikes counted.

## Phase 4C Submission Finish Mechanics

User authorized Phase 4C on 2026-08-13 11:16 America/Chicago.
Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4C_SUBMISSION_FINISHES_2026-08-13.md`
Prompt commit: `a0ed3d38bc0b4c5d0fd8933e021db89e14472c33`.

Implementation commit:
`6de03e753633b9b527f76324a21708a61023317f`.

Implemented design:
- frozen FSR-32 already contains leakage-safe `submission_conversion` and `submission_resistance`; adapter exposes both without rebuilding FSR;
- attacker threat = `0.75 * submission_conversion + 0.25 * submission_pressure`;
- defender resistance = `0.75 * submission_resistance + 0.25 * control_resistance`;
- logit = `-2.20 + (threat - resistance)/12 + position_bonus + 0.50*(attacker_stamina-defender_stamina)`;
- top bonus `0.25`, bottom bonus `0.0`, numerical clipping only;
- submission attempt generation remains unchanged and separate from conversion;
- all coefficients are under `defaults.submission_finish`; real weight-class overrides remain empty;
- existing `RNGStream.SUBMISSION` owns conversion sampling;
- terminal SUB uses shared lifecycle and preserves the already-resolved attempt/outcome exactly once before one `FightFinished`.

Descriptive completed-fight master anchor (not calibration): 8,654 fights, 19.77% SUB finish rate, 0.748 recorded attempts/fight, 42.21% with a recorded attempt, 46.35% SUB finish rate conditional on at least one recorded attempt. This last figure is fight-level descriptive exposure, not attempt-level conversion calibration.

Five-fixture mechanics diagnostic, 50 paths each, reported SUB attempts/path from 0.02 to 0.30 and SUB finishes from 0% to 10% while KO/TKO remained dominant because striking finish calibration is intentionally open.

### Independent review finding — pre-action stamina sequencing
`PhaseCandidate.resolve()` computes the current submission attempt stamina cost in its primary `StateDelta`. The engine applies that delta before calling `SubmissionFinishModel.resolve()`. The current model then reads stamina directly from the post-action `FightState`.

Therefore the current submission attempt's own stamina cost can reduce its own `P(SUB)`. This conflicts with the established Phase 4A timing lock that the current action uses pre-action state and its stamina cost affects subsequent events.

Required correction: calculate current-attempt submission conversion from pre-action stamina while still applying the attempt cost exactly once for future events. No formula coefficients, attempt rates, action costs, RNG ownership, FSR traits, or other mechanics may change.

Governing correction prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4C_SUBMISSION_PREACTION_STAMINA_FIX_2026-08-13.md`
Prompt commit: `11d5b270f5ce1584be6747825e437a5bb0afd269`.

Expected return:
`PHASE 4C SUBMISSION PRE-ACTION STAMINA FIX GATE: PASS` or FAIL.

Next assistant action: independently review the correction and, if clean, accept `PHASE 4C SUBMISSION FINISH MECHANICS GATE: PASS`. Do not authorize judging automatically without user direction.

## Checkpoint history
- 001-010: Phase 0 baseline established and exact FSR-32 frozen; PASS.
- 011-012: Phase 1 implemented; PASS.
- 013-015: Phase 2A implemented; PASS.
- 016-017: Phase 2B wrestling-entry ontology corrected; PASS.
- 018-020: Phase 3 clinch/ground flow implemented; PASS.
- 021-023: Phase 4A stamina/dynamic modifiers implemented; PASS.
- 024-026: Phase 4B1 impact/trauma/KD implemented; architecture PASS, KD overprediction exposed, calibration deferred.
- 027-030: calibration externalization completed after one review fix; PASS; future weight-class seam established with no active overrides.
- 031-032: Phase 4B2 KO/TKO authorized, implemented, independently PASS; finish rates intentionally left uncalibrated.
- 033-036: Codespaces single-fight runner built and review-completed; PASS.
- 037: terminal-action accounting corrected bookkeeping-only after manual trace exposed missing finishing ActionOutcome.
- 038: user authorized Phase 4C submission mechanics; governing prompt committed.
- 039: Phase 4C implemented at `6de03e75...`; traits/config/terminal mechanics/diagnostics added without calibration.
- 040: independent Phase 4C review found current submission attempt cost feeding its own conversion through post-action stamina; narrow pre-action sequencing fix issued at `11d5b270...`.
- 041: submission conversion now reads the immutable pre-action snapshot while the attempt cost remains applied once for later events. Oliveira/Poirier seed 22 remains a red SUB at `180.33041448667973`, with `pSUB` moving from 19.205% to 19.411% solely from corrected stamina timing; Lewis/Daukaus remains unchanged.
