# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 11:16 America/Chicago
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
- Phase 4B2 terminal-action accounting correction: completed/physics-neutral; verify final implementation commit before Phase 4C coding if needed
- Phase 4C submission finish mechanics: AUTHORIZED / current next phase
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
- Single-fight runner initial `05485f0bd0460ae726fb2f0283373a727464cb38`
- Single-fight runner completion `da622892e33753e3f4aa10b00c88963150630f41`

## Hard architecture locks
- one authoritative fight clock;
- engine-only state mutation through typed deltas;
- named deterministic RNG streams;
- scheduler UFC-agnostic;
- sinks/trace tools observer-only;
- FSR-32 read-only, never rebuilt;
- striking power enters physiology once through impact;
- stamina enters power once through pre-action DynamicModifiers;
- KD/KO do not multiply raw power/stamina again;
- cumulative trauma persists; acute vulnerability decays continuously;
- no deterministic health-bar exhaustion finish;
- one immutable effective calibration from `config/event_mc_v1.yaml` with optional future partial weight-class override;
- committed weight-class overrides remain empty;
- terminal action accounting must preserve the already-resolved action/outcome before the single lifecycle finish while blocking all future primary events.

## Reviewed Phase 4B2 result
`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: PASS`.

The finish model consumes existing impact/post-trauma physiology, derives current finish resistance from durability/KD resistance plus trauma/acute vulnerability, permits fresh one-shot finishes, and terminates with one `FightFinished` lifecycle event and no later primary events. Historical KO/TKO anchor is 32.79%; five-fixture mechanics rates were 70-95%, intentionally left uncalibrated.

## Completed Codespaces single-fight runner

Runner completion commit: `da622892e33753e3f4aa10b00c88963150630f41`.

It resolves canonical historical fight IDs to frozen FSR-32 prefight profiles, supports deterministic seeds, prints chronological action/phase/stamina/physiology/KD/finish traces, and prints multi-path KO/TKO/horizon/KD/TD/SUB/phase/control summaries. Lewis/Daukaus ID `4b7ec02b39fc6f70` resolves correctly.

A later manual trace exposed a lifecycle/accounting bug: the finishing strike generated physiology/finish events but its own `ActionOutcome` was skipped after terminal state. The required correction is bookkeeping-only: count the already-resolved finishing action exactly once, preserve the same winner/time/RNG physics, emit one `FightFinished`, and block all future primary events/time advance. Continuity currently records this as corrected; Phase 4C must verify the branch contains that fix and tests pass before adding another terminal system.

## Implemented task — Phase 4C Submission Finish Mechanics

User authorized Phase 4C on 2026-08-13 11:16 America/Chicago.

Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4C_SUBMISSION_FINISHES_2026-08-13.md`

Prompt commit:
`a0ed3d38bc0b4c5d0fd8933e021db89e14472c33`

Phase 4C is mechanics/architecture only, not population calibration.

Current Phase 3 behavior already generates nonterminal `submission_attempt` events from GROUND. Phase 4C must preserve the separation:

```text
attempt generation = how often a fighter attacks submissions
conversion model = probability that a selected attempt finishes
```

Do not retune submission-attempt frequency.

Before coding, Codex must audit:
- frozen FSR-32 schema and current `FighterProfile` traits;
- whether an existing leakage-safe submission-defense trait exists but is not adapted;
- legacy/static submission conversion logic and overrides.

Known current adapter traits include `submission_pressure`, `control_imposition`, `control_resistance`, `reversal_ability`, and stamina traits. Do not invent/rebuild FSR. If no dedicated defense trait exists, derive transparent defender resistance from approved existing traits and expose blend coefficients in YAML.

Target chain:

```text
existing submission attempt event
-> SubmissionFinishModel
-> attacker threat
-> defender resistance
-> small position/context term
-> at most one clean current-stamina effect if justified
-> probabilistic P(SUB)
-> terminal StateDelta(method=SUB, winner=attacker)
-> already-resolved attempt/outcome still counted once
-> exactly one FightFinished
-> no future primary event
```

All new tunables must live under a dedicated `submission_finish` config section and inherit through the existing global-default + optional weight-class override resolver. Committed real weight-class overrides remain empty.

Use the existing SUBMISSION RNG stream unless a documented architecture reason requires otherwise. No hidden RNGs.

Single-fight trace should show submission threat, resistance, position/context/stamina term, P(SUB), and result. Multi-path summary must separate KO/TKO wins, SUB wins, and scheduled horizons and report submission attempts/conversions/P(SUB|attempt).

Required historical diagnostics are descriptive only: SUB finish rate, attempt exposure if supported, and P(SUB|attempt) only where the authoritative data definitions support it. Do not tune to the anchor in Phase 4C.

Non-goals remain:
- SUB attempt-frequency tuning;
- KD or KO/TKO calibration;
- judging;
- age;
- tactical urgency;
- named submission-technique trees;
- body-part/injury systems;
- trauma recovery;
- FSR rebuild;
- phase/stamina retuning;
- real weight-class tuning.

Expected return:
`PHASE 4C SUBMISSION FINISH MECHANICS GATE: PASS` or FAIL.

Next assistant action: independently review actual Phase 4C implementation, trait ownership, formula, terminal lifecycle, RNG ownership, config propagation, single-fight trace, historical diagnostic semantics, tests, scope protection, and frozen checksum before accepting PASS.

Implementation audit and decisions:
- frozen FSR-32 already contains leakage-safe `submission_conversion` and `submission_resistance`; the adapter now exposes both without rebuilding the artifact;
- attacker threat is `0.75 * submission_conversion + 0.25 * submission_pressure`;
- defender resistance is `0.75 * submission_resistance + 0.25 * control_resistance`;
- conversion logit is `-2.20 + (threat - resistance) / 12 + position_bonus + 0.50 * (attacker_stamina - defender_stamina)`, with top bonus `0.25`, bottom bonus `0.0`, and numerical clipping only;
- the legacy mature reservoir's prior win/loss counts, accumulated submission danger, control-time blend, repeated-attempt bonus, defensive-stability deterioration, and multiple energy pathways were rejected as duplicated or unavailable causal state; its attempt eligibility, historical conversion/resistance ownership, positional context, and simple probabilistic conversion were preserved;
- every coefficient is under `defaults.submission_finish`; `weight_classes` remains empty, and a synthetic override test proves inherited resolution;
- the existing SUBMISSION RNG stream owns conversion sampling; attempt generation rates are unchanged;
- terminal SUB uses the shared engine lifecycle and the terminal accounting correction, so its already-resolved attempt/outcome is observed exactly once before the one `FightFinished`.

Descriptive completed-fight master anchor (not calibration): 8,654 fights, 19.77% SUB finish rate, 0.748 recorded attempts/fight, 42.21% with a recorded attempt, and 46.35% SUB finish rate conditional on at least one recorded attempt. The conditional statistic is descriptive of fight-level recorded attempts, not an attempt-level conversion target.

## Checkpoint history
- 001-010: Phase 0 baseline established and exact FSR-32 frozen; PASS.
- 011-012: Phase 1 implemented; PASS.
- 013-015: Phase 2A implemented; PASS.
- 016-017: Phase 2B wrestling-entry ontology corrected; PASS.
- 018-020: Phase 3 clinch/ground flow implemented; PASS.
- 021-023: Phase 4A stamina/dynamic modifiers implemented; PASS.
- 024-026: Phase 4B1 impact/trauma/KD implemented; architecture PASS, KD overprediction exposed, calibration deferred.
- 027-030: calibration externalization completed after one review fix; PASS; future weight-class seam established with no active overrides.
- 031-032: Phase 4B2 KO/TKO authorized, implemented at `3960bae...`, independently PASS; finish rates intentionally left uncalibrated.
- 033-036: Codespaces single-fight runner built and review-completed at `da622892...`; runner PASS.
- 037: manual trace exposed terminal-action accounting bug; bookkeeping-only correction launched/completed without intended physics change.
- 038: user authorized Phase 4C terminal submission mechanics; governing prompt committed at `a0ed3d38...`.
- 039: Phase 4C compositional submission conversion, diagnostics, runner output, and lifecycle tests implemented without attempt-frequency or population calibration.
