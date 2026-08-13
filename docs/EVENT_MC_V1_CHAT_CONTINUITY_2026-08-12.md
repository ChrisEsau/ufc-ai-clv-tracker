# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 10:46 America/Chicago
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
- Terminal submissions, judging, age, tactical urgency, population calibration, real weight-class tuning: NOT AUTHORIZED

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
- Config externalization `1a9f59c583408dc00aee5097d59e028ee3d0a2c3`, completion `b8b2b870595c9cc62255b4d63b7c20de56e9550f`, `77661563c3a833a3e87b60e4b3ae6caecd648cb8`
- Phase 4B2 KO/TKO `3960bae022ac7d3129a593a102704d2b39a46b28`
- Single-fight runner initial `05485f0bd0460ae726fb2f0283373a727464cb38`

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
- committed weight-class overrides remain empty.

## Reviewed Phase 4B2 result
`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: PASS`.

The finish model consumes existing impact/post-trauma physiology, derives current finish resistance from durability/KD resistance plus trauma/acute vulnerability, permits fresh one-shot finishes, and terminates with one `FightFinished` lifecycle event and no later primary events. Historical KO/TKO anchor is 32.79%; five-fixture mechanics rates were 70-95%, intentionally left uncalibrated.

## Completed task — Codespaces single-fight runner

Initial implementation `05485f0bd0460ae726fb2f0283373a727464cb38` is functional and observer-only. It:
- resolves `--fight-id`/`--bout-id` from `data/master/ufc_master.parquet`;
- resolves exact frozen FSR-32 prefight profiles;
- supports deterministic `--seed` with `base + path_index`;
- prints one-path trace and multi-path summary;
- changes no simulator mechanics or calibration.

Independent review found narrow omissions before runner PASS:
1. trace physiology output should explicitly show post-event cumulative trauma and acute vulnerability;
2. phase/controller transitions should print before -> after clearly;
3. aggregate summary should explicitly show finish-round distribution, scheduled-horizon count+rate, red/blue KO/TKO win counts/rates, side-specific TD attempts/completions, and side-specific submission attempts;
4. tests should explicitly prove nondecreasing trace timestamps, no events after terminal `FightFinished`, same-seed discrete reproducibility, and summary arithmetic;
5. explicitly test Lewis/Daukaus ID `4b7ec02b39fc6f70`; if it does not resolve in master, report the blocker instead of inventing a mapping.

The completion fix addresses all five findings without changing simulator behavior. Trace output now exposes post-event trauma/vulnerability and phase/controller transitions; aggregate output explicitly reports finish rounds, scheduled horizons, corner KO/TKO wins, TD attempts/completions, and submission attempts. Focused lifecycle, reproducibility, lookup, and controlled-arithmetic tests cover the diagnostic contract. Lewis/Daukaus ID `4b7ec02b39fc6f70` resolves canonically to Derrick Lewis vs Chris Daukaus on 2021-12-18 with both frozen prefight profiles.

A subsequent lifecycle/accounting review found that the engine stopped observing the already-resolved action consequence after a same-timestamp KO/TKO set terminal state. The accounting-only correction emits that action's single `ActionOutcome` before the single `FightFinished`, while terminal state continues to block all future primary actions and time advances. Lewis/Daukaus seed `20260813` retains winner `blue`, method `KO_TKO`, and time `26.623215196672668`, with the finishing strike now included as the fourth landed blue strike.

Governing completion prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_SINGLE_FIGHT_RUNNER_FIX_2026-08-13.md`
Prompt commit: `6d036829fe989fa04e0a752bc43310ca67c2877b`

No formula, config, RNG, state, FSR, submission, judging, age, or calibration change is authorized.

Expected return:
`PHASE 4B2 SINGLE-FIGHT RUNNER GATE: PASS` or FAIL.

## Checkpoint history
- 001-010: Phase 0 baseline established and exact FSR-32 frozen; PASS.
- 011-012: Phase 1 implemented; PASS.
- 013-015: Phase 2A implemented; PASS.
- 016-017: Phase 2B wrestling-entry ontology corrected; PASS.
- 018-020: Phase 3 clinch/ground flow implemented; PASS.
- 021-023: Phase 4A stamina/dynamic modifiers implemented; PASS.
- 024-026: Phase 4B1 impact/trauma/KD implemented; architecture PASS, KD overprediction exposed, calibration deferred.
- 027-030: calibration externalization completed after one review fix; PASS; future weight-class seam established with no active overrides.
- 031: user authorized Phase 4B2 KO/TKO mechanics.
- 032: Phase 4B2 implemented at `3960bae...`; independently reviewed PASS; finish rates remain uncalibrated.
- 033: user requested Codespaces single-fight sanity runner.
- 034: runner implemented at `05485f0b...`; functional, observer-only, 91 tests reported passing, frozen checksum unchanged.
- 035: independent runner review found missing trace/summary/test details; narrow completion prompt issued at `6d036829...`.
- 036: runner completion fix implemented; all review omissions addressed, Lewis/Daukaus lookup verified, and runner gate PASS without mechanics or calibration changes.
- 037: terminal-action accounting corrected so an already-resolved finishing action emits its one outcome before the one lifecycle finish; deterministic physics unchanged.
