# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 10:30 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions. This file is not the architecture source of truth; canonical architecture and governing phase prompts remain authoritative.

## Update rule
After every new Codex prompt, update this file. Preserve checkpoint history, current gate state, prompt path, hard locks, expected Codex return, and next assistant review.

---

# Current State

- Phase 0 operational baseline: **PASS**.
- Phase 1 generic continuous-time kernel: **PASS after independent ChatGPT review**.
- Phase 2A distance temporal/mechanical parity: **PASS after independent ChatGPT review**.
- Phase 2B wrestling-entry ontology correction: **PASS after independent ChatGPT review**.
- Phase 3 clinch + ground flow: **PASS after independent ChatGPT review**.
- Phase 4A stamina + dynamic modifiers: **PASS after independent ChatGPT review**.
- Phase 4B1 impact + trauma + knockdown implementation: **PASS after independent ChatGPT review**.
- Phase 4B1 KD calibration: **OPEN / intentionally deferred by user**.
- Phase 4B1 config externalization: **PASS after independent ChatGPT review**.
- Phase 4B2 KO/TKO finish mechanics: **IMPLEMENTED by Codex; awaiting independent review**.
- Phase 4B2 single-fight sanity runner: **AUTHORIZED addendum / current diagnostic task**.
- Terminal submissions, judging, age, tactical urgency, population calibration, real weight-class tuning: **NOT AUTHORIZED**.

Frozen FSR-32 path:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Implementation commits:
- Phase 1: `1debecab69a141bf2f81179f3436af569733b750`
- Phase 2A: `5b7574c7689ffa2e55821a49fca47a2c1c937991`
- Phase 2B: `004740e54618c134e08aa553164c381508811481`, `809389bdabe208e93536034bc795bbcf7e1ab038`
- Phase 3: `5a6c15af9c2315f23c231d0f34cd3cefddba4578`
- Phase 4A: `8155bc45de5fa26fa6077dd870716234d54690c9`
- Phase 4B1 physiology: `65a6f2d4e703af5c777f1943f134728b715d4c55`
- Config externalization: `1a9f59c583408dc00aee5097d59e028ee3d0a2c3`, completion `b8b2b870595c9cc62255b4d63b7c20de56e9550f`, `77661563c3a833a3e87b60e4b3ae6caecd648cb8`

Current Phase 4B2 governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`
Prompt commit: `3bc8ec1c0aefc11c49c5f0ab94285dce420e9e63`

Single-fight runner addendum:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_SINGLE_FIGHT_RUNNER_ADDENDUM_2026-08-13.md`
Addendum commit: `14798a03b4ef59463fbe44ea481182a455d0f792`

Development standard locked by user:
**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Preferred loop:
`build one mechanism -> historical replay -> identify systematic miss -> change one module -> replay -> measure`

Codex cloud may use local branch `work`; verify ancestry/content rather than branch-name equality.

---

# Core Architecture Locks

- one authoritative `FightState.fight_time_seconds` clock;
- scheduler remains UFC-agnostic;
- all rates are events/second;
- exact Bernoulli interval probability -> hazard conversion;
- Poisson count processes preserve count intensity directly;
- stable named RNG streams via `SeedSequence([root_seed, stable_stream_id])`;
- engine-owned mutation through typed deltas;
- continuous state advances exact elapsed `dt` before event resolution;
- hard round/fight boundaries owned by engine;
- round start resets DISTANCE and clears positional ownership;
- sinks/trace tools are observer-only;
- no hidden component RNGs;
- inheritance simulator remains untouched;
- FSR-32 remains read-only and is not rebuilt;
- EVENT MC uses composition, not inheritance;
- tune one subsystem at a time;
- evented strikes remain until runtime demonstrates a real optimization need.

Stable RNG stream IDs:
- SCHEDULER 10
- STRIKE_RESOLUTION 20
- TAKEDOWN 30
- SUBMISSION 40
- DAMAGE 50
- KNOCKDOWN_FINISH 60
- JUDGING 70

---

# Reviewed Mechanical Locks

## Phase 2B wrestling ontology
`wrestling_entry` owns intrinsic DISTANCE TD initiation. `wrestling_conversion` vs opponent `td_defense` owns completion. `control_imposition` is post-position persistence only. Phase 2A blend is diagnostic-only.

## Phase 3 flow
Final gate: `PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.
Continuous nonterminal flow supports `DISTANCE <-> CLINCH <-> GROUND`. Ground exit is one hazard partitioned into reversal + escape.

## Phase 4A stamina
Final gate: `PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.
Path-local stamina drives offensive `output_multiplier` and expressed `power_multiplier`. Current action uses pre-action modifiers; cost affects later events. Passive exits are not stamina-suppressed. Stamina must not enter KD/KO directly after modifying impact power.

## Phase 4B1 physiology
Final implementation gate: `PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS`.
Chain: landed strike -> pre-action effective power -> stochastic impact -> durability-moderated primary trauma -> cumulative trauma -> current KD resistance -> probabilistic KD -> acute vulnerability.

Trait ownership:
- striking_power -> impact only;
- damage_durability -> trauma deposited;
- knockdown_resistance -> baseline KD resistance;
- stamina -> pre-action power modifier only.

Historical KD anchor:
- 8,801 fights;
- 0.4298 KDs/fight;
- 64.54% zero-KD;
- 35.46% >=1 KD;
- 6.14% multi-KD.

Five-fixture mechanics runs produced roughly 2.2-3.3 KDs/path. This is an exposed calibration miss, not an architecture failure. User explicitly deferred calibration.

## Config externalization
Final gate: `PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS`.
Calibration source: `config/event_mc_v1.yaml`.
Architecture: global defaults + optional partial string-key weight-class override -> one immutable `EventMCCalibration` threaded through DISTANCE, CLINCH, GROUND, submissions, stamina, DynamicModifiers, physiology, and acute decay.
Committed weight-class mapping remains empty. No real division tuning is active yet.

---

# Phase 4B2 KO/TKO Mechanics

User authorized Phase 4B2 on 2026-08-13 10:15 America/Chicago.

Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`
commit `3bc8ec1c0aefc11c49c5f0ab94285dce420e9e63`.

Phase 4B2 is mechanics/architecture only. Known KD overprediction must not be hidden by suppressing finish conversion.

Target chain:
`landed strike -> impact -> trauma -> KD resistance -> KD -> acute vulnerability -> finish resistance -> probabilistic KO/TKO -> terminal engine delta`.

Hard locks:
- power and stamina enter once through impact;
- finish consumes impact/current finish resistance, not raw power again;
- trauma lowers resistance but never deterministically forces finish;
- fresh one-shot KO/TKO remains possible;
- KD may condition finish probability without duplicating power;
- engine remains sole state mutator;
- exactly one `FightFinished`, no primary events after terminal finish;
- all finish coefficients externalized in `config/event_mc_v1.yaml` and compatible with future weight-class overrides;
- no KD/KO population calibration yet.

Codex continuity checkpoint indicates Phase 4B2 implementation exists and is awaiting independent review. Reported diagnostics exposed very high five-fixture finish rates versus historical descriptive anchor; do not tune until user authorizes calibration.

---

# Current Diagnostic Addendum — Single Fight Runner

User requested a Codespaces-friendly sanity runner that can take a historical fight ID and either show one complete path or summarize many paths. This is diagnostic infrastructure only and does not replace population calibration.

Governing addendum:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_SINGLE_FIGHT_RUNNER_ADDENDUM_2026-08-13.md`
commit `14798a03b4ef59463fbe44ea481182a455d0f792`.

Required CLI shape:

```bash
python -m pipeline.simulation.event_mc_v1.single_fight --fight-id <ID> --paths 1 --trace
```

and:

```bash
python -m pipeline.simulation.event_mc_v1.single_fight --fight-id <ID> --paths 1000
```

Requirements:
- run from Codespaces repo root;
- resolve historical fight/bout ID from existing project data and frozen FSR-32 pre-fight state;
- print fight metadata, fighter profiles, calibration fingerprint and resolved weight-class key;
- `--paths 1 --trace`: chronological lifecycle/action/phase/stamina/physiology/KD/finish trace plus path summary;
- multi-path without `--trace`: compact aggregate summary only;
- explicit deterministic `--seed` option;
- multi-path seeds stable/reproducible;
- runner/trace remains observer-only and does not change mechanics;
- no FSR rebuild, no calibration, no judging, no terminal submissions;
- test chronological event ordering and no events after a terminal KO/TKO;
- return exact Codespaces commands and abbreviated sample outputs.

Single-path trace should expose where applicable:
- timestamp, round/round-clock, phase;
- actor/defender/action/outcome;
- phase/controller transitions;
- stamina before/after and pre-action output/power modifiers;
- impact, primary trauma, cumulative trauma, acute vulnerability;
- KD resistance/probability/result;
- finish resistance/probability/result;
- lifecycle events and terminal winner/method/time.

Multi-path summary should expose at minimum:
- paths, seed, runtime, throughput, calibration fingerprint;
- terminal wins when available, scheduled-horizon counts;
- KO/TKO rates, average finish time, finish-round distribution;
- KDs/path, zero/>=1/multi-KD rates;
- final trauma/stamina averages;
- phase residence, attempts/lands, TD attempts/completions, SUB attempts, control time.

Next assistant action after Codex return: independently review fight-ID lookup correctness, observer-only trace implementation, deterministic path seeding, summary calculations, terminal lifecycle, sample Codespaces commands, tests, and frozen checksum. This runner does not change the Phase 4B2 mechanics gate.

---

# Canonical / Governing Docs

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`
6. `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`
7. `docs/EVENT_MC_V1_CODEX_PHASE2B_WRESTLING_ENTRY_ONTOLOGY_2026-08-13.md`
8. `docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md`
9. `docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`
10. `docs/EVENT_MC_V1_CODEX_PHASE4B1_IMPACT_TRAUMA_KD_2026-08-13.md`
11. `docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_2026-08-13.md`
12. `docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_FIX_2026-08-13.md`
13. `docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`
14. `docs/EVENT_MC_V1_CODEX_PHASE4B2_SINGLE_FIGHT_RUNNER_ADDENDUM_2026-08-13.md`

---

# Checkpoint History

## 001-010
Phase 0 baseline established after recovery/freeze of exact FSR-32 artifact. Final Phase 0 PASS.

## 011-012
Phase 1 implemented at `1debecab...`, independently PASS.

## 013-015
Phase 2A implemented at `5b7574c...`, independently PASS.

## 016-017
Phase 2B implemented at `004740e...` / `809389b...`, independently PASS.

## 018-020
Phase 3 implemented at `5a6c15af...`, independently PASS.

## 021-023
Phase 4A implemented at `8155bc45...`, independently PASS.

## 024-026
Phase 4B1 impact/trauma/KD implemented at `65a6f2d4...`; independent architecture PASS; severe KD overprediction exposed; user deferred calibration.

## 027-030
Config externalization completed after one review correction. Final commits `b8b2b870...`, `77661563...`; independently PASS. Future weight-class override seam active structurally, no real overrides committed.

## 031 — 2026-08-13 10:15 America/Chicago
User authorized Phase 4B2 KO/TKO mechanics. Governing prompt commit `3bc8ec1c...`.

## 032 — 2026-08-13
Codex continuity reports Phase 4B2 implementation completed; independent review still pending. Mechanics diagnostics expose high finish rates; calibration remains deferred.

## 033 — 2026-08-13 10:30 America/Chicago
User requested a Codespaces single-fight sanity runner: historical fight ID lookup, one-path full event/physiology trace, and multi-path compact summary for the same fight. User explicitly said this is for sanity while population calibration remains the calibration strategy.

New diagnostic addendum:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_SINGLE_FIGHT_RUNNER_ADDENDUM_2026-08-13.md`
commit `14798a03b4ef59463fbe44ea481182a455d0f792`.

No simulator tuning or new terminal systems authorized by this addendum.
