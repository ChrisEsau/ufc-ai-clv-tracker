# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 09:20 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions. This file is not the architecture source of truth; canonical architecture and governing phase prompts remain authoritative.

## Update rule
After every new Codex prompt, update this file. Preserve checkpoint history, current gate state, prompt path, hard locks, expected Codex return, and next assistant review.

---

# Current State

Architecture revision: **v0.3**.

- Phase 0 operational baseline: **PASS**.
- Phase 1 generic continuous-time kernel: **PASS after independent ChatGPT review**.
- Phase 2A distance temporal/mechanical parity: **PASS after independent ChatGPT review**.
- Phase 2B wrestling-entry ontology correction: **PASS after independent ChatGPT review**.
- Phase 3 clinch + ground flow: **PASS after independent ChatGPT review**.
- Phase 4A stamina + dynamic modifiers: **PASS after independent ChatGPT review**.
- Phase 4B1 impact + trauma + knockdown implementation: **PASS after independent ChatGPT review**.
- Phase 4B1 KD calibration: **OPEN / intentionally deferred by user**.
- Phase 4B1 config externalization: **IMPLEMENTED by Codex; awaiting independent review**.
- Phase 4B2 KO/TKO: **NOT AUTHORIZED**.

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
- Phase 4B1: `65a6f2d4e703af5c777f1943f134728b715d4c55`

Current governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_2026-08-13.md`

Config-externalization prompt commit:
`9cd3daf5502c81ca49bf1f6cabebe69b45c79762`

Development standard locked by user:
**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Preferred loop:
`build one mechanism -> historical replay -> identify systematic miss -> change one module -> replay -> measure`

Codex cloud may use a local branch named `work`; that is acceptable. Verify ancestry/content rather than requiring local branch-name equality.

---

# Core Architecture Locks

- one authoritative `FightState.fight_time_seconds` clock;
- scheduler remains UFC-agnostic;
- all competing event rates are events/second;
- exact Bernoulli interval probability -> hazard conversion;
- Poisson count processes preserve count intensity directly;
- stable named RNG streams via `SeedSequence([root_seed, stable_stream_id])`;
- engine-owned state mutation through typed deltas;
- continuous state advances exact elapsed `dt` before event resolution;
- hard round/fight boundaries owned by engine;
- round start resets phase to DISTANCE and clears positional ownership;
- sinks remain observer-only;
- no hidden component RNGs;
- inheritance-based simulator remains untouched;
- frozen FSR-32 remains read-only profile source;
- EVENT MC uses composition, not another inheritance chain;
- tune one subsystem at a time and validate historically before claiming improvement;
- individual strike attempts remain evented until runtime demonstrates a real need to optimize.

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

```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention of completion
control_imposition   = persistence after advantageous position
```

DISTANCE TD initiation remains centered `wrestling_entry` only. TD success remains `wrestling_conversion` vs opponent `td_defense`. The Phase 2A blend is diagnostic-only.

## Phase 3 flow

Final gate:
`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

Continuous nonterminal flow supports:
`DISTANCE <-> CLINCH <-> GROUND`.

Ground exit is one hazard partitioned exactly:

```text
lambda_reversal = lambda_ground_exit * P(reversal | exit)
lambda_escape   = lambda_ground_exit * (1 - P(reversal | exit))
```

No double ground-exit clock.

## Phase 4A stamina

Final gate:
`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.

Path-local stamina drives:
- offensive `output_multiplier`;
- expressed `power_multiplier`.

Current action uses pre-action modifiers; stamina cost affects subsequent actions. Passive separation/escape/reversal hazards are not stamina-suppressed. Stamina is not inserted directly into KD/KO later.

## Phase 4B1 physiology

Final implementation gate:
`PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS`.

Target/implemented chain:

```text
landed strike
-> pre-action effective power
-> stochastic impact
-> durability-moderated primary trauma
-> cumulative trauma
-> current KD resistance
-> probabilistic KD
-> acute vulnerability after KD
```

Trait ownership lock:

```text
striking_power        -> impact severity only
damage_durability     -> primary trauma deposited
knockdown_resistance  -> baseline KD resistance
stamina               -> pre-action power modifier only
```

Power/stamina must not enter KD probability again.

Dynamic physiology:
- cumulative trauma persists and does not recover in-fight;
- acute vulnerability decays continuously with exact dt;
- KD is nonterminal;
- no health-bar exhaustion finish.

Phase 4B1 historical KD anchor from completed master rows:
- 8,801 fights;
- 0.4298 KDs/fight;
- 64.54% zero-KD;
- 35.46% >=1 KD;
- 6.14% multi-KD.

Current five-fixture mechanics runs produce about 2.2-3.3 KDs/path, materially above the historical population anchor. Independent review classified this as an exposed **calibration miss**, not an architecture failure. User explicitly chose **not to calibrate yet**.

Therefore:
- KD architecture PASS;
- KD calibration OPEN;
- do not compensate by artificially suppressing later KO/TKO;
- Phase 4B2 remains blocked until explicitly authorized.

---

# Current Task — Config Externalization

User requested that tunable simulator constants stop living directly in Python and be moved to an easy-to-edit external configuration before further calibration/finish work.

User also stated a future goal: eventually tune simulator behavior by weight class because divisions likely behave differently. Current calibration remains population-wide.

Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_2026-08-13.md`

Prompt commit:
`9cd3daf5502c81ca49bf1f6cabebe69b45c79762`

Preferred config path:
`config/event_mc_v1.yaml`

Current phase is a **pure behavior-neutral refactor**.

Required architecture:

```text
load YAML once
-> immutable Event MC calibration object
-> optional future weight-class override resolution
-> inject resolved config into rate/stamina/modifier/physiology components
-> simulate
```

Hard requirements:
- move active tunable constants from DISTANCE/CLINCH/GROUND/stamina/dynamic-modifier/damage/KD code into external config;
- code keeps structural invariants, RNG IDs, state/event identifiers, and numerical safety epsilons;
- no per-event/per-path YAML reads where one loaded object can be reused;
- current global/default values remain exactly unchanged;
- deterministic seeds/default config reproduce pre-refactor physics;
- no KD calibration;
- no KO/TKO;
- no FSR changes.

Future weight-class seam:

```text
global defaults
+ optional partial weight-class override
= effective config
```

No active weight-class numerical differences are authorized now. Override keys should remain simple/flexible strings; do not yet hard-wire whether final segmentation is weight class, UFC division, or sex+division.

Expected Codex gate:
`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS` or `FAIL`.

Next assistant action: independently review actual config refactor, source-to-config mapping, dependency injection, deterministic before/after parity, override resolver behavior, tests, scope protection, and frozen checksum before accepting PASS.

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

---

# Checkpoint History

## 001-010 — 2026-08-12 to early 2026-08-13
Phase 0 baseline was established after recovering and freezing the exact FSR-32 parquet. Multiple stale/no-remote cloud attempts stopped safely without implementation changes. Phase 0 ultimately PASS.

## 011-012
User authorized Phase 1. Generic event kernel implemented at `1debecab...` and independently accepted PASS.

## 013-015
User authorized Phase 2A. DISTANCE temporal/mechanical parity implemented at `5b7574c...` after one safe stale-checkout stop. Independently accepted PASS.

## 016-017
User authorized Phase 2B. Wrestling-entry semantic correction implemented at `004740e...` / `809389b...` and independently accepted PASS.

## 018-020
User authorized Phase 3. Live CLINCH/GROUND flow implemented at `5a6c15af...`. Independent review accepted `PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

## 021-023
User authorized Phase 4A. Stamina/dynamic modifiers implemented at `8155bc45...`. Independent review accepted `PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.

## 024-026
User authorized Phase 4B1 impact/trauma/KD only. Implemented at `65a6f2d4...`; 78 EVENT MC + relevant V0 tests reported passing, five-fixture nonterminal physiology diagnostics completed, frozen checksum unchanged. Independent review accepted architecture/implementation PASS while flagging KD rate as materially above the historical population anchor. User chose not to calibrate yet. Phase 4B2 remained unauthorized.

## 027 — 2026-08-13 09:20 America/Chicago
User authorized the next phase: externalize EVENT MC tunable constants into an editable simulator configuration, with zero default behavior change. User also requested future support for weight-class-specific simulation calibration while keeping current population-wide calibration unchanged.

New governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_2026-08-13.md`

Prompt commit:
`9cd3daf5502c81ca49bf1f6cabebe69b45c79762`

Phase 4B1 config externalization: **AUTHORIZED**.
KD calibration: **DEFERRED**.
Phase 4B2 KO/TKO: **NOT AUTHORIZED**.

## 028 — 2026-08-13
Codex externalized the active EVENT MC calibration into `config/event_mc_v1.yaml`, added an immutable load-once resolver and stable fingerprint, and established behaviorally neutral partial string-key weight-class overrides. Python compatibility aliases now derive from YAML rather than duplicate literals; stamina, DynamicModifiers, and physiology accept resolved calibration injection. No committed weight-class overrides are active.

Deterministic before/after validation compared the five frozen fixtures at identical seeds against commit `65a6f2d4...`: discrete path physics and outcomes matched, with only approximately 1e-15 arithmetic serialization differences from values passing through YAML/config mappings. KD calibration, mechanics, RNG order, and Phase 4B2 were not changed. Await independent review before accepting the config-externalization gate.
