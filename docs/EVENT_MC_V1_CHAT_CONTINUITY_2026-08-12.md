# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 09:46 America/Chicago
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
- Phase 4B1 config externalization: **NOT YET PASS — completion fix authorized/current**.
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
- Phase 4B1 physiology: `65a6f2d4e703af5c777f1943f134728b715d4c55`
- first config externalization attempt: `1a9f59c583408dc00aee5097d59e028ee3d0a2c3`

Current governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_FIX_2026-08-13.md`

Completion prompt commit:
`32b5a041f60a0584f58265e3c2c2cf79a57d4d65`

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

`wrestling_entry` owns intrinsic DISTANCE TD initiation. `wrestling_conversion` vs opponent `td_defense` owns completion. `control_imposition` is post-position persistence only. The old Phase 2A blend remains diagnostic-only.

## Phase 3 flow

Final gate:
`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

Continuous nonterminal flow supports `DISTANCE <-> CLINCH <-> GROUND`. Ground exit is one hazard partitioned exactly into reversal + escape; no double ground-exit clock.

## Phase 4A stamina

Final gate:
`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.

Path-local stamina drives offensive `output_multiplier` and expressed `power_multiplier`. Current action uses pre-action modifiers; action cost affects subsequent events. Passive separation/escape/reversal hazards are not stamina-suppressed. Stamina must not enter KD/KO directly after already modifying impact power.

## Phase 4B1 physiology

Final implementation gate:
`PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS`.

Implemented chain:
`landed strike -> pre-action effective power -> stochastic impact -> durability-moderated primary trauma -> cumulative trauma -> current KD resistance -> probabilistic KD -> acute vulnerability after KD`.

Trait ownership lock:
- striking_power -> impact severity only;
- damage_durability -> primary trauma deposited;
- knockdown_resistance -> baseline KD resistance;
- stamina -> pre-action power modifier only.

Power/stamina must not enter KD probability again. Cumulative trauma persists with no in-fight recovery; acute vulnerability decays continuously with exact dt; KD remains nonterminal; no health-bar exhaustion finish.

Historical KD anchor from completed master rows:
- 8,801 fights;
- 0.4298 KDs/fight;
- 64.54% zero-KD;
- 35.46% >=1 KD;
- 6.14% multi-KD.

Current five-fixture mechanics runs produced about 2.2-3.3 KDs/path, materially above the historical population anchor. Independent review classified this as an exposed calibration miss, not an architecture failure. User explicitly chose **not to calibrate yet**.

Therefore:
- KD architecture PASS;
- KD calibration OPEN;
- do not compensate by artificially suppressing later KO/TKO;
- Phase 4B2 remains blocked until explicitly authorized.

---

# Current Task — Config Externalization Completion

User requested that tunable simulator constants live in an easy-to-edit external config. User also wants future weight-class-specific calibration because divisions likely behave differently, while current calibration remains population-wide.

Initial config externalization implementation at `1a9f59c583408dc00aee5097d59e028ee3d0a2c3` added:
- `config/event_mc_v1.yaml`;
- immutable load-once `EventMCCalibration` / `EventMCConfigResolver`;
- stable calibration fingerprints;
- global defaults + optional string-key partial `weight_classes` overrides;
- empty committed override mapping;
- default deterministic physics parity vs pre-externalization baseline;
- config injection into stamina, DynamicModifiers, and physiology.

Independent review did **not** accept the gate because two requirements remained incomplete.

## Review finding 1 — active calibration coefficients still hard-coded

Examples independently observed in active Python paths:
- DynamicModifiers resilience normalization `(stamina_performance_resilience - 10) / 80`;
- `style_preferences()` coefficients `0.5`, `0.75`, `0.25`;
- active `_modifier()` behavioral clip magnitude `8.0`;
- ground-exit blend `0.60 escape + 0.40 reversal`;
- ground-exit edge clip `1.5`;
- reversal sensitivity `0.75`;
- potentially other active numeric coefficients that materially change rates/physiology and are not merely numerical-safety epsilons or mathematical identities.

These must be audited and externalized without changing values.

## Review finding 2 — weight-class calibration not threaded through full fight flow

`FightFlowRateProvider` / `DistanceActionRateProvider` still invoke many formulas without a resolved calibration object, and several formula functions still read module-level default aliases. Therefore a future weight-class override could modify stamina/damage while leaving DISTANCE/CLINCH/GROUND rates on global defaults.

Required target architecture:

```text
calibration = resolver.for_weight_class(key)
-> same immutable calibration injected across complete fight stack
-> DISTANCE + CLINCH + GROUND + submissions + stamina + modifiers + physiology
```

No global mutable config.

At minimum runtime overrides must reach:
- DISTANCE strikes, TD initiation/success, clinch entry;
- CLINCH strikes, TD initiation, separation;
- GROUND strikes, submissions, exits/reversal;
- stamina costs/recovery;
- DynamicModifiers;
- impact/trauma/KD and acute decay.

A synthetic end-to-end weight-class override test must span multiple subsystems and specifically prove a CLINCH or GROUND override is consumed.

This remains a **pure behavior-neutral refactor**:
- no numerical tuning;
- no KD calibration;
- no active real weight-class differences;
- no KO/TKO;
- no Phase 4B2;
- default deterministic path outcomes/RNG order must remain unchanged.

Governing completion prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_FIX_2026-08-13.md`

Prompt commit:
`32b5a041f60a0584f58265e3c2c2cf79a57d4d65`

Expected final gate after correction:
`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS` or `FAIL`.

Next assistant action: independently review the completion commit, remaining literal inventory, full calibration propagation, synthetic weight-class override tests, deterministic default parity, tests, scope protection, and frozen checksum before accepting PASS.

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

---

# Checkpoint History

## 001-010 — 2026-08-12 to early 2026-08-13
Phase 0 baseline established after recovery/freeze of exact FSR-32 artifact. Multiple stale/no-remote cloud attempts stopped safely. Final Phase 0 PASS.

## 011-012
Phase 1 authorized, implemented at `1debecab...`, independently accepted PASS.

## 013-015
Phase 2A authorized, implemented at `5b7574c...` after one safe stale-checkout stop, independently accepted PASS.

## 016-017
Phase 2B authorized; wrestling-entry correction implemented at `004740e...` / `809389b...`, independently accepted PASS.

## 018-020
Phase 3 authorized; live CLINCH/GROUND flow implemented at `5a6c15af...`, independently accepted PASS.

## 021-023
Phase 4A authorized; stamina/dynamic modifiers implemented at `8155bc45...`, independently accepted PASS.

## 024-026
Phase 4B1 impact/trauma/KD authorized and implemented at `65a6f2d4...`; independent review accepted architecture/implementation PASS but exposed severe KD overprediction. User explicitly deferred KD calibration. Phase 4B2 remained unauthorized.

## 027 — 2026-08-13 09:20 America/Chicago
User authorized config externalization with zero behavioral change and requested future weight-class-specific override support. Governing prompt commit `9cd3daf...`.

## 028 — 2026-08-13
Codex implemented first config externalization at `1a9f59c583408dc00aee5097d59e028ee3d0a2c3`. Default five-fixture deterministic physics matched pre-refactor behavior; empty weight-class map; 82 tests reported passing; frozen checksum unchanged.

## 029 — 2026-08-13 09:46 America/Chicago
ChatGPT independently reviewed `1a9f59c5...` and **did not accept** the config externalization gate. Two incomplete requirements were found: (1) remaining active behavior-changing coefficients were still hard-coded, and (2) resolved calibration was not propagated through the full flow/rate/formula stack, so future weight-class overrides would affect only some subsystems.

A narrow completion prompt was issued under the already-authorized config phase:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_FIX_2026-08-13.md`
commit `32b5a041f60a0584f58265e3c2c2cf79a57d4d65`.

No tuning authorized. KD calibration remains deferred. Phase 4B2 remains unauthorized.

## 030 — 2026-08-13
Codex completed the config-externalization review fixes: remaining style-blend, modifier-clip, ground-exit/reversal, and resilience-normalization coefficients moved to YAML; one resolved immutable calibration now threads through DISTANCE, CLINCH, GROUND, stamina, modifiers, and physiology consumers. A synthetic partial override proves simultaneous DISTANCE, CLINCH, stamina, and damage configuration reach while inheriting unspecified defaults.

Five-fixture exact-seed comparison against pre-externalization commit `65a6f2d4...` preserved discrete physics/RNG behavior with only floating serialization tolerance. No default value, KD calibration, real weight-class tuning, or Phase 4B2 mechanic changed. Await independent review of the completed externalization gate.
