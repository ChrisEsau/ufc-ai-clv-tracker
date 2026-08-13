# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 10:15 America/Chicago
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
- Phase 4B1 config externalization: **PASS after independent ChatGPT review**.
- Phase 4B2 KO/TKO finish mechanics: **AUTHORIZED and current task; not yet passed**.
- Terminal submissions, judging, age, tactical urgency, later calibration: **NOT AUTHORIZED**.

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
- Config externalization initial: `1a9f59c583408dc00aee5097d59e028ee3d0a2c3`
- Config externalization completion: `b8b2b870595c9cc62255b4d63b7c20de56e9550f`, `77661563c3a833a3e87b60e4b3ae6caecd648cb8`

Current governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`

Phase 4B2 prompt commit:
`3bc8ec1c0aefc11c49c5f0ab94285dce420e9e63`

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

`wrestling_entry` owns intrinsic DISTANCE TD initiation. `wrestling_conversion` vs opponent `td_defense` owns completion. `control_imposition` is post-position persistence only. The Phase 2A blend remains diagnostic-only.

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

Power/stamina must not enter KD probability again. Cumulative trauma persists with no in-fight recovery; acute vulnerability decays continuously with exact dt; KD is nonterminal in Phase 4B1; no health-bar exhaustion finish.

Historical KD anchor from completed master rows:
- 8,801 fights;
- 0.4298 KDs/fight;
- 64.54% zero-KD;
- 35.46% >=1 KD;
- 6.14% multi-KD.

Current five-fixture mechanics runs produce roughly 2.2-3.3 KDs/path, materially above the historical population anchor. Independent review classified this as an exposed calibration miss, not an architecture failure. User explicitly chose **not to calibrate yet**.

Therefore:
- KD architecture PASS;
- KD calibration OPEN;
- do not hide the upstream miss by artificially suppressing later KO/TKO conversion.

## Phase 4B1 config externalization

Final gate:
`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS`.

Current calibration source:
`config/event_mc_v1.yaml`

Architecture:

```text
global defaults
+ optional partial string-key weight-class override
= one immutable EventMCCalibration
```

The same resolved calibration is now threaded through:
- DISTANCE rates + resolution;
- CLINCH rates + resolution;
- GROUND rates + resolution;
- submission attempts;
- stamina costs/recovery;
- DynamicModifiers;
- impact/trauma/KD;
- acute-vulnerability decay.

Remaining behavior-changing coefficients found during first review were moved to YAML, including modifier clip, style-preference weights, ground-exit/reversal blend coefficients, and stamina-resilience normalization. A synthetic override test proves simultaneous DISTANCE, CLINCH, stamina, and damage reach while unspecified values inherit defaults.

Committed `weight_classes` mapping remains empty. No real weight-class tuning is active yet. Default five-fixture physics and RNG ordering matched pre-externalization behavior within floating serialization tolerance.

---

# Current Task — Phase 4B2 KO/TKO Finish Mechanics

User said **proceed** on 2026-08-13 10:15 America/Chicago, explicitly authorizing Phase 4B2.

Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`

Prompt commit:
`3bc8ec1c0aefc11c49c5f0ab94285dce420e9e63`

Phase 4B2 is an **architecture/mechanics phase, not a calibration phase**.

Known KD overprediction remains unchanged and must not be compensated for by artificially suppressing finish probability.

Target same-timestamp chain:

```text
landed strike
-> pre-action effective power
-> stochastic impact
-> primary trauma
-> cumulative trauma
-> current KD resistance
-> probabilistic KD
-> acute-vulnerability consequence
-> current finish resistance
-> probabilistic KO/TKO
-> terminal engine delta
```

Hard locks:
- power and stamina enter once through impact;
- finish probability consumes impact/current finish resistance rather than recomputing power;
- cumulative trauma lowers resistance but never forces a deterministic finish;
- acute vulnerability may lower finish resistance;
- KD may explicitly condition finish probability but must not duplicate power;
- fresh one-shot KO/TKO must remain possible;
- one shared physiology/finish pipeline across phases unless legacy tracing strongly proves a small context modifier;
- engine remains sole state mutator;
- successful finish stops future primary events immediately and produces exactly one `FightFinished` lifecycle event;
- all new finish tuning constants live in `config/event_mc_v1.yaml` and support the existing future weight-class override seam;
- committed weight-class overrides remain empty.

Required legacy trace before coding:
- `StaticFSRMCKOTKOV2`;
- `StaticFSRMCKOTKOV2KDCollapse`;
- `StaticFSRMCKOTKOV2RoundRecovery`;
- V3/V3.1/V3.2/V3.3 stamina layers;
- later full-fight/audit overrides if relevant.

Explicitly reject unless independently justified:
- deterministic damage-reservoir exhaustion finish;
- collapse-trauma replay;
- multiple direct power terms;
- hidden defender-stamina finish penalties;
- duplicated recent-KD effects;
- damage recovery;
- age effects.

Required diagnostics:
- five frozen fixtures;
- KO/TKO finish rate;
- average finish time and round distribution;
- KD-strike vs direct non-KD finish share;
- impact ratio / trauma / acute vulnerability at finish;
- KDs before termination;
- scheduled-horizon rate;
- runtime;
- descriptive historical KO/TKO anchor if supported by master data.

Interpret diagnostics mechanically only. Do not tune to the anchor and do not claim predictive performance.

Absolute non-goals:
- KD calibration;
- KO/TKO population calibration;
- terminal submissions;
- judging;
- age;
- tactical urgency;
- body-part/injury/doctor stoppage systems;
- FSR changes;
- phase/stamina retuning.

Expected Codex return:
`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: PASS` or `FAIL`.

Next assistant action: independently review actual implementation commit/diff, legacy finish trace, formula ownership, same-timestamp ordering, terminal lifecycle behavior, RNG ownership, config propagation, tests, diagnostics, historical anchor, scope protection, and frozen checksum before accepting PASS.

Do not authorize terminal submissions, judging, calibration, age, or later phases automatically.

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
Phase 4B1 impact/trauma/KD authorized and implemented at `65a6f2d4...`; independent review accepted architecture/implementation PASS but exposed severe KD overprediction. User explicitly deferred KD calibration.

## 027-030
User authorized behavior-neutral calibration externalization plus future weight-class override support. First implementation at `1a9f59c5...` was not accepted because active coefficients remained in Python and runtime calibration was not threaded through the whole fight-flow stack. Completion prompt `32b5a041...` produced commits `b8b2b870...` and `77661563...`; independent review confirmed complete propagation, externalized remaining tunables, synthetic multi-subsystem override behavior, and default deterministic parity. Final gate accepted:
`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS`.

## 031 — 2026-08-13 10:15 America/Chicago
User said **proceed**, explicitly authorizing Phase 4B2 KO/TKO finish mechanics while KD calibration remains intentionally deferred.

New governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md`

Prompt commit:
`3bc8ec1c0aefc11c49c5f0ab94285dce420e9e63`

Phase 4B2: **AUTHORIZED / current**.
KD calibration: **DEFERRED / unchanged**.
Terminal submissions, judging, age, tactical urgency, later calibration: **NOT AUTHORIZED**.

## 032 — 2026-08-13
Codex implemented Phase 4B2 KO/TKO mechanics as a compositional finish model consuming the existing Phase 4B1 impact outcome and post-trauma state. Finish resistance is derived from equal-weight durability/KD-resistance baseline, cumulative-trauma erosion, and acute-vulnerability erosion; finish probability is a compact impact/resistance sigmoid with an explicit KD conditioning bonus. Power and stamina are not consumed again. Successful finishes set structured terminal winner/method state and the engine emits one lifecycle finish event with no later primary events.

All finish coefficients live under the new YAML `finish` section and accept the same resolved weight-class calibration. Five-fixture mechanics diagnostics exposed 70-95% finish rates versus the descriptive 32.79% historical anchor; this is reported without tuning because KD and KO/TKO population calibration remain deferred. Await independent review before accepting the Phase 4B2 mechanics gate.
