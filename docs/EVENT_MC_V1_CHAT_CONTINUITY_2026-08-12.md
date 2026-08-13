# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 08:42 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

Purpose: persistent handoff for future ChatGPT sessions. This file is not the architecture source of truth; canonical architecture and phase contracts are referenced below.

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
- Phase 4B1 impact + trauma + knockdown: **AUTHORIZED and current task; not yet passed**.
- Phase 4B2 KO/TKO finishes and later mechanics: **NOT AUTHORIZED**.

Frozen FSR-32 path:
`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Frozen Phase 0 compact baseline:
- winner accuracy 59.0%;
- winner Brier 0.27745;
- KO/TKO 25.0%;
- SUB 17.1%;
- DEC 57.9%;
- 5 frozen fixtures;
- 15 deterministic traces;
- five 1,000-path matchup summaries;
- first 200 eligible bouts x 10 paths;
- full 1,565-fight method/submission anchor;
- recorded artifact checksums revalidated.

Implementation commits:
- Phase 1: `1debecab69a141bf2f81179f3436af569733b750`
- Phase 2A: `5b7574c7689ffa2e55821a49fca47a2c1c937991`
- Phase 2B: `004740e54618c134e08aa553164c381508811481`, `809389bdabe208e93536034bc795bbcf7e1ab038`
- Phase 3: `5a6c15af9c2315f23c231d0f34cd3cefddba4578`
- Phase 4A: `8155bc45de5fa26fa6077dd870716234d54690c9`

Current Phase 4B1 governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_IMPACT_TRAUMA_KD_2026-08-13.md`

Phase 4B1 prompt commit:
`fd81b02a271fa3f367a1bdeaab7150e982d4240b`

Development standard locked by user:
**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE. Do not optimize for perfection or exhaustive defensive hardening. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Preferred loop:
`build one mechanism -> historical replay -> identify systematic miss -> change one module -> replay -> measure`

Codex cloud may use a local branch named `work`; that is acceptable. Verify ancestry/content rather than requiring the local branch name to equal the feature branch.

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
- frozen FSR-32 remains the profile source and is never rebuilt during EVENT MC work;
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

Known physiology seam now active in Phase 4B1: same-timestamp landed-strike -> trauma -> KD consequences may require the smallest compositional extension to the current single-Resolution contract. Engine must remain sole state mutator.

---

# Phase 2B Wrestling Ontology Lock

```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention of completion
control_imposition   = persistence after advantageous position
```

Active DISTANCE TD initiation is driven by centered `wrestling_entry` only, using the existing Phase 2B base/scale. The old blended wrestling preference remains diagnostic-only. TD success remains `wrestling_conversion` vs opponent `td_defense`.

---

# Reviewed Phase 3 Result

Final gate:
`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

Implementation commit:
`5a6c15af9c2315f23c231d0f34cd3cefddba4578`

Continuous nonterminal flow now supports:
`DISTANCE <-> CLINCH <-> GROUND`.

Implemented:
- clinch strikes and TD attempts;
- separation;
- top/reduced bottom ground strikes;
- nonterminal submission attempts;
- escape/standup;
- reversal/controller swaps;
- exact phase/control residence.

Ground exit hard invariant:
```text
lambda_reversal = lambda_ground_exit * P(reversal | exit)
lambda_escape   = lambda_ground_exit * (1 - P(reversal | exit))
lambda_reversal + lambda_escape == lambda_ground_exit
```

Validation watchlist, not tuning authorization:
- some grappling fixtures spend roughly 40% of scheduled time on ground;
- some CLINCH TD hazards are high while in phase;
- historical cohort validation must decide whether these need adjustment.

---

# Reviewed Phase 4A Result

Final gate:
`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.

Implementation commit:
`8155bc45de5fa26fa6077dd870716234d54690c9`

Phase 4A added:
- normalized path-local `red_stamina` / `blue_stamina` state;
- immutable StateDelta-owned action stamina costs;
- exact-dt positional stamina expenditure;
- one V3.3-style 40%-of-missing between-round recovery owner;
- `DynamicModifiers(output_multiplier, power_multiplier)`;
- offensive rate suppression through output modifier;
- pre-action power/output capture before the current action cost is applied.

Important locks:
- passive separation / escape / reversal hazards are not stamina-suppressed;
- strike accuracy, TD success/defense, control resistance, durability and KD resistance are not stamina-modified directly;
- full stamina maps exactly to output=1 and power=1;
- future damage consumes the pre-action power modifier once; stamina must not be multiplied into KD/KO again.

Legacy stamina trace accepted:
- V3 supplies reservoir/cost/curve calibration;
- V3.1 establishes pre-action ordering and bypassed old direct output suppression;
- V3.2 supplies sustained positional expenditure;
- V3.3 supplies the single global round-recovery owner;
- Phase 4A intentionally restores the user-requested output pathway once through DynamicModifiers without duplicating inheritance effects.

Diagnostics watchlist, not tuning authorization:
- Holloway/Kattar showed the strongest stamina draw and attempt suppression;
- low late stamina and the severe power curve must be checked against historical round-by-round output and later KO timing before retuning.

Codex reported 72 EVENT MC + relevant V0 tests passed, frozen checksum unchanged, and all five-fixture stamina diagnostics reached horizon. Independent code review accepted the gate.

---

# Phase 4B1 — Current Task

Phase 4B1 is explicitly authorized and governed by:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_IMPACT_TRAUMA_KD_2026-08-13.md`

Goal: implement physiology through **knockdown only**, with no terminal finish.

Target chain:
```text
landed strike
-> pre-action effective power
-> stochastic impact
-> primary trauma
-> engine applies cumulative trauma
-> derive current KD resistance
-> probabilistic KD
-> same-timestamp acute-vulnerability consequence
```

Target dynamic physiology state:
```text
cumulative_trauma
acute_vulnerability
stamina [already owned by Phase 4A]
```

Trait ownership — HARD LOCK:
```text
striking_power        -> impact severity distribution
damage_durability     -> persistent trauma deposited per impact
knockdown_resistance  -> baseline KD resistance
stamina               -> DynamicModifiers.power_multiplier only
```

Power is counted once through impact. Do not separately insert `striking_power` or stamina into KD probability.

Cumulative trauma:
- persistent;
- non-negative;
- only increases from primary trauma;
- no in-fight/round recovery in 4B1;
- lowers future derived resistance;
- does not directly finish the fight.

Acute vulnerability:
- rises after KD;
- decays continuously with exact elapsed dt;
- no hard reset;
- does not finish the fight.

KD:
- probabilistic from impact relative to current derived resistance;
- fresh one-shot KDs must remain possible;
- accumulated trauma and acute vulnerability must raise future KD risk through lower resistance;
- use named `KNOCKDOWN_FINISH` RNG stream.

No KO/TKO, terminal SUB, judging, age, body-part damage, trauma recovery, defender-fatigue resistance penalty, or Phase 4B2 work.

Required five fixtures:
- Lewis/Daukaus;
- Holloway/Kattar;
- Font/Rosas;
- Merab/Yan;
- Oliveira/Poirier.

Also build a historical KD anchor audit if current historical fields support it. Do not globally optimize parameters in 4B1.

Expected Codex return:
`PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS` or `FAIL`.

Next assistant action: independently inspect actual implementation/diff, legacy source trace, same-timestamp ordering, parameter provenance, trauma/KD formulas, RNG ownership, exact recovery behavior, tests, five-fixture diagnostics, historical KD audit, runtime, scope protection, and frozen checksum before accepting the gate.

Do not authorize Phase 4B2 automatically.

---

# Damage / KD / KO Target Architecture

The external review concluded that the old damage reservoir identified a real missing concept — fight memory — but should not be ported literally.

Recommended eventual architecture:
```text
StrikeResolver
-> DamageModel
-> engine trauma delta
-> KnockdownModel
-> engine KD delta
-> KnockdownConsequenceModel
-> engine consequence delta
-> FinishModel
-> engine finish delta
```

Key principles:
- cumulative trauma + short-lived acute vulnerability;
- impact/current resistance is the central KD/finish quantity;
- power route and accumulation route both exist;
- stamina modifies expressed attacking power only;
- cumulative trauma initially has no recovery;
- acute vulnerability recovers continuously;
- same physiology pipeline across phases;
- no health-bar finish threshold;
- no multiple overlapping damage reservoirs.

Phase 4B1 implements this architecture only through knockdown. Phase 4B2 will later own KO/TKO if explicitly authorized.

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

---

# Checkpoint History

## 001 — 2026-08-12 22:33 America/Chicago
Phase 0 operational baseline prompt issued.

## 002 — 2026-08-12 22:37
Codex baseline execution plan approved with exact-remote-branch guardrail.

## 003 — 2026-08-12 22:40
Codex stopped because isolated checkout had no Git remote. Remote-unblock prompt issued.

## 004 — 2026-08-12 22:49
Codex repaired Git environment but frozen FSR-32 parquet was absent. Artifact recovery issued; no rebuild authorized.

## 005 — 2026-08-12 22:59
User initially chose to move on and supply FSR later; Phase 1 prepared but not launched.

## 006 — 2026-08-12 23:08
Exact frozen FSR-32 parquet located and SHA-256 locked.

## 007 — 2026-08-12 23:18
Exact frozen parquet published temporarily as GitHub Release transfer asset.

## 008 — 2026-08-12 23:21
Codex generic workspace lacked remote/auth; bootstrap issued; no implementation changes.

## 009 — 2026-08-12 23:24
Continuity corrected: Phase 1 had not yet launched.

## 010 — 2026-08-13 near 00:00
Phase 0 baseline returned PASS after exact artifact recovery/revalidation.

## 011 — 2026-08-13 00:03
User authorized Phase 1.

## 012 — 2026-08-13 about 00:15
Phase 1 commit `1debecab...` independently reviewed and accepted PASS.

## 013 — 2026-08-13 00:18
User authorized Phase 2A.

## 014 — 2026-08-13 00:27
First Phase 2A Codex attempt stopped safely due stale checkout missing governing prompt. No code changes. Retry/bootstrap issued.

## 015 — 2026-08-13
Phase 2A commit `5b7574c...` independently reviewed and accepted PASS.

## 016 — 2026-08-13 06:34 America/Chicago
User authorized Phase 2B. Governing prompt commit `e4278f...`.

## 017 — 2026-08-13
Phase 2B implementation commits `004740e...` and `809389b...` independently reviewed and accepted PASS.

## 018 — 2026-08-13 07:29 America/Chicago
User authorized Phase 3. Governing prompt commit `0d67599...`.

## 019 — 2026-08-13
Codex implemented Phase 3 at `5a6c15af...`; tests/fixtures/checksum passed in Codex report.

## 020 — 2026-08-13 before 08:01 America/Chicago
ChatGPT independently reviewed Phase 3 and accepted `PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

## 021 — 2026-08-13 08:01 America/Chicago
User authorized Phase 4A. Governing prompt commit `fc5dc0b...`.

## 022 — 2026-08-13
Codex implemented Phase 4A at `8155bc45de5fa26fa6077dd870716234d54690c9`; 72 tests and five-fixture diagnostics reported passing; checksum unchanged.

## 023 — 2026-08-13 before 08:42 America/Chicago
ChatGPT independently reviewed Phase 4A stamina state ownership, action costs, exact-dt positional drain, exactly-once round recovery, DynamicModifiers, pre-action ordering, passive-exit neutrality and tests. Final gate accepted:
`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`.

## 024 — 2026-08-13 08:42 America/Chicago
User said **proceed**, explicitly authorizing the next physiology step. Phase 4B was split so knockdown mechanics can be validated before terminal KO/TKO logic is introduced.

Phase 4B1 authorized scope:
- stochastic impact from `striking_power * pre_action_power_modifier`;
- primary persistent trauma moderated by `damage_durability`;
- cumulative trauma state;
- derived current KD resistance from `knockdown_resistance`, trauma, and acute vulnerability;
- probabilistic knockdown;
- KD acute-vulnerability increment;
- continuous exact-time acute-vulnerability decay;
- physiology/KD diagnostics and historical KD anchor if available.

New governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4B1_IMPACT_TRAUMA_KD_2026-08-13.md`

Prompt commit:
`fd81b02a271fa3f367a1bdeaab7150e982d4240b`

Phase 4B2 KO/TKO remains unauthorized.

Expected Codex return: implementation/tests/diagnostics ending with `PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS` or `FAIL`.

Next assistant action: independently review the actual Phase 4B1 implementation before any Phase 4B2 authorization.
