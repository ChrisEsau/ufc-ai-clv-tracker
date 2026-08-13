# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13 08:01 America/Chicago
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
- Phase 4A stamina + dynamic modifiers: **AUTHORIZED and current task; not yet passed**.
- Phase 4B damage/KD/KO and later mechanics: **NOT AUTHORIZED**.

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

Current Phase 4A governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`

Phase 4A prompt commit:
`fc5dc0b32460e9200ea070db7f3656d1aac5f689`

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
- null/stats/full-trace sinks remain observer-only;
- no hidden component RNGs;
- current inheritance-based simulator remains untouched;
- FSR-32 remains the initial profile source and frozen artifact is never rebuilt during EVENT MC work;
- no giant inheritance chain in EVENT MC; use composition;
- no whole-simulator Codex prompt;
- tune one subsystem at a time and validate historically before claiming improvement.

Stable RNG stream IDs:
- SCHEDULER 10
- STRIKE_RESOLUTION 20
- TAKEDOWN 30
- SUBMISSION 40
- DAMAGE 50
- KNOCKDOWN_FINISH 60
- JUDGING 70

Known non-blocking future seam: the current primary Resolution returns one state delta plus consequence notifications. Damage -> KD -> consequence -> finish may later require sequential consequence modules that each return/apply deltas. Do not retrofit until Phase 4B requires it.

Known performance note: individual strike attempts are currently scheduled as events. Do not preemptively batch them; benchmark and optimize only if runtime becomes a demonstrated problem.

---

# Phase 2B Wrestling Ontology Lock

Correct ontology:
```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention of completion
control_imposition   = persistence after advantageous position
```

Active DISTANCE TD initiation:
```text
entry_delta = wrestling_entry - 50
entry_modifier = exp(clip(entry_delta, -8, 8) / existing_MODIFIER_SCALE)
p_td_10s = existing_DISTANCE_TD_ATTEMPT_BASE_10S * entry_modifier
lambda_td = -ln(1 - p_td_10s) / 10
```

Legacy Phase 2A blend remains diagnostic-only:
```text
0.75*wrestling_entry
+ 0.25*control_imposition
- 0.50*distance_striking_pressure
- 0.50*clinch_striking_pressure
```

TD success remains wrestling_conversion vs opponent td_defense.

---

# Reviewed Phase 3 Result

Final reviewed gate:
`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

Implementation commit:
`5a6c15af9c2315f23c231d0f34cd3cefddba4578`

Phase 3 now supports continuous nonterminal fight flow through:

```text
DISTANCE <-> CLINCH <-> GROUND
```

Authorized/implemented mechanics:
- CLINCH strikes;
- CLINCH takedown attempts and resolution;
- CLINCH separation to DISTANCE;
- GROUND top strikes;
- reduced-rate bottom ground strikes;
- top and bottom submission-attempt generation, nonterminal only;
- ground escape/standup;
- ground reversal/controller swap;
- exact phase residence time;
- exact controller-specific clinch/ground time;
- deterministic full scheduled-horizon diagnostics.

Ground exit hard invariant is preserved:
```text
lambda_reversal = lambda_ground_exit * P(reversal | exit)
lambda_escape   = lambda_ground_exit * (1 - P(reversal | exit))
lambda_reversal + lambda_escape == lambda_ground_exit
```

Phase 3 ported effective V0 bases without broad retuning:
- clinch separation 0.25 / 30 sec;
- clinch TD attempts 0.24 / 30 sec;
- ground exit 0.20 / 30 sec;
- clinch strikes 1.2 / 30 sec;
- ground strikes 1.6 / 30 sec;
- submission attempts 0.045 / 30 sec;
- reversal share 0.18;
- bottom ground strike multiplier 0.20;
- bottom submission multiplier 0.55.

Independent review confirmed:
- scheduler remains generic;
- state transitions are correct;
- submission attempts are nonterminal;
- control-time ledger is observer-only and driven by exact engine dt;
- Phase 2B DISTANCE TD initiation is not contaminated by control_imposition;
- round boundary reset remains authoritative;
- no stamina/damage/KD/KO/judging/age/terminal SUB mechanics slipped in.

Phase 3 diagnostics reported all 500/500 paths reached 900-sec horizon. Runtime approximately 31.84 paths/sec for five frozen fixtures at 100 paths each.

Validation watchlist, not tuning authorization:
- some grappling fixtures spend roughly 40% of scheduled time on ground;
- some CLINCH TD hazards, especially Merab, are high while in phase;
- historical cohort validation must determine whether those are correct before changing them.

Minor non-blocking doc debt: FighterProfile docstring still says DISTANCE parity adapter even though it now carries all-phase inputs.

---

# Phase 4A — Current Task

Phase 4A is explicitly authorized and governed by:

`docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`

Goal: add **stamina reservoir + action costs + dynamic output/power modifiers only**.

Required target behavior:

```text
actions consume stamina
-> stamina persists path-locally
-> round recovery restores some stamina
-> low stamina reduces offensive action frequency
-> low stamina reduces expressed striking power
```

Required architecture seam:

```text
FighterProfile + FightState
        -> DynamicModifiers
              output_multiplier
              power_multiplier
```

Fight-flow and future damage modules consume derived modifiers; they do not call stamina internals.

Phase 4A should trace final legacy behavior across the stamina/rolling-FSR/phase-stamina/global-recovery inheritance layers before deciding what to preserve.

Important Phase 4A locks:
- full stamina must recover reviewed Phase 3 rates exactly;
- offensive output multiplier applies to offensive attempt rates, not blindly to passive phase-exit clocks;
- full stamina output multiplier = 1;
- full stamina power multiplier = 1;
- lower stamina monotonically lowers both;
- current action uses **pre-action stamina** for its modifiers, then action cost reduces stamina for subsequent actions;
- do not reduce strike accuracy, TD success, TD defense, control resistance, durability, or KD resistance in Phase 4A;
- do not implement damage/KD/KO/terminal SUB/judging/age;
- no broad Phase 3 retuning.

Historical fixtures required:
- Lewis/Daukaus;
- Holloway/Kattar;
- Merab/Yan;
- Font/Rosas;
- Oliveira/Poirier.

Expected Codex return:
`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS` or `FAIL`.

When Phase 4A returns, ChatGPT must independently inspect the actual implementation commit/diff, legacy stamina trace, action-cost ownership, round recovery, modifier formulas, pre-action ordering, Phase 3 neutral regression, tests, runtime, and fixture diagnostics before accepting PASS.

Do not authorize Phase 4B automatically.

---

# Future Damage / KD / KO Target Architecture

A separate damage-system review recommended a simpler EVENT MC physiology model. Treat this as target architecture for Phase 4B, not a claim that every legacy constant has been reconstructed.

Future dynamic physiology state should initially be small:
```text
cumulative_trauma
acute_vulnerability
stamina  [owned by stamina system]
```

Future derived values rather than redundant mutable health bars:
```text
current_KD_resistance
current_finish_resistance
effective_power
```

Trait ownership target:
```text
striking_power        -> impact severity distribution
damage_durability     -> persistent trauma deposited per impact
knockdown_resistance  -> baseline acute KD resistance
stamina               -> expressed-power modifier
```

Desired consequence chain:
```text
StrikeResolver
-> DamageModel / impact + primary trauma
-> KnockdownModel / impact vs current resistance
-> KnockdownConsequenceModel / acute vulnerability + optional small collapse trauma
-> FinishModel / probabilistic KO/TKO
```

Important future rules:
- power should not separately multiply damage + KD + KO;
- impact carries power downstream;
- cumulative trauma lowers future resistance rather than acting primarily as a health-bar KO threshold;
- acute vulnerability decays continuously;
- cumulative trauma initially has no in-fight recovery;
- stamina recovery remains separate;
- ground and distance use the same physiology pipeline with context modifiers, not separate damage systems.

Phase 4B is NOT authorized yet.

---

# Canonical / Governing Docs

- `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
- `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE1_GENERIC_KERNEL_PROMPT_2026-08-12.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2A_RETRY_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE2B_WRESTLING_ENTRY_ONTOLOGY_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE3_RETRY_2026-08-13.md`
- `docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`

---

# Checkpoint History

## 001 — 2026-08-12 22:33 America/Chicago
Phase 0 operational baseline prompt issued.

## 002 — 2026-08-12 22:37
Codex baseline execution plan approved with exact-remote-branch guardrail.

## 003 — 2026-08-12 22:40
Codex stopped because isolated checkout had no Git remote. Remote-unblock prompt issued.

## 004 — 2026-08-12 22:49
Codex repaired Git environment but baseline failed because frozen FSR-32 parquet was absent. Artifact recovery issued; no rebuild authorized.

## 005 — 2026-08-12 22:59
User initially chose to move on and supply FSR later; Phase 1 was prepared but not launched.

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
Phase 2B implementation commits `004740e...` and `809389b...` independently reviewed and accepted PASS. 56 EVENT MC + relevant V0 tests reported passing; frozen FSR SHA unchanged.

## 018 — 2026-08-13 07:29 America/Chicago
User authorized Phase 3. Governing prompt commit `0d67599...`.

## 019 — 2026-08-13
Codex implemented Phase 3 at `5a6c15af9c2315f23c231d0f34cd3cefddba4578`. 66 EVENT MC + relevant V0 tests reported passing; 500/500 fixture paths reached horizon; frozen FSR SHA unchanged.

## 020 — 2026-08-13 before 08:01 America/Chicago
ChatGPT independently reviewed the actual Phase 3 commit, active providers, action resolution, formulas, profile adapter, FlowStatsSink, engine boundary behavior, and tests. Final gate accepted:
`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`.

## 021 — 2026-08-13 08:01 America/Chicago
User said **proceed**, explicitly authorizing the next step. Phase 4 was deliberately split into smaller modules rather than one giant physiology/finish rewrite.

Phase 4A authorized scope: stamina reservoir, action stamina costs, round recovery, derived offensive-output multiplier, and derived expressed-power multiplier. No damage/KD/KO/terminal SUB/judging/age.

New governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`

Prompt commit:
`fc5dc0b32460e9200ea070db7f3656d1aac5f689`

Expected Codex return: implementation/tests/fixture diagnostics ending with `PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS` or `FAIL`.

Next assistant action: independently review the actual Phase 4A implementation before any Phase 4B authorization.

Phase 0: **PASS**.
Phase 1: **PASS**.
Phase 2A: **PASS**.
Phase 2B: **PASS**.
Phase 3: **PASS**.
Phase 4A authorized: **YES**.
Phase 4A reviewed/passed: **NO**.
Phase 4B authorized: **NO**.
