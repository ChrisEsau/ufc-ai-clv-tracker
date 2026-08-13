# EVENT MC V1 Chat Continuity / Working Memory

Date created: 2026-08-12
Last updated: 2026-08-13
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
- Phase 2B wrestling-entry ontology correction: **AUTHORIZED and current task; not yet passed**.
- Phase 3 and later mechanics: **NOT AUTHORIZED**.

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

Phase 1 implementation commit:
`1debecab69a141bf2f81179f3436af569733b750`

Phase 2A implementation commit:
`5b7574c7689ffa2e55821a49fca47a2c1c937991`

Current Phase 2B governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2B_WRESTLING_ENTRY_ONTOLOGY_2026-08-13.md`

Phase 2B prompt commit:
`e4278f62359e500055eea8c4521d8be6eba6fa2b`

Development standard locked by user:
**Working + predictive + modular + easy to iterate. Do not optimize for perfection or exhaustive defensive hardening. Ultimate success is moneyline/prop predictive accuracy and betting usefulness. Protect critical invariants and calculation seams, but favor flexibility and rapid historical validation over speculative abstraction.**

Damage/KO external review may run in parallel. Do not hold EVENT MC progress for it. Preserve clean replaceable damage/KD/KO subsystem seams so a later design can be swapped in.

Known performance note: Phase 2A schedules individual strike attempts as events. Do not optimize/batch them preemptively; benchmark later and optimize only if runtime becomes a real problem.

Known non-blocking Phase 1 seam: primary resolution currently returns one state delta plus consequence notifications. Later damage -> KD -> finish chains may need sequential consequence modules that each return/apply their own deltas. Do not retrofit this until required.

---

# Phase 1 Locks Preserved

- one authoritative `FightState.fight_time_seconds` clock;
- scheduler remains UFC-agnostic;
- all rates are events/second;
- exact interval probability -> per-second hazard conversion;
- stable named RNG streams via `SeedSequence([root_seed, stable_stream_id])`;
- engine-owned state mutation through typed deltas;
- continuous advance over exact elapsed `dt` before event resolution;
- hard round/fight boundaries owned by engine;
- round start resets phase to DISTANCE and clears positional ownership;
- null/stats/full-trace sinks remain observer-only;
- no hidden component RNGs.

Stable RNG stream IDs:
- SCHEDULER 10
- STRIKE_RESOLUTION 20
- TAKEDOWN 30
- SUBMISSION 40
- DAMAGE 50
- KNOCKDOWN_FINISH 60
- JUDGING 70

---

# Reviewed Phase 2A Result

Phase 2A successfully ported the six authorized DISTANCE primary action families:
1. red strike attempt;
2. blue strike attempt;
3. red TD attempt;
4. blue TD attempt;
5. red clinch entry;
6. blue clinch entry.

Phase 2A preserved current V0 formulas/semantics and changed timing only.

Reviewed formula behavior:
- DISTANCE strike attempt intensity preserves the legacy Poisson mean: base 5 attempts/30 sec multiplied by the existing pressure modifier, then expected count / 30 sec;
- strike landing remains `sigmoid(logit(0.40) + (precision - opponent defense)/12)`;
- legacy TD initiation uses 30-sec base 0.10 rescaled to 10 sec, then the legacy blended wrestling preference;
- TD success remains `sigmoid((wrestling_conversion - opponent_td_defense)/12 - 0.40)`;
- clinch entry preserves current V0 style preference logic and 0.60 cap;
- final 10-sec transition probabilities convert exactly to per-second hazards.

Legacy Phase 2A wrestling preference:
```text
legacy_wrestling_preference =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Phase 2A tests compare new formulas directly against actual `StaticFSRMCV0` consumers, not only copied constants.

Reviewed Phase 2A diagnostic highlights:

Rob Font:
- wrestling_entry 48.59305;
- legacy blended preference -4.44615;
- TD p/10 sec 1.64486%;
- continuous hazard/sec 0.00165854;
- legacy matched-distance TD attempts/15 min 1.3852;
- EVENT MC matched-distance TD attempts/15 min 1.5030;
- EVENT MC TD success 42.37%.

Raul Rosas Jr.:
- wrestling_entry 54.43824;
- legacy blended preference 6.38876;
- TD p/10 sec 10.00891%;
- continuous hazard/sec 0.01054595;
- legacy matched-distance TD attempts/15 min 8.8052;
- EVENT MC matched-distance TD attempts/15 min 9.5456;
- EVENT MC TD success 53.33%.

Merab Dvalishvili vs Petr Yan showed the largest expected timing difference: Merab legacy matched-distance TD attempts/15 min 17.4524 vs continuous 19.8816. This was classified as expected continuous-event/segment-suppression behavior, not a formula bug. Do not tune it away during ontology work.

Phase 2A validation:
- no formula mismatch observed;
- no unit/conversion mismatch observed;
- continuous event competition differences observed as expected;
- missing clinch/ground downstream flow intentionally out of scope;
- no Phase 2B semantics introduced during Phase 2A;
- 32 EVENT MC tests passed in Codex report;
- 45 EVENT MC + relevant V0 tests passed;
- 17 downstream simulator regressions passed;
- frozen FSR SHA unchanged.

Final reviewed gate:
`PHASE 2A DISTANCE TEMPORAL PARITY GATE: PASS`.

---

# Phase 2B — Current Task

Phase 2B makes **one deliberate semantic correction**:

`wrestling_entry` becomes the intrinsic DISTANCE TD initiation driver.

Correct ontology:
```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete a shot
td_defense           = opponent prevention of completion
control_imposition   = persistence/behavior after advantageous position
```

Phase 2B active TD initiation contract:
```text
entry_delta = wrestling_entry - 50.0
entry_modifier = exp(clip(entry_delta, -8, 8) / existing_MODIFIER_SCALE)

p_td_10s = existing_DISTANCE_TD_ATTEMPT_BASE_10S * entry_modifier
p_td_10s = clip(p_td_10s, 0, 1 - epsilon)

lambda_td = -ln(1 - p_td_10s) / 10
```

Reuse the existing Phase 2A modifier scale and TD base. **No tuning.**

Phase 2B separation requirements:
- `wrestling_entry` changes TD initiation;
- `control_imposition` no longer changes intrinsic TD initiation;
- distance striking pressure no longer directly suppresses intrinsic TD initiation;
- clinch striking pressure no longer directly suppresses intrinsic TD initiation;
- wrestling_conversion does not change initiation;
- opponent td_defense does not change initiation;
- TD success formula stays conversion vs defense exactly as in Phase 2A;
- context multiplier remains neutral 1.0 in Phase 2B; preserve a seam for future contextual opportunity without inventing coefficients.

Keep the Phase 2A legacy formula available as an explicit A/B diagnostic comparator where practical.

Required Phase 2B A/B diagnostics include Font/Rosas, Merab/Yan, Holloway/Kattar, and Lewis/Daukaus. Font/Rosas remains non-replaceable.

Do not assume Phase 2B improves prediction. Measure what moves; historical validation/tuning comes later.

---

# Hard Locks / Non-Goals

Do not:
- modify the current inheritance-based simulator;
- rebuild/rewrite FSR-32;
- alter FSR builders/ratings/ontology/maturity/leakage rules;
- retune base TD probability or modifier scale;
- add opponent td_defense to initiation;
- hide control/striking pressure back into TD initiation;
- change strike mechanics;
- change clinch-entry mechanics;
- change TD success mechanics;
- add clinch internals;
- add ground internals;
- add submissions;
- add stamina;
- add damage/KD/KO;
- add recovery;
- add age transforms;
- add judging;
- add tactical urgency;
- add arbitrary cooldowns;
- add MatReturn;
- implement Phase 3 or later work.

Future damage/KD/KO systems must remain replaceable behind clean interfaces, but their redesign must not block Phase 2B.

---

# Assistant Review Required When Phase 2B Returns

Do not accept Codex PASS automatically. Independently inspect the actual commit/diff and verify:
- Phase 2A was starting point or later documentation-only fast-forward;
- only TD initiation semantics changed materially;
- new formula centers wrestling_entry at 50 and reuses existing base/scale without tuning;
- control_imposition and striking pressures no longer affect Phase 2B intrinsic initiation;
- conversion and td_defense remain resolution-only;
- TD success formula unchanged;
- strike/clinch mechanics unchanged;
- Phase 1 clock/scheduler/RNG/state/sink invariants preserved;
- Phase 2A legacy comparator retained cleanly where useful;
- Font/Rosas and Merab/Yan A/B results are interpretable;
- no full-fight conclusions are drawn before clinch/ground mechanics exist;
- no downstream physiology/finish systems slipped in;
- implementation remains small and easy to iterate.

Only after independent review may ChatGPT mark:
`PHASE 2B WRESTLING ENTRY ONTOLOGY GATE: PASS`.

Do not authorize Phase 3 automatically.

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
User said to move on and supply FSR later. Phase 1 prompt prepared but not launched.

## 006 — 2026-08-12 23:08
User found exact frozen parquet and established frozen SHA-256.

## 007 — 2026-08-12 23:18
Exact parquet published as temporary GitHub Release asset for transfer.

## 008 — 2026-08-12 23:21
Codex generic workspace lacked remote/auth; changed nothing. Bootstrap prompt issued.

## 009 — 2026-08-12 23:24
User clarified Phase 1 had never launched. Continuity corrected.

## 010 — 2026-08-13 near 00:00
Phase 0 returned PASS after exact artifact recovery/revalidation.

## 011 — 2026-08-13 00:03
User explicitly authorized Phase 1.

## 012 — 2026-08-13 about 00:15
Phase 1 commit `1debecab...` independently reviewed and accepted.

## 013 — 2026-08-13 00:18
User explicitly authorized Phase 2A.

## 014 — 2026-08-13 00:27
First Phase 2A attempt stopped safely due stale checkout missing the governing prompt; no implementation changes occurred. Retry/bootstrap prompt added.

## 015 — 2026-08-13
Phase 2A implementation returned at commit `5b7574c7689ffa2e55821a49fca47a2c1c937991`. ChatGPT independently reviewed the actual commit, formulas, direct V0 parity tests, and diagnostics and accepted:
`PHASE 2A DISTANCE TEMPORAL PARITY GATE: PASS`.

## 016 — 2026-08-13 06:34 America/Chicago
User said **proceed with the next step**, explicitly authorizing Phase 2B wrestling-entry ontology correction.

New governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE2B_WRESTLING_ENTRY_ONTOLOGY_2026-08-13.md`

Prompt commit:
`e4278f62359e500055eea8c4521d8be6eba6fa2b`

Expected Codex return: implementation/tests/A-B diagnostics ending with `PHASE 2B WRESTLING ENTRY ONTOLOGY GATE: PASS` or `FAIL`.

Next assistant action: independently review the actual Phase 2B commit before any Phase 3 authorization.

Phase 0: **PASS**.
Phase 1: **PASS**.
Phase 2A: **PASS**.
Phase 2B authorized: **YES**.
Phase 2B reviewed/passed: **NO**.
Phase 3 authorized: **NO**.
