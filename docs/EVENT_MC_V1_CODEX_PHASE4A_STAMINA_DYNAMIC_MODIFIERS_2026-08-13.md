# Codex Prompt — EVENT MC V1 Phase 4A Stamina + Dynamic Modifiers

Date: 2026-08-13

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Source branch: `feature/fsr-32-stamina-shadow`
Architecture revision: v0.3

## Status before this phase

- Phase 0 operational baseline: PASS.
- Phase 1 generic continuous-time kernel: PASS.
- Phase 2A DISTANCE temporal/mechanical parity: PASS.
- Phase 2B wrestling-entry ontology correction: PASS.
- Phase 3 CLINCH + GROUND flow: PASS after independent ChatGPT review.
- Phase 4A is explicitly authorized by the user.
- Phase 4B damage/KD/KO, terminal submissions, judging, and later work are NOT authorized.

Codex cloud may use a local branch named `work`. That is acceptable. Verify ancestry/content, not local branch name.

Before implementation verify Phase 3 is present:

```bash
git fetch origin --prune
git merge-base --is-ancestor 5a6c15af9c2315f23c231d0f34cd3cefddba4578 HEAD
```

If the cloud checkout is stale, safely rebase onto:

`origin/feature/fsr-32-stamina-shadow`

Then verify these documents exist:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md
```

# Read first

Read before changing code:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
3. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md`
6. the current legacy stamina classes/modules and their final effective consumers
7. this prompt

# Development philosophy

The user has explicitly locked:

**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE**

Do not try to build the final perfect physiology system in one step.

The intended loop is:

`build one mechanism -> historical replay -> identify systematic miss -> change one module -> replay`

Phase 4A therefore implements stamina and the dynamic-modifier seam only.

# Goal

Add a fighter-specific stamina reservoir to EVENT MC V1 so that:

1. actions consume stamina;
2. stamina state persists continuously within each Monte Carlo path;
3. fighters recover some stamina between rounds according to the traced/selected stamina model;
4. low stamina reduces offensive output;
5. low stamina reduces expressed striking power through a single derived power modifier;
6. later damage/KD/KO code can consume that power modifier without reaching into stamina internals;
7. all existing DISTANCE/CLINCH/GROUND flow remains modular and functional.

The user specifically wants fighters to be more dangerous while fresh and for expressed power/output to decline materially as the fight gets expensive.

# Absolute non-goals

Do NOT add in Phase 4A:

- damage;
- cumulative trauma;
- acute vulnerability;
- knockdowns;
- KO/TKO;
- terminal submission finishes;
- judging;
- age transformations;
- score urgency/tactical strategy;
- body-part damage;
- damage recovery;
- new MatReturn behavior;
- broad calibration retuning;
- changes to the current inheritance-based simulator;
- FSR builder/rating/ontology changes;
- changes to the frozen FSR-32 parquet;
- Phase 4B or later work.

Do not use stamina as a hidden way to retune Phase 3 ground/clinch residence broadly.

# Frozen FSR input

Use only the existing frozen FSR-32 artifact for historical diagnostics:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rebuild/rewrite it.

# Source-tracing requirement

Before implementing, reconstruct the **effective final stamina behavior** in the current simulator inheritance chain, especially the layers corresponding to:

- `StaticFSRMCKOTKOV3Stamina`
- `StaticFSRMCKOTKOV31RollingFSR`
- `StaticFSRMCKOTKOV32PhaseStamina`
- `StaticFSRMCKOTKOV33GlobalRecovery`
- any later audit/full-fight subclasses that override or alter stamina behavior

Trace effective behavior, not only the earliest class.

At minimum document:

- stamina state representation and initial value/capacity;
- which FSR traits feed stamina capacity, efficiency, recovery, or degradation;
- action-cost logic by action family/phase;
- whether costs occur on attempt, success, or both;
- whether elapsed time itself changes stamina;
- round recovery formula;
- any continuous/global recovery formula;
- how stamina modifies output;
- how stamina modifies power;
- whether stamina currently modifies accuracy, TD success, defense, durability, KD resistance, or other traits;
- any clamps/floors/exponents;
- any duplicated stamina effects caused by RollingFSR or phase-specific layers.

Then explicitly separate:

1. legacy behavior worth porting now;
2. legacy behavior deliberately deferred because it double-counts or entangles stamina with future physiology.

Do not blindly copy the inheritance stack.

# Target EVENT MC architecture

## 1. Authoritative stamina state

Add stamina as future-relevant `FightState` physiology state owned by the engine.

Use a simple normalized or clearly bounded representation. Prefer an interpretable per-fighter fraction such as `[0,1]` unless direct source tracing gives a strong reason to preserve another bounded scale.

Conceptually:

```text
red_stamina
blue_stamina
```

or a similarly small typed structure.

The state must:

- be path-local;
- be deterministic under the named RNG architecture;
- never be stored only in a sink/ledger;
- never introduce another clock;
- mutate only through engine-applied typed deltas.

## 2. Action stamina cost

Each physically meaningful action can consume stamina through one small `StaminaModel` / action-cost component.

Trace current effective cost behavior first.

At minimum consider these current EVENT MC action families:

```text
distance_strike
clinch_strike
ground_strike
distance_takedown
clinch_takedown
clinch_entry
submission_attempt
ground_escape
ground_reversal
clinch_separation
```

Do not invent dozens of unique costs unless the current model has justified differences.

Prefer a small interpretable cost table/family mapping.

Costs should generally be charged for **attempted work**, not only successful actions, unless the current effective formula clearly justifies success-specific additional cost.

No random stamina cost draw is required unless legacy behavior genuinely depends on one.

## 3. Round recovery

Stamina may recover at round boundaries.

Use one clear recovery owner.

Do not reproduce overlapping round recovery + global recovery + rolling-FSR recovery if they represent the same physiology multiple times.

If current legacy behavior has both continuous and between-round recovery, determine whether both are materially distinct. For Phase 4A favor the smallest coherent mechanism that preserves the useful effect.

Round recovery must happen at the authoritative hard round boundary and must not bypass the Phase 1 lifecycle.

Clamp to the valid stamina range.

## 4. DynamicModifiers seam

Do not let `FightFlowRateProvider` or future `DamageModel` reach into stamina implementation details.

Add a compact derived contract conceptually equivalent to:

```text
DynamicModifiers(
    output_multiplier=...,
    power_multiplier=...,
)
```

A provider derives these from:

```text
FighterProfile + FightState -> DynamicModifiers
```

The exact type/name can vary, but the separation is required.

Future modules should consume the derived modifiers, not call private stamina functions.

# Output modifier

Phase 4A should make low stamina reduce **offensive action frequency**.

Apply the output multiplier to offensive attempt rates, not to every transition indiscriminately.

The intended Phase 4A offensive families are:

- distance strikes;
- clinch strikes;
- ground strikes;
- DISTANCE TD initiation;
- CLINCH TD initiation;
- clinch entry initiation;
- submission-attempt generation.

Do **not** automatically multiply passive/positional exit clocks such as clinch separation or total ground exit by the same output factor. That would confound fatigue with Phase 3 residence calibration.

If source tracing strongly supports stamina effects on escape/reversal, expose that as a separate future modifier seam rather than silently using the offensive-output multiplier everywhere.

Required invariant at full stamina:

```text
output_multiplier == 1.0
```

So Phase 3 mechanics are exactly recovered when fighters are fresh.

The modifier must be bounded and monotonic: lower stamina cannot increase output.

# Power modifier

The damage-system review established the desired future ownership:

```text
striking_power -> impact distribution
stamina -> expressed-power modifier
```

Phase 4A should therefore expose a single bounded monotonic `power_multiplier` derived from current stamina.

Conceptually a curve like:

```text
PowerModifier(s) = floor + (1-floor) * s^gamma
```

is acceptable, but do not invent constants casually. Trace the current effective stamina-to-power relationship first and preserve/reuse a small existing calibration where sensible.

Required behavior:

- full stamina -> power multiplier 1.0;
- lower stamina -> lower expressed power;
- multiplier remains positive/bounded;
- stamina modifies power **once**.

Do not also make Phase 4A separately lower damage, KD probability, or KO probability. Those systems do not exist yet.

# Accuracy / success / defense protection

For Phase 4A, do not automatically reduce:

- strike accuracy;
- TD completion probability;
- TD defense;
- control resistance;
- knockdown resistance;
- damage durability;
- submission success probability.

Those would create additional fatigue pathways and make later attribution difficult.

If legacy RollingFSR changes all of these, document it as legacy behavior but defer it unless there is a compelling architecture reason.

Phase 4A should initially establish the two pathways the user explicitly wants:

```text
stamina -> output
stamina -> expressed power
```

# Continuous-time behavior

Actions happen at exact event timestamps.

When an action occurs:

1. event resolves normally;
2. action stamina cost is applied through the engine-owned state mutation contract;
3. subsequent candidate rates are recomputed from the new stamina state;
4. no 10-second stamina bucket or segment rounding is introduced.

If there is an authorized continuous recovery term, it must use the Phase 1 exact `dt` continuous-advance hook rather than a second timer.

Do not add a second fight clock.

# Event ordering / future damage seam

Phase 4A must not break the future same-timestamp chain.

Eventually a landed strike should be able to behave conceptually as:

```text
strike resolution
-> stamina cost / state update
-> damage impact using current derived power modifier under a clearly documented ordering
-> KD
-> consequence
-> finish
```

Do not implement that chain now.

But avoid hard-wiring stamina into strike resolution in a way that makes later ordering impossible to control.

Document whether the power modifier associated with an action is the pre-action or post-action stamina state. Prefer **pre-action stamina for that action's expressed power**, then charge the action cost for future events, unless the current effective model strongly requires another ordering. This prevents an action from weakening itself merely because it was attempted.

Required test/contract:

```text
current action uses pre-action stamina-derived modifiers
then action cost lowers stamina for subsequent actions
```

# FSR trait mapping

Do not invent trait names.

Inspect the frozen FSR-32 schema and current simulator consumers to determine which existing FSR trait(s) represent stamina/cardio/recovery/efficiency.

Keep mapping explicit and small.

If more than one stamina-related FSR trait exists, assign distinct interpretable roles only when supported by existing construction/meaning. Do not average arbitrary ratings simply to use them all.

Do not change FSR ratings or builders.

# Observability

Extend observer-only diagnostics to report at minimum per fighter:

- starting stamina;
- stamina after each round;
- ending stamina;
- minimum stamina;
- action stamina spent by family;
- action counts by family;
- average output multiplier by round;
- average power multiplier by round;
- optionally timestamped stamina trace in diagnostic/full-trace mode only.

For compact cohort runs, do not require full per-event traces.

Sinks remain observer-only.

# Required tests

## A. State/invariant tests

- stamina begins at valid full/default state;
- costs lower only the acting fighter's stamina;
- stamina never drops below lower bound;
- round recovery raises stamina but never above cap;
- action cost does not change fight time;
- sinks do not affect stamina physics;
- deterministic seed gives deterministic state/result.

## B. Modifier tests

- full stamina produces output=1 and power=1;
- lower stamina monotonically lowers output;
- lower stamina monotonically lowers power;
- modifiers remain bounded;
- output modifier changes attempt rates but not strike landing probability or TD completion probability;
- Phase 2B DISTANCE TD intrinsic base/ontology remains intact before multiplication by the external stamina output modifier;
- ground/clinch passive exit clocks remain unchanged by offensive output multiplier.

## C. Action-order test

Explicitly verify the current action uses the **pre-action** stamina-derived power/output state and that its cost affects subsequent events/rates.

## D. Round-boundary test

Start near a round boundary with depleted stamina and verify:

- sampled event past the boundary is truncated;
- round lifecycle occurs;
- stamina round recovery happens exactly once;
- positional reset still occurs;
- next-round rates reflect recovered stamina.

## E. Phase 3 regression

At full stamina or with stamina effects neutralized, Phase 3 flow formulas/rates should reproduce the reviewed Phase 3 behavior exactly or within deterministic floating tolerance.

Do not silently retune Phase 3 mechanics while adding stamina.

# Historical / fixture diagnostics

Run deterministic diagnostics on at least:

1. Derrick Lewis vs Chris Daukaus — power/early-danger fixture;
2. Max Holloway vs Calvin Kattar — high-volume stamina stress fixture;
3. Merab Dvalishvili vs Petr Yan — wrestling/output stress fixture;
4. Rob Font vs Raul Rosas Jr. — ontology + wrestling stress fixture;
5. Charles Oliveira vs Dustin Poirier — mixed grappling/submission-attempt fixture.

Use full scheduled 900-second nonterminal paths because damage/finishes remain out of scope.

For each fighter report approximately:

- R1/R2/R3 start and end stamina;
- total stamina spent;
- major action counts;
- mean output multiplier by round;
- mean power multiplier by round;
- phase shares;
- whether every path reaches horizon.

Also include a Phase-3-neutral A/B where stamina modifiers/costs are disabled versus active Phase 4A so we can see exactly what stamina changes.

Do not claim winner/method predictive improvement from this phase.

# Runtime

Report paths/second for the five-fixture diagnostic and compare roughly with the Phase 3 diagnostic (~31.8 paths/sec reported at 100 paths/fixture).

Do not optimize individual-strike eventing yet unless the new stamina implementation creates an obvious pathological slowdown.

# Frozen protections

Verify before completion:

- current inheritance-based simulator files unchanged;
- frozen FSR artifact SHA unchanged;
- no FSR builder/rating changes;
- no Phase 4B physiology/finish files introduced;
- no terminal outcomes introduced;
- Phase 3 tests remain passing.

# Expected implementation style

Prefer small modules, conceptually:

```text
stamina.py                 # state/cost/recovery calculations
modifiers.py               # DynamicModifiers + provider
```

Names may differ.

Do not put all stamina formulas in `actions.py`.

Do not create an inheritance hierarchy.

Keep one owner for each mutation.

# Required Codex return

Return:

1. starting SHA;
2. final SHA;
3. files changed;
4. legacy source trace and what was preserved/deferred;
5. exact stamina state/cost/recovery formulas;
6. exact output and power modifier formulas;
7. FSR trait mapping;
8. fixture diagnostic summary;
9. runtime;
10. tests run/results;
11. frozen artifact SHA verification;
12. scope-protection statement;
13. commit/push status;
14. final gate:

`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: PASS`

or

`PHASE 4A STAMINA + DYNAMIC MODIFIERS GATE: FAIL`

Do not authorize or implement Phase 4B automatically.
