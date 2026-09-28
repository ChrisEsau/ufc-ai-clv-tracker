# Codex Prompt — EVENT MC V1 Phase 4B1 Impact + Trauma + Knockdown

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Source branch: `feature/fsr-32-stamina-shadow`
Architecture revision: v0.3

## Status before this phase

- Phase 0 operational baseline: PASS.
- Phase 1 continuous-time kernel: PASS.
- Phase 2A DISTANCE parity: PASS.
- Phase 2B wrestling-entry ontology correction: PASS.
- Phase 3 CLINCH + GROUND flow: PASS.
- Phase 4A stamina + DynamicModifiers: PASS after independent ChatGPT review.
- Phase 4B1 is explicitly authorized by the user.
- Phase 4B2 KO/TKO finishes, terminal submissions, judging, age, and later work are NOT authorized.

Codex cloud may use a local branch named `work`. That is acceptable. Verify ancestry/content, not local branch name.

Before implementation:

```bash
git fetch origin --prune
git merge-base --is-ancestor 8155bc45de5fa26fa6077dd870716234d54690c9 HEAD
```

If stale, safely rebase onto:

`origin/feature/fsr-32-stamina-shadow`

Then verify:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE4B1_IMPACT_TRAUMA_KD_2026-08-13.md
```

# Read first

Read before changing code:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
3. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE4A_STAMINA_DYNAMIC_MODIFIERS_2026-08-13.md`
6. the current inheritance-based damage/KD classes and their final effective consumers
7. this prompt

Also treat the reviewed EVENT MC damage-system design as the target architecture:

```text
landed strike
-> effective power
-> stochastic impact
-> primary trauma
-> engine applies persistent trauma
-> derive current KD resistance
-> probabilistic KD
-> engine applies KD consequence / acute vulnerability
```

No KO/TKO finish is allowed in 4B1.

# Development philosophy

**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE**

Build and validate one physiology mechanism at a time.

The intended ablation logic is:

```text
impact/KD baseline
-> cumulative trauma
-> acute post-KD vulnerability
-> continuous vulnerability recovery
-> later KO/TKO finish model
```

Do not create the final finish engine now.

# Phase 4B1 goal

Build a clean physiology layer that turns landed strikes into:

1. one stochastic impact value;
2. persistent cumulative trauma on the defender;
3. derived current knockdown resistance;
4. probabilistic knockdowns;
5. short-lived acute vulnerability after a KD;
6. continuous exact-time decay of acute vulnerability;
7. observer-visible impact/trauma/KD diagnostics.

The result should support both:

```text
large fresh impact -> possible one-shot KD
```

and:

```text
repeated landed strikes -> cumulative trauma -> lower future resistance -> later KD becomes more likely
```

A fighter must never be knocked out in Phase 4B1.

# Absolute non-goals

Do NOT add:

- KO/TKO finishes;
- referee stoppages;
- terminal submission finishes;
- judging;
- age transforms;
- body-part damage;
- head/body/leg reservoirs;
- cuts;
- multiple health bars;
- cumulative-trauma recovery;
- tactical urgency;
- defender-stamina-to-KD-resistance effects;
- accuracy degradation from damage;
- TD-defense degradation from damage;
- new phase-flow retuning;
- new stamina calibration;
- modifications to the inheritance-based simulator;
- FSR builder/rating/ontology changes;
- changes to the frozen FSR-32 parquet;
- Phase 4B2 or later work.

# Frozen FSR input

Use only:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rebuild or rewrite it.

# Source-tracing requirement

Before implementation, reconstruct the **final effective legacy damage/KD behavior**, not merely the earliest base class.

At minimum inspect the inheritance path around:

- `StaticFSRMCDamageV1`
- `StaticFSRMCDamageV1Ground017`
- `StaticFSRMCKOTKOV2`
- `StaticFSRMCKOTKOV2KDCollapse`
- `StaticFSRMCKOTKOV2RoundRecovery`
- `StaticFSRMCKOTKOV3Stamina`
- `StaticFSRMCKOTKOV31RollingFSR`
- `StaticFSRMCKOTKOV32PhaseStamina`
- `StaticFSRMCKOTKOV33GlobalRecovery`
- later audit / KO-SUB / full-fight subclasses that override relevant behavior

Document:

- where landed strikes become damage/impact;
- how striking_power enters;
- how damage_durability enters;
- how knockdown_resistance enters;
- whether power is double/triple counted in legacy damage/KD/KO;
- whether KD-collapse adds damage again;
- which recovery layers operate on damage/resistance;
- whether stamina alters raw power upstream or damage downstream;
- exact useful constants/formulas worth preserving;
- legacy behaviors intentionally NOT ported because they conflate responsibilities.

The external review explicitly did not certify all final legacy formulas. Source tracing in this task must resolve what the code actually does before claiming parity.

# Target trait ownership — HARD LOCK

Each FSR trait gets one primary physiological responsibility:

```text
striking_power
    -> impact severity distribution

damage_durability
    -> persistent trauma deposited for a given impact

knockdown_resistance
    -> baseline acute KD resistance

stamina traits
    -> stamina state
    -> DynamicModifiers.power_multiplier
```

Do not use `striking_power` separately again in KD probability.

Do not use stamina separately again in KD probability.

Power and stamina must be carried downstream through **impact**.

# Dynamic state

Add only the smallest necessary physiology state to `FightState`:

```text
red_cumulative_trauma
blue_cumulative_trauma
red_acute_vulnerability
blue_acute_vulnerability
```

These names may vary slightly if a cleaner typed structure fits current conventions, but semantics must remain explicit.

Properties:

### cumulative_trauma
- defender-specific;
- persistent within the fight;
- non-negative;
- only increases from primary damaging impacts in 4B1;
- no between-round or continuous recovery in 4B1;
- never directly causes a finish;
- lowers future derived resistance.

### acute_vulnerability
- defender-specific;
- non-negative;
- rises after a KD;
- may optionally receive a smaller severe-impact increment only if source tracing clearly supports it; default: KD-owned increment only;
- decays continuously with exact elapsed fight time;
- does not itself terminate the fight.

Keep knockdown count in diagnostics/ledger unless causal state is truly required. Do not add unnecessary mutable fields.

# Landed-strike physiology contract

Only **landed strikes** enter damage/KD physiology.

Missed strikes must not create impact, trauma, or KD.

For every landed DISTANCE / CLINCH / GROUND strike, produce an immutable result/observation carrying enough information for the physiology chain, conceptually:

```text
LandedStrikeResult(
    attacker,
    defender,
    phase,
    pre_action_dynamic_modifiers,
    ...
)
```

Do not hard-code damage inside the strike-rate provider.

Do not make the scheduler aware of damage.

# Effective power — HARD LOCK

Use Phase 4A's pre-action power modifier.

Conceptually:

```text
effective_power = base_power * power_multiplier * strike_type_modifier * context_modifier
```

For 4B1:

- `base_power` comes from `striking_power`;
- `power_multiplier` comes from the pre-action `DynamicModifiers` captured before stamina cost;
- strike type/context multipliers should remain neutral unless current source tracing provides a simple justified phase/ground value worth porting;
- do not invent multiple phase multipliers just to force target KD rates.

Use a normalized/interpretable scaling so ratings near 50 produce sensible baseline impact.

# Stochastic impact

Impact must be stochastic even for the same fighter at the same stamina.

Conceptually:

```text
Impact = EffectivePower * SeverityDraw
```

Use the existing named DAMAGE RNG stream for impact/severity sampling.

Prefer a positive distribution with a small parameter count, e.g. lognormal/gamma, unless the effective legacy source clearly provides a better directly portable distribution.

Requirements:

- positive impact;
- higher striking_power stochastically shifts impact upward;
- lower stamina power multiplier shifts impact downward;
- identical seed/input reproduces identical impact sequence;
- no hidden RNG.

Do not calibrate many distribution parameters now. Keep them centralized and explicit.

# Primary trauma

Conceptually:

```text
primary_trauma = impact * durability_modifier(defender.damage_durability)
```

Higher `damage_durability` must mean less trauma for identical impact.

Then:

```text
defender.cumulative_trauma += primary_trauma
```

This is the **only primary persistent damage mutation** in Phase 4B1.

Do not add the original strike impact again after a KD.

Do not replay primary trauma through a collapse event.

# Derived current KD resistance

Do not store mutable remaining-KD-resistance state initially.

Derive it from:

```text
base knockdown_resistance
x cumulative-trauma erosion modifier
x acute-vulnerability modifier
```

Conceptually:

```text
current_kd_resistance =
    base_kd_resistance
    * trauma_modifier(cumulative_trauma)
    * acute_modifier(acute_vulnerability)
```

Required directions:

- higher `knockdown_resistance` -> higher current resistance;
- more cumulative trauma -> lower current resistance;
- more acute vulnerability -> lower current resistance;
- all terms remain positive and bounded away from zero.

Keep the number of calibration parameters small.

# Knockdown probability

Use **impact relative to current resistance** as the central quantity.

Conceptually:

```text
impact_ratio = impact / current_kd_resistance
p_kd = sigmoid(kd_slope * (log(impact_ratio) - kd_midpoint))
```

Equivalent monotonic forms are acceptable if source tracing gives a compelling simpler expression.

Hard requirements:

- probabilistic, never a hard threshold;
- fresh one-shot KDs must be possible;
- high trauma raises future KD probability through lower resistance;
- acute vulnerability raises immediate repeat-KD probability;
- `striking_power` is NOT separately inserted into p(KD);
- stamina is NOT separately inserted into p(KD);
- p(KD) stays strictly within [0,1].

Use the named `KNOCKDOWN_FINISH` RNG stream for KD sampling.

# Same-timestamp consequence ordering

The desired event chain is:

```text
landed strike resolution
-> calculate impact
-> calculate/apply primary trauma
-> derive resistance after primary trauma
-> sample KD
-> if KD, apply KD consequence / acute vulnerability
```

No extra simulation time elapses between those same-strike consequence stages.

The engine remains the sole mutator.

If the current single-Resolution contract cannot express this cleanly, make the **smallest necessary compositional extension** for sequential same-timestamp consequence modules/deltas. This is the known Phase 1 seam that was intentionally deferred until physiology required it.

Do not make action components mutate `FightState` directly.

Do not create another inheritance stack.

# Knockdown consequence

A KD should:

- increment defender acute vulnerability by one small explicit amount or bounded transformation;
- generate observer-visible KD outcome with attacker/defender/timestamp/phase/impact/current resistance/p(KD);
- optionally cause a simple positional consequence only if current event architecture or effective legacy behavior makes it essential.

Default Phase 4B1 behavior should **not** force a phase transition merely because a KD occurred unless source tracing supports a clean existing convention.

Do not add collapse trauma by replaying the original strike.

If legacy collapse damage appears useful, document it as deferred and do not port it unless there is a clear small non-duplicative formulation.

# Acute vulnerability recovery

Acute vulnerability decays continuously using the exact engine `dt`.

Conceptually:

```text
v(t + dt) = v(t) * exp(-lambda * dt)
```

or half-life equivalent.

Requirements:

- exact elapsed time;
- one recovery owner;
- no segment buckets;
- no second clock;
- no between-round special reset required; the same continuous law runs through fight time advances;
- clamp small numerical tails safely to non-negative values.

Cumulative trauma has **no recovery** in 4B1.

# Phase handling

Use the same physiology pipeline for DISTANCE, CLINCH and GROUND landed strikes.

Do not create separate durability reservoirs by phase.

Ground strikes may use a neutral context modifier initially.

Any non-neutral ground impact multiplier must be directly justified by source tracing and clearly centralized for later ablation.

# Diagnostics / observability

Add observer-visible diagnostics without making sinks causal.

At minimum record per fighter / per path:

- landed strike count entering physiology;
- impact values;
- mean/max impact;
- primary trauma deposited;
- cumulative trauma trajectory;
- current KD resistance at damaging events;
- p(KD) at damaging events;
- KD count;
- KD timestamps;
- acute vulnerability trajectory / values after KD;
- final cumulative trauma;
- final acute vulnerability;
- stamina/power modifier at impact;
- phase of impact/KD.

A compact production-style stats sink is enough; full traces may be optional.

# Required validation layers

## A. Formula/unit tests

Verify:

- full-stamina `power_multiplier=1` preserves base effective-power scale;
- higher striking_power -> higher impact distribution / deterministic comparable draw;
- lower stamina power multiplier -> lower impact for same severity draw;
- higher damage_durability -> less primary trauma for same impact;
- higher knockdown_resistance -> lower p(KD) for same impact/state;
- more cumulative trauma -> higher p(KD);
- more acute vulnerability -> higher p(KD);
- no double use of striking_power or stamina in KD formula;
- probabilities remain valid.

## B. State tests

Verify:

- miss -> no physiology mutation;
- landed strike -> exactly one primary trauma increment;
- non-KD landed strike -> no acute-vulnerability increment;
- KD -> one acute-vulnerability increment;
- cumulative trauma never decreases;
- acute vulnerability decays with exact `dt`;
- stamina mutation/order from Phase 4A still works;
- no KO/TKO finish can occur.

## C. Ordering test

Construct a deterministic landed strike and verify:

1. pre-action power modifier captured;
2. impact uses that pre-action modifier;
3. primary trauma is applied once;
4. KD probability derives from resistance after that primary trauma;
5. KD consequence occurs at same timestamp;
6. action stamina cost affects only subsequent actions.

## D. Phase regression

At neutral/no-physiology configuration, preserve Phase 3/4A action rates and phase transitions.

Damage/KD must not alter:

- strike accuracy;
- TD success;
- TD initiation ontology;
- separation/ground-exit clocks;
- judging (not implemented);
- submission attempts.

## E. Frozen-fixture diagnostics

Use at minimum:

- Lewis vs Daukaus — one-shot power / early-KD stress case;
- Holloway vs Kattar — high-volume accumulation case;
- Font vs Rosas — mixed striking/wrestling case;
- Merab vs Yan — wrestling/control case;
- Oliveira vs Poirier — mixed damage/grappling case.

Use deterministic common seeds.

Do not make winner/method claims because KO/TKO and judging are still absent.

Report:

- impacts per path;
- trauma per fighter;
- KDs per fighter/fight;
- KD timing by round;
- KDs per 100 landed strikes;
- mean/max impact;
- final trauma;
- acute vulnerability behavior;
- stamina/power interaction;
- runtime/throughput.

## F. Historical KD anchor diagnostic

This is important enough to add now if the repository has sufficient historical KD fields.

Build a non-leaky historical mechanics audit over a useful mature cohort and compare at least:

- historical knockdowns/fight;
- simulated knockdowns/fight;
- historical zero-KD fight rate vs simulated;
- historical >=1 KD fight rate vs simulated;
- KD rate by `striking_power` bucket;
- KD rate by `knockdown_resistance` bucket;
- KD rate by landed-strike volume bucket;
- round distribution of KDs if available.

Do not tune many coefficients to make all buckets match in this phase. The purpose is to expose the first calibration error cleanly.

If historical KD data are not available in the current artifact path, document the exact blocker and still complete fixture-level mechanics validation.

# Initial calibration policy

The external design review recommends a small physiology parameter set eventually. For 4B1 expose only what is needed for impact/trauma/KD:

- impact scale;
- severity dispersion;
- durability sensitivity;
- trauma-to-resistance erosion strength;
- KD midpoint;
- KD slope;
- KD acute-vulnerability increment;
- acute-vulnerability half-life.

Do not proliferate parameters.

Use source-traced legacy values where they map cleanly to this architecture. Where they do not, choose explicit provisional values and label them **initial calibration candidates**, not historical facts.

No global optimization in this task.

# Performance

Benchmark actual runtime, but do not batch evented strikes unless performance is demonstrated to be a blocker.

# Scope protection

Before finishing, prove:

- inheritance simulator unchanged;
- FSR-32 checksum unchanged;
- Phase 2B TD initiation unchanged;
- Phase 3 flow formulas unchanged;
- Phase 4A stamina formulas unchanged;
- no KO/TKO finish logic;
- no terminal SUB;
- no judging;
- no age;
- no damage recovery;
- no hidden RNG;
- engine remains sole state mutator.

# Required Codex return

Return:

1. starting and final SHAs;
2. exact files changed;
3. legacy damage/KD source trace;
4. final Phase 4B1 architecture;
5. parameter/constants table with provenance: `legacy`, `derived`, or `provisional`;
6. tests and exact counts;
7. five-fixture diagnostic table;
8. historical KD anchor audit if available;
9. runtime;
10. frozen FSR checksum;
11. explicit scope-protection statement;
12. working-tree/remote status.

End with exactly one of:

`PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: PASS`

or

`PHASE 4B1 IMPACT + TRAUMA + KNOCKDOWN GATE: FAIL`

Do not authorize or implement Phase 4B2 automatically.
