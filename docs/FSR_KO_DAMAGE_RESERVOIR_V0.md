# FSR KO / Damage Reservoir V0 — Design Checkpoint

Status: research/design lock only. No finish mechanics are promoted to the simulator by this document.

## Core damage-reservoir model

Each fighter has an individual damage-reservoir capacity. Reservoir size is fighter-specific rather than a universal fixed health pool.

At fight start:

- `reservoir_current = reservoir_capacity`
- landed significant strikes deplete the reservoir
- the depletion caused by one strike is a function of strike severity and attacker power versus defender resistance
- high-power fighters should have severity distributions with fatter damaging tails rather than every landed strike behaving like a bomb

## Dynamic resistance

Damage resistance is not static during the fight.

As the reservoir empties, effective damage resistance decreases. Therefore identical incoming strikes can cause larger reservoir losses later in a damaging sequence than they would against a fresh fighter.

Conceptually:

`effective_damage_resistance = base_damage_resistance × reservoir_condition_modifier × post_KD_modifier`

The exact functional form is not yet locked and must be researched/calibrated.

## Knockdowns

Knockdowns are driven by sudden reservoir depletion rather than a simple fixed reservoir threshold.

The key acute quantity is the strike-level reservoir delta normalized by fighter-specific capacity:

`shock_fraction = reservoir_delta / reservoir_capacity`

A large shock fraction can create a knockdown even when substantial reservoir remains. A depleted fighter may be knocked down by a smaller shock because effective resistance has already deteriorated.

Knockdown probability should therefore depend on at least:

- normalized strike-level reservoir delta;
- current reservoir fraction / vulnerability;
- fighter-specific resistance state.

## Post-knockdown vulnerability

A knockdown causes a sharp temporary reduction in effective damage resistance.

This is a central mechanism, not a separate permanent fighter trait:

`KD → resistance collapse → follow-up strikes cause larger reservoir deltas → elevated KO/TKO probability`

The size and duration of the post-KD penalty are not yet locked and must be backed by historical evidence where possible.

## KO/TKO pathways

The reservoir model should naturally support both:

1. **Catastrophic shock:** one unusually large strike-level reservoir delta can cause an immediate KO or KD even when the fighter was relatively fresh.
2. **Accumulation:** repeated reservoir losses reduce effective resistance until ordinary follow-up strikes become increasingly damaging and the fighter reaches a near-exhausted state associated with KO/TKO risk.

A knockdown should not automatically add arbitrary extra reservoir damage. Instead, it should alter resistance and fight context so subsequent landed strikes become more damaging.

## Recovery

Reservoir can refill during the fight. Recovery rate will eventually depend on fighter recovery traits and context. Exact recovery mechanics are deferred until the dynamic-state design is locked.

## Trait simplification goal

The reservoir architecture is intended to avoid unnecessary parallel hidden states such as separate acute-stress, consciousness, hurt, or finish-danger meters unless historical evidence later demonstrates they are necessary.

The target is a small number of path-state variables with many derived effects.

## Historical validation — next five studies

Before implementing the finish engine, research these five questions:

1. Does prior-round damaging exposure predict increased next-round knockdown susceptibility?
2. Does absorbing a knockdown strongly increase KO/TKO loss probability?
3. How does KO/TKO loss probability change from zero to one to multiple knockdowns absorbed?
4. Do knockdown rounds show elevated significant-strike, head-strike, and ground-strike absorption versus comparable non-KD rounds?
5. Does the existing historical `damage_resistance` signal identify fighters who survive damaging exposure and/or knockdowns better than peers?

### Data limitation

UFCStats round data does not identify exact within-round strike ordering. Therefore round-level studies can support associations such as `KD round → elevated damaging exposure / finish risk`, but cannot by themselves prove that specific strikes occurred after the knockdown. Research outputs must preserve this limitation explicitly.

## Implementation boundary

No finish mechanics should be implemented or promoted from this document until the five historical validation studies are completed and reviewed.
