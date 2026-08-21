# Event Clock MC V2

Event Clock MC V2 is the isolated FSR V3 challenger to the frozen
`pipeline/simulation/event_clock_mc_v1/` baseline.

## Non-negotiable boundary

V2 does **not** authorize a redesign or retune of Event Clock fight mechanics.
The frozen V1 calibration remains the reference for:

- path-budget and finite-time mechanics
- TD/ground hurdle mechanics
- control occurrence and ownership
- standing free-time mechanics
- submission mechanics
- stamina
- KD and KO/TKO
- decision judging
- timing and finish resolution

The initial V2 question is narrower:

> Does the validated FSR V3 fighter state improve Event Clock prediction when
> the simulator mechanics are otherwise held fixed?

## Required comparisons

A. `ECV1 + FSR V2` — frozen baseline

B. `ECV2 + FSR V3` — posterior means only

C. `ECV2 + FSR V3` — posterior means plus validated path-level epistemic draws

B - A measures the value of the new trait means/semantics.

C - B measures the value of uncertainty propagation.

## FSR V3 semantic adapter

`fsr_v3_adapter.py` owns the V3-to-runtime boundary.

### Standing

```
effective standing rate =
    attacker standing tendency
    * defender standing suppression
```

`standing_suppression < 1` means better suppression.

Landing probability:

```
sigmoid(
    logit(population standing accuracy)
    + attacker standing offense
    - defender standing defense
)
```

### Takedowns

```
effective TD rate =
    attacker TD tendency
    * defender TD suppression
```

`TD_suppression < 1` means better suppression.

Completion probability:

```
sigmoid(
    logit(population TD completion)
    + attacker TD offense
    - defender TD defense
)
```

Any age transform already present in the frozen Event Clock execution layer must
remain exactly where it currently lives; V3 does not decay persisted FSR state.

### Ground striking

V3 ground tendency is a sustained ground-and-pound slope per 15 minutes of the
attacker's own control.  It is not an ordinary all-ground-seconds event rate.

```
ground slope =
    attacker ground tendency
    * defender ground suppression

expected ground attempts =
    global burst
    + own_control_seconds / 900 * ground slope
```

The defender multiplier applies only to the slope.  It never scales the global
burst.

Ground landing probability is attacker-only:

```
sigmoid(
    logit(population ground accuracy)
    + attacker ground offense
)
```

There is intentionally no `ground_striking_defense` in FSR V3.

## Epistemic uncertainty

Path-level epistemic sampling is enabled only for:

- `takedown_tendency`
- `takedown_suppression`
- `standing_striking_tendency`
- `standing_striking_suppression`

One latent draw is made per fighter at path initialization and then remains
fixed throughout the path.

Every other rebuilt V3 trait uses its posterior mean deterministically.

NB2 `alpha` and Beta-Binomial `rho` describe fight/observation noise.  They are
not fighter epistemic variance and are not sampled by this adapter.

The V3 uncertainty parquet currently publishes posterior mean and SD rather than
full posterior weights.  ECV2 therefore uses a moment-matched Gamma projection
for the four positive sampled traits:

```
shape = (mean / sd)^2
scale = sd^2 / mean
```

This preserves positivity, posterior mean, and posterior SD.  The means-only
comparison bypasses this projection entirely.

## Current implementation stage

Implemented:

- canonical FSR V3 prefight/latest loaders
- historical exact fight/fighter lookup
- uncertainty lookup
- immutable per-path fighter trait state
- means-only and epistemic modes
- standing/TD multiplicative suppression transforms
- standing/TD paired success transforms
- ground burst + suppressed-slope transform
- attacker-only ground landing transform
- deterministic adapter tests

Not yet implemented at this checkpoint:

- a copied/frozen V1 simulation runner under the V2 namespace
- V3 refit of V1 direct-output bundle components
- full historical A/B/C replay

Those pieces require auditing the exact frozen V1 module interfaces before
copying or wrapping them.  Do not guess those interfaces or silently feed V3
semantic values into a V1 bundle fitted on V2 feature scales.
