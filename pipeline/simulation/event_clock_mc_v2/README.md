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

The frozen Event Clock TD age translation remains in the matchup feature layer.
FSR V3 itself does not decay persisted fighter state with age.

### Ground striking

V3 ground tendency is a sustained ground-and-pound slope per 15 minutes of the
attacker's own control. It is not an ordinary all-ground-seconds event rate.

```
ground slope =
    attacker ground tendency
    * defender ground suppression

expected ground attempts =
    global burst
    + own_control_seconds / 900 * ground slope
```

The defender multiplier applies only to the slope. It never scales the global
burst.

Ground landing probability is attacker-only:

```
sigmoid(
    logit(population ground accuracy)
    + attacker ground offense
)
```

There is intentionally no `ground_striking_defense` in canonical FSR V3.

## Epistemic uncertainty

Path-level epistemic sampling is enabled only for:

- `takedown_tendency`
- `takedown_suppression`
- `standing_striking_tendency`
- `standing_striking_suppression`

One latent draw is made per fighter at path initialization and remains fixed for
that entire path. Every other rebuilt V3 trait uses its posterior mean.

NB2 `alpha` and Beta-Binomial `rho` describe fight/observation noise. They are
not fighter epistemic variance and are not sampled here.

The V3 uncertainty parquet currently publishes posterior mean and SD rather than
full posterior weights. ECV2 therefore uses a moment-matched Gamma projection
for the four positive sampled traits:

```
shape = (mean / sd)^2
scale = sd^2 / mean
```

This preserves positivity, posterior mean, and posterior SD. The means-only arm
bypasses this projection entirely.

## Frozen-mechanics inheritance

The V2 bundle is deliberately parented from the persisted frozen V1 bundle:

`data/models/event_clock_mc_v1/event_clock_frozen_bundle.joblib`

`fit_event_clock_bundle.py` deep-copies that context and replaces only:

1. FSR V2 canonical snapshots with FSR V3 snapshots;
2. V1 direct inference models with models refit on the V3 feature semantics.

It does **not** refit the Stage-9 path mechanics, control calibration, judge,
submission scalar/offset, stamina, KD, or KO/TKO systems.

The V3 direct model uses the same V1 model classes, ridge constants, hurdle
architecture, control architecture, and standing-free-time machinery. The
models must be refit because V3 tendency/suppression values have different
semantics and scales from V2. A V1 direct-inference bundle must never be used
with V3 features.

## Variance-arm random-number rule

In `--epistemic validated` mode, uncertainty is sampled with a dedicated RNG.
The Stage-9 budget RNG and detailed-fight RNG retain the same seeds used by the
means-only arm. This gives the B/C comparison common random numbers without
letting epistemic draws merely shift the mechanics RNG stream.

## Legacy profile-parser compatibility

The frozen V1 detailed path still constructs its physical/stamina profiles via
the legacy FSR V2 fight parser, whose schema expects `ground_striking_defense`.
ECV2 supplies a temporary neutral zero field only to that parser copy.

That field is **not** added back to canonical FSR V3 and is never used to
calculate V3 ground accuracy or path budgets. Ground accuracy has already been
resolved by the V3 attacker-only feature/inference layer before the frozen
mechanics run.

## Implemented files

- `fsr_v3_adapter.py` — canonical V3 loading, semantics, uncertainty draws
- `feature_builder.py` — V3 direct-model features and sampled path features
- `inference.py` — V3-fitted version of the frozen direct inference layer
- `fit_event_clock_bundle.py` — frozen V1 parent + V3 direct-model bundle
- `run_event_or_fight_frozen.py` — means-only / validated-epistemic runner

Tests cover:

- allowed epistemic traits only
- positivity and deterministic seeded sampling
- multiplicative standing/TD suppression
- ground burst + suppressed slope
- attacker-only ground effectiveness
- paired offense/defense directions
- propagation of sampled latent traits into direct-model features

## Build the V2 bundle

FSR V3 must already be published and the frozen V1 bundle must exist.

```bash
PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle
```

Outputs:

```text
data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle.joblib
data/models/event_clock_mc_v2/event_clock_v2_fsr_v3_bundle_manifest.json
```

## Run a historical event or fight

Means-only B arm:

```bash
PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen \
  --fight-id <FIGHT_ID> --paths 2000 --epistemic off
```

Validated epistemic C arm:

```bash
PYTHONPATH=. python -m pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen \
  --fight-id <FIGHT_ID> --paths 2000 --epistemic validated
```

The same event/date/fighter selectors supported by the frozen V1 runner are
also supported.

## Current validation gate

Implementation is complete enough for bundle build and seeded smoke testing.
It is **not** promoted or calibrated yet.

Next required sequence:

1. build the V2 bundle;
2. run a seeded means-only smoke fight;
3. run the same fight with validated epistemic sampling;
4. compare path distributions and verify no mechanical regressions;
5. build the historical A/B/C replay and compare probability quality.

Do not tune fight mechanics during this gate.
