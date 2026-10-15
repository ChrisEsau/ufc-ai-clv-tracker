# FSR V3

FSR V3 is a parallel, non-destructive successor to FSR V2.

## Frozen baselines

The following must not be modified by V3 work:

- `pipeline/fsr_v2/`
- `data/fsr_v2/`
- `pipeline/simulation/event_clock_mc_v1/`

FSR V3 publishes only under `data/fsr_v3/`.

## Statistical rule

FSR V3 separates two kinds of variance:

- NB2 `alpha` / Beta-Binomial `rho` = aleatoric observation/fight noise;
  these are never sampled as fighter uncertainty.
- posterior uncertainty in the latent fighter trait = epistemic uncertainty;
  it is propagated only where chronological predictive validation supported it.

All historical states use strictly prior-date information. Same-date fights use
the same prefight state and update only after the entire date batch.

## Validated V3 replacements

### Takedown tendency

`Y ~ NB2(E/900 * q_fighter, alpha)`

- native exposure: `td_tendency_exposure_seconds = round_elapsed - opponent_CTRL`
- population rate from prior-date data
- Gamma shrinkage: `K=468.48` seconds
- study population reference: about `5.1087 TD attempts / 15m opportunity`
- NB2 observation dispersion reference: `alpha=.2432`
- posterior mean is the FSR value
- epistemic posterior uncertainty is enabled (`c=1`)

### Takedown suppression

`expected = attacker_prefight_q * exposure / 900`

`Y ~ NB2(expected * s_defender, alpha)`

- `s < 1` means the defender suppresses opponent TD generation
- Gamma population heterogeneity shape `8.5281`
- study population multiplier about `.9885`
- NB2 observation dispersion reference `alpha=.4574`
- posterior mean is the FSR value
- epistemic posterior uncertainty is enabled (`c=1`)

### Takedown effectiveness

`logit P(TD success) = beta_population + O_attacker - D_defender`

- Beta-Binomial `rho=.12`
- offense prior sigma `.35`
- defense prior sigma `.50`
- posterior means are the FSR values
- epistemic sampling disabled (`c=0`)

### Standing striking tendency

`Y ~ NB2(E/900 * q_fighter, alpha)`

- standing striking is DISTANCE significant striking only
- native exposure: modeled standing exposure
- population rate from prior-date data
- Gamma shrinkage `K=87.78` seconds
- study population reference about `169.527 attempts / 15m standing`
- NB2 observation dispersion reference `alpha=.0824`
- posterior mean is the FSR value
- epistemic posterior uncertainty is enabled (`c=1`)

### Standing striking suppression

`expected = attacker_prefight_q * standing_exposure / 900`

`Y ~ NB2(expected * s_defender, alpha)`

- `s < 1` suppresses opponent distance-strike generation
- Gamma population heterogeneity shape `28.7138`
- study population multiplier about `1.0495`
- NB2 observation dispersion reference `alpha=.0863`
- posterior mean is the FSR value
- epistemic posterior uncertainty is enabled (`c=1`)

### Standing striking effectiveness

`logit P(land) = beta_population + O_attacker - D_defender`

- distance landed / distance attempted
- Beta-Binomial `rho=.035`
- offense prior sigma `.30`
- defense prior sigma `.30`
- posterior means are the FSR values
- epistemic sampling disabled (`c=0`)

### Ground striking tendency

`Y ~ NB2(burst + own_CTRL/900 * q_fighter, alpha)`

- own UFCStats control seconds define opportunity
- global burst is separate from the fighter rating
- Gamma shrinkage `K=90` own-control seconds
- posterior mean is the FSR value
- epistemic sampling disabled (`c=0`)

### Ground striking suppression

`mu = burst + s_defender * (own_CTRL/900 * q_attacker)`

- suppression acts only on the attacker slope, never on the global burst
- Gamma prior shape `2`
- posterior mean is the FSR value
- epistemic sampling disabled (`c=0`)

### Ground striking effectiveness

`logit P(land) = beta_population + O_attacker`

- Beta-Binomial `rho=.08`
- attacker prior `O ~ Normal(0, .25^2)`
- no defender ground-effectiveness trait
- posterior mean is the FSR value
- epistemic sampling disabled (`c=0`)

### Striking power

- attacker-only KD-production latent effect
- Beta-Binomial `rho=.01`
- attacker prior `Normal(0, .50^2)`
- no KO-win bonus and no persisted age adjustment
- posterior mean only (`c=0`)
- Event Clock V2 translates the native latent into the frozen KD/KO coordinate
  without refitting the frozen consequence mechanics

## Paired-effect replay note

The standing and takedown paired families were selected by chronological
Beta-Binomial predictive tests.  V3 generates per-fight prefight states with a
leakage-safe conditional posterior filter using the same likelihood, priors,
signs, and rolling population intercept.  Because `c=0` for these families,
only posterior means are simulator-facing.  Their V3 snapshot output should be
historically audited before Event Clock V2 integration.

## Event Clock V2 consumer status for inherited V2 fields

Do not spend V3 prior/variance research effort on an inherited field merely
because the frozen V2 compatibility object still requires it.  A field is
"active" below only when changing its value can currently affect Event Clock V2
prediction probabilities through direct inference or detailed path mechanics.

### Active inherited fields — still require V3 prior/variance validation

- `escape_offense`
- `escape_defense`
- `escape_population_mean_seconds`
  - explicitly form `retention_mean_base` and `control_pressure` in the V3
    feature builder and are also present in the fitted direct-feature matrix
- `submission_tendency`
- `submission_suppression`
  - directly determine the submission attempt clock rate
- `submission_offense`
- `submission_defense`
  - remain inputs to the fitted V3 direct-feature matrix; they do not currently
    enter the explicit submission-conversion formula, but are prediction-active
    through the fitted direct models
- `submission_conversion_baseline`
  - directly determines submission conversion probability after the frozen
    conversion offset
- `knockdown_resistance`
  - directly enters the frozen shadow KD hazard as the defender KD-resistance
    term; it does not enter the KO hazard

These are the remaining inherited families that should receive native-target
prior-strength and between-fighter/posterior-variance validation before the FSR
V3 uncertainty contract is considered complete.

### Obsolete for current Event Clock V2 prediction

The following fields are retained temporarily only because the frozen V2
profile/schema boundary still expects them.  They are prediction-inert in the
current Event Clock V2 path and should not receive new V3 prior/variance work
unless a future simulator mechanic explicitly reintroduces a consumer.

- `head_strike_tendency`
- `body_strike_tendency`
- `leg_strike_tendency`
  - Event Clock V2 predicts standing/ground strike budgets without target
    location and the detailed path never samples head/body/leg targets
- `damage_durability`
  - the active shadow KO/KD model explicitly does not use damage durability or
    cumulative trauma
- `stamina_depletion_resistance`
  - the frozen stamina object updates an internal stamina state, but current
    Event Clock V2 budgets are already fixed upstream; dynamic output/power
    modifiers are not applied to those budgets, and shadow KD/KO stamina
    coefficients are zero, so this field cannot change current outcomes
- `stamina_performance_resilience`
  - it only changes dynamic modifiers whose result is not consumed by the
    current detailed Event Clock path; therefore it cannot change prediction
    probabilities
- `stamina_capacity`
  - fixed at `100.0`; not a learned fighter trait and not a prior/variance
    research target
- `reversal_tendency`
  - experimental V2 trait with no Event Clock V2 prediction consumer

`ground_striking_defense` is not obsolete compatibility state: it was rejected
by V3 validation and is intentionally removed from V3 publication.

Do not physically delete the obsolete compatibility columns yet.  The frozen
`FSRV2FighterInput` constructor still requires several of them when Event Clock
V2 creates the parent mechanics-profile object.  Removal belongs to a later
interface-cleanup change after the frozen mechanics boundary is replaced.

Age transforms remain matchup/simulator translations. They do not rewrite or
decay persisted FSR ratings. Freshness/layoff effects are not part of this V3
build.

## Publication contract

V3 starts from frozen V2 prefight/latest publication and replaces only the
validated families above. `ground_striking_defense` is removed because the
robustness study rejected a persistent defender ground-accuracy effect.

Outputs:

- `data/fsr_v3/fsr_v3_prefight_snapshots.parquet`
- `data/fsr_v3/fsr_v3_latest.parquet`
- `data/fsr_v3/fsr_v3_prefight_uncertainty.parquet`
- `data/fsr_v3/history/*.parquet`

The uncertainty table explicitly publishes posterior mean, posterior SD,
variance multiplier, sampling flag, and posterior-family metadata.

## Build

Build the canonical V3 overlay with all validated replacements:

```bash
python -m pipeline.fsr_v3.build --all
```

Individual history families can also be rebuilt without publication:

```bash
python -m pipeline.fsr_v3.build --takedowns --no-publish
python -m pipeline.fsr_v3.build --standing-striking --no-publish
python -m pipeline.fsr_v3.build --ground-striking --no-publish
```
