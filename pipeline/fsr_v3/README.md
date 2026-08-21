# FSR V3

FSR V3 is a parallel, non-destructive successor to FSR V2.

## Frozen baselines

The following must not be modified by V3 work:

- `pipeline/fsr_v2/`
- `data/fsr_v2/`
- `pipeline/simulation/event_clock_mc_v1/`

FSR V3 publishes only under `data/fsr_v3/`.

## Current implementation checkpoint

Implemented and historically validated:

### Ground striking tendency

`Y ~ NB2(burst + own_CTRL/900 * q_fighter, alpha)`

- own UFCStats control seconds define exposure
- global burst is separate from the fighter rating
- Gamma shrinkage equivalent to `K=90` own-control seconds
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

- Beta-Binomial `rho=0.08`
- attacker prior `O ~ Normal(0, 0.25^2)`
- no defender ground-effectiveness trait
- posterior mean is the FSR value
- epistemic sampling disabled (`c=0`)

## Publication contract

V3 starts from the frozen V2 prefight/latest publication and replaces only the
validated V3 fields.  The rejected V2 `ground_striking_defense` field is
removed from the V3 snapshot.

Outputs:

- `data/fsr_v3/fsr_v3_prefight_snapshots.parquet`
- `data/fsr_v3/fsr_v3_latest.parquet`
- `data/fsr_v3/fsr_v3_prefight_uncertainty.parquet`
- `data/fsr_v3/history/*.parquet`

The uncertainty table keeps epistemic uncertainty explicit.  For the ground
family, posterior SD is retained for audit but `variance_multiplier=0` and
`sampling_enabled=False`.

## Build

```bash
python -m pipeline.fsr_v3.build --ground-striking
```

or, at this checkpoint:

```bash
python -m pipeline.fsr_v3.build --all
```

`--all` currently means all V3 families that have actually been implemented,
which is intentionally only the completed ground-striking family.  TD and
standing V3 implementations are not approximated here; they should be ported
from their validated research math in a later checkpoint.
