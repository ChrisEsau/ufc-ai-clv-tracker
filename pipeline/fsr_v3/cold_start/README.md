# FSR V3 Cold Start

Status: **research / validation only**. Nothing in this package is wired into the published FSR V3 snapshot or Event Clock.

## Goal

Reduce the information deficit for fighters with little or no UFC evidence without pretending external MMA data is equivalent to UFC data.

The model hierarchy is:

```text
UFC population prior
    + objective external pre-fight evidence
    -> fighter-specific cold-start prior
    + subsequent UFC observations
    -> normal FSR V3 posterior
```

If there is no usable external evidence, the cold-start layer contributes zero equivalent evidence seconds and the prior is exactly the current FSR V3 population prior.

## Hard boundaries

- No betting odds or market probabilities.
- No LLM-generated or manually assigned fighter ratings.
- No subjective scouting grades.
- No future fight results, future profile records, or current-profile fields in historical validation.
- No fuzzy name matches hidden inside the model. Ambiguous aliases must be resolved in an explicit auditable mapping.
- External evidence is never inserted as if it were UFC observation data.
- Cross-promotion Elo for this layer is computed from **non-UFC fights only**, preventing UFC results from being re-imported and double-counted.
- No production FSR or Event Clock promotion without held-out native-target improvement.

## v0.1 evidence

The first runnable source adapter is `mma_global.py`, which accepts the public longitudinal MMA fight database containing dated fight history across thousands of organizations. Only dated fight facts are consumed. From those facts the pipeline derives:

- pre-target professional record and win rate;
- KO/TKO, submission and decision win/loss mix;
- activity and layoff;
- fight duration / early-finish tendencies;
- number of organizations and major-organization experience;
- promotion-specific experience (PFL, Bellator, LFA, Cage Warriors, Rizin, ACA, KSW, Oktagon);
- leakage-safe non-UFC Elo and opponent-quality summaries;
- physical measurements when present.

The current repository does **not** persist a broad historical external-MMA observation table, so the source database is supplied at study/build time and is not committed into the repo.

Optional interfaces already exist for:

- pathway technical statistics such as DWCS / Road to UFC / PFL;
- objective wrestling, BJJ, judo and sambo pedigree.

Those sources are only enabled when a dated structured table is supplied. They are not inferred from prose.

## Prior math

For a positive FSR rate trait with current population mean `q_pop` and prior strength `K_pop` seconds, the current prior is:

```text
shape_pop = q_pop * K_pop / 900
rate_pop  = K_pop / 900
```

The external model predicts `q_ext` from objective pre-target features. Held-out calibration chooses an additional equivalent evidence strength `K_ext`.

The combined prior is:

```text
shape = (q_pop*K_pop + q_ext*K_ext) / 900
rate  = (K_pop + K_ext) / 900
mean  = (q_pop*K_pop + q_ext*K_ext) / (K_pop + K_ext)
```

`K_ext=0` exactly reproduces the current prior.

## Learning and validation

`ColdStartNB2RateModel` is a deterministic ridge-regularized NB2 model trained against the **next UFC native target**, not fight outcome or betting market.

For v0.1 the first targets are:

- standing striking tendency: next-fight distance significant-strike attempts / standing exposure;
- takedown tendency: next-fight takedown attempts / eligible takedown exposure.

Default chronological split:

- model training: 2012-01-01 through 2021-12-31;
- external-strength calibration: 2022-01-01 through 2023-12-31;
- untouched test: 2024-01-01 through 2025-12-31.

The public MMA source currently extends through 2026-01-31, so the default historical gate deliberately ends at 2025-12-31. That prevents later 2026 targets from being penalized by source staleness rather than model quality.

The test scores:

- UFC debut (`0` prior observations);
- UFC fight #2 (`1` prior observation);
- UFC fight #3 (`2` prior observations).

For fight #2/#3 the model adds the exact accumulated V3 UFC NB2 likelihood state to the cold-start prior, preserving the production same-date delayed update semantics. The validation asserts that `K_ext=0` reproduces the current V3 prefight rating before any cold-start comparison is accepted.

Primary gate: posterior-predictive native log likelihood, with posterior-mean plug-in likelihood, count MAE, evidence coverage, year splits, calibration tables and fight-cluster bootstrap as secondary diagnostics.

## Commands

Run focused tests:

```bash
pytest -q tests/fsr_v3/test_cold_start.py
```

Run the historical study with a local longitudinal MMA DuckDB:

```bash
python -m pipeline.fsr_v3.cold_start.study \
  --mma-global-duckdb /path/to/database.duckdb
```

Outputs are written to `data/diagnostics/fsr_v3_cold_start/` by default.

## Promotion rule

Do not promote because external data sounds informative. A trait/source combination is eligible only if:

1. the external model has adequate historical coverage;
2. calibrated `K_ext` is greater than zero on a pre-test calibration period;
3. held-out native posterior-predictive likelihood improves, preferably with a fight-cluster bootstrap interval excluding zero;
4. the gain is not confined to one tiny evidence bucket or one year;
5. established UFC fighters are unchanged except through normal accumulation of UFC evidence.

If these conditions fail, the correct cold-start behavior remains the current population prior plus wide uncertainty.
