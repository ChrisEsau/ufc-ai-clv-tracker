# FSR MC Age Adjustment Lock V1

Status: locked for the shadow KO/TKO Monte Carlo on `feature/fsr-18-shadow`.

## Purpose

Translate leakage-safe historical FSR evidence into current-age effective resistance for the KO/TKO Monte Carlo without rewriting stored FSR ratings.

## Locked curve

For a supplied fighter age:

```text
age_penalty = 2.0 * max(age - 30.0, 0.0)
effective_trait = clip(raw_trait - age_penalty, 10.0, 90.0)
```

## Traits affected

Only these traits receive the age adjustment:

- `knockdown_resistance`
- `damage_durability`

No other FSR trait is age-adjusted by this lock. In particular, the mechanic does not modify:

- `striking_power`
- striking pressure, precision, or defense
- wrestling or control traits
- submission traits
- fatigue, resilience, recovery, or adversity traits

Those families require their own empirical aging studies before any age mechanic may be applied.

## Simulator interpretation

- Raw/pre-fight FSR values remain the historical evidence scores.
- The Monte Carlo creates working profile copies.
- The age curve is applied only to the two validated physiological resistance traits in those working copies.
- `damage_durability` therefore changes effective reservoir capacity through the existing reservoir mapping.
- `knockdown_resistance` therefore changes acute KD susceptibility through the existing KD probability equation.
- The stored FSR artifact is never mutated.

## Evidence used for the lock

The mature 2020+ studies showed that age added predictive value for both KD absorption and KO/TKO loss after controlling for opponent power and realized exposure. Candidate curve search selected `linear_on30_s2` as the strongest simple effective-trait translation for both traits. In the strong KD-collapse R1-KO replay, this candidate improved winner-direction hit rate from 39.1% to 44.6%, produced 14 wrong-to-right flips versus 2 right-to-wrong flips, and improved direction hit rate from 31.1% to 51.1% when the actual loser was age 37+.

A matched-control replay also improved R1-KO/control discrimination modestly (AUC 0.5762 to 0.5820; separation 0.0156 to 0.0201), supporting retention of the mechanic while showing that age is more useful for identifying the vulnerable fighter than for determining whether a KO occurs at all.

## Code contract

The lock is implemented in:

`scripts/experimental/fsr_static_mc_ko_tko_v2.py`

Constructor inputs:

```python
StaticFSRMCKOTKOV2(
    red,
    blue,
    red_age=<age on fight date>,
    blue_age=<age on fight date>,
    ...,
)
```

The strong KD-collapse subclass inherits this behavior through the KO/TKO V2 constructor.

If an age is not supplied for a fighter, no age adjustment is applied to that fighter. Callers responsible for historical or live matchup simulation must therefore supply age on the target fight date when it is available.
