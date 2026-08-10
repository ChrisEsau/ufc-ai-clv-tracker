# FSR KO/TKO V3 Hybrid Shadow Architecture

Status: **shadow / research only**

This candidate preserves prior simulator baselines and does not change stored FSR history.

## Structural diagnosis

The reservoir-exhaustion V2 architecture produced the wrong finish-time shape in the mature 2020+ population audit:

- actual total KO/TKO: 31.44%
- simulated total KO/TKO: 44.42%
- actual R1 KO/TKO: 14.06%
- simulated R1 KO/TKO: 8.06%
- actual mean KO round: 1.835
- simulated mean KO round: 2.313

That pattern is consistent with too much reliance on one-way cumulative reservoir depletion: too few acute early stoppages and too many eventual late stoppages.

## V3 design decision

Keep the reservoir as the single persistent accumulated-damage state, but stop requiring reservoir exhaustion for every KO/TKO.

Dynamic state remains intentionally small:

- damage reservoir: persistent accumulated punishment
- recent KD: short-lived vulnerability window

There is no additional consciousness meter, hurt meter, TKO meter, or hidden health state.

## Finish routes

### 1. Acute KO

A landed strike first uses the existing locked Damage V1 severity model and locked KD probability model.

If the strike causes a confirmed KD, V3 evaluates a conditional immediate-KO hazard based on:

- KD-causing strike shock / reservoir capacity
- current accumulated reservoir depletion

Because the hazard is conditional on a confirmed KD, KD resistance is not applied a second time inside the acute-KO equation.

An acute KO can occur while substantial reservoir remains.

### 2. Post-KD TKO

A landed strike can trigger the TKO hazard only if `recent_kd` was already active before that strike.

Inputs are:

- follow-up strike shock / reservoir capacity
- current reservoir depletion
- defender `recovery_ability`
- phase context
  - ground top position receives the largest provisional bonus
  - clinch receives a smaller provisional bonus

The KD-causing strike itself is not treated as a follow-up strike.

### 3. Cumulative exhaustion

If reservoir reaches zero, the fight still ends by KO/TKO as an absolute terminal safeguard.

This route is retained primarily to represent extreme accumulated punishment. It is no longer intended to be the dominant universal finish mechanism.

## Removed from the V3 candidate

V3 does **not** use:

- KD-collapse bonus reservoir damage
- the 2x post-KD follow-up strike-damage multiplier

Those mechanisms remain preserved in the earlier shadow engines for comparison.

## Between-round recovery

V3 uses the existing provisional `recovery_ability` curve:

- rating 10 -> about 5% of missing reservoir restored
- rating 50 -> about 20%
- rating 90 -> about 35%

Recovery occurs after completed non-final rounds only.

The recent-KD timer is cleared during the one-minute corner break because the current recent-KD state represents only a 30-second vulnerability window.

No KD-based recovery suppression is implemented yet.

## Locked mechanics preserved

V3 preserves:

- Damage V1 strike-severity distribution
- `striking_power` upper-tail behavior
- locked KD probability coefficients
- `knockdown_resistance`
- `damage_durability`
- locked age adjustment on KD resistance and durability only
- phase and transition mechanics inherited from the current static simulator

## Provisional V3 hazard constants

The acute-KO and post-KD-TKO coefficients are architecture-enabling first-pass values only.

They are **not calibration locks**.

Do not tune individual coefficients before the first full population diagnostic establishes:

- total KO/TKO rate
- R1/R2/R3 KO/TKO rates
- mean finish round
- acute-KO share
- post-KD-TKO share
- cumulative-exhaustion share
- reservoir remaining at finish
- Any-KO and R1-KO AUC/Brier/log loss
- KO winner direction
- age-band calibration

## First validation

Run:

```bash
PYTHONPATH=. pytest -q tests/experimental/test_fsr_static_mc_ko_tko_v3_hybrid.py
```

Then:

```bash
PYTHONPATH=. python scripts/experimental/fsr_mature_2020plus_mc_10path_hybrid_v3_audit.py
```

The population audit uses the same mature 2020+ cohort and seed convention as the earlier 10-path audits and prints progress approximately every 1,000 simulated paths.

## Interpretation target

The desired structural movement is not merely lower total KO/TKO frequency.

V3 should ideally:

- increase R1 KO/TKO toward the historical rate through acute finishes
- reduce excessive R2/R3 cumulative finishes through recovery and removal of KD collapse
- reduce the share of finishes caused only by reservoir exhaustion
- preserve or improve occurrence discrimination and KO winner direction

If total rates improve but finish-route composition is implausible, do not lock the coefficients. Diagnose the route mix first.
