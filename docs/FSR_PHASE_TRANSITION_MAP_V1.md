# FSR Phase Transition Map V1

Status: research contract only; not yet a simulator probability contract.

## Purpose

Define the allowed 30-second Monte Carlo phase transitions and map each transition to the existing FSR-26 traits that should drive the transition. This document deliberately separates:

1. phase/action choice,
2. action execution/success,
3. within-phase offense,
4. later dynamic-state modifiers.

No new FSR traits are introduced here.

## Core rule

A fighter first chooses an action/transition from the actions available in the current phase. If that action requires execution (for example a takedown), a separate skill matchup determines whether it succeeds. The resulting phase then uses its own pressure/precision/defense traits to generate offense.

Static pre-fight phase-choice utilities are derived from FSR-26. Fatigue, damage, adversity, score urgency, and other path-specific dynamic modifiers are intentionally deferred.

## State map

### DISTANCE

Allowed actions:

- stay at distance
- enter clinch
- attempt takedown

#### Distance -> Distance

Primary choice signal:

- `distance_striking_pressure`

Relative competitors:

- `clinch_striking_pressure`
- `wrestling_entry`

Optional modest matchup adjustment:

- opponent `td_defense`
- opponent `control_resistance`

Research support:

- Phase-preference matchup-adjusted distance score vs realized distance attempt share: Spearman 0.2187.
- Q1 mean realized distance attempt share: 0.7397.
- Q7 mean realized distance attempt share: 0.8888.

Interpretation:

This is evidence for relative distance preference, not a literal per-segment stay-distance probability.

#### Distance -> Clinch

Primary choice signal:

- `clinch_striking_pressure`

Relative competitors:

- `distance_striking_pressure`
- `wrestling_entry`

Research support:

- Pressure-only clinch preference vs realized clinch attempt share: Spearman 0.1813.
- Q1 mean realized clinch attempt share: 0.0513.
- Q7 mean realized clinch attempt share: 0.1237.

General control adjustments did not improve this signal, so V1 should keep clinch-entry desire primarily intrinsic.

Important limitation:

UFCStats does not directly record clinch entries or exact clinch duration. This transition remains latent and must later be calibrated against aggregate phase mix.

#### Distance -> Takedown Attempt

Primary choice signal:

- `wrestling_entry`

Secondary choice signal:

- modest contribution from `control_imposition`

Research support:

- Control-blend wrestling preference vs realized TD attempts/round: Spearman 0.2673.
- Q1 mean TD attempts/round: 0.6077.
- Q7 mean TD attempts/round: 1.7961.

Do not heavily suppress attempt desire with opponent `td_defense`. Takedown desire and takedown success are separate stages.

Execution after choice:

- `wrestling_conversion` vs opponent `td_defense`

Success:

- transition toward ground/control

Failure:

- later calibration chooses return to distance vs continuation into clinch

## CLINCH

Allowed actions:

- remain in clinch
- separate to distance
- attempt takedown

### Clinch -> Clinch

Primary persistence signal:

- `clinch_striking_pressure`

Secondary positional signal:

- `control_imposition` vs opponent `control_resistance`

Evidence status:

- `clinch_striking_pressure` strongly predicts realized clinch striking volume.
- `control_imposition - control_resistance` predicts general sustained positional control, but does not predict clinch striking ownership.
- Exact clinch persistence is not directly observed by UFCStats.

Therefore V1 may use these ratings as a latent persistence utility, but the numerical hazard is not yet locked.

### Clinch -> Distance

Primary separation signal:

- inverse of own clinch persistence utility
- opponent `control_resistance` relative to own `control_imposition`

Evidence status:

Latent. UFCStats does not directly record clinch separation events or clinch duration. No dedicated separation rating is justified yet.

### Clinch -> Takedown Attempt

Primary choice signal:

- `wrestling_entry`

Secondary choice signal:

- modest `control_imposition`

Execution after choice:

- `wrestling_conversion` vs opponent `td_defense`

Important limitation:

UFCStats does not identify whether a historical takedown attempt originated from distance or clinch. V1 therefore uses the same wrestling-entry trait in both standing contexts.

## GROUND / CONTROL

Allowed transitions:

- remain controlled / remain ground
- control break / escape toward standing
- reversal

Ground striking and submission choices occur within the ground state and do not themselves require a phase transition.

### Ground -> Ground / Continued Control

Primary persistence matchup:

- `control_imposition` vs opponent `control_resistance`

Historical support:

Control edge is strongly monotonic with realized control seconds/round. Existing control ratings are therefore suitable for sustained-control opportunity.

Within-phase offense while ground persists:

- `ground_striking_pressure`
- `ground_striking_precision` vs opponent `ground_striking_defense`
- `submission_pressure`
- `submission_conversion` vs opponent `submission_resistance`

### Ground -> Escape / Control Break

Primary signal:

- bottom fighter `control_resistance` minus top fighter `control_imposition`

Historical support:

Higher resistance edge corresponds to sharply lower opponent control seconds/round and much higher probability of keeping opponent control below 30 seconds/round.

Important limitation:

Exact historical get-ups/escapes are not directly observed. This is an inferred escape hazard calibrated from control suppression.

### Ground -> Reversal

Primary signal:

- bottom fighter `reversal_ability` vs top fighter `control_imposition`

Historical support:

Reversal-edge buckets show monotonic realized reversal occurrence from about 8.7% in the lowest bucket to 18.8% in the highest bucket among opponent-control exposures.

A successful reversal changes positional ownership while remaining in the ground/control family unless later transition logic moves the fight elsewhere.

## Within-phase striking contracts

These do not choose the phase. They resolve offense once the phase is active.

### Distance striking

- attempt volume: `distance_striking_pressure`
- landing: `distance_striking_precision` vs opponent `distance_striking_defense`
- damage: `striking_power` vs opponent `chin_resistance` / `damage_resistance`

### Clinch striking

- attempt volume: `clinch_striking_pressure`
- landing: `clinch_striking_precision` vs opponent `clinch_striking_defense`
- damage: `striking_power` vs opponent `chin_resistance` / `damage_resistance`

### Ground striking

- attempt volume: `ground_striking_pressure`
- landing: `ground_striking_precision` vs opponent `ground_striking_defense`
- damage: `striking_power` vs opponent `chin_resistance` / `damage_resistance`

## Submission contract

Within ground/control opportunity:

- attack propensity: `submission_pressure`
- finish probability after attack: `submission_conversion` vs opponent `submission_resistance`

## Static phase-choice research conclusion

FSR-26 already contains enough information to derive useful static phase preferences without adding dedicated phase-preference Elo ratings.

Observed research results:

- distance preference: best candidate Spearman 0.2187 vs realized distance attempt share
- clinch preference: simple pressure-relative candidate Spearman 0.1813 vs realized clinch attempt share
- wrestling preference: control-blend candidate Spearman 0.2673 vs realized TD attempts/round
- wrestling/control matchup candidate: Spearman 0.2467 vs realized control seconds/round

These support using state-specific utilities rather than one universal phase-preference rating.

## Not yet locked

The following remain research/calibration tasks:

- utility-to-probability transform
- softmax/logit temperature
- absolute per-30-second transition hazards
- distance-vs-clinch destination after failed takedown
- clinch separation hazard
- exact control-break timing
- dynamic fatigue modifiers
- dynamic damage/adversity modifiers
- score/round/time urgency modifiers

Historical aggregate phase shares must not be treated as literal transition probabilities.

## Dynamic-state boundary

The static transition policy should be validated first. Later the dynamic engine will modify the static utilities on each Monte Carlo path using path-specific state, for example:

- fatigue
- accumulated damage
- acute adversity
- recovery
- score urgency
- remaining time

The base FSR profile remains fixed for a simulated fight; dynamic state changes independently on each Monte Carlo path.
