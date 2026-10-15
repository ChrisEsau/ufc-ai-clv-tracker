# RFS Monte Carlo V1 Architecture Specification

## Document Status

- Project: UFC AI CLV Tracker
- System: RFS Monte Carlo V1
- Branch: `feature/rfs-mc-v1`
- Status: Phase 0 architecture specification
- Runtime status: Shadow-only
- Production integration: Prohibited until explicit promotion approval
- Existing Monte Carlo engine: Preserved as a frozen comparison baseline

## 1. Purpose

RFS Monte Carlo V1 is a new fight-state simulation engine built from leakage-safe
Round Fighter State profiles.

The simulator will replay a matchup thousands of times while allowing each
simulation path to develop differently. Outcomes should emerge from the
interaction between:

1. each fighter's historical tendencies
2. the opponent matchup
3. events generated during the simulated fight
4. evolving simulated fatigue, damage, defense, and positional state

The simulator will ultimately model:

- striking pace and efficiency
- opponent suppression
- cardio and pace decay
- defensive deterioration
- takedown pressure
- control conversion
- ground offense
- submission pressure
- damage accumulation
- chin risk
- durability and recovery
- KO/TKO and submission risk
- round scoring and decision outcomes

## 2. Core Architecture

```text
Historical UFCStats round observations
        ↓
Leakage-safe Round Fighter State
        ↓
Pre-fight fighter simulation profiles
        ↓
Opponent matchup interaction
        ↓
30-second simulated fight segments
        ↓
Striking, wrestling, control, and submission events
        ↓
Dynamic fatigue, damage, defense, and positional state
        ↓
Competing KO/TKO, submission, and no-finish hazards
        ↓
Round scoring or terminal outcome
        ↓
Thousands of Monte Carlo paths
        ↓
Outcome and statistical distributions
```

## 3. New Engine Boundary

The new engine must be implemented separately under:

```text
pipeline/simulation/rfs_mc_v1/
```

The existing simulator must not be rewritten in place.

The existing simulator remains:

- a historical comparison baseline
- a regression reference
- a research fallback
- evidence of previously tested mechanics

No existing production prediction, feature, model, betting, or artifact path may
be changed during the initial RFS Monte Carlo V1 build.

## 4. Simulation Resolution

The simulator will use ten 30-second segments per standard five-minute round.

Historical UFCStats observations remain round-level. Segment resolution is a
simulation mechanism, not a claim that exact historical event sequences are
known.

Segment resolution allows:

- natural finish timing
- incremental fatigue and damage
- knockdown and follow-up pressure
- phase changes
- control persistence
- submission-danger accumulation
- termination without generating activity after the finish

## 5. Historical State Versus Dynamic State

### 5.1 Historical Fighter Profile

The historical profile is fixed entering a target fight and must use only
information available before that fight.

Candidate profile domains:

- baseline striking pace and variance
- pace sustainability and cardio decay
- striking accuracy and target mix
- damage conversion
- strike defense and defensive deterioration
- opponent suppression
- takedown pressure and persistence
- takedown defense
- control conversion and stability
- ground-offense conversion
- submission pressure and escape resistance
- damage susceptibility
- chin risk
- durability and recovery
- adversity response
- phase identity and phase imposition
- sample depth and parameter uncertainty

### 5.2 Dynamic Simulated State

Dynamic state differs in every Monte Carlo path.

Candidate variables:

```text
energy
head_damage
body_damage
leg_damage
chin_integrity
defensive_stability
recovery_reserve
confidence
tactical_urgency
current_phase
control_position
submission_danger
cumulative_strike_activity
cumulative_wrestling_activity
knockdowns
rounds_won
score_state
```

Dynamic state may only be updated from events generated earlier in the current
simulation path.

## 6. Fighter Simulation Profile Contract

A future `FighterSimulationProfile` contract will expose stable pre-fight
parameters to the simulator.

It must include:

- identity
- prior-fight and valid-observation counts
- fallback source
- uncertainty
- pace and cardio
- striking offense and defense
- suppression
- wrestling offense and defense
- control conversion
- submission offense and defense
- damage conversion
- damage susceptibility
- chin risk
- durability and recovery
- phase preferences

Sparse fighters must be shrunk toward subgroup and population priors.

Candidate fallback hierarchy:

```text
fighter estimate
    ↓
weight class + gender + scheduled rounds
    ↓
weight class + gender
    ↓
gender
    ↓
global population
```

Every parameter must identify its source and effective sample depth.

## 7. Matchup Interaction

The fighters must not be simulated independently.

Conceptually:

```text
expected Fighter A activity =
    Fighter A offensive tendency
  × Fighter B defensive allowance
  × Fighter B suppression effect
  × matchup phase probability
  × current dynamic-state effects
```

The exact statistical form will be selected and calibrated later.

## 8. Segment Event Generation

Each segment may generate:

1. phase selection
2. significant-strike attempts
3. strikes landed
4. head, body, and leg targeting
5. distance, clinch, or ground striking
6. takedown attempts
7. takedown completions
8. control time
9. ground strikes
10. submission attempts
11. reversals or escapes
12. knockdowns

Initial candidate distributions:

| Quantity | Initial candidate |
|---|---|
| Strike attempts | Negative binomial |
| Landed strikes | Beta-binomial or calibrated binomial |
| Takedown attempts | Hurdle negative binomial |
| Takedown success | Binomial conditional on attempts |
| Control duration | Zero-inflated bounded distribution |
| Submission attempts | Hurdle count model |
| Knockdowns | Bernoulli plus conditional count |
| Phase selection | Multinomial model |

These are candidate starting points, not permanently locked algorithms.

## 9. Fatigue and Cardio

Fatigue must emerge from simulated workload.

Energy expenditure may depend on:

- striking volume
- high-output exchanges
- failed and successful takedowns
- control work
- being controlled
- body damage
- recovery between segments and rounds

RFS pace sustainability and cardio features determine how strongly workload
changes later output and defense.

## 10. Damage, Chin, Durability, and Recovery

These concepts must remain separate:

- Defensive avoidance: likelihood that an attack lands.
- Damage susceptibility: damage created by a landed attack.
- Chin risk: likelihood of acute destabilization or knockdown.
- Durability: likelihood of surviving sustained damage.
- Recovery: restoration after damaging exchanges or between rounds.
- Defensive deterioration: becoming easier to hit as damage and fatigue increase.

The engine may track:

```text
head_damage
body_damage
leg_damage
chin_integrity
defensive_stability
recovery_reserve
```

These are latent simulation states and must be calibrated against observable
historical proxies.

## 11. Finish Mechanics

Hard finish thresholds are prohibited.

Do not use:

```python
if attacker_output > defender_absorption:
    finish = True
```

At the end of each segment, calculate competing probabilities for:

```text
no finish
red KO/TKO
red submission
blue KO/TKO
blue submission
```

### KO/TKO hazard inputs

- recent head-damage pressure
- cumulative head damage
- ground-strike pressure
- knockdown shock
- follow-up pressure
- defender fatigue
- defensive deterioration
- remaining chin integrity
- durability
- recovery

### Submission hazard inputs

- current phase and control
- recent takedown success
- submission attempts
- accumulated submission danger
- attacker conversion ability
- defender fatigue
- escape resistance
- recovery

The pressure-versus-resistance margin changes finish probability but does not
guarantee a finish.

## 12. Scoring and Decisions

When no finish occurs, the simulator will determine round and fight winners from
simulated activity.

Candidate scoring inputs:

- effective striking
- damaging strikes
- knockdowns
- takedowns
- control
- submission pressure
- judge uncertainty

The decision engine must be calibrated independently and checked for corner bias.

## 13. Point-in-Time Rule

For target fight N:

```text
completed fights through N-1
        ↓
historical RFS profile entering N
        ↓
simulation of N
```

The target fight's realized statistics, outcome, or future information must never
enter its simulation profile or mechanics.

## 14. Experience Cohorts

Primary model-selection cohort:

```text
red_prior_fights >= 3
and
blue_prior_fights >= 3
```

Low-experience fights remain in diagnostics but do not select or reject V1.

Sparse profiles must show:

- stronger prior shrinkage
- greater uncertainty
- explicit fallback provenance

## 15. Reproducibility

Every stochastic function must accept an explicit random generator.

Every simulation and calibration run must record:

- simulator version
- profile version
- calibration version
- data versions
- parameter set
- seed
- path count
- eligibility cohort
- fallback usage

Identical inputs, parameters, and seeds must reproduce identical outputs.

## 16. Production Boundary

RFS Monte Carlo V1 remains shadow-only until a separate promotion review.

During initial development it must not:

- alter production predictions
- alter betting recommendations
- write to production prediction artifacts
- change production schemas
- modify production feature contracts
- replace the current simulator
- merge without explicit approval

## 17. Initial Non-Goals

The following are out of scope for initial V1:

- market odds and betting logic
- parlays
- user-interface work
- live play-by-play reconstruction
- exact cage coordinates
- individual strike animation
- referee-specific models
- judge-specific models
- neural networks or reinforcement learning
- proprietary event data not currently available
- production deployment

## 18. Approval Gate

Phase 0 passes only when:

- this architecture is reviewed
- the roadmap is reviewed
- the initial decision log is accepted
- package and artifact boundaries are approved
- no simulator implementation has begun prematurely
