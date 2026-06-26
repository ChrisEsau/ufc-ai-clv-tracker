# Round Fighter State Feature Registry

## Purpose

This registry defines approved Round Fighter State feature families.

It is the bridge between:

```text
docs/ROUND_FIGHTER_STATE_ONTOLOGY.md
```

and future implementation code.

The ontology defines the latent-state architecture.

This registry defines which feature families are approved for engineering, what each family measures, what raw observations it may use, what outputs it may create, and how each family must be validated before promotion.

This document does not implement features.

---

# Core Rule

Round Fighter State features must never be raw round statistics exposed directly to the model.

Correct:

```text
round stats
    ↓
fighter-state feature family
    ↓
point-in-time fighter state
    ↓
side-aware model feature view
```

Incorrect:

```text
round stats
    ↓
model
```

---

# Approved Initial Build Scope

The initial Round Fighter State build is limited to P0 feature families.

## P0 Families

1. Round Trajectory and Decay
2. Opponent Suppression and Phase Imposition
3. Wrestling Control Conversion

No P1 or P2 feature family should be implemented until the P0 families are individually validated.

---

# Global Feature Rules

## Point-in-Time Rule

For a target fight on date `D`, every Round Fighter State feature must be calculated using only fights completed before date `D`.

The target fight itself must not contribute to the feature row used to predict that fight.

---

## Fighter-State Grain

Historical state file:

```text
data/features/round_fighter_state_history.parquet
```

Expected grain:

```text
one row per fighter per completed fight
```

Latest state file:

```text
data/features/round_latest_fighter_state.parquet
```

Expected grain:

```text
one row per fighter
```

---

## Model View Grain

The model feature view should be fight-level and side-aware.

Raw fighter-state columns should be transformed into:

```text
r_rfs_*
b_rfs_*
rfs_*_diff
```

Example:

```text
r_rfs_traj_late_output_ratio
b_rfs_traj_late_output_ratio
rfs_traj_late_output_ratio_diff
```

---

# Naming Convention

All Round Fighter State features must begin with:

```text
rfs_
```

Approved family prefixes:

```text
rfs_traj_*       round trajectory, pace sustainability, cardio decay
rfs_suppress_*   opponent suppression
rfs_phase_*      phase control and phase imposition
rfs_wrestle_*    wrestling pressure and control conversion
rfs_def_*        defensive stability and degradation
rfs_adv_*        adversity response and recovery
rfs_adapt_*      tactical adaptation and round adjustment
rfs_finish_*     damage conversion and finishing pressure
rfs_identity_*   fight identity and style persistence
```

---

# P0.1 — Round Trajectory and Decay

## Purpose

Measure whether a fighter sustains, improves, or deteriorates as rounds progress.

This family is designed to capture:

* pace sustainability
* cardio decay
* late-round output
* offensive persistence
* defensive deterioration
* recovery after expensive rounds

---

## Parent Domains

* Sustainability and Resilience
* Tactical Intelligence

---

## Latent States

* Pace Sustainability State
* Cardio State
* Defensive Stability State
* Recovery State

---

## Raw Inputs Allowed

From `data/fight_details/ufc_round_stats.parquet`:

* `round`
* significant strikes landed / attempted
* total strikes landed / attempted
* takedowns landed / attempted
* control seconds
* knockdowns
* opponent significant strikes landed / attempted
* opponent total strikes landed / attempted
* opponent control seconds

---

## Approved Feature Concepts

### Offensive Trajectory

Allowed feature concepts:

```text
rfs_traj_sig_attempt_slope
rfs_traj_total_attempt_slope
rfs_traj_sig_landed_slope
rfs_traj_total_landed_slope
rfs_traj_late_sig_attempt_ratio
rfs_traj_late_total_attempt_ratio
```

Purpose:

Measure whether the fighter’s offensive output holds up across rounds.

---

### Efficiency Trajectory

Allowed feature concepts:

```text
rfs_traj_sig_accuracy_slope
rfs_traj_total_accuracy_slope
rfs_traj_late_sig_accuracy_ratio
```

Purpose:

Measure whether the fighter remains efficient as fatigue and damage accumulate.

---

### Wrestling Persistence

Allowed feature concepts:

```text
rfs_traj_td_attempt_slope
rfs_traj_td_accuracy_slope
rfs_traj_late_td_attempt_ratio
```

Purpose:

Measure whether the fighter continues wrestling after early rounds.

---

### Defensive Decay

Allowed feature concepts:

```text
rfs_traj_opp_sig_accuracy_allowed_slope
rfs_traj_opp_total_accuracy_allowed_slope
rfs_traj_opp_late_output_allowed_ratio
rfs_traj_late_control_allowed_ratio
```

Purpose:

Measure whether the fighter becomes easier to hit or control as rounds progress.

---

## Exclusions

Do not include:

* raw career totals
* raw per-round rows
* unshifted target-fight round stats
* features that require in-round timestamps
* spatial ringcraft assumptions

---

## Validation Requirements

Before promotion, this family must be tested for:

* feature completeness
* missingness by fighter
* stability across eras
* correlation with existing EWM features
* ablation impact on log loss
* ablation impact on calibration
* SHAP contribution
* ROI impact
* CLV impact

---

# P0.2 — Opponent Suppression and Phase Imposition

## Purpose

Measure how much worse opponents perform against a fighter compared with their own prior baseline.

This is a flagship Round Fighter State family.

The core question is not:

```text
How good is Fighter A?
```

The core question is:

```text
How much does Fighter A suppress Fighter B below Fighter B's normal state?
```

---

## Parent Domains

* Phase Control
* Fight Identity
* Tactical Intelligence

---

## Latent States

* Opponent Suppression State
* Phase Imposition State
* Initiative State
* Range Ownership State
* Identity Disruption State

---

## Raw Inputs Allowed

From `data/fight_details/ufc_round_stats.parquet`:

* opponent significant strike attempts
* opponent total strike attempts
* opponent takedown attempts
* opponent takedown success
* opponent control seconds
* opponent distance strike attempts
* opponent clinch strike attempts
* opponent ground strike attempts
* opponent accuracy by phase
* opponent round-level output

From prior fighter-state history:

* opponent historical output baseline
* opponent historical phase mix baseline
* opponent historical accuracy baseline
* opponent historical wrestling baseline
* opponent historical control baseline

---

## Baseline Requirement

Opponent suppression features require a point-in-time opponent baseline.

For a fight on date `D`, the opponent baseline must use only the opponent’s fights before date `D`.

Correct:

```text
opponent fights before D
    ↓
opponent expected baseline
    ↓
realized opponent performance against fighter
    ↓
suppression delta
```

Incorrect:

```text
opponent full career including future fights
    ↓
suppression delta
```

---

## Approved Feature Concepts

### Output Suppression

Allowed feature concepts:

```text
rfs_suppress_opp_sig_attempt_delta
rfs_suppress_opp_total_attempt_delta
rfs_suppress_opp_late_output_delta
```

Purpose:

Measure whether the fighter reduces the opponent’s normal activity.

---

### Accuracy Suppression

Allowed feature concepts:

```text
rfs_suppress_opp_sig_accuracy_delta
rfs_suppress_opp_total_accuracy_delta
rfs_suppress_opp_distance_accuracy_delta
```

Purpose:

Measure whether the fighter makes the opponent less efficient.

---

### Wrestling Suppression

Allowed feature concepts:

```text
rfs_suppress_opp_td_attempt_delta
rfs_suppress_opp_td_accuracy_delta
rfs_suppress_opp_control_delta
```

Purpose:

Measure whether the fighter denies the opponent’s wrestling and control game.

---

### Phase-Mix Disruption

Allowed feature concepts:

```text
rfs_suppress_opp_distance_share_delta
rfs_suppress_opp_clinch_share_delta
rfs_suppress_opp_ground_share_delta
rfs_suppress_opp_phase_mix_disruption
```

Purpose:

Measure whether the fighter forces opponents out of their normal fight shape.

---

## Exclusions

Do not include:

* future opponent baselines
* raw opponent totals without baseline comparison
* assumptions about cage cutting without spatial data
* subjective labels like “pressure” unless backed by measurable suppression

---

## Validation Requirements

Before promotion, this family must be tested for:

* point-in-time opponent baseline correctness
* opponent baseline availability
* fighter coverage
* missing baseline handling
* correlation with existing opponent-adjusted features
* ablation impact on log loss
* ablation impact on calibration
* SHAP contribution
* ROI impact
* CLV impact

---

# P0.3 — Wrestling Control Conversion

## Purpose

Measure whether wrestling activity becomes meaningful control, damage, submission threat, or opponent suppression.

This family separates empty wrestling from effective wrestling.

Raw control time is not enough.

The important question is:

```text
Does the fighter convert wrestling into consequential offense?
```

---

## Parent Domains

* Conversion and Enforcement
* Phase Control

---

## Latent States

* Pressure Wrestling State
* Control Conversion State
* Transition Linkage State
* Damage Conversion State

---

## Raw Inputs Allowed

From `data/fight_details/ufc_round_stats.parquet`:

* takedowns landed
* takedowns attempted
* control seconds
* ground strikes landed / attempted
* significant ground strikes landed / attempted
* submission attempts
* reversals
* opponent strike output
* opponent control seconds

---

## Approved Feature Concepts

### Takedown-to-Control Conversion

Allowed feature concepts:

```text
rfs_wrestle_control_per_td_attempt
rfs_wrestle_control_per_td_landed
rfs_wrestle_td_to_control_conversion
```

Purpose:

Measure whether takedown attempts produce actual control.

---

### Control-to-Damage Conversion

Allowed feature concepts:

```text
rfs_wrestle_ground_strikes_per_control_min
rfs_wrestle_sig_ground_strikes_per_control_min
rfs_wrestle_control_to_damage_score
```

Purpose:

Measure whether control creates meaningful offense.

---

### Control-to-Submission Threat

Allowed feature concepts:

```text
rfs_wrestle_sub_attempts_per_control_min
rfs_wrestle_submission_pressure_score
```

Purpose:

Measure whether control creates submission threat.

---

### Control Without Reversal

Allowed feature concepts:

```text
rfs_wrestle_reversal_allowed_per_control_min
rfs_wrestle_control_stability_score
```

Purpose:

Measure whether the fighter controls safely without giving up reversals.

---

### Wrestling Pressure Persistence

Allowed feature concepts:

```text
rfs_wrestle_td_attempt_slope
rfs_wrestle_td_persistence_score
rfs_wrestle_failed_td_persistence
```

Purpose:

Measure whether wrestling pressure continues across rounds.

---

## Exclusions

Do not include:

* raw takedown totals alone
* raw control time alone
* control time treated as automatically positive
* unadjusted wrestling volume without conversion
* sequence assumptions that require timestamped data

---

## Validation Requirements

Before promotion, this family must be tested for:

* division-by-zero handling
* control time outlier handling
* fighter coverage
* correlation with existing takedown features
* ablation impact on log loss
* ablation impact on calibration
* SHAP contribution
* ROI impact
* CLV impact

---

# P1 — Deferred Families

The following families are approved conceptually but should not be implemented until P0 has been validated.

## P1.1 Defensive Degradation

Purpose:

Measure whether a fighter becomes easier to hit, control, or pressure as rounds progress.

Candidate prefix:

```text
rfs_def_*
```

---

## P1.2 Adversity Response

Purpose:

Measure how a fighter responds after losing rounds, absorbing damage, being controlled, or failing takedowns.

Candidate prefix:

```text
rfs_adv_*
```

---

## P1.3 Tactical Adaptation Proxies

Purpose:

Measure whether a fighter changes phase or strategy after poor results.

Candidate prefix:

```text
rfs_adapt_*
```

---

## P1.4 Finishing Snowball / Dominant-Round Conversion

Purpose:

Measure whether a fighter converts dominant moments into follow-up dominance, damage, or finishes.

Candidate prefix:

```text
rfs_finish_*
```

---

# P2 — Future Data Required

The following families require richer data before serious implementation.

## P2.1 Spatial Ringcraft

Requires:

* cage position
* center control
* fence position
* lateral movement
* exit denial

---

## P2.2 Cage Cutting

Requires:

* opponent retreat tracking
* fence trapping events
* angle control
* pressure direction

---

## P2.3 Strike-to-Shot Sequencing

Requires:

* event timestamps
* strike sequence data
* takedown entry timing
* setup chains

---

## P2.4 True Tempo / Rhythm

Requires:

* in-round timestamps
* burst/lull detection
* minute-level or second-level event data

---

## P2.5 In-Round Momentum

Requires:

* timestamped strikes
* damage events
* control transitions
* live scoring or inferred momentum windows

---

# Weak Beliefs and Modeling Cautions

The following concepts should not be over-weighted without stronger evidence or better measurement.

## Raw Reach and Wingspan

Reach and wingspan may matter as context, but they should not be treated as strong direct predictors by themselves.

Better framing:

```text
reach usage
stance geometry
range ownership
phase access
```

---

## Bonus Incentive Narratives

Do not assume performance bonuses materially change fighter behavior unless there is measurable evidence.

---

## Raw Control Time

Control time alone is not equivalent to effective grappling.

Prefer:

```text
control-to-damage
control-to-submission-threat
control-to-advancement
control-to-opponent-suppression
```

---

## Aura, Momentum, and Championship Experience

These should not be modeled as vague narrative variables.

If useful, decompose them into measurable states:

```text
recovery after adversity
late-round output
defensive stability
dominant-round conversion
tactical adjustment
```

---

# Promotion Criteria

A Round Fighter State family may be promoted only if it passes both technical and betting evaluation.

## Technical Evaluation

Required:

* schema validation
* point-in-time validation
* feature completeness audit
* missingness audit
* duplicate-key audit
* infinite-value audit
* distribution stability
* correlation review
* ablation testing
* SHAP review
* calibration review

---

## Betting Evaluation

Required:

* ROI impact
* CLV impact
* bet count impact
* edge distribution review
* confidence bucket behavior
* probability calibration by bucket
* watchlist/official-bet behavior
* live feature availability

---

# Initial Development Sequence

The approved sequence is:

```text
1. Finalize ontology
2. Create this feature registry
3. Design P0.1 Round Trajectory formulas
4. Implement P0.1 only
5. Validate P0.1
6. Run ablation against baseline
7. Decide keep / revise / reject
8. Repeat for P0.2
9. Repeat for P0.3
10. Only then consider P1
```

---

# Canonical Summary

This registry approves only the first wave of Round Fighter State feature development.

The P0 build scope is:

```text
P0.1 Round Trajectory and Decay
P0.2 Opponent Suppression and Phase Imposition
P0.3 Wrestling Control Conversion
```

Round Fighter State development must stay incremental, interpretable, point-in-time safe, and independently validated.

No feature family should become part of production unless it improves the model in ways that matter for betting:

* better log loss
* better calibration
* better ROI
* better CLV
* better live usability
