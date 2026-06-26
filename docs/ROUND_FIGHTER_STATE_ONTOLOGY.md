# Round Fighter State Ontology

## Purpose

The **Round Fighter State** architecture is a second-generation UFC feature system built from UFCStats round-level observations.

Its purpose is not to add more raw statistics to the model.

Its purpose is to infer how fighters behave as fights progress:

* who controls where the fight happens
* who suppresses the opponent
* who sustains pace
* who converts position into damage
* who deteriorates under pressure
* who adapts between rounds
* who recovers after adversity
* who repeatedly imposes a durable fight identity

Round statistics are treated as **raw observations**.

Round Fighter State features are treated as **derived latent-state features**.

The intended architecture is:

```text
data/fight_details/ufc_round_stats.parquet
        ↓
Round Fighter State feature builder
        ↓
data/features/round_fighter_state_history.parquet
        ↓
data/features/round_latest_fighter_state.parquet
        ↓
training / live feature views
        ↓
model experiments
```

This document is the canonical ontology for future Round Fighter State development.

No engineered feature family should be implemented unless it maps clearly into this ontology.

---

# Core Philosophy

## 1. Model Behavior, Not Just Totals

Traditional fight models often ask:

> How many strikes did Fighter A land historically?

Round Fighter State asks deeper questions:

> Did Fighter A maintain output as rounds progressed?

> Did Fighter A reduce the opponent’s normal output?

> Did Fighter A convert wrestling attempts into meaningful control?

> Did Fighter A adjust when the original plan failed?

> Did Fighter A’s opponent perform worse than expected?

This feature layer is designed to capture **fight behavior**, not merely cumulative production.

---

## 2. Round Stats Are Not Model Features

`ufc_round_stats.parquet` is a raw observation table.

It should not be joined directly into the model feature view.

Instead, it feeds a state-building layer:

```text
round observations
        ↓
fighter-state history
        ↓
latest pre-fight fighter state
        ↓
side-aware model features
```

This protects the model from feature explosion, leakage, and unstable raw-stat joins.

---

## 3. Every Feature Must Have a Purpose

Every future feature must answer at least one of these questions:

* Does this fighter control the phase?
* Does this fighter suppress the opponent?
* Does this fighter sustain pace?
* Does this fighter degrade defensively?
* Does this fighter recover after adversity?
* Does this fighter convert opportunities into damage or control?
* Does this fighter adapt tactically?
* Does this fighter impose a repeatable identity?

If a proposed feature does not answer one of these questions, it should not be added.

---

# Data Grain

## Raw Round Statistics Grain

Input file:

```text
data/fight_details/ufc_round_stats.parquet
```

Expected grain:

```text
one row per fighter per fight per round
```

Each fight round should produce two rows:

* one red-corner fighter row
* one blue-corner fighter row

Recommended primary key:

```text
fight_id
fighter_id
round
corner
```

Expected identity columns:

* `event_id`
* `fight_id`
* `fighter_id`
* `opponent_id`
* `fighter_name`
* `opponent_name`
* `round`
* `corner`
* `event_name`
* `date`

Expected round statistic columns:

* knockdowns
* significant strikes landed / attempted
* total strikes landed / attempted
* takedowns landed / attempted
* submission attempts
* reversals
* control seconds
* head strikes landed / attempted
* body strikes landed / attempted
* leg strikes landed / attempted
* distance strikes landed / attempted
* clinch strikes landed / attempted
* ground strikes landed / attempted

---

## Round Fighter State History Grain

Output file:

```text
data/features/round_fighter_state_history.parquet
```

Expected grain:

```text
one row per fighter per completed fight
```

This table represents the fighter’s state **after a completed historical fight has been processed**.

Recommended keys:

* `fighter_id`
* `fighter_name`
* `fight_id`
* `event_id`
* `event_name`
* `date`
* `opponent_id`
* `opponent_name`
* `corner`

This table may include both:

* fight-level state observations from the completed fight
* rolling / expanding / EWM state values available after that fight

However, this table must not be used directly for a target fight without proper point-in-time shifting.

---

## Latest Fighter State Grain

Output file:

```text
data/features/round_latest_fighter_state.parquet
```

Expected grain:

```text
one row per fighter
```

This table represents each fighter’s latest known Round Fighter State entering the next prediction run.

Recommended key:

```text
fighter_id
```

This table is used by live prediction feature builders.

---

# Point-in-Time Rules

Point-in-time correctness is mandatory.

## Training Rule

For historical model training:

> A fighter’s Round Fighter State entering Fight N must be based only on fights completed before Fight N.

The target fight itself must not contribute to the features used to predict that same fight.

Correct:

```text
Fighter history through Fight N-1
        ↓
features for Fight N
```

Incorrect:

```text
Fight N round stats
        ↓
features for Fight N
```

---

## Live Prediction Rule

For live prediction:

> Live Round Fighter State must use only completed fights available before the upcoming fight.

Upcoming fight data must never be present in:

* `round_fighter_state_history.parquet`
* `round_latest_fighter_state.parquet`
* live feature views
* prediction artifacts

---

## Append Rule

A completed fight may be appended into Round Fighter State history only after the fight is final.

The lifecycle should be:

```text
upcoming fight
        ↓
prediction uses latest prior state
        ↓
fight occurs
        ↓
round stats scraped
        ↓
round stats validated
        ↓
fighter state updated
        ↓
latest state refreshed
```

---

# Parent Domains

Round Fighter State is organized into five parent domains:

1. Phase Control
2. Sustainability and Resilience
3. Conversion and Enforcement
4. Tactical Intelligence
5. Fight Identity

Each parent domain contains latent states.

Each latent state maps to observable signals.

Each observable signal may later produce engineered features.

---

# Domain 1 — Phase Control

## Core Question

Who decides where the fight happens?

Phase Control captures whether a fighter can impose the fight environment:

* distance
* clinch
* ground
* cage control
* wrestling threat
* opponent suppression

A fighter with strong Phase Control does not merely perform well inside a phase. They influence which phase occurs.

---

## Latent State: Initiative State

### Question

Who is forcing exchanges instead of reacting?

### Observable Signals

* strike attempt share by round
* first-round output advantage
* round-opening output advantage
* takedown attempt share
* total offensive action share
* opponent defensive shell behavior

### Candidate Feature Concepts

* `rfs_phase_initiative_attempt_share`
* `rfs_phase_first_round_initiative`
* `rfs_phase_late_round_initiative`
* `rfs_phase_offensive_action_share`

---

## Latent State: Range Ownership State

### Question

Who controls whether the fight occurs at striking range?

### Observable Signals

* distance strike share
* opponent distance strike suppression
* distance accuracy advantage
* distance volume advantage
* round-to-round distance control persistence

### Candidate Feature Concepts

* `rfs_phase_distance_share`
* `rfs_phase_distance_attempt_advantage`
* `rfs_phase_distance_accuracy_advantage`
* `rfs_phase_opponent_distance_suppression`

---

## Latent State: Cage Geography State

### Question

Who controls constrained-space phases such as clinch and cage control?

### Observable Signals

* clinch strike share
* control seconds
* takedown attempts from pressure sequences
* opponent reduction in distance volume
* control-heavy round frequency

### Candidate Feature Concepts

* `rfs_phase_clinch_share`
* `rfs_phase_control_time_share`
* `rfs_phase_control_round_rate`
* `rfs_phase_distance_denial_rate`

---

## Latent State: Opponent Suppression State

### Question

How much worse does the opponent become when facing this fighter?

This is one of the highest-priority feature families.

Instead of asking:

> How good is Fighter A?

Opponent Suppression asks:

> How much does Fighter A reduce Fighter B below Fighter B’s normal baseline?

### Observable Signals

* opponent strike attempts versus opponent baseline
* opponent significant strike accuracy versus baseline
* opponent takedown attempts versus baseline
* opponent takedown success versus baseline
* opponent control time versus baseline
* opponent phase mix deviation from baseline
* opponent late-round output collapse

### Candidate Feature Concepts

* `rfs_suppress_opp_sig_attempt_delta`
* `rfs_suppress_opp_total_attempt_delta`
* `rfs_suppress_opp_td_attempt_delta`
* `rfs_suppress_opp_td_success_delta`
* `rfs_suppress_opp_control_delta`
* `rfs_suppress_opp_accuracy_delta`
* `rfs_suppress_opp_phase_mix_disruption`

### Implementation Note

Opponent baselines must be computed point-in-time.

For a fight on date D, the opponent baseline must only include the opponent’s fights before date D.

---

## Latent State: Phase Imposition State

### Question

Does the fighter force the opponent into uncomfortable fight phases?

### Observable Signals

* opponent phase mix versus baseline
* opponent distance/clinch/ground deviation
* forced grappling-heavy fights
* forced low-volume striking fights
* forced defensive wrestling rounds
* repeated phase disruption across opponents

### Candidate Feature Concepts

* `rfs_phase_imposition_score`
* `rfs_phase_opp_identity_disruption`
* `rfs_phase_forced_grappling_rate`
* `rfs_phase_forced_low_volume_rate`

---

# Domain 2 — Sustainability and Resilience

## Core Question

Who keeps functioning when the fight gets expensive?

This domain captures output persistence, cardio, durability, defensive stability, recovery, and response to adversity.

---

## Latent State: Pace Sustainability State

### Question

Does the fighter maintain output across rounds?

### Observable Signals

* round-to-round strike attempt slope
* round-to-round significant strike slope
* third-round output versus first-round output
* late-round offensive action share
* output after high-volume rounds
* output after control-heavy rounds

### Candidate Feature Concepts

* `rfs_traj_sig_attempt_slope`
* `rfs_traj_total_attempt_slope`
* `rfs_traj_late_output_ratio`
* `rfs_traj_round3_vs_round1_output`
* `rfs_traj_post_high_pace_output`

---

## Latent State: Cardio State

### Question

Does the fighter’s performance decay as the fight progresses?

### Observable Signals

* decreasing output
* decreasing accuracy
* increasing opponent accuracy allowed
* increased control allowed
* reduced takedown defense late
* reduced offensive wrestling late

### Candidate Feature Concepts

* `rfs_traj_cardio_decay_score`
* `rfs_traj_accuracy_decay`
* `rfs_traj_defense_decay`
* `rfs_traj_late_control_allowed`
* `rfs_traj_late_td_defense_decay`

---

## Latent State: Defensive Stability State

### Question

Does the fighter remain defensively sound across rounds?

### Observable Signals

* opponent accuracy by round
* significant strikes absorbed by round
* head strikes absorbed by round
* knockdowns absorbed
* late-round defensive deterioration
* opponent output acceleration

### Candidate Feature Concepts

* `rfs_def_opp_accuracy_allowed_slope`
* `rfs_def_sig_absorbed_slope`
* `rfs_def_head_absorbed_slope`
* `rfs_def_late_damage_allowed`
* `rfs_def_deterioration_score`

---

## Latent State: Durability State

### Question

Does the fighter absorb damage without immediate collapse?

### Observable Signals

* knockdowns absorbed
* output after knockdown
* output after high-damage round
* defensive recovery after high-damage round
* fight continuation after damaging rounds

### Candidate Feature Concepts

* `rfs_adv_post_kd_output_ratio`
* `rfs_adv_post_damage_output_ratio`
* `rfs_adv_post_damage_defense_recovery`
* `rfs_adv_durability_response_score`

---

## Latent State: Recovery State

### Question

Does the fighter recover between rounds after losing exchanges?

### Observable Signals

* improved output after losing prior round
* improved accuracy after poor prior round
* reduced damage allowed after high-damage round
* increased wrestling after striking adversity
* reduced opponent momentum after adversity

### Candidate Feature Concepts

* `rfs_adv_rebound_output_score`
* `rfs_adv_rebound_accuracy_score`
* `rfs_adv_rebound_defense_score`
* `rfs_adv_momentum_reset_score`

---

## Latent State: Adversity Response State

### Question

What does the fighter do after things go badly?

### Observable Signals

* response after knockdown
* response after losing round
* response after being controlled
* response after failed takedowns
* response after absorbing high damage
* response after opponent output spike

### Candidate Feature Concepts

* `rfs_adv_after_lost_round_output`
* `rfs_adv_after_controlled_round_output`
* `rfs_adv_after_failed_td_phase_shift`
* `rfs_adv_after_damage_phase_shift`
* `rfs_adv_response_score`

---

# Domain 3 — Conversion and Enforcement

## Core Question

Who turns opportunities into meaningful damage or control?

This domain separates activity from effectiveness.

A fighter may attempt many takedowns but fail to convert them into control.

A fighter may hold control but fail to convert it into damage.

A fighter may hurt opponents but fail to build finishing pressure.

---

## Latent State: Pressure Wrestling State

### Question

Does the fighter create wrestling pressure across rounds?

### Observable Signals

* takedown attempts by round
* takedown attempt share
* takedown persistence after failed attempts
* takedown attempts after losing striking exchanges
* opponent output suppression after wrestling attempts

### Candidate Feature Concepts

* `rfs_wrestle_td_attempt_share`
* `rfs_wrestle_td_attempt_slope`
* `rfs_wrestle_failed_td_persistence`
* `rfs_wrestle_striking_to_wrestling_shift`
* `rfs_wrestle_pressure_score`

---

## Latent State: Control Conversion State

### Question

Does the fighter convert takedown attempts into control?

### Observable Signals

* control seconds per takedown attempt
* control seconds per takedown landed
* control share by round
* opponent output reduction after control
* repeated control-heavy rounds

### Candidate Feature Concepts

* `rfs_wrestle_control_per_td_attempt`
* `rfs_wrestle_control_per_td_landed`
* `rfs_wrestle_control_share`
* `rfs_wrestle_control_repeatability`
* `rfs_wrestle_control_conversion_score`

---

## Latent State: Transition Linkage State

### Question

Does the fighter chain wrestling, control, submissions, and ground strikes?

### Observable Signals

* ground strikes after control
* submission attempts after takedowns
* reversals following opponent control
* control continuation across rounds
* ground strike volume per control minute

### Candidate Feature Concepts

* `rfs_wrestle_ground_strikes_per_control_min`
* `rfs_wrestle_sub_attempts_per_control_min`
* `rfs_wrestle_transition_activity_score`
* `rfs_wrestle_control_to_damage_score`

---

## Latent State: Damage Conversion State

### Question

Does the fighter convert opportunities into meaningful damage?

### Observable Signals

* significant strikes per total attempt
* head strike share
* ground strikes per control second
* knockdowns per significant strike landed
* opponent defensive degradation after damage

### Candidate Feature Concepts

* `rfs_finish_sig_damage_share`
* `rfs_finish_head_strike_share`
* `rfs_finish_ground_damage_per_control_min`
* `rfs_finish_kd_per_sig_landed`
* `rfs_finish_damage_conversion_score`

---

## Latent State: Finishing Pressure State

### Question

Does the fighter snowball dominant moments?

### Observable Signals

* follow-up output after knockdowns
* output after dominant rounds
* control after damaging exchanges
* increased finishing activity after opponent damage
* opponent late-round collapse

### Candidate Feature Concepts

* `rfs_finish_post_kd_followup_output`
* `rfs_finish_post_dominant_round_output`
* `rfs_finish_snowball_score`
* `rfs_finish_late_pressure_score`
* `rfs_finish_dominant_round_conversion`

---

# Domain 4 — Tactical Intelligence

## Core Question

Who changes correctly when the first plan does not work?

This domain attempts to model adaptation, discipline, tactical switching, and momentum management.

Because true tactical intelligence is difficult to observe directly, this domain should begin with conservative proxy features.

---

## Latent State: Adaptability State

### Question

Does the fighter change strategy after poor results?

### Observable Signals

* increased wrestling after losing striking exchanges
* increased striking after failed grappling
* improved accuracy after poor round
* reduced damage allowed after poor defensive round
* phase mix changes after losing round

### Candidate Feature Concepts

* `rfs_adapt_phase_shift_after_lost_round`
* `rfs_adapt_wrestling_after_striking_loss`
* `rfs_adapt_striking_after_failed_grappling`
* `rfs_adapt_accuracy_rebound`
* `rfs_adapt_defensive_rebound`

---

## Latent State: Fight IQ State

### Question

Does the fighter make effective tactical adjustments?

### Observable Signals

* adjustment followed by improved output
* adjustment followed by reduced damage allowed
* adjustment followed by increased control
* opponent output reduction after phase change
* late-round trend reversal

### Candidate Feature Concepts

* `rfs_adapt_successful_phase_change`
* `rfs_adapt_adjustment_payoff_score`
* `rfs_adapt_late_trend_reversal`
* `rfs_adapt_opp_output_reduction_after_shift`

---

## Latent State: Tactical Discipline State

### Question

Does the fighter avoid inefficient or deteriorating tactical choices?

### Observable Signals

* repeated failed takedown attempts without payoff
* declining accuracy with increasing volume
* loss of defense during pressure attempts
* control without damage
* output spikes followed by collapse

### Candidate Feature Concepts

* `rfs_adapt_failed_td_overcommitment`
* `rfs_adapt_volume_efficiency_decay`
* `rfs_adapt_pressure_defense_tradeoff`
* `rfs_adapt_empty_control_rate`
* `rfs_adapt_overextension_score`

---

## Latent State: Momentum State

### Question

Does the fighter gain, lose, or reverse momentum across rounds?

### Observable Signals

* round-to-round output swings
* damage swings
* control swings
* opponent output swings
* late-round reversal
* response after dominant or poor round

### Candidate Feature Concepts

* `rfs_adapt_momentum_swing_score`
* `rfs_adapt_positive_momentum_rate`
* `rfs_adapt_negative_momentum_rate`
* `rfs_adapt_momentum_reversal_score`

---

## Latent State: Round Adjustment State

### Question

Does the fighter improve between rounds?

### Observable Signals

* accuracy improvement from prior round
* output improvement from prior round
* damage allowed reduction from prior round
* control improvement from prior round
* phase selection improvement from prior round

### Candidate Feature Concepts

* `rfs_adapt_round_to_round_accuracy_gain`
* `rfs_adapt_round_to_round_output_gain`
* `rfs_adapt_round_to_round_defense_gain`
* `rfs_adapt_round_to_round_control_gain`

---

# Domain 5 — Fight Identity

## Core Question

What behavioral pattern does this fighter repeatedly impose?

Fight Identity captures repeatable stylistic behavior.

This domain should help answer:

* Is this fighter consistently a pressure wrestler?
* Is this fighter consistently a distance striker?
* Is this fighter consistently a pace fighter?
* Does this fighter abandon their identity under pressure?
* Does this fighter disrupt the opponent’s identity?

---

## Latent State: Preferred Fight Graph

### Question

What sequence of fight phases does this fighter tend to create?

### Observable Signals

* distance-to-clinch transitions
* clinch-to-ground transitions
* ground-control-to-ground-strike sequences
* striking-to-wrestling shifts
* repeated phase sequences across fights

### Candidate Feature Concepts

* `rfs_identity_distance_first_rate`
* `rfs_identity_wrestling_chain_rate`
* `rfs_identity_control_damage_sequence_rate`
* `rfs_identity_phase_sequence_consistency`

---

## Latent State: Style Persistence State

### Question

Does the fighter maintain their preferred identity across opponents?

### Observable Signals

* phase mix consistency
* output profile consistency
* wrestling frequency consistency
* control usage consistency
* distance striking reliance consistency

### Candidate Feature Concepts

* `rfs_identity_phase_mix_stability`
* `rfs_identity_output_profile_stability`
* `rfs_identity_wrestling_usage_stability`
* `rfs_identity_control_usage_stability`

---

## Latent State: Identity Disruption State

### Question

Does the fighter lose their normal identity under pressure?

### Observable Signals

* deviation from normal phase mix
* reduced preferred-phase usage
* forced defensive wrestling
* forced low-volume striking
* loss of usual pressure pattern

### Candidate Feature Concepts

* `rfs_identity_self_disruption_rate`
* `rfs_identity_preferred_phase_loss`
* `rfs_identity_pressure_disruption_score`

---

## Latent State: Tempo State

### Question

What pace environment does this fighter create?

### Observable Signals

* combined fight output
* strike attempt pace
* takedown attempt pace
* control-heavy pace suppression
* round-to-round pace changes
* opponent pace deviation

### Candidate Feature Concepts

* `rfs_identity_combined_pace`
* `rfs_identity_sig_attempt_pace`
* `rfs_identity_td_attempt_pace`
* `rfs_identity_control_pace_suppression`
* `rfs_identity_opp_pace_disruption`

---

# Priority Feature Family Roadmap

## P0 — Build First

P0 families should be built before all others because they are high-signal, interpretable, and directly supported by current round-level data.

### 1. Round Trajectory and Decay

Purpose:

Model whether a fighter improves, sustains, or deteriorates as rounds progress.

Primary domains:

* Sustainability and Resilience
* Tactical Intelligence

Core concepts:

* output slope
* accuracy slope
* defensive deterioration
* late-round output ratio
* cardio decay

---

### 2. Opponent Suppression and Phase Imposition

Purpose:

Model how much worse opponents perform against this fighter compared with their own prior baseline.

Primary domains:

* Phase Control
* Fight Identity

Core concepts:

* opponent output suppression
* opponent accuracy suppression
* opponent takedown suppression
* opponent control suppression
* opponent phase-mix disruption

This is one of the most important families in the architecture.

---

### 3. Wrestling Control Conversion

Purpose:

Separate empty wrestling activity from meaningful control and damage conversion.

Primary domains:

* Conversion and Enforcement
* Phase Control

Core concepts:

* control per takedown attempt
* control per takedown landed
* ground strikes per control minute
* submission attempts per control minute
* opponent output suppression after control

---

## P1 — Build Second

P1 families should be built only after P0 families are validated through ablation, SHAP, calibration, ROI, and CLV review.

### 4. Defensive Degradation

Purpose:

Model whether a fighter becomes easier to hit or control as the fight progresses.

Primary concepts:

* opponent accuracy allowed slope
* head strikes absorbed slope
* late-round damage allowed
* late takedown defense decay

---

### 5. Adversity Response

Purpose:

Model how a fighter responds after losing rounds, absorbing damage, being controlled, or failing takedowns.

Primary concepts:

* output after lost round
* output after knockdown
* defense after high-damage round
* tactical switch after adversity

---

### 6. Tactical Adaptation Proxies

Purpose:

Model whether a fighter changes phase or strategy after poor results.

Primary concepts:

* wrestling increase after losing striking
* striking increase after failed grappling
* accuracy rebound
* defensive rebound
* phase shift payoff

---

### 7. Finishing Snowball / Dominant-Round Conversion

Purpose:

Model whether a fighter builds on damaging or dominant moments.

Primary concepts:

* follow-up output after knockdown
* output after dominant round
* ground damage after control
* late finishing pressure

---

## P2 — Future Data Required

P2 families may require richer data than UFCStats round tables currently provide.

These should not be prioritized until data quality and sourcing are solved.

### 8. Spatial Ringcraft

Potential future signals:

* cage position
* center control
* lateral movement
* back-to-cage frequency

---

### 9. Cage Cutting

Potential future signals:

* opponent retreat patterns
* fence engagement frequency
* forced clinch entries
* reduced opponent movement space

---

### 10. Strike-to-Shot Sequencing

Potential future signals:

* strike combinations before takedown attempts
* level-change setups
* feint-to-entry sequences
* reactive shots after opponent strikes

---

### 11. True Tempo / Rhythm

Potential future signals:

* in-round burst timing
* lull patterns
* pace changes inside rounds
* rhythm disruption

---

### 12. In-Round Momentum

Potential future signals:

* minute-by-minute momentum
* damage bursts
* control swings
* late-round surges
* immediate post-event reactions

---

# Feature Naming Convention

All Round Fighter State features should use the prefix:

```text
rfs_
```

Recommended family prefixes:

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

Training and live model views should convert fighter-level state features into side-aware features:

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

The raw fighter-state store should remain fighter-centric.

The model feature view should be side-aware.

---

# Artifact Flow

## Raw Input

```text
data/fight_details/ufc_round_stats.parquet
```

Contains validated round-level observations.

This is not a model input.

---

## Historical State Output

```text
data/features/round_fighter_state_history.parquet
```

Contains fighter-state rows over time.

Expected grain:

```text
one row per fighter per completed fight
```

Used for:

* historical training feature generation
* state audits
* fighter development tracking
* feature-family validation

---

## Latest State Output

```text
data/features/round_latest_fighter_state.parquet
```

Contains one row per fighter with latest known Round Fighter State.

Used for:

* live prediction feature building
* upcoming card prediction
* dashboard display
* feature completeness audits

---

## Training Feature View Integration

Training flow:

```text
data/fight_details/ufc_round_stats.parquet
        ↓
data/features/round_fighter_state_history.parquet
        ↓
point-in-time shifted fighter state
        ↓
moneyline training feature view
        ↓
model experiment
```

Round Fighter State should be added as a feature store, not as a modification to `ufc_master.parquet`.

---

## Live Prediction Integration

Prediction flow:

```text
data/features/latest_fighter_state.parquet
        +
data/features/round_latest_fighter_state.parquet
        ↓
live feature builder
        ↓
prediction artifact
        ↓
action board / CLV tracking
```

Round Fighter State must not replace the existing fighter feature store.

It should augment it.

---

# Validation Requirements

Round Fighter State development must include validation at each layer.

## Raw Round Stats Validation

Already expected:

* required columns present
* no duplicate fighter/fight/round/corner keys
* no null fighter IDs
* no null fight IDs
* numeric columns are numeric
* landed values do not exceed attempted values
* control seconds are valid
* corners are valid
* point-in-time safety is preserved

---

## State Feature Validation

Required checks:

* one row per fighter per completed fight in history
* one row per fighter in latest state
* no duplicate fighter/fight rows
* no target-fight leakage
* no future dates included in training state rows
* no infinite values
* missing values handled intentionally
* feature names follow `rfs_` convention
* feature family metadata is available
* feature provenance is traceable to round stats

---

## Model View Validation

Required checks:

* side-aware columns created correctly
* red and blue fighter states matched correctly
* diff columns computed consistently
* no accidental use of target fight round stats
* feature completeness tracked
* missing fighter state audited
* feature families can be toggled for ablation

---

# Evaluation Requirements

No Round Fighter State feature family should be promoted permanently without evaluation.

Each new family should be tested with:

* baseline model comparison
* family-only ablation
* family-added ablation
* log loss
* ROC-AUC
* accuracy
* calibration
* Brier score if available
* SHAP contribution
* feature stability
* ROI impact
* CLV impact
* live feature completeness

A feature family may be rejected even if it improves accuracy but damages calibration, ROI, or CLV.

Model quality should be judged by betting usefulness, not only classification metrics.

---

# Development Rules

## Rule 1 — Do Not Modify Production First

Round Fighter State development should begin as a separate feature store and experiment path.

It should not alter production prediction behavior until validated.

---

## Rule 2 — Build One Family at a Time

Do not build all ontology families at once.

Recommended order:

```text
P0.1 Round Trajectory and Decay
P0.2 Opponent Suppression and Phase Imposition
P0.3 Wrestling Control Conversion
P1.1 Defensive Degradation
P1.2 Adversity Response
P1.3 Tactical Adaptation Proxies
P1.4 Finishing Snowball
```

Each family should be independently validated before adding the next.

---

## Rule 3 — Prefer Interpretable Features

Avoid creating hundreds of weak, redundant features.

Prefer compact, meaningful feature families that can be explained:

* to the model developer
* in SHAP analysis
* in dashboard audits
* in betting decision review

---

## Rule 4 — Preserve Existing Contracts

Round Fighter State must not break:

* master schema
* production scraper
* historical market outcome files
* existing fighter feature store
* current live prediction pipeline
* current CLV tracking pipeline
* dashboard artifact expectations

---

## Rule 5 — Keep Raw, State, and Model Layers Separate

Do not collapse layers.

Correct separation:

```text
raw round stats
        ↓
fighter-state feature store
        ↓
side-aware model feature view
```

Incorrect:

```text
raw round stats
        ↓
model
```

---

# Non-Goals

Round Fighter State is not intended to:

* replace the master dataset
* replace the existing fighter feature store
* directly expose raw round stats to the model
* create an uncontrolled feature explosion
* build spatial or in-round tracking without proper data
* optimize only for accuracy
* bypass point-in-time safety
* modify production pipelines before validation

---

# Initial Canonical Build Scope

The first implementation pass should include only the P0 families.

## P0.1 Round Trajectory and Decay

Goal:

Measure how fighter performance changes as rounds progress.

Expected feature themes:

* output slope
* late output ratio
* accuracy slope
* defensive deterioration
* cardio decay

---

## P0.2 Opponent Suppression and Phase Imposition

Goal:

Measure how much a fighter reduces the opponent below that opponent’s prior baseline.

Expected feature themes:

* opponent output suppression
* opponent accuracy suppression
* opponent wrestling suppression
* opponent control suppression
* opponent phase-mix disruption

---

## P0.3 Wrestling Control Conversion

Goal:

Measure whether wrestling activity becomes control, damage, or submission threat.

Expected feature themes:

* control per takedown attempt
* control per takedown landed
* ground strikes per control minute
* submission attempts per control minute
* opponent output reduction after control

---

# Canonical Summary

Round Fighter State is a latent-state feature architecture.

It exists to model how fighters behave over the course of fights.

The architecture is based on five parent domains:

1. Phase Control
2. Sustainability and Resilience
3. Conversion and Enforcement
4. Tactical Intelligence
5. Fight Identity

The first build should focus only on:

1. Round Trajectory and Decay
2. Opponent Suppression and Phase Imposition
3. Wrestling Control Conversion

Round statistics are raw observations.

Round Fighter State is the feature layer.

The model should consume point-in-time fighter states, not raw round rows.

Every feature must be interpretable, validated, and justified before becoming part of the production model pipeline.
