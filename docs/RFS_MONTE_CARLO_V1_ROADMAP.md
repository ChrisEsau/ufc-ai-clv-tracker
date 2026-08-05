# RFS Monte Carlo V1 Implementation Roadmap

## Status

- Branch: `feature/rfs-mc-v1`
- Current phase: Phase 0 — Architecture Lock
- Runtime: Shadow-only
- Existing simulator: Frozen comparison baseline
- Primary evaluation cohort: Both fighters have at least three prior fights

## Operating Rule

Only one roadmap phase may be active at a time.

A phase cannot advance until:

1. its deliverables are complete
2. its tests and audits have run
3. its gate result is documented
4. the next phase is explicitly approved

Ideas outside the active phase go into a backlog and do not interrupt the work.

---

# Phase 0 — Architecture Lock

## Goal

Freeze the V1 design before implementation.

## Deliverables

- `docs/RFS_MONTE_CARLO_V1_SPEC.md`
- `docs/RFS_MONTE_CARLO_V1_ROADMAP.md`
- `docs/RFS_MONTE_CARLO_V1_DECISION_LOG.md`
- package-boundary decision
- artifact-boundary decision
- evaluation-cohort decision
- calibration-order decision
- explicit non-goals

## Gate

No mechanics code may be implemented before these documents are reviewed and
approved.

---

# Phase 1 — Data and RFS Readiness Audit

## Goal

Determine which simulator inputs already exist and which RFS families must be
added or completed.

## Audit Areas

- round-stat coverage
- result and finish-time coverage
- trajectory and cardio coverage
- opponent suppression
- phase imposition
- wrestling pressure
- control conversion
- defensive stability
- damage conversion
- damage susceptibility
- chin and durability proxies
- recovery and adversity response
- submission pressure and escape proxies
- prior-fight and valid-state coverage
- fighter and subgroup sample size

## Deliverables

```text
data/audits/rfs_mc_v1_data_readiness.json
data/audits/rfs_mc_v1_feature_coverage.csv
docs/RFS_MONTE_CARLO_V1_RFS_GAP_ANALYSIS.md
```

## Gate

The exact RFS families required for V1 must be approved before any RFS schema or
builder changes are made.

No simulator mechanics are built during this phase.

---

# Phase 2 — Fighter Simulation Profile Contract

## Goal

Create one leakage-safe pre-fight simulation profile per fighter and target date.

## Proposed Contract

```python
FighterSimulationProfile
```

## Profile Families

- identity and sample depth
- uncertainty and fallback provenance
- pace and cardio
- striking offense and defense
- opponent suppression
- wrestling offense and defense
- control conversion
- submission offense and defense
- damage conversion
- damage susceptibility
- chin risk
- durability
- recovery
- phase preferences and imposition

## Work

- map approved RFS features into profile parameters
- implement point-in-time historical profiles
- implement latest live profiles
- implement hierarchical shrinkage
- implement subgroup and population fallbacks
- represent uncertainty
- validate schema and ranges

## Proposed Deliverables

```text
pipeline/simulation/rfs_mc_v1/contracts.py
pipeline/simulation/rfs_mc_v1/profile_builder.py
data/model_lab/simulation/rfs_mc_v1/historical_profiles.parquet
data/model_lab/simulation/rfs_mc_v1/latest_profiles.parquet
```

## Gate

- no target leakage
- deterministic rebuild
- one profile per eligible fighter and target date
- source and fallback recorded for every parameter
- sparse fighters show stronger shrinkage and uncertainty
- profile coverage report passes

---

# Phase 3 — Segment Activity Engine

## Goal

Generate realistic 30-second activity without dynamic finishes.

## Scope

- phase selection
- strike attempts
- landed strikes
- target mix
- takedown attempts and success
- control
- ground strikes
- submission attempts
- knockdowns as event outputs

## Proposed Modules

```text
phase_model.py
event_models.py
segment_engine.py
activity_calibration.py
```

## Gate

On walk-forward validation data:

- segment totals aggregate to realistic round totals
- zero rates and overdispersion are realistic
- round-specific activity distributions are reasonable
- matchup suppression acts in the expected direction
- repeated seeds are deterministic
- activity improves over or is non-inferior to the frozen baseline

---

# Phase 4 — Dynamic State Engine

## Goal

Carry path-specific state between segments and rounds.

## Proposed State

```python
DynamicFighterState
```

## State Variables

- energy
- head, body, and leg damage
- chin integrity
- defensive stability
- recovery reserve
- confidence or urgency
- phase and control position
- submission danger
- cumulative activity
- knockdowns
- score state

## Work

- workload and energy costs
- segment and round recovery
- damage accumulation
- defensive degradation
- post-adversity behavior
- tactical urgency
- phase persistence

## Gate

The engine must reproduce held-out historical patterns for:

- pace decay
- accuracy decay
- defensive deterioration
- late wrestling persistence
- post-adversity rebound
- opponent suppression

No finish engine may be added until state trajectories are stable.

---

# Phase 5 — KO/TKO and Submission Finish Engine

## Goal

Make finishes emerge from simulated events and dynamic state.

## Work

- competing-risk finish contract
- KO/TKO pressure model
- submission pressure model
- defender resistance
- hazard normalization
- probabilistic finish sampling
- segment-level finish time
- recovery and survival behavior

## Proposed Modules

```text
damage_model.py
submission_model.py
finish_engine.py
finish_calibration.py
```

## Gate

Validation must cover:

- overall finish rate
- KO/TKO rate
- submission rate
- finish round
- finish time
- KO/TKO after knockdown
- survival after knockdown
- submission after control or submission attempt
- method calibration

Hard finish thresholds are prohibited.

---

# Phase 6 — Scoring and Decision Engine

## Goal

Determine round and fight winners when no finish occurs.

## Work

- effective-striking score
- damage and knockdown weighting
- wrestling and control weighting
- submission-pressure weighting
- round winner
- judge uncertainty
- decision winner

## Gate

- reasonable decision rate
- decision-winner accuracy
- realistic score-margin distribution
- no systematic red/blue bias
- no leakage from actual winners

---

# Phase 7 — End-to-End Calibration

## Goal

Tune latent mechanics without allowing one mechanics family to hide another
family's errors.

## Calibration Order

```text
1. exposure and normalization
2. phase and activity counts
3. accuracy and conversion
4. cardio and trajectory
5. wrestling and control
6. damage and knockdowns
7. submission danger
8. KO/TKO and submission hazards
9. scoring and decisions
10. end-to-end global calibration
```

## Work

- define versioned parameter registry
- freeze directly estimated event models
- tune latent-state coefficients
- tune finish coefficients
- tune scoring coefficients
- run historical simulations
- record every parameter set and seed

## Proposed Deliverables

```text
calibration_registry.yaml
calibration_runs.parquet
selected_parameters.json
calibration_summary.md
```

## Gate

Every lower-level mechanics gate must pass before end-to-end outcome metrics are
used to select a parameter set.

---

# Phase 8 — Walk-Forward Historical Evaluation

## Goal

Compare RFS Monte Carlo V1 with current baselines.

## Primary Cohort

Both fighters have at least three prior fights.

## Secondary Cohort

At least one fighter has fewer than three prior fights.

## Comparators

- frozen heuristic simulator
- current survival-finish provider configuration
- historical population baseline
- RFS Monte Carlo V1

## Required Metrics

- winner Brier score
- winner log loss
- winner accuracy as secondary
- method log loss and accuracy
- goes-distance Brier and accuracy
- finish-time MAE and bias
- significant-strike attempt MAE
- takedown-attempt MAE
- control-time MAE
- knockdown calibration
- calibration curves
- subgroup metrics
- paired bootstrap intervals

## Promotion Rule

Do not promote from one favorable point estimate.

V1 must:

1. improve or remain non-inferior on winner probability quality
2. improve or remain non-inferior on method probability quality
3. improve path realism
4. avoid material distance and timing degradation
5. avoid severe subgroup failures
6. pass leakage and reproducibility audits

Mixed results keep V1 as a research challenger.

---

# Phase 9 — Future-Card Shadow Runner

## Goal

Run V1 on upcoming fights without affecting production.

## Work

- build latest fighter profiles
- handle low-experience fighters
- generate simulation summaries
- store versions, seeds, and parameter identifiers
- log uncertainty and fallback usage
- compare with production predictions

## Proposed Deliverables

```text
run_rfs_mc_v1.py
latest_rfs_mc_v1_predictions.parquet
latest_rfs_mc_v1_summary.json
```

## Gate

- no production writes
- no wagering decisions
- full profile provenance
- deterministic reruns
- explicit low-experience flags
- completed-card shadow monitoring

---

# Phase 10 — Promotion Review

Promotion requires a separate explicit decision after:

- multi-year walk-forward evidence
- completed-card shadow monitoring
- calibration review
- ROI and CLV evaluation
- schema review
- production integration plan
- rollback plan

Promotion is not automatic.

---

# Engineering Rules

1. Review before modification.
2. Explain architecture impact before implementing.
3. Work only on the active roadmap phase.
4. Do not skip phase gates.
5. Preserve the existing simulator.
6. Keep V1 shadow-only.
7. Do not change production paths or schemas without approval.
8. Enforce point-in-time correctness.
9. Pass explicit random generators into stochastic functions.
10. Record versions, parameters, seeds, and source artifacts.
11. Keep mechanics functions separate from I/O.
12. Do not create fighter-specific exceptions.
13. Do not tune on locked holdouts.
14. Update the decision log when architecture changes.
15. Stop progression when a phase gate fails.

# Anti-Side-Tracking Rules

Out of scope until their roadmap phase:

- production integration
- wagering and market odds
- parlays
- live scraping
- user-interface work
- neural networks
- reinforcement learning
- exact cage coordinates
- individual strike animation
- referee-specific models
- judge-specific models
- unrelated RFS expansion
- ROI optimization before probability calibration

# Phase Handoff Template

```text
RFS MONTE CARLO V1 HANDOFF

Repository:
Branch:
Commit:
Pull request:

Current phase:
Phase objective:

Approved scope:
- ...

Explicitly out of scope:
- ...

Completed:
- ...

Files created or modified:
- ...

Artifacts generated:
- ...

Tests executed:
- ...

Validation results:
- ...

Known failures or risks:
- ...

Decisions made:
- ...

Decision-log entries:
- ...

Current gate status:
PASS / FAIL / NOT RUN

Exact next action:
- ...

Do not proceed to:
- ...

Commands to reproduce:
- ...
```
