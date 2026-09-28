# RFS Monte Carlo V1 Decision Log

This file records architecture decisions for RFS Monte Carlo V1.

## Decision Template

```text
Decision ID:
Date:
Phase:
Question:
Options considered:
Decision:
Reason:
Contracts affected:
Artifacts affected:
Validation required:
Approved by:
```

---

## RFS-MC-001 — Build a Separate V1 Engine

- Date: 2026-08-05
- Phase: Phase 0
- Question: Should the existing simulator be rewritten or should V1 be isolated?
- Options considered:
  - rewrite the existing simulator
  - extend the existing component-provider engine
  - build a new isolated V1 package
- Decision: Build a new package under `pipeline/simulation/rfs_mc_v1/`.
- Reason: The existing simulator remains useful as a frozen comparison baseline.
  Isolation prevents accidental production or research regression and permits a
  staged rebuild.
- Contracts affected: Future V1 contracts only.
- Artifacts affected: Future V1 model-lab artifacts only.
- Validation required: Existing simulator tests must remain unchanged and pass.
- Approved by: Chris Esau and architecture review.

## RFS-MC-002 — Use 30-Second Segments

- Date: 2026-08-05
- Phase: Phase 0
- Question: What simulation time resolution should V1 use?
- Options considered:
  - full-fight simulation
  - round-level simulation
  - 30-second segments
  - individual-event simulation
- Decision: Use ten 30-second segments per standard five-minute round.
- Reason: Segment resolution supports natural finish timing and incremental state
  updates without pretending exact historical event sequences are available.
- Contracts affected: Future segment and state contracts.
- Artifacts affected: Future segment-level simulation outputs.
- Validation required: Segment totals must aggregate to realistic held-out round
  distributions.
- Approved by: Chris Esau and architecture review.

## RFS-MC-003 — Separate Historical and Dynamic State

- Date: 2026-08-05
- Phase: Phase 0
- Question: Should RFS features be directly overwritten during simulation?
- Options considered:
  - modify RFS values in place
  - use one combined state object
  - separate fixed pre-fight profiles from dynamic path state
- Decision: Keep historical `FighterSimulationProfile` values fixed and maintain
  a separate `DynamicFighterState` for each simulation path.
- Reason: This preserves interpretability, point-in-time correctness, and
  reproducibility.
- Contracts affected: Future profile and dynamic-state contracts.
- Artifacts affected: Profile and simulation trace artifacts.
- Validation required: No target-fight observations may enter either pre-fight
  profile.
- Approved by: Chris Esau and architecture review.

## RFS-MC-004 — Use Probabilistic Competing Finish Hazards

- Date: 2026-08-05
- Phase: Phase 0
- Question: Should output exceeding absorption force a finish?
- Options considered:
  - deterministic threshold
  - separate independent finish checks
  - competing probabilistic hazards
- Decision: Use mutually exclusive KO/TKO, submission, and no-finish hazards.
- Reason: Offensive pressure should increase finish probability without making
  outcomes deterministic. KO/TKO and submission require different mechanics.
- Contracts affected: Future finish-engine contract.
- Artifacts affected: Finish calibration and simulation summaries.
- Validation required: Method, timing, and conditional finish calibration.
- Approved by: Chris Esau and architecture review.

## RFS-MC-005 — Keep Durability Concepts Separate

- Date: 2026-08-05
- Phase: Phase 0
- Question: Should defense, chin, durability, and recovery be one parameter?
- Options considered:
  - one absorption rating
  - two offense/defense ratings
  - separate latent concepts
- Decision: Separate defensive avoidance, damage susceptibility, chin risk,
  durability, recovery, and defensive deterioration.
- Reason: Fighters can be hittable but durable, evasive but fragile, or recover
  differently from similar damage.
- Contracts affected: Future profile and dynamic-state contracts.
- Artifacts affected: Profile calibration artifacts.
- Validation required: Each parameter must map to distinct observable proxies and
  show stable calibration.
- Approved by: Chris Esau and architecture review.

## RFS-MC-006 — Use Experienced Fights for Primary Selection

- Date: 2026-08-05
- Phase: Phase 0
- Question: Which fights should select or reject V1?
- Options considered:
  - all fights
  - fighters with any prior history
  - both fighters with at least three prior fights
- Decision: Primary model selection uses fights where both fighters have at least
  three completed prior UFC fights.
- Reason: Prior analysis showed that RFS and simulator comparisons are strongly
  affected by sparse fighter history.
- Contracts affected: Evaluation and cohort contracts.
- Artifacts affected: Historical replay reports.
- Validation required: Low-experience fights must still be reported separately.
- Approved by: Chris Esau and architecture review.

## RFS-MC-007 — Calibrate Bottom-Up

- Date: 2026-08-05
- Phase: Phase 0
- Question: Should V1 be tuned directly for winner accuracy?
- Options considered:
  - optimize winner accuracy first
  - optimize all metrics jointly from the start
  - calibrate mechanics families in sequence
- Decision: Calibrate exposure, activity, conversion, trajectory, wrestling,
  damage, finish, scoring, and finally end-to-end outcomes.
- Reason: A simulator can match winner accuracy for unrealistic reasons.
- Contracts affected: Calibration registry and phase gates.
- Artifacts affected: Calibration runs and evaluation reports.
- Validation required: Lower-level mechanics gates must pass before global tuning.
- Approved by: Chris Esau and architecture review.

## RFS-MC-008 — Keep V1 Shadow-Only

- Date: 2026-08-05
- Phase: Phase 0
- Question: When can V1 affect production predictions or wagering?
- Options considered:
  - integrate as components become available
  - integrate after one holdout
  - remain shadow-only through multi-year evaluation
- Decision: V1 remains shadow-only until a separate promotion review.
- Reason: The new engine changes core simulation mechanics and requires extensive
  calibration and completed-card monitoring.
- Contracts affected: Runtime and artifact boundaries.
- Artifacts affected: V1 writes only to model-lab/shadow paths.
- Validation required: Multi-year walk-forward and shadow-card evidence.
- Approved by: Chris Esau and architecture review.

## Open Phase 0 Questions

The following require explicit review before Phase 0 is marked complete:

1. Confirm `pipeline/simulation/rfs_mc_v1/` as the permanent V1 package path.
2. Confirm 30-second segments as the permanent V1 resolution.
3. Confirm the primary experienced cohort threshold of three prior fights per
   fighter.
4. Confirm the proposed model-lab artifact root:
   `data/model_lab/simulation/rfs_mc_v1/`.
5. Confirm that Phase 1 performs an audit only and does not modify RFS builders.
