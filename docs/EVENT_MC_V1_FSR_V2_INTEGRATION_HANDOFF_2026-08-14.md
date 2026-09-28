# EVENT MC V1 — FSR V2 Structural Refactor / Branch Migration Handoff

**Date:** 2026-08-14  
**Repository:** `ChrisEsau/ufc-ai-clv-tracker`  
**TARGET / ACTIVE BRANCH:** `feature/fsr-v2`  
**SOURCE BRANCH FOR CURRENT EVENT MC CODE:** `feature/fsr-32-stamina-shadow`

---

# 0. Purpose

This document is the implementation handoff for the next chat.

The next chat owns **all remaining work** required to:

1. move the current EVENT MC V1 simulator code onto `feature/fsr-v2`;
2. preserve the completed FSR V2 work on that branch;
3. structurally refactor EVENT MC from its legacy phase model to the new two-state model;
4. integrate the validated 21-core-trait FSR V2 skill layer;
5. preserve the current damage / KD / KO / power / durability / recovery / stamina systems;
6. get complete fights running from FSR V2;
7. only after the simulator runs correctly, begin calibration and diagnostics.

This handoff supersedes older implementation sequencing where it conflicts with the decisions below.

---

# 1. Branch rule — critical

The active development branch for this work is:

```text
feature/fsr-v2
```

Do **not** continue implementation on:

```text
feature/fsr-32-stamina-shadow
```

That older branch is now the source of the current EVENT MC V1 implementation that must be ported forward.

Before making any change:

```bash
git fetch origin
git checkout feature/fsr-v2
git pull --ff-only origin feature/fsr-v2
git status
```

Confirm the working tree is clean and confirm:

```bash
git branch --show-current
```

returns:

```text
feature/fsr-v2
```

All new commits for this work go to `feature/fsr-v2`.

---

# 2. Immediate branch-migration task

The current EVENT MC simulator lives on:

```text
feature/fsr-32-stamina-shadow
```

The validated FSR V2 implementation lives on:

```text
feature/fsr-v2
```

The next chat must bring the EVENT MC code forward so the simulator and FSR V2 live together on the same branch.

## Do not blindly merge the entire old branch

Do not merge or overwrite unrelated legacy FSR/RFS work into `feature/fsr-v2`.

Instead:

1. compare the two branches;
2. identify EVENT MC-owned files and any direct simulator dependencies;
3. port only the simulator code/config/tests/docs required to preserve the current EVENT MC behavior;
4. preserve all FSR V2 files and changes already present on `feature/fsr-v2`.

Primary simulator path expected:

```text
pipeline/simulation/event_mc_v1/
```

Also inspect for associated:

- EVENT MC tests;
- EVENT MC config/defaults;
- EVENT MC runners/CLI;
- EVENT MC diagnostics;
- EVENT MC-specific docs;
- simulator-specific utility modules outside the primary directory.

Use repository inspection rather than assumptions.

A good first audit is:

```bash
git diff --name-status feature/fsr-v2..feature/fsr-32-stamina-shadow -- pipeline/simulation
```

Then inspect related paths referenced by imports.

If cherry-picking simulator-specific commits is clean and does not pull obsolete FSR code, that is acceptable. Otherwise port the necessary files directly from the source branch.

---

# 3. Preserve the completed FSR V2 product

FSR V2 core validation is complete.

Core branch:

```text
feature/fsr-v2
```

Canonical outputs:

```text
data/fsr_v2/fsr_v2_prefight_snapshots.parquet
data/fsr_v2/fsr_v2_latest.parquet
```

Core traits: **21**

1. `standing_striking_tendency`
2. `standing_striking_suppression`
3. `standing_striking_offense`
4. `standing_striking_defense`
5. `head_strike_tendency`
6. `body_strike_tendency`
7. `leg_strike_tendency`
8. `takedown_tendency`
9. `takedown_suppression`
10. `takedown_offense`
11. `takedown_defense`
12. `escape_offense`
13. `escape_defense`
14. `ground_striking_tendency`
15. `ground_striking_suppression`
16. `ground_striking_offense`
17. `ground_striking_defense`
18. `submission_tendency`
19. `submission_suppression`
20. `submission_offense`
21. `submission_defense`

Experimental only:

22. `reversal_tendency`

Reversal must not be required for the core simulator to run.

---

# 4. FSR V2 status already achieved

Treat the FSR V2 work as validated/frozen unless a direct integration defect is discovered.

Final core history audit:

- 281,190 total history rows;
- 21 unique core traits;
- 13,390 fighter-fight rows;
- 13,390 rows per trait;
- date range 2010-03-21 through 2026-08-01;
- duplicate fighter-fight-trait rows: 0;
- no NaN/inf in pre/post ratings.

Snapshot audit:

- prefight rows: 13,390;
- latest rows: 2,190;
- all 21 traits present;
- no NaN;
- no inf.

Latest-state recentering bug has already been fixed.

Commit:

```text
80ea443a
Fix FSR V2 latest snapshot recentering
```

Existing regression suite after that fix:

```text
28 passed
```

Existing command:

```bash
git diff --check &&
PYTHONPATH=. python -m pytest -q tests/fsr_v2 tests/round_stats
```

Do not redesign FSR V2 during initial MC integration.

---

# 5. Small FSR publication hardening

Before freezing canonical outputs, inspect whether canonical publication still loads every `*.parquet` from the trait-history directory.

If so, harden publication so it explicitly loads only the validated core replay groups.

Core replay groups:

1. `standing_striking_tendency`
2. `standing_striking_suppression`
3. `standing_striking_effectiveness`
4. `head_body_tendency`
5. `leg_strike_tendency`
6. `takedown_tendency`
7. `takedown_suppression`
8. `takedown_effectiveness`
9. `escape_effectiveness`
10. `ground_striking_tendency`
11. `ground_striking_suppression`
12. `ground_striking_effectiveness`
13. `submission_tendency`
14. `submission_suppression`
15. `submission_effectiveness`

Canonical publication must never accidentally include experimental `reversal_tendency` merely because a history parquet exists.

This is a narrow publisher-hardening task only.

---

# 6. New EVENT MC structural ontology — locked

EVENT MC must be refactored to two persistent fight states:

```text
STANDING
├─ standing strike clock
└─ takedown clock
       ↓ successful takedown

GROUND
├─ escape clock
├─ ground strike clock
├─ submission clock
└─ reversal clock [experimental]
```

This is the structural model to implement.

## Remove the legacy three-phase dependence

The new persistent state model is **not**:

```text
DISTANCE
CLINCH
GROUND
```

There is no persistent CLINCH state in the new structure.

Do not preserve legacy phase-pressure behavior merely for compatibility.

The current continuous-time competing-clock kernel is already the correct kernel and should be preserved.

This is a **state/event model refactor**, not a scheduler rewrite.

---

# 7. Kernel behavior to preserve

The current EVENT MC already uses competing exponential clocks / rates.

Conceptually:

```text
active event rates
    ↓
sample waiting time for competing events
    ↓
earliest event fires
    ↓
resolve event
    ↓
update dynamic state
    ↓
recompute/resample active clocks
```

Preserve this architecture.

Do not replace it with fixed 30-second turns or a discrete event table.

---

# 8. STANDING state

While the fight is `STANDING`, both fighters can generate:

```text
standing strike events
takedown events
```

These clocks compete in continuous time.

## Standing strike generation

Event frequency comes from attacker `standing_striking_tendency` and opponent `standing_striking_suppression`.

Semantic direction must be:

```text
higher attacker tendency
→ more attacker standing strike events

higher defender suppression
→ fewer attacker standing strike events
```

Do not reuse legacy `distance_striking_pressure` as the primary rate trait.

## Standing strike resolution

Once a standing strike event fires:

```text
attacker standing_striking_offense
vs
defender standing_striking_defense
→ landed / missed
```

The validated FSR V2 standing-striking family is **DISTANCE significant striking only**.

---

# 9. Strike target selection

Stored traits:

```text
leg = leg_strike_tendency
head_cond = head_strike_tendency
body_cond = body_strike_tendency
```

Reconstruction:

```text
P(leg) = leg
P(head) = (1 - leg) * head_cond
P(body) = (1 - leg) * body_cond
```

Required:

```text
P(head) + P(body) + P(leg) = 1
```

Leg strikes are STANDING-only.

GROUND striking target selection should not allow leg strikes.

Do not redesign the downstream damage system during this work.

---

# 10. Takedown event

While `STANDING`:

```text
attacker takedown_tendency
+ defender takedown_suppression
→ takedown attempt clock
```

Directionality:

```text
higher attacker tendency
→ more TD attempts

higher defender suppression
→ fewer attacker TD attempts
```

When the TD event fires:

```text
attacker takedown_offense
vs
defender takedown_defense
→ success / failure
```

A failed takedown:

```text
STANDING → STANDING
```

A successful takedown:

```text
STANDING → GROUND
```

and establishes top/bottom ownership.

---

# 11. GROUND state

While `GROUND`, active events are:

```text
escape
ground strike
submission
reversal [experimental]
```

No standing-strike clock and no takedown clock should run while the fight is already in GROUND.

---

# 12. Escape clock

Escape is a direct **successful escape clock**.

Do not model invisible failed escape attempts because UFCStats does not provide escape-attempt counts.

Matchup:

```text
bottom escape_offense
vs
top escape_defense
```

Validated semantics:

```text
higher escape_offense
→ shorter/faster escape time

higher escape_defense
→ longer/slower escape time
```

Intended clock:

```text
T_escape ~ Exponential(mean μ)
```

When escape fires:

```text
GROUND → STANDING
```

Cancel ground-only clocks and restart STANDING clocks.

Do not add arbitrary explicit round-number escape multipliers.

Stored FSR ratings themselves must not be mutated.

---

# 13. Ground striking

Historical FSR V2 ground striking is **TRUE ground significant striking only**.

Clinch is not included.

Generation:

```text
attacker ground_striking_tendency
+ defender ground_striking_suppression
→ ground strike clock
```

Resolution:

```text
attacker ground_striking_offense
vs
defender ground_striking_defense
→ landed / missed
```

Do not fold clinch strikes into this FSR family.

---

# 14. Submission

While GROUND:

```text
attacker submission_tendency
+ defender submission_suppression
→ submission attempt clock
```

When an attempt event fires:

```text
attacker submission_offense
vs
defender submission_defense
→ finish / survive
```

If the submission succeeds:

```text
fight ends by SUB
```

If it fails:

```text
remain GROUND
```

---

# 15. Reversal — experimental only

Reversal is not required to get the core simulator running.

If implemented:

```text
GROUND
top/bottom ownership flips
GROUND
```

After reversal:

1. cancel the previous escape clock;
2. swap top/bottom ownership;
3. resample the escape clock for the new bottom/top matchup;
4. continue GROUND clocks.

Do not let reversal work block the initial integration milestone.

---

# 16. Core semantic rule for all event families

Keep these concepts separate.

## Event frequency

```text
tendency + opponent suppression
```

or an equivalent calibrated rate-space transform.

This controls **whether/how often the event occurs**.

## Event success

```text
offense vs defense
```

This controls **what happens after the event occurs**.

Do not combine these into a single pressure/style/composite rating.

---

# 17. Stored FSR vs dynamic path state

FSR V2 values are persistent fighter baseline traits.

They must not be mutated during a Monte Carlo path.

Correct architecture:

```text
stored FSR baseline
    ↓
dynamic state modifier
    ↓
effective runtime trait
    ↓
event rate / event resolution
```

Example:

```text
stored escape_offense
    ↓
current stamina effect
    ↓
effective escape_offense
    ↓
escape clock
```

Do not overwrite the stored FSR value.

---

# 18. Systems frozen during structural refactor

Preserve the current implementation of:

- stamina reservoir;
- stamina expenditure;
- stamina recovery;
- stamina effect on output/power;
- damage reservoir / accumulated trauma;
- striking power;
- durability;
- knockdowns;
- KD resistance;
- KO/TKO stoppages;
- damage recovery;
- finish mechanics.

Do not redesign them while performing the FSR V2 / state-model migration.

---

# 19. Legacy concepts that should not survive as required core mechanics

Do not require:

- `distance_striking_pressure`;
- `clinch_striking_pressure`;
- `ground_striking_pressure`;
- generic phase-pressure ratings;
- three-phase style preference;
- a persistent CLINCH phase;
- a fighter-level phase-tendency trait;
- legacy RFS/FSR composite pressure inputs.

Do not silently map new FSR traits back into those old semantics.

---

# 20. Why the structural refactor is necessary

The old EVENT MC audit found a major ontology problem.

Legacy `distance_striking_pressure` mixed:

1. strike activity/volume;
2. phase/style allocation.

EVENT MC then used that mixed trait both as a direct strike-rate modifier and inside style/phase transition logic.

This double-use amplified modest rating differences into large simulated volume gaps and contributed to bad winner discrimination.

Fresh-100 predictive replay with the old FSR/MC state had:

- winner accuracy: 49%;
- winner Brier: 0.3133;
- winner log loss: 0.8687;
- many wrong high-confidence predictions.

The new FSR V2 architecture eliminates that ambiguous pressure concept.

---

# 21. Porting current EVENT MC code — preserve before refactor

Before structural changes, get the current EVENT MC implementation onto `feature/fsr-v2` and prove it still passes its existing tests.

Recommended order:

## Stage A — inspect

On `feature/fsr-v2`, inspect current branch contents and compare against `feature/fsr-32-stamina-shadow`.

Identify all EVENT MC-owned files.

## Stage B — port

Bring current simulator files/config/tests forward.

Do not overwrite completed FSR V2 code.

## Stage C — regression baseline

Before structural edits, run the simulator test suite and record the baseline.

Commit the pure port separately.

Suggested commit message:

```text
Port EVENT MC V1 onto FSR V2 branch
```

---

# 22. Structural refactor implementation stages

After the clean port:

## Stage 1 — state model

Introduce/replace phase representation with:

```text
STANDING
GROUND
```

Remove CLINCH as a persistent phase from the new core path.

Preserve scheduler/kernel.

## Stage 2 — fighter input adapter

Create a clear FSR V2 fighter-input object/adapter consuming the 21 canonical traits.

Avoid scattering parquet column names throughout the simulator.

## Stage 3 — standing clocks

Wire:

- standing strike generation;
- standing strike offense/defense;
- target selection;
- TD generation;
- TD offense/defense.

## Stage 4 — ground clocks

Wire:

- escape;
- ground striking;
- submission;
- optional reversal.

## Stage 5 — preserve dynamic systems

Reconnect the existing stamina/damage/finish systems to the new event paths without changing their mechanics.

## Stage 6 — complete-fight runner

Make sure existing runner/Monte Carlo APIs can run full fights from FSR V2 prefight/latest rows.

---

# 23. Input source by use case

Historical replay:

```text
data/fsr_v2/fsr_v2_prefight_snapshots.parquet
```

Current/upcoming fight:

```text
data/fsr_v2/fsr_v2_latest.parquet
```

Historical simulations must use the appropriate historical prefight snapshot, never a current/latest fighter row.

No future leakage.

---

# 24. Required mapping tests

Add deterministic tests proving semantic direction.

## Standing striking

- higher `standing_striking_tendency` → higher standing-strike rate;
- higher opponent `standing_striking_suppression` → lower rate;
- higher offense → higher landing probability;
- higher defense → lower opponent landing probability.

## Targets

- reconstructed head/body/leg probabilities sum to 1;
- higher leg tendency increases leg selection;
- no leg target from GROUND.

## Takedowns

- higher TD tendency → more attempts;
- higher TD suppression → fewer opponent attempts;
- higher TD offense → higher completion;
- higher TD defense → lower opponent completion;
- failed TD remains STANDING;
- successful TD enters GROUND and sets controller.

## Escape

- higher bottom escape offense → faster escape;
- higher top escape defense → slower escape;
- successful escape returns to STANDING.

## Ground striking

- tendency increases ground-strike generation;
- suppression reduces opponent generation;
- offense/defense move landing probability in correct direction.

## Submission

- tendency increases attempt generation;
- suppression reduces opponent attempts;
- offense increases finish probability;
- defense reduces opponent finish probability.

## Reversal

- not required by core input schema;
- simulator runs with reversal disabled/absent.

---

# 25. Required frozen-system regression tests

The refactor must prove that the following mechanics were not intentionally changed:

- stamina cost/update functions;
- stamina recovery;
- power modulation;
- damage accumulation;
- KD calculation;
- KO/TKO calculation;
- durability;
- recovery;
- finish state behavior.

Existing tests should continue to pass.

---

# 26. Smoke-test milestone

The first major success criterion is:

> EVENT MC can simulate complete UFC fights using FSR V2 on `feature/fsr-v2`, under the STANDING/GROUND competing-clock structure, while existing stamina, damage, KD, KO, power, durability, recovery, and finish systems remain intact.

Required seeded smoke tests should include:

- striker vs striker;
- wrestler vs striker;
- grappler vs wrestler;
- high-volume vs low-volume fighter;
- strong TD offense vs strong TD defense;
- strong escape offense vs strong control/escape defense;
- submission specialist vs strong submission defense.

At minimum confirm:

- simulation terminates;
- no impossible state transitions;
- no NaN/inf event rates;
- no negative probabilities;
- no negative clocks;
- result is always valid;
- decisions still occur when no finish happens;
- full scheduled-round duration is handled.

---

# 27. Calibration comes after integration

Do **not** broad-calibrate during the structural port.

Once full fights run correctly, begin a new calibration phase.

Calibration targets should include standing strike attempts/accuracy/target mix, TD attempts/completion, ground episode duration/control, ground strike attempts/accuracy, submission attempts/conversion, KD, method shares, finish timing, and decision rate.

Do not tune damage/stamina simultaneously unless diagnostics later prove those systems are the active source of error.

---

# 28. Predictive validation comes after calibration

After the event environment is calibrated:

1. run historical fights;
2. compare winner probabilities to actual winners;
3. calculate Brier/log loss;
4. compare method probabilities;
5. compare to prefight market where appropriate;
6. inspect high-confidence misses;
7. diagnose one subsystem at a time.

Do not judge the new FSR V2 winner signal before the simulator's event environment is reasonably calibrated.

---

# 29. Continuity docs

The old EVENT MC working-memory file is:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
```

It was maintained on:

```text
feature/fsr-32-stamina-shadow
```

Port this continuity file to `feature/fsr-v2` if it is not already present.

Update its branch header to `feature/fsr-v2` and append the new FSR V2 / STANDING-GROUND integration state. Do not delete the historical calibration record.

---

# 30. Recommended commit sequence

Suggested commits:

```text
1. Port EVENT MC V1 onto FSR V2 branch
2. Harden FSR V2 canonical core publication
3. Add FSR V2 EVENT MC input adapter
4. Refactor EVENT MC to standing-ground state model
5. Wire FSR V2 standing and takedown events
6. Wire FSR V2 ground escape striking and submission events
7. Restore full-fight EVENT MC runners and regression coverage
8. Add FSR V2 EVENT MC smoke diagnostics
9. Update EVENT MC continuity documentation
```

Avoid one giant opaque commit.

---

# 31. Codex execution convention

For every Codex prompt, include near the top:

> **Begin immediately when this prompt is received. Do not ask for confirmation or approval to start the task. Execute the authorized work, tests, diagnostics, commits, and push without waiting for another user message. Only stop if a required operation is technically blocked by the environment or would exceed the explicit scope of this prompt.**

After each substantial task, update the continuity MD on `feature/fsr-v2`.

---

# 32. Do not overengineer the migration

Design priorities remain:

```text
WORKING
PREDICTIVE
MODULAR
EASY TO ITERATE
```

Prefer a clear adapter and straightforward event modules over excessive abstraction.

---

# 33. Final target architecture

```text
FSR V2 PREFIGHT FIGHTER TRAITS
            │
            ▼
   EVENT MC INPUT ADAPTER
            │
            ▼
┌─────────────────────────────┐
│       DYNAMIC PATH STATE    │
│ stamina / damage / position │
└─────────────────────────────┘
            │
            ▼
STANDING
├─ standing strike clock
│   ├─ tendency + suppression → frequency
│   └─ offense vs defense → land/miss
│
└─ takedown clock
    ├─ tendency + suppression → frequency
    └─ offense vs defense → success/fail
                              │
                       successful TD
                              ▼
GROUND
├─ escape clock
│   └─ escape offense vs escape defense
│
├─ ground strike clock
│   ├─ tendency + suppression → frequency
│   └─ offense vs defense → land/miss
│
├─ submission clock
│   ├─ tendency + suppression → frequency
│   └─ offense vs defense → finish/survive
│
└─ reversal clock [experimental]
    └─ controller swap only

LANDED STRIKES / STATE EVENTS
            │
            ▼
EXISTING FROZEN CONSEQUENCE SYSTEMS
├─ stamina
├─ power
├─ damage / trauma
├─ KD
├─ KO/TKO
├─ durability
├─ recovery
└─ judging / result
```

---

# Appendix A — FINAL FSR V2 Mathematical Trait Contract for EVENT MC

**Source of truth date:** 2026-08-14  
**Repository:** `ChrisEsau/ufc-ai-clv-tracker`  
**Active integration branch:** `feature/fsr-v2`

This appendix contains the FINAL VALIDATED simulator-facing mathematical meaning of the 21 core FSR V2 traits. Where this appendix conflicts with any earlier preliminary FSR formula in this handoff, **this appendix wins**.

The central architectural separation is:

```text
EVENT FREQUENCY
tendency + opponent suppression

EVENT SUCCESS
offense vs defense

STATE RESIDENCE
emerges from event outcomes and clocks
```

Do not recombine these into legacy pressure or phase-style ratings.

## A1. Core conventions

All FSR historical states are chronological and leakage-safe.

For a historical fight at date D, all population baselines, fighter states, and opponent states must use information strictly before D. Same-date fights use the same prefight state and update only after the date batch.

For latest/current profiles, behavior/composition traits are recentered against the final/current population baseline.

Stored FSR values are immutable baseline fighter traits. EVENT MC may derive temporary effective values from stamina or dynamic path state, but must not mutate stored ratings.

Useful functions:

```text
sigmoid(x) = 1 / (1 + exp(-x))
logit(p)   = ln(p / (1-p))
epsilon    = 1e-6
```

For retained paired O/D families:

```text
p_expected =
    sigmoid(
        logit(B)
        + offense
        - defense
    )
```

because the validated rating scale is `1.0`.

Neutral offense/defense are both 0, which returns the natural population baseline B rather than 50%.

## A2. Generic paired Elo constants

Applies to standing striking O/D, ground striking O/D, and submission O/D.

```text
observed = successes / opportunities

expected =
    sigmoid(
        logit(population_baseline)
        + offense_pre
        - defense_pre
    )

evidence_strength =
    1 - exp(-opportunities / 12)

delta =
    0.35
    × evidence_strength
    × (observed - expected)

offense_post = offense_pre + delta
defense_post = defense_pre - delta
```

Locked constants:

```text
prior_rating                 = 0.0
rating_scale                 = 1.0
elo_k                        = 0.35
evidence_saturation_attempts = 12
```

These are centered matchup-effect values, not 0-100 scores.

## A3. Behavioral rate shrinkage

For time-exposure behavioral traits:

```text
fighter_rate =
    (
        fighter_cumulative_events
        + population_rate × 900
    )
    /
    (
        fighter_cumulative_exposure
        + 900
    )
```

Locked:

```text
behavior_prior_seconds = 900
```

Stored units are events per modeled exposure second.

Per-minute display only:

```text
rate_per_minute = trait × 60
```

For a literal exponential clock:

```text
mean_wait_seconds = 1 / rate_per_second
```

EVENT MC may require explicit population calibration multipliers because modeled exposure and historical exposure are not guaranteed to be identical.

## A4. Suppression equation

For opponent j facing defender i:

```text
expected_opponent_rate = opponent prefight tendency
actual_opponent_rate   = opponent_events / relevant_exposure
residual               = expected_opponent_rate - actual_opponent_rate
```

Persistent suppression:

```text
suppression_i =
    Σ(residual × exposure)
    /
    (Σ exposure + 900)
```

Locked:

```text
suppression_prior_seconds = 900
```

Positive suppression means the fighter reduced opponent event generation.

Suppression has the same units as its tendency.

Natural raw semantic starting point:

```text
effective_rate ≈ attacker_tendency - defender_suppression
```

but the exact MC rate-space transform is not yet calibrated.

Do not use FSR validation regression beta coefficients as simulator multipliers.

## A5. Standing striking — distance only

Critical lock:

```text
standing_striking_* = DISTANCE significant striking only
```

Clinch is excluded.

### `standing_striking_tendency`

Numerator:

```text
distance significant-strike attempts
```

Exposure:

```text
standing_exposure_seconds =
    round_elapsed_seconds
    - combined UFCStats control seconds
```

Stored state is the 900-second-prior shrunk cumulative rate.

Units: distance attempts / standing exposure second.

### `standing_striking_suppression`

```text
expected_B_rate = B prefight standing_striking_tendency
actual_B_rate   = B distance attempts / standing exposure
residual_A      = expected_B_rate - actual_B_rate
```

Exposure-weighted and shrunk toward zero with 900 seconds.

Units: distance attempts/second.

### `standing_striking_offense` / `standing_striking_defense`

Raw observation:

```text
distance_sig_landed / distance_sig_attempted
```

Population baseline is prior-date UFC distance significant-strike accuracy.

Natural MC equation:

```text
P(distance strike lands) =
    sigmoid(
        logit(B_distance_accuracy)
        + standing_striking_offense_attacker
        - standing_striking_defense_defender
    )
```

## A6. Target composition — final hierarchical system

The old `LEG / DISTANCE` formula is invalid and must never be restored.

Define:

```text
H = head significant-strike attempts
B = body significant-strike attempts
L = leg significant-strike attempts
T = H + B + L
```

Locked prior:

```text
target_composition_prior_attempts = 200
```

### `leg_strike_tendency`

```text
leg =
    (
        fighter_L
        + population_leg_rate × 200
    )
    /
    (
        fighter_T
        + 200
    )
```

### `head_strike_tendency`

Conditional on non-leg target:

```text
head_cond = H / (H + B)
```

shrunk with 200-attempt prior.

### `body_strike_tendency`

```text
body_cond = B / (H + B)
body_cond = 1 - head_cond
```

### EVENT MC reconstruction

```text
P(LEG)  = leg
P(HEAD) = (1 - leg) × head_cond
P(BODY) = (1 - leg) × body_cond
```

Invariant:

```text
P(HEAD) + P(BODY) + P(LEG) = 1
```

Leg targets are STANDING-only. Ground target selection must not generate leg strikes.

## A7. Takedown tendency and suppression — final exposure

For fighter A:

```text
TD_OPPORTUNITY_A =
    round_elapsed_seconds
    - opponent_control_seconds
```

For fighter A suppressing opponent B:

```text
TD_SUPPRESSION_EXPOSURE_A =
    round_elapsed_seconds
    - fighter_A_control_seconds
```

No artificial floor.

### `takedown_tendency`

```text
TD attempts / TD opportunity seconds
```

with 900-second rate prior.

### `takedown_suppression`

Opponent prefight expected TD attempt rate minus actual opponent TD attempt rate, using corresponding opportunity exposure.

Positive = fewer opponent attempts than expected.

## A8. Takedown offense / defense — final career formulation

Takedown O/D did **not** retain Elo.

Locked prior:

```text
TD_EFFECTIVENESS_PRIOR_ATTEMPTS = 10
```

Let:

```text
pTD   = prior-date population TD completion probability
pSTOP = 1 - pTD
```

### `takedown_offense`

```text
shrunk_TD_accuracy =
    (
        fighter_TD_landed
        + pTD × 10
    )
    /
    (
        fighter_TD_attempted
        + 10
    )

takedown_offense =
    logit(shrunk_TD_accuracy)
    - logit(pTD)
```

### `takedown_defense`

```text
TD_stopped = opp_TD_attempted - opp_TD_landed

shrunk_TD_stop_rate =
    (
        TD_stopped
        + pSTOP × 10
    )
    /
    (
        opp_TD_attempted
        + 10
    )

takedown_defense =
    logit(shrunk_TD_stop_rate)
    - logit(pSTOP)
```

Natural MC equation:

```text
P(TD success) =
    sigmoid(
        logit(pTD)
        + takedown_offense_attacker
        - takedown_defense_defender
    )
```

## A9. Ground-entry qualification for escape

A qualified ground/control entry exists if actual TD landed > 0, OR a zero-TD inferred entry where:

```text
control_seconds >= 5
AND independent true-ground evidence exists
```

Locked threshold:

```text
zero_td_control_threshold_seconds = 5
```

Independent ground evidence includes true ground significant strikes, submission attempts, reversal evidence, or relevant true-ground evidence from either fighter.

Validation produced 20,827 qualified entries.

## A10. Escape offense / defense — final equations

There is no `escape_tendency` and no `escape_suppression`.

Locked prior:

```text
ESCAPE_PRIOR_ENTRIES = 5
```

Earlier value 3 is obsolete.

Let:

```text
μ_pop = leakage-safe population mean qualified ground-control duration per qualified entry
```

### `escape_offense`

```text
μ_suffered =
    (
        fighter_suffered_duration
        + μ_pop × 5
    )
    /
    (
        fighter_suffered_entries
        + 5
    )

escape_offense = ln(μ_pop / μ_suffered)
```

Positive = faster-than-population escape.

### `escape_defense`

```text
μ_inflicted =
    (
        fighter_inflicted_duration
        + μ_pop × 5
    )
    /
    (
        fighter_inflicted_entries
        + 5
    )

escape_defense = ln(μ_inflicted / μ_pop)
```

Positive = longer-than-population control.

### Escape clock

```text
T_escape ~ Exponential(mean μ_matchup)
```

Directionality:

```text
higher bottom escape_offense
→ lower μ_matchup
→ faster escape

higher top escape_defense
→ higher μ_matchup
→ slower escape
```

Successful escape:

```text
GROUND → STANDING
```

Do not add an explicit round-number escape multiplier.

Exact mapping coefficients are not yet locked. Natural calibration form:

```text
log μ_matchup =
    log μ_population
    - β_escape × escape_offense_bottom
    + β_control × escape_defense_top
```

but `β_escape` and `β_control` must be historically calibrated.

## A11. Qualified ground exposure

Earlier all-control / ground+clinch formulations are rejected.

Ground exposure is qualified only when actual ground evidence exists, including:

- TD landed;
- true ground significant-strike attempt;
- submission attempt;
- reversal.

Use combined capped UFCStats control only when ground-qualified.

If explicit true-ground activity exists but recorded control is zero:

```text
zero_control_ground_fallback_seconds = 5
```

Validation:

```text
raw control seconds       = 3,455,672
qualified ground exposure = 3,170,794
removed                   = 284,878 sec
removed share             = 8.24%
```

## A12. Ground striking — true ground only

Critical lock:

```text
ground_striking_* = TRUE GROUND significant striking only
```

Clinch is excluded.

### `ground_striking_tendency`

```text
ground_sig_attempted
/
qualified_ground_exposure_seconds
```

with 900-second behavior prior.

### `ground_striking_suppression`

```text
expected_B_ground_rate = B prefight ground_striking_tendency
actual_B_ground_rate   = B true-ground attempts / qualified ground exposure
residual_A             = expected - actual
```

Exposure-weighted, shrunk toward zero with 900-second prior.

### `ground_striking_offense` / `ground_striking_defense`

Raw:

```text
true_ground_sig_landed / true_ground_sig_attempted
```

Natural MC equation:

```text
P(ground strike lands) =
    sigmoid(
        logit(B_ground_accuracy)
        + ground_striking_offense_attacker
        - ground_striking_defense_defender
    )
```

## A13. Submission effective attempts

Historical effective submission opportunities:

```text
effective_submission_attempts =
    max(
        recorded_SUB_ATT,
        submission_finish_indicator
    )
```

where submission finish indicator is 1 if fighter won by submission, else 0.

## A14. Submission tendency

```text
effective_submission_attempts
/
qualified_ground_exposure_seconds
```

with 900-second behavior prior.

Units: submission attempts per qualified ground second.

## A15. Submission suppression

```text
expected_B_submission_rate = B prefight submission_tendency
actual_B_submission_rate   = B effective submission attempts / qualified ground exposure
residual_A                 = expected - actual
```

Exposure-weighted and shrunk toward zero with 900-second prior.

## A16. Submission offense / defense

Opportunity count:

```text
effective_submission_attempts
```

Success numerator:

```text
submission_finish_indicator
```

Observed conversion:

```text
submission_finish_indicator / effective_submission_attempts
```

when effective attempts > 0.

Population baseline is prior-date UFC submission finishes per effective submission attempt.

Natural MC equation:

```text
P(SUB finish | generated attempt) =
    sigmoid(
        logit(B_submission_conversion)
        + submission_offense_attacker
        - submission_defense_defender
    )
```

Keep submission tendency separate from submission offense.

## A17. Reversal — experimental only

If retained:

```text
reversal_tendency = successful reversals / qualified ground exposure
```

No `reversal_suppression`, `reversal_offense`, or `reversal_defense`.

Core simulator must run without reversal.

If reversal fires:

```text
GROUND
→ swap controller / controlled fighter
→ remain GROUND
→ cancel and resample escape clock
```

## A18. Final trait-scale summary

Rate/probability-like traits:

```text
standing_striking_tendency      rate per standing second
standing_striking_suppression   rate reduction per standing second
head_strike_tendency            conditional probability
body_strike_tendency            conditional probability
leg_strike_tendency             unconditional target probability
takedown_tendency               TD attempts per opportunity second
takedown_suppression            TD attempt-rate suppression
ground_striking_tendency        true-ground attempts per ground second
ground_striking_suppression     ground attempt-rate suppression
submission_tendency             SUB attempts per ground second
submission_suppression          SUB attempt-rate suppression
```

Centered logit matchup traits:

```text
standing_striking_offense
standing_striking_defense
ground_striking_offense
ground_striking_defense
submission_offense
submission_defense
```

Neutral = 0. Higher = better. Matchup enters success logit as `+ offense - defense`.

Takedown centered logit traits:

```text
takedown_offense
takedown_defense
```

Neutral = 0; constructed from cumulative career state with 10-attempt prior.

Log-duration traits:

```text
escape_offense = ln(pop_duration / fighter_suffered_duration)
escape_defense = ln(fighter_inflicted_duration / pop_duration)
```

Positive offense → faster escape. Positive defense → longer control.

## A19. Final simulator-relevant constant table

```text
GENERIC PAIRED ELO
prior_rating                         = 0.0
rating_scale                         = 1.0
elo_k                                = 0.35
evidence_saturation_attempts         = 12

RATE TRAIT SHRINKAGE
behavior_prior_seconds               = 900 sec
suppression_prior_seconds            = 900 sec

TARGET COMPOSITION
target_composition_prior_attempts    = 200 attempts

TAKEDOWN EFFECTIVENESS
takedown_effectiveness_prior         = 10 attempts

ESCAPE
escape_prior_entries                 = 5 entries
zero_td_control_threshold_seconds    = 5 sec

GROUND EXPOSURE FALLBACK
zero_control_ground_fallback_seconds = 5 sec

ROUND PHYSICAL MAXIMUM
maximum_round_seconds                = 300 sec

RATE DISPLAY CONVERSION
rate_seconds                         = 60 sec

LOGIT NUMERICAL CLIP
epsilon                              = 1e-6
```

## A20. Old constants that must not be copied blindly

Do not use legacy FSR V1 / MC population constants simply because they already exist.

Examples not to treat as permanent V2 constants:

```text
distance_attempts_per_round = 30.976531
distance_accuracy           = 0.401885
TD attempts/round           = 1.201188
TD completion               = 0.364353
TD scale                    = 1.75
submission hazard           = 0.12
```

EVENT MC must calculate/freeze new population baselines from the FSR V2-era historical calibration cohort.

## A21. Suppression regression coefficients are not simulator constants

FSR validation regressions showed suppression adds predictive information. Example coefficients such as standing suppression beta around 4.2 are **not** authorized as MC multipliers.

Do not implement `effective rate = tendency - 4.2 × suppression` unless EVENT MC calibration specifically re-estimates and validates such a transform.

## A22. Clinch — critical final rule

Clinch is excluded from both `standing_striking_*` and `ground_striking_*`.

FSR standing striking = distance only.

FSR ground striking = true ground only.

No persistent CLINCH state is required in the new EVENT MC.

## A23. Two-state EVENT MC structure

Persistent states:

```text
STANDING
GROUND
```

STANDING active clocks:

```text
distance strike
takedown
```

GROUND active clocks:

```text
escape
ground strike
submission
reversal [optional experimental]
```

Transitions:

```text
successful TD:     STANDING → GROUND
failed TD:         STANDING → STANDING
successful escape: GROUND → STANDING
failed submission: GROUND → GROUND
successful SUB:    FIGHT TERMINATES
reversal:          GROUND → GROUND, controller swaps
```

## A24. Continuous-time kernel

Preserve competing exponential clocks.

If active rates are `λ1 ... λn`:

```text
Λ = Σ λi
T ~ Exponential(rate = Λ)
mean(T) = 1 / Λ
P(event i fires next) = λi / Λ
```

After event resolution, update dynamic state, determine active events, recompute rates, and resample clocks.

Do not replace this with fixed 30-second segments.

## A25. Quantities EVENT MC can compute directly

Targets:

```text
P_leg  = leg
P_head = (1-leg) × head_cond
P_body = (1-leg) × body_cond
```

Standing landing:

```text
P_land =
    sigmoid(
        logit(B_standing_accuracy)
        + standing_O
        - standing_D
    )
```

TD completion:

```text
P_TD =
    sigmoid(
        logit(B_TD_completion)
        + TD_O
        - TD_D
    )
```

Ground landing:

```text
P_ground_land =
    sigmoid(
        logit(B_ground_accuracy)
        + ground_O
        - ground_D
    )
```

Submission finish per generated attempt:

```text
P_SUB_finish =
    sigmoid(
        logit(B_SUB_conversion)
        + SUB_O
        - SUB_D
    )
```

Event-generation semantics:

```text
higher tendency → higher event rate
higher defender suppression → lower attacker event rate
```

Natural raw starting point:

```text
λ_raw = tendency - opponent_suppression
```

subject to positivity and eventual calibration.

Escape:

```text
T_escape ~ Exponential(mean μ_matchup)
∂μ / ∂escape_offense_bottom < 0
∂μ / ∂escape_defense_top  > 0
```

## A26. What still requires EVENT MC calibration

FSR V2 validation does **not** finish MC calibration.

The simulator still needs to calibrate:

1. standing strike event-rate mapping;
2. tendency/suppression clock-space transform;
3. standing population landing baseline;
4. TD attempt event-rate mapping;
5. TD population completion baseline;
6. ground strike event-rate mapping;
7. ground population landing baseline;
8. submission attempt event-rate mapping;
9. submission finish baseline per generated attempt;
10. escape population duration baseline;
11. escape offense coefficient;
12. escape defense coefficient;
13. any global rate multipliers needed because historical exposure and simulated state time are not perfectly identical.

Do not modify FSR equations to fix MC calibration errors.

## A27. Diagnostic ownership

```text
standing attempts wrong, accuracy correct
→ inspect standing tendency/suppression rate mapping

standing attempts correct, accuracy wrong
→ inspect standing offense/defense probability mapping

TD attempts wrong, completion correct
→ inspect TD tendency/suppression clock

TD attempts correct, completion wrong
→ inspect TD offense/defense conversion

TD attempts/completion correct, control duration wrong
→ inspect escape clock

ground duration correct, ground attempts wrong
→ inspect ground tendency/suppression

ground attempts correct, ground accuracy wrong
→ inspect ground offense/defense

SUB attempts wrong
→ inspect submission tendency/suppression

SUB attempts correct, finishes wrong
→ inspect submission offense/defense
```

## A28. Historical vs current FSR inputs

Historical EVENT MC validation:

```text
data/fsr_v2/fsr_v2_prefight_snapshots.parquet
```

Use exact fighter-fight prefight rows. Never use current/latest ratings for a historical fight.

Current/upcoming EVENT MC:

```text
data/fsr_v2/fsr_v2_latest.parquet
```

Latest behavior/composition ratings are already recentered against the final/current population baseline.

## A29. Final simulator locks

Locked:

- 21 core traits;
- zero-centered O/D scales;
- distance-only standing FSR;
- true-ground-only ground FSR;
- hierarchical target composition;
- target prior = 200 attempts;
- takedown effectiveness prior = 10 attempts;
- escape prior = 5 entries;
- inferred zero-TD control threshold = 5 sec;
- zero-control true-ground fallback = 5 sec;
- paired Elo K = 0.35 for retained Elo families;
- paired evidence saturation = 12 attempts;
- rating scale = 1.0;
- behavior time prior = 900 sec;
- suppression prior = 900 sec;
- effective submission attempts = max(SUB ATT, SUB finish);
- escape clock is exponential;
- no explicit round escape multiplier;
- stored FSR values never mutate during a path;
- reversal optional/experimental;
- clinch excluded from standing and ground FSR;
- tendency = event frequency;
- suppression = opponent frequency reduction;
- offense/defense = event resolution.

Not yet locked:

- exact MC population event rates;
- exact tendency/suppression rate transform;
- exact escape matchup coefficients;
- global clock calibration multipliers.

Those are EVENT MC calibration tasks, not FSR redesign tasks.

## A30. Most important warning

Do not make up a simulator equation and then modify FSR to fit it.

Correct workflow:

```text
VALIDATED FSR TRAITS
        ↓
literal semantic mapping
        ↓
complete fights run
        ↓
historical population calibration
        ↓
single-family diagnostics
        ↓
winner/method validation
```

Do not return to legacy composite pressure ratings.

Do not mix style and skill.

Do not tune all subsystems simultaneously.

---

# 34. Definition of completion for this handoff

This implementation phase is complete when all of the following are true:

1. active branch is `feature/fsr-v2`;
2. current EVENT MC code has been ported from `feature/fsr-32-stamina-shadow`;
3. FSR V2 canonical outputs remain intact;
4. only the 21 core traits are required;
5. Event MC persistent state is STANDING/GROUND;
6. continuous-time competing clocks remain intact;
7. standing strike and TD clocks operate from FSR V2;
8. ground escape/strike/submission clocks operate from FSR V2;
9. reversal remains optional/experimental;
10. stored FSR ratings are immutable within paths;
11. damage/KD/KO/power/durability/recovery mechanics are unchanged;
12. stamina mechanics are unchanged;
13. seeded full fights complete successfully;
14. directionality tests pass;
15. legacy EVENT MC regressions remain green or are updated only for intentional state-ontology changes;
16. continuity documentation has been moved/updated on `feature/fsr-v2`;
17. all work is committed and pushed to `feature/fsr-v2`.

The next phase after this is **EVENT MC calibration**, not another FSR redesign.
