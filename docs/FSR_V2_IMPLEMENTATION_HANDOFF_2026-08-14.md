# FSR V2 Implementation Handoff and Trait Specification

**Date:** 2026-08-14  
**Repository:** `ChrisEsau/ufc-ai-clv-tracker`  
**Working branch:** `feature/fsr-v2`  
**Base checkpoint:** `feature/fsr-32-stamina-shadow`  
**Project:** UFC Model Pipeline / EVENT MC V1  
**Purpose:** Implementation handoff for a new modular FSR V2, built for eventual use by EVENT MC V1.

---

## 0. Authorization / execution boundary

This document defines the **FSR V2 implementation and validation work**.

**Begin immediately when this prompt is received. Do not ask for confirmation or approval to start the task. Execute the authorized work, tests, diagnostics, commits, and push without waiting for another user message. Only stop if a required operation is technically blocked by the environment or would exceed the explicit scope of this prompt.**

### Authorized now

1. Audit the actual source schemas.
2. Implement the modular FSR V2 build system.
3. Implement the trait definitions in this document.
4. Build chronological leakage-safe trait histories and prefight snapshots.
5. Build incremental/checkpointed replay.
6. Run extensive sanity checks and diagnostics on the new traits.
7. Produce ranked/representative outputs that can be manually checked.
8. Commit and push the FSR V2 implementation and diagnostic tooling.

### Not authorized yet

- Do **not** change EVENT MC mechanics.
- Do **not** replace current EVENT MC FSR inputs.
- Do **not** remove the current DISTANCE/CLINCH/GROUND phase implementation yet.
- Do **not** recalibrate EVENT MC clocks yet.
- Do **not** change the existing damage / KD / KO system.
- Do **not** change the existing stamina system.
- Do **not** promote FSR V2 into production.

The next gate is **FSR V2 trait sanity validation**. EVENT MC integration happens only after those traits are reviewed and approved.

---

# 1. Core design objective

FSR V2 exists to provide EVENT MC with a complete, leakage-safe, modular prefight fighter profile.

The design philosophy is:

> UFCStats provides observations. FSR V2 converts those observations into persistent fighter traits. EVENT MC consumes those traits. Historical UFC data is also used separately to validate the simulator.

FSR V2 should be:

- modular;
- easy to add/remove traits from;
- directly traceable to raw data;
- free of hidden legacy transformations;
- leakage-safe;
- opponent-adjusted where appropriate;
- resumable/incremental;
- cheap to modify one trait without replaying every unrelated trait;
- suitable for direct EVENT MC consumption later.

---

# 2. Authoritative data sources

## 2.1 Primary source

Use:

`data/fight_details/ufc_round_stats.parquet`

This remains the authoritative round-level statistical source.

Do **not** modify this parquet.

Do **not** create a persisted fighter-fight observation parquet as an additional source layer.

Trait modules should read directly from the raw round parquet and derive the observations they need in code.

## 2.2 Secondary source

The UFC master table is permitted **only when information required for a trait is unavailable in the round stats parquet**.

Examples may include:

- fight winner;
- finish method;
- actual fight elapsed time if not available in the round parquet;
- scheduled rounds;
- fight-level metadata required for a leakage-safe trait.

The schema must be audited before assuming a master-table join is required.

## 2.3 Validation use of raw data

Historical round stats and master data may always be used to compare EVENT MC outputs against reality.

Those comparison labels do **not** need to live in FSR V2 unless they are required to construct a prefight fighter trait.

---

# 3. No intermediate observation datastore

Do not persist a general:

`fighter_fight_observations.parquet`

FSR V2 should remain close to the authoritative data.

Preferred flow:

```text
ufc_round_stats.parquet
        +
UFC master when required
        ↓
trait-specific observation code
        ↓
trait-specific chronological replay
        ↓
trait history/checkpoint
        ↓
published FSR V2 prefight snapshots
```

Raw round data may be loaded once into memory during a build and passed to multiple requested trait modules.

The additional compute cost is acceptable. Data visibility and simplicity are preferred over another transformed data layer.

---

# 4. Modular replay architecture

Each trait or tightly coupled trait group must be independently rebuildable.

A change to one trait must **not** require replaying all other FSR traits.

Examples:

```text
change standing_striking_tendency
    → replay standing_striking_tendency only

change takedown_offense equation
    → replay takedown offense/defense group only

add new trait
    → replay only that trait/group through history
```

## 4.1 Recommended structure

```text
pipeline/
  fsr_v2/
    __init__.py

    sources/
      round_stats.py
      master.py

    traits/
      registry.py
      standing_striking.py
      targets.py
      takedowns.py
      escapes.py
      ground_striking.py
      submissions.py
      reversals_experimental.py
      preserved_traits.py

    replay/
      engine.py
      checkpoint.py
      dependency_graph.py
      versions.py

    publish/
      snapshots.py
      latest.py

    diagnostics/
      trait_sanity.py
      trait_rankings.py
      replay_audit.py
      source_audit.py
```

Exact paths may be adjusted to fit repository conventions, but preserve the separation of concerns.

---

# 5. Replay groups and dependencies

A trait with no coupled opponent rating can replay independently.

Paired competitive traits should replay as a small group.

Examples:

```text
standing_striking_effectiveness:
  standing_striking_offense
  standing_striking_defense

ground_striking_effectiveness:
  ground_striking_offense
  ground_striking_defense

takedown_effectiveness:
  takedown_offense
  takedown_defense

escape_effectiveness:
  escape_offense
  escape_defense

submission_effectiveness:
  submission_offense
  submission_defense
```

Changing one member invalidates that replay group, not the entire FSR V2 build.

Suppression traits depend on the corresponding opponent tendency being available prefight. Declare those dependencies explicitly.

---

# 6. Checkpoints and incremental replay

The chronological replay is expensive and must be resumable.

Persist enough state to continue from the latest processed date.

A replay-group checkpoint should contain at least:

- fighter ID;
- current rating/state;
- meaningful update/exposure count;
- current population baseline state required by that trait;
- last processed event date;
- trait/group version fingerprint;
- source schema/version fingerprint;
- dependency versions.

## Routine new UFC data

When new fight data arrives and equations have not changed:

```text
load latest valid checkpoint
    ↓
process only dates after checkpoint
    ↓
append trait history
    ↓
write new checkpoint
    ↓
publish new snapshots/latest profile
```

Do not replay full UFC history for routine weekly data updates.

## Equation change

If one trait/group changes:

```text
invalidate only that trait/group
    ↓
full chronological replay of that trait/group
    ↓
unrelated histories remain valid
```

---

# 7. Version fingerprints

Every trait definition should have a deterministic version/fingerprint based on:

- raw source columns;
- observation definition;
- evidence-quality definition;
- opponent/counter trait;
- population baseline rule;
- update/rating parameters;
- dependencies;
- missing/zero-opportunity rules.

The build should clearly report:

```text
CACHE VALID
REPLAY REQUIRED
NEW TRAIT
DEPENDENCY INVALIDATED
```

Do not rely on human memory to determine which histories are stale.

---

# 8. Published artifacts

## 8.1 Historical prefight snapshots

Publish a historical leakage-safe artifact such as:

`fsr_v2_prefight_snapshots.parquet`

One row per fighter per historical fight, containing only information available **before that fight**.

## 8.2 Latest fighter profile

Publish a current/latest artifact such as:

`fsr_v2_latest.parquet`

One current row per fighter for upcoming simulation.

## 8.3 Trait audit histories

Each trait/replay group should preserve an auditable historical output containing enough information to explain every rating change.

For competitive ratings, retain fields such as:

- fighter/fight/date;
- pre-rating;
- opponent pre-rating;
- raw opportunity counts;
- observed performance;
- expected performance;
- evidence strength;
- update;
- post-rating.

For behavioral/suppression states, preserve equivalent transparent fields.

The exact file layout can be chosen by implementation, but the audit history must be queryable.

---

# 9. EVENT MC architecture decision — locked for later integration

The current EVENT MC kernel already uses a continuous-time competing-risk scheduler with event rates.

**Do not redesign the kernel.**

The future simulator ontology will be simplified to two states:

```text
STANDING
GROUND
```

Clinch will no longer be a separate simulation phase when integration is eventually authorized.

Historical UFCStats mapping for the future two-state abstraction:

```text
distance significant strikes → STANDING strikes

clinch significant strikes
+
ground significant strikes
→ GROUND-side strikes

UFCStats control seconds
→ GROUND-side exposure proxy
```

This is an explicit model abstraction, not a claim that all real UFC clinch control is literally ground control.

Phase residence will emerge from events rather than fighter phase-tendency ratings.

Expected later flow:

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

A successful escape returns the fight to STANDING.

Do not implement these MC changes during the current FSR V2 build/sanity-check gate.

---

# 10. Standard FSR trait roles

Use four common names where the underlying event supports them.

## `*_tendency`

> How frequently does the fighter generate/attempt the event?

Used to drive an event clock.

## `*_suppression`

> How much does the opponent reduce the fighter's normal event-generation frequency?

Used to modify the opponent's event clock.

## `*_offense`

> Once the event occurs, how effective is the initiator?

Used in event success resolution.

## `*_defense`

> Once the event occurs, how effective is the opponent at stopping/resisting it?

Used in event success resolution.

Not every event must have all four roles. Do not invent a trait when UFCStats cannot identify it.

---

# 11. Chronological state conventions

## Behavioral tendencies

Tendency traits are not automatically Elo.

They should be leakage-safe chronological persistent fighter states based on event rate per meaningful exposure.

New observations update the prior fighter state.

More exposure should mean more evidence.

Established fighters should be more stable than fighters with little evidence.

The exact smoothing/update hyperparameters may be calibrated after diagnostic output, but the raw observation must remain visible.

## Suppression

Suppression is separate from defense.

Example:

```text
striking defense:
opponent throws but misses

striking suppression:
opponent throws less than expected
```

Suppression should compare:

```text
opponent prefight expected event rate
vs
opponent actual event rate against this fighter
```

Positive suppression means the fighter pushed the opponent's event generation below that opponent's prefight norm.

## Competitive offense/defense ratings

Where a genuine attacker-vs-defender success interaction exists, use the opponent-adjusted Elo-style architecture.

Conceptually:

```text
actual success
vs
prefight expected success based on:
  attacker offense rating
  defender defense rating
  prior-date UFC population baseline
```

Better-than-expected performance improves attacker offense and reduces defender defense; worse-than-expected performance does the opposite.

Opportunity/sample size controls evidence strength.

Neutral ratings must respect the natural historical UFC success rate rather than assuming every event has a 50% population baseline.

All same-date updates must be applied leakage-safely/simultaneously so fights on the same date cannot contaminate each other.

---

# 12. Exposure definitions — preliminary

These are deliberate two-state approximations that must be sanity checked.

For a shared fighter round:

```text
ground_exposure_seconds =
    min(round_elapsed_seconds,
        fighter_A_control_seconds + fighter_B_control_seconds)
```

```text
standing_exposure_seconds =
    round_elapsed_seconds - ground_exposure_seconds
```

Use actual elapsed time for incomplete finishing rounds.

Both fighters share the same inferred exposure denominator for a given round because the fight phase is shared.

### Caveat

Real UFCStats control includes both standing clinch control and ground control.

In FSR V2, that combined control time is intentionally mapped to the future MC's GROUND-side exposure because the future simulator has only STANDING and GROUND states.

This approximation must be measured and documented.

---

# 13. LOCKED PRELIMINARY TRAIT DEFINITIONS

The definitions below are the current implementation target.

They are intentionally called **preliminary** because the new FSR V2 must first undergo sanity checks before any EVENT MC integration.

---

## 13.1 Standing striking event

### `standing_striking_tendency`

**Role:** tendency  
**Type:** chronological behavioral rate  
**Raw UFCStats evidence:** distance significant-strike attempts  
**Exposure:** standing exposure

Preliminary observation:

```text
distance significant-strike attempts
/
standing exposure
```

Interpretation:

> While the fight is in the modeled STANDING state, how frequently does this fighter generate significant-strike attempts?

Later MC use:

```text
standing_striking_tendency
× opponent standing_striking_suppression
→ standing strike event rate
```

---

### `standing_striking_suppression`

**Role:** suppression  
**Type:** chronological opponent-effect state

Preliminary observation:

```text
opponent prefight expected distance-strike rate per standing exposure
-
opponent actual distance-strike rate per standing exposure
```

Weight by meaningful standing exposure.

Interpretation:

> How much does this fighter cause opponents to generate fewer standing strikes than they normally would?

Later MC use:

> modifies the opponent's standing-strike clock.

---

### `standing_striking_offense`

**Role:** offense  
**Type:** ELO paired with `standing_striking_defense`

Raw fight performance:

```text
distance significant strikes landed
/
distance significant strikes attempted
```

Update based on how that accuracy compares with the prefight expectation from:

```text
attacker standing_striking_offense
vs
defender standing_striking_defense
```

with the prior-date UFC population distance-strike accuracy as the neutral baseline.

More attempts = stronger evidence.

Interpretation:

> Once the fighter generates a standing strike, how good is the fighter at landing it?

---

### `standing_striking_defense`

**Role:** defense  
**Type:** ELO paired with `standing_striking_offense`

Same strike interaction from the defender side.

Interpretation:

> Once an opponent generates a standing strike, how good is this fighter at preventing it from landing?

An opponent landing below expectation improves defense; landing above expectation reduces defense.

---

# 14. Strike target tendencies

The raw round source contains head/body/leg significant-strike information. Preserve and use it even though no anatomical damage redesign is authorized now.

A future body-location damage system may consume these traits.

## State rule

Leg strikes are only eligible in the future `STANDING` state.

GROUND target selection is head/body only.

Because UFCStats does not provide a clean target × phase cross-tab, target selection remains an approximation.

---

### `head_strike_tendency`

**Role:** target tendency  
**Type:** behavioral composition

Preliminary observation:

```text
head significant-strike attempts
/
(head significant-strike attempts + body significant-strike attempts)
```

Interpretation:

> Conditional on a non-leg target, how strongly does the fighter favor the head?

---

### `body_strike_tendency`

**Role:** target tendency  
**Type:** behavioral composition

Preliminary observation:

```text
body significant-strike attempts
/
(head significant-strike attempts + body significant-strike attempts)
```

By construction:

```text
head_strike_tendency + body_strike_tendency = 1
```

---

### `leg_strike_tendency`

**Role:** standing-only target tendency  
**Type:** behavioral propensity

Preliminary proxy:

```text
leg significant-strike attempts
/
distance significant-strike attempts
```

Interpretation:

> How strongly does the fighter generate leg targets relative to the historical STANDING-strike proxy?

This is a proxy because UFCStats target and phase categories are not cross-tabulated.

Sanity diagnostics must inspect its range and edge cases before it is used by EVENT MC.

---

# 15. Takedown event

## `takedown_tendency`

**Role:** tendency  
**Type:** chronological behavioral rate

Preliminary observation:

```text
takedown attempts
/
standing exposure
```

Interpretation:

> While the fight is STANDING, how frequently does this fighter generate a takedown attempt?

Later MC use:

> drives the takedown event clock.

---

## `takedown_suppression`

**Role:** suppression  
**Type:** chronological opponent-effect state

Preliminary observation:

```text
opponent prefight expected takedown-attempt rate per standing exposure
-
opponent actual takedown-attempt rate per standing exposure
```

Interpretation:

> How much does this fighter cause opponents to attempt fewer takedowns than they normally would?

Later MC use:

> reduces opponent takedown clock.

---

## `takedown_offense`

**Role:** offense  
**Type:** ELO paired with `takedown_defense`

Raw performance:

```text
takedowns landed
/
takedowns attempted
```

Compare actual completion against prefight expected completion based on:

```text
attacker takedown_offense
vs
defender takedown_defense
```

with prior-date UFC population TD completion as the neutral baseline.

Interpretation:

> Once the fighter attempts a takedown, how good is the fighter at completing it?

---

## `takedown_defense`

**Role:** defense  
**Type:** ELO paired with `takedown_offense`

Interpretation:

> Once an opponent attempts a takedown, how good is this fighter at stopping it?

Opponent conversion below expectation improves defense; above expectation reduces it.

---

## `takedown_persistence_tendency`

**Status:** NOT part of initial locked core.

Do not implement as a required production FSR V2 trait yet.

Rationale:

After a failed TD, the future event-driven MC can simply remain STANDING and resample the takedown clock. Repeated attempts may emerge naturally from ordinary `takedown_tendency`.

Keep the concept available for future research if historical replay shows fighter-specific re-shot behavior cannot be reproduced without it.

---

# 16. Escape event / ground-duration system

There is no explicit UFCStats escape-attempt count.

Therefore do **not** create:

- `escape_tendency`;
- `escape_suppression`.

Instead, the future EVENT MC will model a **successful escape clock** directly.

A successful TD begins a ground episode.

The escape clock is determined by:

```text
bottom fighter escape_offense
vs
top fighter escape_defense
```

When the escape event fires:

```text
GROUND → STANDING
```

Ground strikes and submission clocks compete during that exposure.

---

## `escape_offense`

**Role:** offense  
**Type:** paired opponent-adjusted rating with `escape_defense`; ELO-like

Meaning:

> How good is the bottom fighter at terminating opponent ground control sooner than expected?

Primary historical evidence:

```text
opponent control seconds
relative to
opponent successful ground-entry proxy
```

Initial ground-entry proxy:

```text
opponent takedowns landed
```

Primary preliminary duration measure:

```text
opponent control seconds
/
opponent takedowns landed
```

Shorter-than-expected episodes are positive `escape_offense` evidence.

### Zero-TD control fallback

Cases will exist with control seconds > 0 but TD landed = 0.

Do not divide by zero.

During source diagnostics, quantify these cases.

Initial fallback may infer one ground entry when meaningful control exists, but the chosen threshold/fallback must be printed in diagnostics and treated as a tunable preliminary rule.

### Repeated TD evidence

Multiple takedowns in the same round imply at least some returns to standing between successful takedowns.

Preserve this as a sanity/audit signal even if it is not part of the first rating equation.

---

## `escape_defense`

**Role:** defense  
**Type:** paired opponent-adjusted rating with `escape_offense`; ELO-like

Meaning:

> How good is the top fighter at maintaining ground control and preventing the bottom fighter from returning to standing?

Longer-than-expected ground-control episodes are positive `escape_defense` evidence.

Shorter-than-expected episodes reduce it.

---

## Escape clock interpretation for later MC integration

When integration is authorized:

```text
successful takedown
    ↓
GROUND begins
    ↓
schedule successful escape clock
based on:
bottom.escape_offense vs top.escape_defense
```

Other ground events run concurrently.

The escape clock is cancelled if:

- fight finishes;
- ground state ends;
- control ownership changes due to an experimental reversal.

Ground episode duration should emerge from this matchup.

This clock is expected to be the main future mechanism controlling simulated control time.

---

# 17. Ground striking event

Historical mapping:

```text
ground-side strikes =
    UFCStats ground significant strikes
    +
    UFCStats clinch significant strikes
```

This follows the two-state abstraction.

---

## `ground_striking_tendency`

**Role:** tendency  
**Type:** chronological behavioral rate

Preliminary observation:

```text
(ground sig attempts + clinch sig attempts)
/
ground exposure
```

with:

```text
ground exposure ≈ combined control seconds
```

Interpretation:

> While the fight is in the modeled GROUND state, how frequently does this fighter generate significant strikes?

### Zero-control edge case

Cases can exist where ground/clinch strikes > 0 but recorded combined control = 0.

The implementation must:

1. quantify these cases;
2. report them;
3. implement a transparent minimum/fallback exposure rule only after the frequency is understood.

Do not silently divide by zero or discard them.

---

## `ground_striking_suppression`

**Role:** suppression  
**Type:** chronological opponent-effect state

Preliminary observation:

```text
opponent prefight expected ground-side strike rate per ground exposure
-
opponent actual ground-side strike rate per ground exposure
```

Interpretation:

> How much does this fighter cause opponents to generate fewer ground-side strikes than expected?

Later MC use:

> modifies opponent ground-strike clock.

---

## `ground_striking_offense`

**Role:** offense  
**Type:** ELO paired with `ground_striking_defense`

Raw performance:

```text
(ground sig landed + clinch sig landed)
/
(ground sig attempted + clinch sig attempted)
```

Compare actual accuracy against prefight expectation from:

```text
attacker ground_striking_offense
vs
defender ground_striking_defense
```

with prior-date UFC population ground-side accuracy as the neutral baseline.

---

## `ground_striking_defense`

**Role:** defense  
**Type:** ELO paired with `ground_striking_offense`

Meaning:

> Once an opponent generates a ground-side strike, how well does this fighter prevent it from landing?

---

# 18. Submission event

Submissions occur only during the modeled GROUND state.

---

## `submission_tendency`

**Role:** tendency  
**Type:** chronological behavioral rate

Preliminary observation:

```text
submission attempts
/
ground exposure
```

Interpretation:

> Given ground exposure, how frequently does this fighter generate submission attempts?

---

## `submission_suppression`

**Role:** suppression  
**Type:** chronological opponent-effect state

Preliminary observation:

```text
opponent prefight expected submission-attempt rate per ground exposure
-
opponent actual submission-attempt rate per ground exposure
```

Interpretation:

> How much does this fighter suppress the opponent's normal submission-attempt behavior?

---

## `submission_offense`

**Role:** offense  
**Type:** paired opponent-adjusted success rating with `submission_defense`

Round stats provide SUB attempts.

Use UFC master result/method when required to determine whether the fight ended by submission.

Preliminary interpretation:

> Given that a submission attempt was generated, how well does this fighter convert submission opportunities into finishes relative to opponent submission defense?

The first implementation should retain:

- number of SUB attempts;
- whether a submission finish occurred;
- opponent prefight defense;
- evidence/quality count.

Do not hide the coarse nature of UFCStats `SUB ATT`.

---

## `submission_defense`

**Role:** defense  
**Type:** paired opponent-adjusted success rating with `submission_offense`

Meaning:

> Given opponent submission attempts, how well does this fighter survive them relative to expectation?

If an opponent expected to be dangerous generates attempts but does not finish, defense receives positive evidence.

A submission loss is strong negative evidence.

---

# 19. Reversal — experimental only

Reversals are not part of the required core V2 trait set.

UFCStats records successful reversals but not reversal attempts.

Initial shadow-only candidate:

## `reversal_tendency`

```text
successful reversals
/
ground exposure
```

Interpretation:

> Rate of successful ownership-switch events given modeled ground exposure.

Do not initially build:

- `reversal_suppression`;
- `reversal_offense`;
- `reversal_defense`.

If later integrated into EVENT MC, a successful reversal event would switch ground controller and restart the escape clock for the new top/bottom pairing.

Keep all reversal work experimental/shadow until it proves useful.

---

# 20. Damage / KD / KO system

**Preserve the current EVENT MC damage, knockdown, KO/TKO, durability, power, and damage-recovery implementation exactly as it is currently defined.**

This FSR V2 task does not redesign it.

When constructing the eventual FSR V2 profile, preserve/provide whatever current fighter inputs that system requires, using the existing definitions.

Do not silently reinterpret or rename the current damage-system traits.

The raw head/body/leg strike fields remain available for future anatomical damage work, but no anatomical damage mechanics are authorized now.

---

# 21. Stamina system

**Preserve the current EVENT MC stamina and recovery system exactly as it is currently defined.**

This FSR V2 task does not redesign it.

When constructing the eventual FSR V2 profile, preserve/provide whatever current fighter inputs the stamina system requires, using the existing definitions.

Do not add new capacity/efficiency/recovery traits unless the existing system already uses them.

---

# 22. Complete initial core trait inventory

## Standing striking

1. `standing_striking_tendency`
2. `standing_striking_suppression`
3. `standing_striking_offense` **[ELO]**
4. `standing_striking_defense` **[ELO]**

## Strike targets

5. `head_strike_tendency`
6. `body_strike_tendency`
7. `leg_strike_tendency`

## Takedowns

8. `takedown_tendency`
9. `takedown_suppression`
10. `takedown_offense` **[ELO]**
11. `takedown_defense` **[ELO]**

## Ground exit / control duration

12. `escape_offense` **[ELO-like paired]**
13. `escape_defense` **[ELO-like paired]**

## Ground striking

14. `ground_striking_tendency`
15. `ground_striking_suppression`
16. `ground_striking_offense` **[ELO]**
17. `ground_striking_defense` **[ELO]**

## Submissions

18. `submission_tendency`
19. `submission_suppression`
20. `submission_offense` **[paired opponent-adjusted]**
21. `submission_defense` **[paired opponent-adjusted]**

## Experimental

22. `reversal_tendency` — experimental/shadow only

## Preserved existing systems

- existing damage/KD/KO/durability/power/recovery traits required by current EVENT MC;
- existing stamina/recovery traits required by current EVENT MC.

---

# 23. Trait-to-future-MC mapping

When EVENT MC integration is eventually authorized:

```text
STANDING
│
├─ STANDING STRIKE CLOCK
│    attacker standing_striking_tendency
│    modified by defender standing_striking_suppression
│
│    if event fires:
│      attacker standing_striking_offense
│      vs defender standing_striking_defense
│      → land / miss
│
│      target selection:
│        leg is standing-only
│        otherwise head/body
│
└─ TAKEDOWN CLOCK
     attacker takedown_tendency
     modified by defender takedown_suppression

     if event fires:
       attacker takedown_offense
       vs defender takedown_defense
       → success / failure

       success:
         STANDING → GROUND

GROUND
│
├─ ESCAPE CLOCK
│    bottom escape_offense
│    vs top escape_defense
│    → successful escape
│    → GROUND → STANDING
│
├─ GROUND STRIKE CLOCK
│    attacker ground_striking_tendency
│    modified by defender ground_striking_suppression
│
│    if event fires:
│      attacker ground_striking_offense
│      vs defender ground_striking_defense
│      → land / miss
│
├─ SUBMISSION CLOCK
│    attacker submission_tendency
│    modified by defender submission_suppression
│
│    if event fires:
│      attacker submission_offense
│      vs defender submission_defense
│      → finish / survive
│
└─ REVERSAL CLOCK [EXPERIMENTAL]
     reversal_tendency
     → successful ownership switch
```

The existing damage and stamina systems remain attached to the relevant actions as they are now.

---

# 24. Sanity-check gate — REQUIRED BEFORE MC CHANGES

The user explicitly wants to sanity check the new traits **before any EVENT MC change**.

The implementation is not complete until these diagnostics exist and are run.

## 24.1 Source/schema audit

Print and save:

- exact columns used from round stats;
- exact master fields used;
- null rates;
- reciprocal fighter/opponent integrity;
- round-duration handling;
- control-time ranges;
- cases where combined control exceeds round duration;
- cases with control > 0 and zero TD landed;
- cases with ground/clinch strikes > 0 and zero control;
- SUB attempt distributions;
- target-location consistency checks.

## 24.2 Trait coverage

For every trait:

- number of eligible fighter-fight observations;
- number of fighters with at least 1/3/5/10 meaningful observations;
- missing-rate;
- zero-opportunity rate;
- evidence/exposure distribution.

## 24.3 Distribution sanity

For every published trait:

- min;
- p01;
- p05;
- p25;
- median;
- p75;
- p95;
- p99;
- max;
- mean;
- standard deviation.

Flag:

- collapsed traits;
- extreme outliers;
- impossible values;
- mass at prior/default;
- unstable sparse traits.

## 24.4 Fighter ranking sanity

For every core trait, print at least:

- top 20 fighters;
- bottom 20 fighters;
- representative middle fighters;
- sample size/exposure beside rating.

This output is specifically intended for manual inspection and later comparison to known fighter styles and postfight analysis.

## 24.5 Chronology/leakage audit

For sampled historical fights prove:

```text
prefight snapshot date < target fight information
```

and prove the target fight itself does not enter its own prefight trait.

Same-date fights must not leak into one another.

## 24.6 Opponent-adjustment sanity

For each ELO/paired trait show examples where:

- same observed result against strong vs weak opposition produces appropriately different updates;
- strong defender suppresses expected success;
- weak defender raises expected success;
- update direction is correct.

## 24.7 Suppression sanity

For suppression traits, print examples of:

```text
opponent prior expected rate
actual opponent rate in fight
raw suppression residual
resulting suppression state
```

Confirm:

- positive means opponent did less than expected;
- negative means opponent did more than expected;
- short exposure has smaller influence than large exposure.

## 24.8 Escape sanity

This is especially important.

Print historical examples showing:

- TD landed;
- control seconds;
- control per TD proxy;
- inferred/fallback entry handling;
- bottom escape offense before/after;
- top escape defense before/after.

Compare fighter rankings against intuitive control/escape behavior.

Do not wire this into EVENT MC until these outputs are reviewed.

## 24.9 Submission sanity

Print:

- ground exposure;
- SUB attempts;
- SUB attempts per ground minute;
- submission finish yes/no;
- attacker offense;
- defender defense;
- suppression residual.

Check that fighters with high attempt tendency are not automatically treated as high finish offense.

## 24.10 Stability / next-fight persistence

For behavioral traits, measure whether prior prefight state predicts the fighter's next relevant observable better than:

- population average;
- simple career average;
- simple last-fight value where appropriate.

For competitive traits, measure whether rating differential predicts the relevant next-fight success measure.

This is a trait-validity diagnostic, not an EVENT MC outcome test.

---

# 25. Required sanity-check output package

Codex should produce a machine-readable diagnostic artifact plus a readable report.

Recommended outputs:

```text
artifacts/fsr_v2/
  source_audit.json
  trait_coverage.csv
  trait_distribution.csv
  trait_rankings/
  update_examples/
  escape_audit.csv
  submission_audit.csv
  replay_validation.json
```

Exact repository convention may be used instead.

Also print a concise final summary containing:

- trait list built;
- traits passing basic sanity;
- traits requiring review;
- source edge cases;
- replay runtime;
- checkpoint/resume behavior;
- no-leakage result;
- no EVENT MC files changed.

---

# 26. Build commands / ergonomics

The system should support selective rebuilds.

Examples of desired CLI behavior:

```bash
python -m pipeline.fsr_v2.build --traits standing_striking
```

```bash
python -m pipeline.fsr_v2.build --traits takedown_effectiveness
```

```bash
python -m pipeline.fsr_v2.build --traits submission_tendency,submission_suppression
```

```bash
python -m pipeline.fsr_v2.build --all
```

```bash
python -m pipeline.fsr_v2.build --resume
```

```bash
python -m pipeline.fsr_v2.publish
```

Exact command names can differ, but selective trait/group rebuild, full build, resume, and publish must all be straightforward.

---

# 27. Tests

Minimum automated tests should cover:

- source loading;
- reciprocal opponent pairing;
- no future leakage;
- same-date update isolation;
- correct exposure computation;
- control capping;
- zero-opportunity behavior;
- tendency update direction;
- suppression sign;
- offense/defense update direction;
- paired replay determinism;
- checkpoint/resume equivalence to full replay;
- independent trait invalidation;
- dependency invalidation;
- snapshot assembly;
- latest-profile assembly;
- experimental trait isolation;
- no mutation of authoritative source parquet.

A resumed replay must produce the same result as a full replay for the same trait version and data.

---

# 28. Calibration values are not yet locked

Do not blindly inherit old FSR numerical constants simply because they existed previously.

The useful old architecture to preserve conceptually is:

- opponent-adjusted expectations;
- natural prior-date population baselines;
- evidence/opportunity weighting;
- decreasing volatility with accumulated evidence;
- leakage-safe chronological replay;
- simultaneous same-date updates.

The exact numerical values for:

- K;
- rating scale;
- evidence saturation;
- behavioral smoothing;
- suppression smoothing;
- escape duration transform;
- any fallback exposure constants;

should be visible, configurable, and evaluated during the sanity-check stage.

Avoid hidden magic numbers.

---

# 29. What should not return from the old architecture

Do not recreate:

- three independent distance/clinch/ground pressure ratings;
- phase tendencies that force fight residence;
- hidden composites inherited from RFS/FSR32;
- a separate clinch MC phase;
- arbitrary phase-share FSRs;
- persisted generic fighter-fight observation parquet;
- simulator mappings that mix style, event frequency, and skill into one rating.

The new separation is intentional:

```text
event frequency:
  tendency × opponent suppression

event success:
  offense vs defense

state residence:
  emerges from event clocks and outcomes
```

---

# 30. Post-sanity-check future work — NOT AUTHORIZED YET

After the user manually reviews the new trait outputs:

## Future Stage A — EVENT MC adapter

- preserve current kernel/scheduler;
- simplify phase ontology to STANDING/GROUND;
- wire the new FSR V2 traits into event rates;
- implement successful escape clock;
- keep reversals experimental;
- preserve damage and stamina systems.

## Future Stage B — calibration

Then calibrate population/event-rate mappings using historical data.

Key calibration targets should include:

### STANDING
- standing strike attempts per standing exposure;
- strike accuracy;
- TD attempts per standing exposure;
- TD completion.

### GROUND
- control/ground episode duration;
- ground-side strike attempts per ground exposure;
- ground-side strike accuracy;
- submission attempts per ground exposure;
- submission finish conversion;
- escape-rate distribution.

### Fight level
- total significant strikes;
- TDs;
- control time;
- ground strikes;
- SUB attempts;
- KO/SUB/DEC method shares;
- finish timing;
- winner prediction;
- eventually round-level output.

Only after the global event environment is calibrated should winner/method predictive performance be judged as a final system.

---

# 31. Final implementation philosophy

The new architecture should remain easy to diagnose.

If historical replay later shows:

```text
TD attempts correct
TD success correct
control time too high
```

look at the escape clock, not takedown tendency.

If:

```text
control time correct
ground strikes too low
```

look at ground-striking tendency/rate, not escape.

If:

```text
SUB attempts correct
SUB finishes too high
```

look at submission offense/defense, not submission tendency.

If:

```text
standing strike attempts wrong
accuracy correct
```

look at standing tendency/suppression, not striking offense/defense.

That modular diagnosability is a core project requirement.

---

# 32. Gate for returning to EVENT MC work

FSR V2 is ready for EVENT MC integration only when:

1. source/schema audit passes;
2. replay is leakage-safe;
3. selective trait rebuild works;
4. checkpoint/resume equals full replay;
5. all core traits have reasonable coverage/distributions;
6. ranking sanity output has been manually reviewed;
7. suppression signs and magnitudes make sense;
8. escape trait outputs are plausible;
9. submission tendency and submission offense are clearly separated;
10. no major trait is obviously redundant/broken;
11. the user explicitly approves proceeding to MC integration.

Until then:

> **Do not change EVENT MC.**
