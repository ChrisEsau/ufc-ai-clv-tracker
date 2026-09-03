# Codex Prompt — EVENT MC V1 Phase 3 Clinch + Ground Flow

Date: 2026-08-13

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch/source branch: `feature/fsr-32-stamina-shadow`
Architecture revision: **v0.3**

Status before this phase:
- Phase 0 operational baseline: **PASS**.
- Phase 1 generic continuous-time kernel: **PASS**.
- Phase 2A DISTANCE temporal/mechanical parity: **PASS**.
- Phase 2B wrestling-entry ontology correction: **PASS**.
- Phase 3 is explicitly authorized by the user.
- Phase 4 and later are **NOT AUTHORIZED**.

## Cloud-worktree note

Codex cloud may place the task on a local branch named `work`. That is acceptable.

Do **not** require `git branch --show-current` to equal the feature branch name.

Instead verify that the checkout contains Phase 2B history and the governing documents. At minimum:

```bash
git rev-parse HEAD
git log --oneline -12
git merge-base --is-ancestor 809389bdabe208e93536034bc795bbcf7e1ab038 HEAD
```

The ancestry check must succeed, and these must exist:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md
```

If the cloud checkout is stale but has an authenticated remote, fetch/fast-forward/rebase safely before implementation. Do not discard existing Phase 2B work.

# Read first

Before changing code, read:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
3. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
6. `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`
7. `docs/EVENT_MC_V1_CODEX_PHASE2B_WRESTLING_ENTRY_ONTOLOGY_2026-08-13.md`
8. this prompt

The architecture audit v0.3 remains canonical where not superseded by reviewed later phase contracts.

# Goal

Implement the **CLINCH and GROUND fight-flow layer** on EVENT MC V1 so a path can move naturally through all major UFC phases for the full scheduled horizon without relying on inert CLINCH/GROUND states.

This phase is about:

- phase behavior;
- action opportunities;
- position/control persistence;
- phase transitions;
- continuous residence time;
- submission **attempt generation**;
- observability and historical/mechanical diagnostics.

This phase is **not** about physiology or terminal outcomes.

By the end of Phase 3, a nonterminal path should be able to do something conceptually like:

```text
DISTANCE
  -> strike
  -> clinch entry
CLINCH
  -> clinch strike
  -> takedown
GROUND
  -> ground strike
  -> submission attempt
  -> escape / reversal / standup
DISTANCE or GROUND
  -> continue
...
round boundary -> DISTANCE reset
```

The simulator should now be capable of completing full scheduled fight time while moving through phases.

# Development standard

The user's standard remains:

**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE**

Do not over-engineer.

Prefer:

1. trace current effective mechanics;
2. port the useful formula/behavior;
3. fit it into clean EVENT MC composition;
4. expose diagnostics;
5. test the important invariants;
6. historical replay later decides what deserves tuning.

Do not create speculative abstractions or dozens of defensive checks.

# Absolute non-goals

Do NOT in Phase 3:

- modify the current inheritance-based simulator;
- rebuild or alter FSR-32;
- alter FSR builders/ratings/ontology/maturity/leakage rules;
- undo Phase 2B wrestling-entry semantics;
- retune calibration constants broadly;
- add stamina depletion/recovery;
- add damage;
- add knockdowns;
- add KO/TKO;
- add terminal submission finishes;
- add damage/KO recovery;
- add age transforms;
- add judging/scoring;
- add tactical urgency/score-aware strategy;
- add arbitrary cooldowns/refractory periods;
- add body-part damage;
- add MatReturn unless the existing mechanics absolutely require it (default: do not add it);
- implement Phase 4 or later work.

The separate damage/KD/KO redesign must not hold this phase up. Only preserve clean interfaces so that physiology can be attached later.

# Frozen FSR input

When real historical diagnostics need fighter profiles, use only:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rebuild or rewrite it.

# Phase 2B lock

DISTANCE TD initiation remains Phase 2B ontology-correct:

```text
wrestling_entry = intrinsic TD initiation frequency
```

Active DISTANCE TD rate must continue using the Phase 2B path.

Do not reintroduce:

```text
0.75*wrestling_entry
+ 0.25*control_imposition
- 0.5*distance_striking_pressure
- 0.5*clinch_striking_pressure
```

into active DISTANCE TD initiation.

The legacy blend remains diagnostic-only.

TD success remains:

```text
wrestling_conversion vs opponent td_defense
```

with the existing Phase 2A/2B formula unchanged unless Phase 3 needs the same resolution mechanic in CLINCH.

# Locked control ontology

`control_imposition` now belongs **after advantageous position is established**.

Its conceptual job is persistence / ability to maintain control, not intrinsic shot frequency.

Phase 3 is the first phase where this trait can be used in its intended place.

Do not mechanically force it into every ground formula. Trace current consumers and use it where it has an interpretable persistence/retention role.

# Source-tracing requirement

Before coding the new mechanics, trace the actual current simulator/V0 consumers for CLINCH and GROUND behavior.

At minimum reconstruct the effective formulas/constants for:

## CLINCH
- clinch strike attempt frequency;
- clinch strike hit/miss probability;
- clinch TD attempt frequency;
- clinch TD success probability;
- clinch separation / return-to-distance behavior;
- any controller-dependent asymmetry already present.

## GROUND
- top ground-strike attempt frequency;
- bottom ground-strike attempt frequency if present;
- ground strike accuracy / landed logic;
- submission-attempt frequency;
- which fighter is eligible to attempt submissions under current semantics;
- ground escape / standup frequency;
- reversal frequency / probability;
- controller persistence;
- any direct ground-to-distance transition;
- any controller swap without leaving ground.

Trace **effective final behavior**, not just the earliest base class.

The current simulator has an inheritance stack. Port useful formulas/semantics, not the inheritance architecture.

Document the source module/function/class used for every Phase 3 action family.

# Core architecture

Continue using composition under:

`pipeline/simulation/event_mc_v1/`

The scheduler stays generic.

Phase-specific providers decide which candidates exist.

A clean structure might be conceptually:

```text
DistanceActionRateProvider
ClinchActionRateProvider
GroundActionRateProvider
```

or one small composed provider delegating by phase.

Do not create a giant monolithic fight class.

Do not create a new inheritance stack.

# Primary action families

The exact implementation may vary slightly if current legacy mechanics combine/split actions differently, but Phase 3 should support the following behavioral families.

## DISTANCE
Already implemented and must remain working:
- red strike attempt;
- blue strike attempt;
- red TD attempt;
- blue TD attempt;
- red clinch entry;
- blue clinch entry.

## CLINCH
Implement enough mechanics for CLINCH not to be inert:

- red clinch strike attempt;
- blue clinch strike attempt;
- red clinch TD attempt;
- blue clinch TD attempt;
- separation / return to DISTANCE.

If the current mechanics represent separation as fighter-specific escape attempts rather than one symmetric event, preserve the current effective semantics in a clean way.

If clinch ownership materially changes action rates in current V0, preserve that distinction explicitly.

Successful clinch TD:

```text
CLINCH -> GROUND
attacker becomes ground controller
clinch controller clears
```

Failed clinch TD:
- remains CLINCH unless current semantics explicitly change position.

Separation:

```text
CLINCH -> DISTANCE
all positional ownership clears
```

## GROUND
Implement enough mechanics for GROUND to be live and escapable:

- top ground strike attempt;
- bottom ground strike attempt if current mechanics allow it;
- submission attempt(s) according to current effective semantics;
- escape/standup;
- reversal.

Ground primary-action design must preserve controller identity.

Possible transitions:

### Escape/standup

```text
GROUND -> DISTANCE
ground controller clears
```

### Reversal

```text
GROUND -> GROUND
ground controller swaps
```

No new time should elapse merely because a consequence changes controller.

# Ground exit partition — HARD INVARIANT

Do **not** independently schedule a full escape clock and a full reversal clock if both were derived from the same legacy ground-exit opportunity. That doubles ground exit frequency.

Where the existing mechanic conceptually has one total ground-exit hazard and a conditional reversal probability, preserve the exact partition:

```text
lambda_ground_exit = total exit hazard
P_reversal_given_exit = conditional reversal probability

lambda_reversal = lambda_ground_exit * P_reversal_given_exit
lambda_escape   = lambda_ground_exit * (1 - P_reversal_given_exit)
```

Required invariant:

```text
lambda_reversal + lambda_escape == lambda_ground_exit
```

within floating tolerance.

If the current simulator genuinely has separate independent mechanisms with separate opportunities, document evidence before representing them as independent clocks.

# Control-time behavior

EVENT MC should not need a fake control-time counter incremented in chunks.

Control time should emerge from exact elapsed residence in CLINCH/GROUND under controller ownership.

Use the existing engine ordering:

```text
sample dt
-> advance continuous state over exact dt
-> advance authoritative clock
-> resolve event / boundary
```

For diagnostics/stat sinks, account for exact elapsed time in controlled phases without making the ledger feed back into physics.

At minimum expose:
- total clinch residence time;
- clinch control time by fighter if ownership is meaningful;
- total ground residence time;
- ground control time by fighter;
- number/timestamps of controller switches;
- phase transitions.

Do not add a second mutable fight clock.

# Probability / rate migration

For legacy Bernoulli-style event opportunities, preserve the **final effective interval probability** first, then convert exactly:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

For legacy Poisson count processes, preserve the expected count intensity:

```text
lambda_per_second = expected_count / interval_seconds
```

Do not convert a Poisson mean as though it were a Bernoulli probability.

This distinction was correctly handled for DISTANCE strikes in Phase 2A and must remain consistent.

For every Phase 3 action family, diagnostics should make the migration auditable:
- legacy interval/count representation;
- interval seconds;
- final probability or expected count;
- continuous rate/sec;
- major FSR inputs;
- controller/phase context.

# Strike resolution in CLINCH/GROUND

Port the current effective hit/miss consumers for those phases.

A landed strike in Phase 3 produces only an observation such as:

```text
ActionAttempt
ActionOutcome(landed/missed)
```

Do not attach:
- damage;
- KD;
- KO;
- stamina cost.

Those come later behind replaceable interfaces.

It should be straightforward later for a landed-strike observation/result to feed a DamageModel without rewriting phase flow.

# Submission attempts

Implement **submission attempt generation and observation only** in Phase 3.

Do NOT terminate fights by submission yet.

We have a frozen full-method submission anchor and a neutral P(SUB|attempt)=34% used by the legacy validation program, but Phase 3 should not retune or wire terminal submission success into fight results yet.

The purpose here is to establish realistic opportunity/attempt generation from ground position.

At minimum record:
- attacker;
- defender;
- timestamp;
- ground controller at attempt;
- attempt count.

If current mechanics distinguish top/bottom submission attempts, preserve that distinction if it is useful and supported by source behavior.

Keep the SUBMISSION RNG stream reserved/usable for future success resolution, but do not consume it unnecessarily for a terminal finish that is out of scope.

# Round boundaries

Phase 1 round reset remains authoritative:

At each new round:

```text
phase = DISTANCE
ground_controller = None
clinch_controller = None
```

No Phase 3 component may bypass this.

If a ground/clinch event's sampled time crosses a round boundary, the engine truncates at the boundary and the event does not occur before the reset.

Test this in a non-DISTANCE phase.

# Context and modifiers

Do not add tactical urgency.

Do not add score state.

Do not add stamina modifiers yet.

Do not add age modifiers.

Use only contextual multipliers that are already part of the current effective mechanics or are necessary to express controller/top/bottom state.

Keep future modifier seams neutral rather than inventing coefficients.

# Action availability / cooldowns

Phase 1 reserved an availability/cooldown extension point.

Do not activate arbitrary cooldowns in Phase 3 unless the current effective mechanics truly require a duration that cannot be represented by competing event rates.

Default remains memoryless continuous clocks.

# MatReturn

Do not add MatReturn in Phase 3 unless direct source tracing proves that omitting it makes the core ground flow impossible to represent.

The default architectural decision remains:

**MatReturn is optional/future, not required initially.**

# Observability

Extend the compact observer/stat sink enough to report:

## Striking
- distance strike attempts/landed;
- clinch strike attempts/landed;
- ground strike attempts/landed;
- by fighter and phase.

## Wrestling
- DISTANCE TD attempts/landed;
- CLINCH TD attempts/landed;
- TD success percentage by phase;
- controller after landed TD.

## Clinch
- entries;
- separations;
- total clinch time;
- control time by fighter if applicable.

## Ground
- entries;
- total ground time;
- ground control time by fighter;
- escapes/standups;
- reversals;
- controller swaps;
- submission attempts.

## Phase flow
- transition counts;
- transition timestamps;
- percentage of scheduled fight time in DISTANCE / CLINCH / GROUND;
- check that phase-time shares sum to scheduled nonterminal time within tolerance.

Sinks remain observer-only.

# Required validation layers

## A. Formula-level tests

For every new CLINCH/GROUND formula port:
- compare to the actual current legacy consumer where direct comparison is possible;
- verify correct time units;
- verify Bernoulli hazard round trip or Poisson intensity mapping as applicable;
- verify major FSR inputs move the intended output direction;
- verify Phase 2B DISTANCE TD initiation remains unchanged.

## B. State-transition tests

At minimum test:

### CLINCH
- only CLINCH candidates emitted in CLINCH;
- landed clinch TD -> GROUND with correct controller;
- failed clinch TD stays CLINCH;
- separation -> DISTANCE and clears controllers;
- clinch strikes do not change phase.

### GROUND
- only GROUND candidates emitted in GROUND;
- escape -> DISTANCE and clears ground controller;
- reversal -> stays GROUND and swaps controller;
- ground strikes do not change phase;
- submission attempt does not terminate fight;
- controller-dependent candidates use current controller correctly.

## C. Ground-exit invariant test

Explicitly test:

```text
lambda_reversal + lambda_escape == lambda_ground_exit
```

and show that adding the reversal pathway does not double total exit opportunity.

## D. Boundary tests

Start a path in CLINCH or GROUND near a round boundary and verify:
- hard boundary wins when sampled event lies beyond it;
- exact continuous residence time is credited only up to boundary;
- next round resets to DISTANCE;
- controllers clear.

## E. Determinism / sink invariance

With the same root seed:
- Null/Stats/Trace observers must not change fight physics;
- repeated runs produce identical event/transition sequence;
- no hidden RNG introduced.

## F. Full nonterminal synthetic fight

Run enough synthetic paths to show that the engine can now traverse all three phases and complete the scheduled horizon without getting permanently stuck merely because CLINCH/GROUND lack mechanics.

Report:
- phase shares;
- action counts;
- transition counts;
- control time;
- submission attempts.

Do not require a terminal winner yet.

# Historical diagnostics

Use the frozen FSR artifact and the existing stress fixtures.

Required where inputs resolve cleanly:

1. Rob Font vs Raul Rosas Jr. — wrestling-entry/control stress
2. Merab Dvalishvili vs Petr Yan — sustained wrestling/control
3. Max Holloway vs Calvin Kattar — high-volume striking / low wrestling
4. Derrick Lewis vs Chris Daukaus — power profiles, although physiology remains out of scope
5. Charles Oliveira vs Dustin Poirier — grappling/submission-attempt stress

For each, run a reasonable number of **nonterminal full scheduled-time paths** and report mechanics only:

- DISTANCE strike attempts/landed;
- CLINCH strike attempts/landed;
- GROUND strike attempts/landed;
- DISTANCE TD attempts/landed;
- CLINCH TD attempts/landed;
- clinch entries/separations;
- ground entries;
- escapes;
- reversals;
- submission attempts;
- average DISTANCE / CLINCH / GROUND time;
- control time by fighter;
- whether all paths reached scheduled horizon.

Do not claim winner/method predictive performance yet.

# Font/Rosas and Merab/Yan special audit

For Font/Rosas and Merab/Yan specifically report:
- Phase 2B DISTANCE TD rate for each fighter;
- CLINCH TD rate for each fighter;
- average TD attempts by phase;
- TD landed by phase;
- average ground entries;
- average ground control seconds by fighter;
- escape/reversal counts;
- time shares by phase.

This will tell us whether the Phase 2B initiation correction produces sensible behavior once downstream phase flow exists.

Do not tune from these two fights alone.

# Submission anchor caution

The frozen mature baseline has:
- historical SUB 16.23%;
- simulated SUB 16.49%;
- neutral P(SUB|attempt)=34%;
- historical submission attempts/fight 0.5655;
- simulated attempts/path 0.4994;
- historical >=1 attempt 35.02%;
- simulated >=1 attempt 35.08%.

Phase 3 only owns **attempt opportunity generation**, not terminal SUB probability.

Use the attempt-frequency figures as a sanity reference, not a tuning target requiring exact agreement before full physiology/finish integration.

Do not change the 34% neutral finish anchor in Phase 3.

# Existing code protection

Before finalizing confirm:
- current simulator unchanged;
- FSR builders/ratings/ontology unchanged;
- frozen parquet unchanged and uncommitted;
- Phase 2B DISTANCE TD semantics unchanged;
- Phase 1 clock/scheduler/RNG/boundary invariants unchanged;
- no stamina/damage/KD/KO/recovery/age/judging code added;
- no terminal submission finish added;
- no broad calibration tuning;
- no new inheritance hierarchy.

# Testing standard

Prioritize calculation/state/transition tests over exhaustive defensive hardening.

Run at minimum:

1. all `tests/simulation/event_mc_v1`;
2. relevant V0/current legacy tests for formulas being ported;
3. relevant downstream regression tests sufficient to prove old simulator untouched;
4. `compileall` on touched source/tests;
5. Phase 3 synthetic full-horizon diagnostics;
6. frozen historical fixture mechanics diagnostics;
7. frozen FSR SHA check;
8. `git diff --check`.

Do not turn this into repository-wide cleanup.

# Performance note

Individual strike attempts are currently primary events. Do not batch or optimize them in Phase 3 unless a measured catastrophic runtime problem prevents validation.

Record approximate paths/second or elapsed runtime for the Phase 3 historical diagnostic if easy, so we begin building a performance baseline.

Do not trade away mechanical clarity prematurely.

# Required return

Return a concise but complete report containing:

1. starting SHA and final SHA;
2. commits created;
3. files added/changed;
4. exact legacy source functions/classes traced for each CLINCH/GROUND action family;
5. CLINCH action families and formulas/rates;
6. GROUND action families and formulas/rates;
7. exact ground exit partition and invariant;
8. how `control_imposition` is used after position is established;
9. submission-attempt generation behavior;
10. state-transition behavior;
11. exact control/residence-time accounting;
12. formula/state/boundary/determinism test results;
13. synthetic full-horizon flow results;
14. frozen fixture diagnostics, especially Font/Rosas, Merab/Yan, Oliveira/Poirier;
15. approximate diagnostic runtime/paths/sec if available;
16. confirmation Phase 2B remains intact;
17. confirmation current simulator/FSR/calibrations are untouched;
18. confirmation no stamina/damage/KD/KO/terminal SUB/recovery/age/judging/Phase4 work was introduced;
19. push/PR/working-tree status;
20. final line exactly one of:

`PHASE 3 CLINCH + GROUND FLOW GATE: PASS`

or

`PHASE 3 CLINCH + GROUND FLOW GATE: FAIL`

# Stop condition

Stop after Phase 3 implementation, tests, diagnostics, commit/push, and PASS/FAIL report.

Do **not** begin Phase 4 without explicit user authorization.