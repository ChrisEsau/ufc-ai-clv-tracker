# Codex Prompt — EVENT MC V1 Phase 2A Distance Temporal / Mechanical Parity

Date: 2026-08-13

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`
Architecture revision: **v0.3**

Status:
- Phase 0 operational baseline: **PASS**.
- Phase 1 generic kernel: **PASS** at commit `1debecab69a141bf2f81179f3436af569733b750` after independent review.
- Phase 2A is now explicitly authorized by the user.
- Phase 2B remains **NOT AUTHORIZED**.

## Read first

Before touching code, read:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
5. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
7. `docs/EVENT_MC_V1_CODEX_PHASE1_EXECUTION_2026-08-13.md`
8. this prompt

The architecture audit v0.3 remains canonical.

# Goal

Implement the **first real UFC mechanics** on the new continuous-time engine, but only the minimal DISTANCE-phase mechanics required to test temporal/mechanical parity against the current simulator.

Phase 2A must preserve the current legacy semantics/formulas as closely as practical and change only the timing architecture.

This phase is intentionally **not** the wrestling ontology correction.

The key experimental question is:

> If we hold current mechanics/semantics fixed and move them from the legacy discrete simulator onto the event-driven clock, what changes purely because of continuous timing and competing hazards?

# Development standard

Do not over-engineer. The user explicitly prioritizes a working, measurable, modular simulator that can be iterated against moneyline and prop accuracy over exhaustive defensive hardening.

Keep the implementation small, explicit, observable, and easy to revise.

Preserve the critical invariants from Phase 1, but do not add elaborate abstraction or error-proofing unless needed for correctness or later modular swapping.

# Absolute non-goals

Do NOT in Phase 2A:

- implement the ontology-correct `wrestling_entry -> intrinsic base TD rate` model;
- change the legacy blended takedown-attempt consumer semantics;
- tune any constants;
- modify the current inheritance-based simulator;
- change FSR-32 ratings/builders/ontology/maturity/leakage rules;
- add clinch-phase internal mechanics beyond entering CLINCH;
- add ground-phase internal mechanics;
- add submissions;
- add stamina;
- add damage;
- add knockdowns or KO/TKO;
- add recovery;
- add age transforms;
- add judging;
- add tactical urgency/score state;
- add arbitrary cooldown/refractory constants;
- add MatReturn;
- implement Phase 2B.

A Phase 2A path may enter `CLINCH` or `GROUND` when a distance action succeeds, but no new clinch/ground action scheduler is required in this phase unless the architecture absolutely requires a minimal inert holding behavior to complete the test harness. Prefer narrow distance-focused runs if full-fight progression would otherwise require premature mechanics.

# Frozen FSR input

Use the existing frozen FSR-32 artifact only when real fixture/matchup diagnostics need it:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Frozen SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rebuild or rewrite it.

# Locked wrestling semantics for Phase 2A

The corrected FSR ontology remains conceptually locked:

```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete shot
td_defense           = opponent prevention
control_imposition   = persistence after advantageous position
```

However, the **current simulator consumer** still uses the legacy blended wrestling preference approximately as:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Phase 2A must reproduce that current blended consumer for TD attempt initiation.

Do not "fix" it here.

Phase 2B will deliberately replace this with the ontology-correct initiation model later.

# Scope to port

Implement only these DISTANCE primary action families:

1. Red strike attempt
2. Blue strike attempt
3. Red takedown attempt
4. Blue takedown attempt
5. Red clinch-entry attempt
6. Blue clinch-entry attempt

Port the current effective mechanics needed to determine:

- strike attempt frequency;
- strike hit/miss outcome;
- takedown attempt frequency using the legacy blended consumer;
- takedown landed/failed outcome using current `wrestling_conversion` vs opponent `td_defense` semantics;
- clinch-entry frequency / success according to current effective logic;
- resulting phase transition for successful takedown/clinch entry.

Do not port downstream damage/finish physiology.

# Migration rule: preserve final interval probabilities, then convert exactly to hazards

Do not reinterpret old interval probabilities as rates.

For any current action family that ends with a final probability over a legacy interval, preserve that final legacy probability first, then convert to a per-second hazard using:

```text
lambda_per_second = -ln(1 - p_interval) / interval_seconds
```

Use the actual interval associated with the current formula/consumer.

If the current implementation rescales a 30-second base to a 10-second step or otherwise composes several factors, identify the **final effective probability actually consumed by the current simulator** and convert that to the continuous hazard.

Document the source formula/module/function used for each Phase 2A action family.

Do not silently invent new base rates.

# Current known TD timing reference

Earlier audit context found current 30-second TD bases approximately:

- DISTANCE: `0.10`
- CLINCH: `0.24`

with current logic rescaled to the legacy 10-second step.

Do not blindly hardcode these from this prompt if the current code reveals a more precise final consumer path. Trace and port the actual effective implementation.

Known Font/Rosas hazard diagnostic with seed 20260811 was approximately:

- Font distance TD hazard: 1.64% per legacy interval
- Rosas distance TD hazard: 10.01%
- Font clinch TD hazard: 4.17%
- Rosas clinch TD hazard: 25.35%

These are **sanity evidence, not tuning targets**.

# Profiles and modifiers

Create only the minimal profile/input adapter required for Phase 2A.

Do not embed the full future physiology architecture yet.

Phase 2A needs enough immutable fighter/profile data to reproduce the current distance action-rate and conversion formulas, likely including the current relevant FSR traits such as:

- distance_striking_pressure;
- distance_striking_precision / current accuracy consumer;
- striking_defense / current defense consumer;
- wrestling_entry;
- control_imposition;
- clinch_striking_pressure;
- wrestling_conversion;
- td_defense;
- any other trait already used by the exact current distance/clinch-entry formula being ported.

Use explicit names and keep the adapter separate from `FightState`.

Do not put immutable fighter identity/FSR values into mutable path state.

Dynamic modifiers in Phase 2A should default to neutral (`1.0`) unless the current distance formula being ported genuinely requires an existing non-physiology multiplier that is part of mechanical parity.

Do not port stamina/age/damage modifiers yet.

# Action rate model

Introduce a compositional distance action-rate provider/model that converts current effective legacy action probabilities into event rates.

It should consume something conceptually like:

```text
fighter immutable profile
+ opponent immutable profile
+ current FightState / phase
+ neutral/current Phase 2A context
-> EventRate candidates in events/second
```

Keep each action family auditable. It should be possible in diagnostics to print, for each fighter:

- legacy interval probability;
- interval seconds;
- converted hazard per second;
- major trait inputs / blended wrestling preference where relevant.

Do not make the generic scheduler know these formulas.

# Strike attempt and strike resolution

Port the current effective DISTANCE strike attempt generation and hit/miss logic only.

Required outputs/observations at minimum:

- strike attempt event;
- landed vs missed consequence/result;
- fighter side;
- current timestamp;
- no damage/KD/KO side effects.

If the legacy simulator generates attempt counts via a pressure-to-attempt formula inside a 10-second segment, derive the corresponding Phase 2A continuous hazard in a way that preserves the current expected/final attempt process as closely as practical.

Document any parity approximation where the discrete simulator can generate multiple attempts in one segment but the event model schedules attempts one at a time.

Do not compensate by tuning.

# Takedown attempt and result

A takedown attempt is a primary event.

At DISTANCE:

```text
TakedownAttempt
-> TakedownFailed -> remain DISTANCE
or
-> TakedownLanded -> transition to GROUND
```

The attempt event itself must be counted whether it succeeds or fails.

Use the legacy blended wrestling consumer for attempt frequency.

Use current `wrestling_conversion` vs opponent `td_defense` logic for success/failure.

Do not let `control_imposition` become the new intrinsic TD rate in this phase; it appears only through the legacy blended consumer because parity requires it.

No ground mechanics after landing are required in Phase 2A beyond phase/controller state sufficient to represent the transition.

# Clinch entry

Port current effective DISTANCE -> CLINCH entry logic as a scheduled primary event or attempt family in a way consistent with the existing simulator.

Successful entry should transition phase to `CLINCH` and assign positional ownership only if the current semantics support it.

Do not add clinch strikes, clinch TDs, separation, cage control, or other Phase 3 behavior.

# Event/state consequences

Reuse Phase 1 engine-owned state mutation.

Do not modify state directly from the action-rate provider.

For Phase 2A, a primary action resolver may return the minimal delta needed for:

- phase change;
- controller assignment/clearing;
- terminal state is not expected because finishes are out of scope.

Use consequence events for landed/missed/failed/landed observations as needed.

Do not yet generalize the full later damage -> KD -> finish sequential-delta pipeline unless Phase 2A actually requires it. The Phase 1 one-delta seam is a known future extension, not a Phase 2A blocker.

# Stats / diagnostics

Add compact Phase 2A observation support sufficient to compare old vs new mechanics without using stats as engine communication.

At minimum be able to calculate per path / aggregate:

- significant/distance strike attempts;
- significant/distance strikes landed;
- strike landing percentage;
- TD attempts;
- TD landed;
- TD success percentage;
- clinch entries;
- transition timestamps / inter-event timing;
- phase reached after TD/clinch success.

For diagnostics, expose per-fighter action-rate audit rows containing the effective legacy probability and continuous rate.

# Required validation layers

## A. Formula-level parity tests

For controlled synthetic/profile inputs, test that:

- current final legacy probability is reproduced exactly or to floating tolerance;
- probability -> per-second rate conversion is exact;
- converting the rate back over the same interval recovers the original probability;
- TD attempt frequency uses the legacy blended wrestling consumer;
- `wrestling_conversion` changes TD success probability without changing the intrinsic pre-conversion attempt hazard except where the current legacy code explicitly couples them;
- opponent `td_defense` changes TD success, not the legacy attempt calculation except where the actual current consumer explicitly does so;
- no ontology-correct Phase 2B behavior has leaked in.

## B. Matched-state Monte Carlo parity

Build a small direct parity harness that can compare the current legacy DISTANCE mechanics and EVENT MC V1 Phase 2A under matched fighter inputs.

Because the time processes differ, do not expect path-by-path equality.

Compare distributional/aggregate outputs over enough simulation time/paths for stable diagnostics:

- strike attempts per minute;
- strike landing %;
- TD attempts per 15 minutes or per comparable distance exposure;
- TD success %;
- clinch entries per minute;
- event timing distribution where useful.

Use stable seeds.

Do not tune Phase 2A to force exact agreement. Report differences and identify whether they arise from discrete multi-event-per-segment behavior vs continuous competing hazards.

## C. Frozen real-fixture diagnostics

Use the existing frozen FSR-32 artifact and at least these cases if they resolve cleanly:

1. Rob Font vs Raul Rosas Jr. — required, non-replaceable wrestling stress case
2. Derrick Lewis vs Chris Daukaus — striking/power profile, but only distance attempts/landing are in scope
3. Max Holloway vs Calvin Kattar — high-volume striking
4. Merab Dvalishvili vs Petr Yan — sustained wrestling tendency

Charles Oliveira vs Dustin Poirier may be included for distance/grappling initiation diagnostics, but submission mechanics are not in scope.

The goal is not full fight winner prediction yet. Compare only the Phase 2A observables.

For Font/Rosas specifically report both:

- each fighter's FSR inputs relevant to the blended wrestling consumer;
- computed legacy blended wrestling preference;
- current effective distance TD interval probability;
- EVENT MC per-second TD hazard;
- aggregate TD attempts over equal DISTANCE exposure;
- TD success rate.

Do not tune to reproduce the previous full-fight 4.49 Rosas TD attempts/path because Phase 2A does not yet include full clinch/ground/re-entry flow.

# Important parity interpretation

Continuous competing hazards can legitimately change aggregate behavior even when each individual hazard is derived from the same legacy interval probability because the legacy simulator may allow several action families/counts inside one discrete segment.

Therefore classify discrepancies into categories such as:

1. exact formula mismatch — bug;
2. unit/conversion mismatch — bug;
3. event competition effect — expected architectural difference;
4. discrete multi-attempt count vs one-at-a-time event process — expected/needs later modeling decision;
5. missing downstream phase mechanics — out of scope for Phase 2A;
6. semantic wrestling correction — defer to Phase 2B.

Do not erase categories 3–5 by calibration tuning in this phase.

# Package direction

Add only what is needed under `pipeline/simulation/event_mc_v1/`, likely small modules such as:

```text
components/profiles.py
components/action_rates.py
components/strikes.py
components/takedowns.py
components/phase.py
```

or an equivalently compact structure.

Do not create empty future modules.

Tests go under:

`tests/simulation/event_mc_v1/`

Diagnostics may live under:

`pipeline/simulation/event_mc_v1/diagnostics/`

or a small experimental script if that better matches repo conventions.

# Existing code protection

Before finalizing, confirm:

- current simulator files unchanged;
- FSR builders/ratings/ontology unchanged;
- calibration constants unchanged;
- frozen parquet unchanged/uncommitted;
- Phase 1 generic scheduler/clock invariants preserved;
- Phase 2B semantics not introduced;
- no stamina/damage/KD/KO/sub/recovery/age/judging mechanics introduced.

# Testing standard

Prioritize tests that protect calculations and modular swap points over exhaustive defensive validation.

Run:

1. Phase 1 + Phase 2A EVENT MC tests;
2. relevant existing simulator regression tests for the formulas/consumers referenced;
3. compileall on touched source trees;
4. compact parity diagnostics.

Do not turn this phase into a repository-wide cleanup task.

# Required return

Return a concise but complete report containing:

1. exact starting and final SHA;
2. files added/changed;
3. current legacy source functions/formulas identified for each action family;
4. profile/input adapter design;
5. exact strike attempt probability/rate mapping;
6. exact strike landing mapping;
7. exact legacy blended TD-attempt mapping and formula;
8. exact TD success mapping;
9. exact clinch-entry mapping;
10. formula-level parity test results;
11. matched-state Monte Carlo parity summary;
12. frozen fixture diagnostic summary, especially Font/Rosas;
13. all meaningful differences classified as bug vs expected continuous-time effect vs out-of-scope downstream behavior;
14. tests/commands/results;
15. confirmation current simulator/FSR/calibration untouched;
16. confirmation Phase 2B and all later mechanics were not started;
17. commit/PR status;
18. final line exactly:

`PHASE 2A DISTANCE TEMPORAL PARITY GATE: PASS`

or

`PHASE 2A DISTANCE TEMPORAL PARITY GATE: FAIL`

with blockers.

Do not begin Phase 2B.