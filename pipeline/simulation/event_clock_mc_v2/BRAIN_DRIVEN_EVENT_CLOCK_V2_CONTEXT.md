# Brain-Driven Event Clock V2 — Persistent Architecture Context

Date: 2026-08-24
Branch: `research/weight-class-audit-20260823`
Status: authoritative architecture context for the upcoming clean refactor

## Purpose

This document exists to preserve the agreed architecture and implementation constraints across chats, Codex tasks, code reviews, and future refactor stages.

If an implementation choice conflicts with this document, stop and resolve the conflict before coding further.

This document is the integration source of truth for the brain-driven Event Clock V2 refactor. The older `STANDARD_FIGHTER_V1_DESIGN.md` remains useful for Standard Fighter behavioral concepts, but any older wording that implies the brain merely biases independently generated Event Clock events is superseded by this document.

---

## Core causal contract

The simulator must obey this causal order:

`CURRENT FIGHT STATE -> BRAIN DECIDES WHEN TO ACT -> BRAIN CHOOSES LEGAL ACTION FAMILY -> ACTION EVENT ENTERS AUTHORITATIVE TIMELINE -> EXISTING/EXPLICIT MECHANICS RESOLVE EVENT -> CONSEQUENCES UPDATE STATE -> BRAIN LATER REEVALUATES`

### Ownership

The Standard Fighter brain owns:

- fighter-initiated event timing;
- fighter-initiated action-family selection;
- state-responsive tactical intent;
- stochastic variation in action choice;
- stochastic variation in when the fighter next initiates an action.

Mechanics own:

- success/failure of the attempted action;
- strike landing;
- takedown completion/defense;
- clinch entry success if modeled as contested;
- clinch takedown success;
- escape/reversal success;
- submission conversion;
- damage;
- knockdowns;
- KO/TKO stoppage;
- stamina costs and recovery where already defined;
- judging and scoring consequences.

The brain must never directly award itself a successful outcome.

### Hard rule

Every fighter-initiated timeline event must originate from a brain decision.

Do not retain a parallel independent fighter-action scheduler that can generate strikes, takedowns, clinch entries, ground attacks, or submissions without a brain decision.

Reactive and consequence events may be emitted by mechanics when they are causally downstream of a fighter-initiated event. Examples include damage, knockdown, stoppage, defensive resolution, score updates, and state-transition consequences.

---

## Authoritative fight timeline

Event Clock V2 must maintain a real chronological fight state.

The timeline is not reconstructed after the fact and is not inferred from event families.

At every instant the fight has exactly one authoritative physical phase:

- `STANDING`
- `CLINCH`
- `GROUND`

For policy purposes, `GROUND` plus the authoritative ground controller is translated into:

- `GROUND_TOP`
- `GROUND_BOTTOM`

The MVP does not require detailed ground positions.

### Phase timeline invariants

- Every elapsed second belongs to exactly one phase.
- Phase segments cannot overlap.
- There are no gaps between phase segments while the fight is active.
- A standing strike cannot occur outside `STANDING`.
- A clinch strike cannot occur outside `CLINCH`.
- A ground strike cannot occur outside `GROUND`.
- `clinch_controller` is populated only when meaningful in `CLINCH`.
- `ground_controller` is populated only when meaningful in `GROUND`.
- Round start returns the authoritative phase to `STANDING` unless a later explicit design decision says otherwise.

A phase segment should be explainable by:

- start time;
- end time;
- phase;
- controller if applicable;
- entry cause;
- exit cause.

---

## MVP phase graph

Required transitions:

`STANDING -> CLINCH` via successful clinch entry.

`STANDING -> GROUND` via successful direct takedown.

`CLINCH -> STANDING` via separation/break.

`CLINCH -> GROUND` via successful clinch takedown.

`GROUND -> STANDING` via successful escape/get-up or deliberate disengage when legal.

`GROUND -> GROUND` with controller flip via successful reversal.

No hidden phase transitions.

Every phase transition must be emitted by an explicit resolved event or round boundary.

---

## Legal action menus

The brain may only select actions legal in the current authoritative phase.

### Standing

- `STAND_ATTACK`
- `STAND_COUNTER`
- `PRESSURE`
- `RESET_RANGE`
- `CLINCH_ENTRY`
- `TAKEDOWN_ENTRY`

### Clinch

- `CLINCH_STRIKE`
- `CLINCH_CONTROL`
- `CLINCH_TAKEDOWN`
- `BREAK_CLINCH`

### Ground top

- `GROUND_STRIKE`
- `ADVANCE_POSITION`
- `SUBMISSION_ATTACK`
- `CONTROL`
- `DISENGAGE`

### Ground bottom

- `ESCAPE_STAND`
- `IMPROVE_POSITION`
- `REVERSAL`
- `SUBMISSION_ATTACK`
- `BOTTOM_STRIKE`

Some actions may initially be observational/non-terminal or have deliberately simple mechanics during staged implementation. They must still obey phase legality and ownership boundaries.

---

## Brain outputs

The brain must answer two distinct questions.

### 1. When should this fighter initiate the next action?

This is the fighter activity/initiative process.

Conceptually:

`next_action_delay = f(base_activity, phase, stamina, urgency, hurt_state, opponent_hurt_state, recent_state)`

The exact MVP formula is to be specified and validated separately.

Do not conflate activity rate with action mix.

If total action volume is wrong, activity timing is a candidate source.

### 2. What legal action should the fighter choose?

Conceptually:

`P(action | current_state, matchup_capability, recent_memory, urgency, phase)`

If total activity is correct but strike/TD/clinch mix is wrong, action selection is the candidate source.

This separation is required for diagnosability.

---

## FSR role

FSR V3 remains capability and matchup information, not a separate event generator.

Examples:

- standing rate/accuracy traits inform the value and/or baseline feasibility of standing offense;
- takedown tendency/completion matchup informs the value and feasibility of wrestling entries;
- ground traits inform the value and feasibility of ground offense;
- power, durability, KD resistance, submissions, escape traits continue to feed the mechanics that resolve consequences.

Do not add a separate path-level latent intensity-prior layer in the MVP unless later diagnostics demonstrate that validated FSR + brain activity cannot represent observed activity.

Avoid double counting the same fighter tendency in both a hidden event generator and the brain.

---

## Clinch is restored as a real phase

Clinch must not be treated as a decorative label or a transient alias of standing.

The authoritative phase model includes persistent `CLINCH` exposure.

The MVP must support:

- standing-to-clinch entry;
- clinch duration;
- clinch strikes;
- clinch control intent;
- clinch takedowns;
- clinch-to-standing separation;
- clinch-to-ground transition.

Current Standard Fighter capability translation uses a neutral placeholder for clinch. That is acceptable during early structural validation only.

Do not invent a fake fighter-specific clinch trait merely to make the refactor run.

A separate empirical clinch-capability workstream can follow once the causal timeline is structurally correct.

---

## Time concepts

Keep these concepts distinct.

### Fight clock

Absolute elapsed fight time.

### Phase clock

Elapsed time since the current phase began.

### Brain/action clock

Time until a fighter next initiates an action.

The brain should not necessarily reroll tactical intent after every landed jab. The implementation must support meaningful reevaluation cadence and event-triggered reevaluation without coupling policy ticks to every mechanical consequence.

---

## Clean refactor requirements

This is a clean refactor, not an adapter layer around the current scattered-event path.

### Required code-quality constraints

- No monkey patches.
- No runtime method replacement.
- No hidden mutation hooks.
- No compatibility wrappers whose only purpose is to keep two contradictory simulator architectures alive indefinitely.
- No duplicate authoritative fight states.
- No second shadow timeline once the causal engine becomes authoritative.
- No broad inheritance hierarchy.
- Prefer composition over inheritance.
- Use inheritance only when there is a clear stable interface and genuine substitutability; default expectation is minimal or zero custom inheritance.
- No named-fighter special cases.
- No archetype classes such as Boxer/Wrestler/Brawler.
- No giant God object that owns policy, mechanics, state, logging, and calibration together.
- No circular imports between brain, mechanics, state, and diagnostics.
- No silent fallback to old Event Clock event generation when the new causal engine is active.

### End-state requirement

Temporary research scaffolding is allowed only if it has an explicit deletion stage.

At the end of the refactor:

- there is one authoritative Event Clock V2 causal execution path;
- dead wrappers/adapters/temporary compatibility shims are deleted;
- old non-causal event-generation code is either removed from the new execution namespace or clearly retained only as archived/frozen reference code outside the production path;
- tests target the final architecture rather than patching around legacy behavior.

The finished codebase should be simpler to reason about than the starting codebase.

---

## Frozen boundaries during structural refactor

Unless a later explicitly approved stage says otherwise, do not redesign or tune:

- FSR V2;
- FSR V3 validated trait construction;
- damage formulas;
- knockdown formulas;
- KO/TKO formulas;
- submission conversion mechanics;
- stamina mechanics;
- judging mechanics;
- sportsbook/market calibration;
- cold-start solving.

Cold starts remain reporting/flagging only for this workstream.

Event MC V1 is reference material only. Borrow clean architectural concepts; do not inherit from or wrap Event MC V1 production classes as the new design.

---

## Reference lessons from Event MC V1

Useful ideas to preserve conceptually:

- an authoritative mutable fight state;
- one authoritative clock;
- explicit state deltas/transitions;
- phase-specific legal actions;
- explicit clinch and ground transitions;
- event sinks/diagnostics separated from mechanics;
- engine-owned state application.

Do not reproduce Event MC V1's later two-state standing/ground collapse.

Do not make Event MC V1 a parent class or runtime dependency of the new Event Clock V2 causal engine.

---

## Validation philosophy

Structural validity precedes predictive calibration.

First prove:

- timeline conservation;
- phase legality;
- deterministic state transitions;
- brain event ownership;
- mechanics outcome ownership;
- reproducibility with seeded RNG streams;
- no event/state overlap or impossible sequencing.

Then prove behavioral plausibility with single-path traces.

Then validate population mechanics and exposure.

Only after the architecture is structurally and behaviorally credible should moneyline or method calibration influence changes.

---

## Canonical debugging decomposition

When output is wrong, diagnose in this order:

1. Did the brain initiate actions too frequently or too slowly?
2. Given an action opportunity, did the brain choose the wrong family mix?
3. Was the action legal in the current phase?
4. Did the transition mechanic resolve correctly?
5. Did consequence mechanics resolve correctly?
6. Did the resulting state update correctly?
7. Did the brain observe the updated state correctly on its next decision?

Examples:

- Too few total actions -> inspect brain activity timing.
- Correct action count but too few strikes -> inspect action selection.
- Correct TD attempts but too many completions -> inspect TD resolution mechanics, not policy.
- Correct landed strikes but too many KOs -> inspect damage/KD/finish mechanics, not brain timing.

This decomposition is a core reason for the refactor.

---

## Current research state entering refactor

Standard Fighter V1 exists in an isolated research namespace and has passed:

- synthetic directional scenario tests;
- magnitude audit;
- real FSR V3 capability translation audit;
- cold-start reporting audit;
- one single-path shadow trace on Aleksandar Rakic vs Marcin Tybura.

The shadow trace demonstrated useful state-responsive behavior but also exposed the current Event Clock limitation: the frozen detailed path does not maintain a true phase timeline, so phase had to be inferred from event family.

That limitation is the immediate reason for this refactor.

The brain currently has zero causal influence on the production Event Clock path.

---

## Definition of architectural success

The refactor succeeds when a single fight can be explained chronologically as:

`state -> fighter brain initiates action -> action appears on timeline -> mechanics resolve -> phase/consequence state updates -> next brain decision`

with real standing, clinch, and ground exposure and no parallel hidden action generator.
