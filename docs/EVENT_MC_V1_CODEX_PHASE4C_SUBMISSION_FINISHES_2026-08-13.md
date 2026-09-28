# EVENT MC V1 — Phase 4C Submission Finish Mechanics

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Status entering this phase

- Phase 0: PASS
- Phase 1: PASS
- Phase 2A: PASS
- Phase 2B: PASS
- Phase 3 CLINCH/GROUND flow + observational SUB attempts: PASS
- Phase 4A stamina + dynamic modifiers: PASS
- Phase 4B1 impact/trauma/KD mechanics: PASS
- Phase 4B1 config externalization: PASS
- Phase 4B2 KO/TKO finish mechanics: PASS
- Phase 4B2 single-fight runner: PASS
- KD/KO population calibration: intentionally deferred
- real weight-class tuning: not active
- judging: not authorized

A narrow terminal-action accounting fix is currently being completed separately. **Do not begin Phase 4C implementation until that fix is present on the branch and all EVENT MC tests pass.** If the branch does not yet contain that fix, stop safely and report the blocker rather than coding around it.

The known accounting bug/fix contract is: the already-resolved finishing action must still emit/count its own `ActionOutcome`, while no new future primary event occurs after terminal state; exactly one `FightFinished` remains.

## Objective

Turn existing nonterminal `submission_attempt` events into a clean, probabilistic terminal submission system while preserving the current separation between:

1. **attempt generation** — how often a fighter attacks submissions; and
2. **attempt conversion** — how dangerous a given attempt is once it occurs.

Do not retune submission-attempt frequency in this phase.

This is a **mechanics/architecture phase, not population calibration**.

## First task: audit existing submission traits and legacy conversion logic

Before coding:

1. inspect the current frozen FSR-32 schema and `FighterProfile` adapter;
2. identify exactly which existing FSR traits can legitimately own submission offense and defense;
3. trace the legacy/static simulator submission conversion implementation and later overrides;
4. document what concepts should be preserved, rejected, or simplified.

Current `FighterProfile` definitely exposes `submission_pressure`, `control_imposition`, `control_resistance`, `reversal_ability`, stamina traits, and other Phase 3 fields. Do **not** invent a new FSR trait or rebuild FSR-32 merely because there is no dedicated `submission_defense` field in the current adapter.

If frozen FSR-32 contains an already-existing leakage-safe submission-defense-style trait that is simply not yet adapted, report it and propose the smallest adapter addition. If no such trait exists, use a transparent derived defensive resistance from already-approved traits and expose the blend coefficients in config.

## Required architecture

The active ground event flow should become conceptually:

```text
submission attempt is selected by existing competing-risk scheduler
-> existing primary ActionAttempt / ActionOutcome semantics remain intact
-> SubmissionFinishModel evaluates that attempt
-> attacker submission threat
-> defender submission resistance
-> positional/context adjustment
-> optional current stamina adjustment
-> probabilistic P(SUB)
-> if success: terminal StateDelta(method=SUB, winner=attacker)
-> exactly one FightFinished
-> no later primary events
```

Use composition. Do not add submission physics to the generic scheduler.

## Trait ownership and modeling constraints

### Submission attempt frequency

Keep Phase 3 ownership unchanged:

- `submission_pressure` drives submission-attempt generation;
- top/bottom attempt multipliers remain in the existing attempt-rate layer;
- do not retune those values in Phase 4C.

### Submission conversion

Submission conversion must be a separate model.

Preferred initial structure:

```text
base attacker threat = approved submission-offense signal
base defender resistance = approved defensive signal(s)
matchup edge = threat - resistance
+ small explicit positional/context adjustment
+ at most one clean current-stamina effect if justified
-> sigmoid/logistic P(SUB)
```

Do not force this exact formula if the existing FSR ontology/legacy trace supports a cleaner equivalent, but preserve the separation of concerns.

### Stamina

Unlike striking power, current stamina may legitimately influence submission conversion because sustained submission offense/defense is physically costly. However:

- keep it simple;
- do not create multiple fatigue pathways;
- do not use stamina to alter attempt frequency again here;
- do not double-count stamina through both several attacker and defender terms unless independently justified;
- all stamina coefficients must be explicit config values.

A clean first version may use a single matchup stamina-edge term or one attacker/defender multiplier pair. Prefer the smallest explainable model.

### Position/context

Submission attempts currently occur from GROUND and may be top or bottom. Conversion may distinguish top vs bottom if justified, but do not create submission-type/body-part/position trees.

Do not add named techniques (RNC, armbar, guillotine, etc.) in this phase.

## Configuration

Add a dedicated section to `config/event_mc_v1.yaml`, e.g. `submission_finish`.

Every behavior-changing numerical coefficient introduced by this phase must live there and inherit through the existing:

```text
global defaults
+ optional partial weight-class override
= effective EventMCCalibration
```

The committed `weight_classes` mapping remains empty/behaviorally neutral. No real division-specific tuning yet.

Do not hard-code tunable submission-finish numbers in Python.

## RNG

Use the existing stable SUBMISSION RNG stream unless architecture review shows a compelling reason otherwise. Do not create hidden RNGs or consume scheduler/DAMAGE/KD streams.

The same root seed must remain reproducible.

## Terminal lifecycle

Submission success must use the existing terminal state contract:

- `finished=True`
- `winner=<attacker side>`
- `finish_method="SUB"` (or the repo's exact normalized method string if a convention already exists)
- meaningful `finish_reason`
- exactly one `FightFinished`
- the already-resolved submission attempt/outcome must still be observed/accounted for
- no new primary events after terminal state
- no post-finish time advance

Do not duplicate KO/TKO terminal machinery unnecessarily; reuse/generalize the smallest shared lifecycle seam if useful, but avoid a large refactor.

## Single-fight runner

Extend the existing Codespaces runner so a traced submission attempt displays, where applicable:

```text
SUBMISSION CHECK attacker->defender
threat=...
resistance=...
position=top|bottom
stamina/context term=...
pSUB=...
finished=True|False
```

On success, path summary must report winner and `SUB` method.

Multi-path summary should distinguish at minimum:

- KO/TKO wins
- SUB wins
- scheduled horizon
- submission attempts/path by side
- submission conversions / finishes
- P(SUB | attempt) overall and by side when denominator > 0

Do not add judging yet, so scheduled-horizon paths remain unresolved terminal horizons for now.

## Historical diagnostics — descriptive only

Build a descriptive historical submission anchor from authoritative master data if available:

- completed fights
- SUB finish rate
- submission attempts/fight if the master supports it cleanly
- share with >=1 submission attempt if supported
- P(SUB finish | >=1 recorded attempt) only if the data supports that interpretation without leakage/definition ambiguity

Also run the five frozen fixtures or another small approved mechanics set and report:

- SUB attempt count/path
- SUB finish rate
- P(SUB | attempt)
- top vs bottom attempts and conversion if available
- average submission finish time / round
- KO/TKO vs SUB vs scheduled horizon distribution
- runtime

Do **not** tune to these anchors in Phase 4C. Expose obvious misses and stop.

## Required tests

At minimum:

1. higher attacker submission offense raises P(SUB);
2. stronger defender resistance lowers P(SUB);
3. any configured top/bottom adjustment moves probability in the intended direction;
4. any stamina term moves probability monotonically in the intended direction;
5. finite values always remain probabilistic, not a hard threshold;
6. deterministic same-seed sampling;
7. stochastic variation across seeds at intermediate probability;
8. successful SUB sets structured terminal state;
9. exactly one `FightFinished`;
10. the finishing submission attempt/outcome is still counted;
11. no later primary event occurs;
12. weight-class override reaches the submission-finish curve while unspecified values inherit defaults;
13. single-fight trace displays submission check fields;
14. aggregate summary separates KO/TKO, SUB, and scheduled-horizon outcomes;
15. frozen FSR-32 SHA remains exactly unchanged.

## Non-goals / prohibited work

Do not add or change:

- submission-attempt frequency calibration;
- KD calibration;
- KO/TKO calibration;
- judging/decision scoring;
- age effects;
- tactical urgency;
- body-part damage;
- named submission techniques;
- doctor/injury stoppages;
- trauma recovery;
- FSR rebuild;
- wrestling-entry ontology;
- Phase 3 phase-rate tuning;
- Phase 4A stamina tuning;
- real weight-class values;
- legacy inheritance simulator.

## Scope philosophy

Keep this small, modular, predictive-oriented, and easy to iterate.

We expect later population calibration. The goal now is not a perfect grappling simulator; the goal is a coherent terminal submission mechanism whose errors can be diagnosed independently from attempt generation.

## Required delivery

Return:

1. legacy submission conversion trace and preserve/reject decisions;
2. exact traits used for attacker threat and defender resistance, with rationale;
3. exact formula and config keys/initial mechanics values;
4. implementation commit SHA;
5. tests and results;
6. five-fixture/small mechanics diagnostic output;
7. descriptive historical submission anchor;
8. exact single-fight Codespaces command showing a submission attempt/check if one can be reproduced deterministically;
9. frozen FSR checksum;
10. explicit statement that no population tuning occurred.

Stop after implementation, diagnostics, tests, commit/push, and report.

Expected final line:

`PHASE 4C SUBMISSION FINISH MECHANICS GATE: PASS`

or

`PHASE 4C SUBMISSION FINISH MECHANICS GATE: FAIL`
