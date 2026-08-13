# Codex Prompt — EVENT MC V1 Phase 4B2 KO/TKO Finishes

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Source branch: `feature/fsr-32-stamina-shadow`
Architecture revision: v0.3

## Status before this phase

- Phase 0: PASS.
- Phase 1 generic continuous-time kernel: PASS.
- Phase 2A DISTANCE parity: PASS.
- Phase 2B wrestling-entry ontology: PASS.
- Phase 3 CLINCH + GROUND flow: PASS.
- Phase 4A stamina + dynamic modifiers: PASS.
- Phase 4B1 impact + trauma + knockdown architecture: PASS.
- Phase 4B1 KD calibration: OPEN / intentionally deferred by user.
- Phase 4B1 config externalization: PASS after independent review of commits `b8b2b870595c9cc62255b4d63b7c20de56e9550f` and `77661563c3a833a3e87b60e4b3ae6caecd648cb8`.
- Phase 4B2 KO/TKO mechanics: explicitly authorized by the user.
- Terminal submissions, judging, age, tactical urgency, and later phases: NOT authorized.

Codex cloud may use local branch `work`. Verify ancestry/content rather than branch-name equality.

Before implementation:

```bash
git fetch origin --prune
git merge-base --is-ancestor 77661563c3a833a3e87b60e4b3ae6caecd648cb8 HEAD
```

If stale, safely rebase onto:
`origin/feature/fsr-32-stamina-shadow`

Then verify:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE4B2_KO_TKO_FINISHES_2026-08-13.md
config/event_mc_v1.yaml
```

## Development philosophy

The user has locked:

**WORKING + PREDICTIVE + MODULAR + EASY TO ITERATE**

This phase builds terminal KO/TKO mechanics on top of the reviewed physiology chain. It is NOT a calibration phase.

The current KD rate is known to be materially above the historical population anchor. Do not hide or compensate for that upstream miss by artificially lowering finish conversion. Build the finish layer cleanly and expose its behavior; calibration comes later.

## Goal

Add a modular probabilistic KO/TKO finish model so a landed strike can terminate a path when the same-timestamp physiology state indicates a stoppage-worthy event.

Target consequence chain:

```text
landed strike
-> pre-action effective power
-> stochastic impact
-> primary trauma
-> cumulative trauma update
-> current KD resistance
-> probabilistic KD
-> acute-vulnerability consequence
-> current finish resistance
-> probabilistic KO/TKO finish
-> engine applies terminal delta
```

No simulation time elapses between these stages.

## Hard ownership locks

### Power
`striking_power` and stamina-powered expression enter physiology ONCE through impact.

Do NOT multiply either directly into finish probability again.

### Trauma
`cumulative_trauma` lowers future/current resistance. It is not a deterministic health-bar threshold and must never directly force a finish.

### Acute vulnerability
Acute vulnerability may lower current finish resistance. Do not also add redundant recent-KD multipliers that count the same collapse twice unless the legacy trace proves a distinct effect and the implementation documents it.

### KD
KD may be an explicit finish-conditioning input because a knocked-down fighter is in a different stoppage state. Keep this separate from power ownership.

## Required legacy trace before coding

Trace the final effective KO/TKO behavior through the relevant inheritance chain, including at minimum:

- `StaticFSRMCKOTKOV2`
- `StaticFSRMCKOTKOV2KDCollapse`
- `StaticFSRMCKOTKOV2RoundRecovery`
- V3/V3.1/V3.2/V3.3 stamina layers
- any later audit/full-fight subclasses that override finish behavior

Document:

- which values feed KO/TKO probability;
- whether finish is attempted only after KD or after any sufficiently damaging strike;
- whether direct one-shot KO without a recorded KD can occur;
- recent-KD/collapse effects;
- follow-up damage behavior;
- any deterministic reservoir exhaustion;
- phase-specific differences;
- round-recovery effects;
- any duplicated power/damage/KD terms.

Then explicitly separate:
1. useful concepts to preserve;
2. inherited/double-counted behavior to reject.

Do not port the health-bar exhaustion architecture.

## Finish model architecture

Create a small compositional `FinishModel` (name may vary) that consumes the existing Phase 4B1 physiology outcome/state rather than recomputing impact from fighter ratings.

Conceptually:

```text
FinishInputs:
    attacker
    defender
    phase
    impact
    primary_trauma
    cumulative_trauma
    acute_vulnerability
    knockdown
    current_finish_resistance

FinishOutcome:
    finish_probability
    finished
    method = KO_TKO
```

The finish model must not reach into stamina internals.

## Current finish resistance

Use a derived value, not another mutable reservoir.

Prefer the smallest coherent relationship supported by the existing FSR traits and the Phase 4B1 architecture.

If FSR-32 contains a distinct valid finish/durability trait appropriate for baseline finish resistance, use it only after verifying its ontology and frozen availability.

If no distinct trait exists, derive finish resistance from the already-owned durability/KD-resistance traits rather than inventing a new FSR rating.

Whatever choice is made, document it clearly.

Required directionality:

- higher baseline resistance -> lower finish probability;
- more cumulative trauma -> lower current finish resistance;
- more acute vulnerability -> lower current finish resistance;
- higher impact/current-finish-resistance ratio -> higher finish probability;
- a KD must not reduce finish probability.

## Finish probability

Use one compact probabilistic curve over an interpretable impact/current-resistance quantity, optionally conditioned on KD.

Fresh one-shot KO/TKO must remain possible; do not require prior cumulative trauma.

Do not create a deterministic trauma threshold.

Do not introduce a second independent power term.

Do not create separate DISTANCE/CLINCH/GROUND physiology models. Phase may enter only through a small context modifier if legacy tracing strongly supports it; otherwise use one shared finish model initially.

## KO vs TKO naming

For Phase 4B2, one terminal method bucket `KO_TKO` is sufficient unless the current architecture already has a clean, evidence-supported distinction.

Do not spend this phase building referee ontology, doctor stoppages, injury stoppages, or technical-decision rules.

## Engine terminal semantics

The engine remains sole state mutator.

A successful finish must:

- set terminal state exactly once;
- preserve the exact current timestamp;
- identify the winning side in structured form if the current state/result contract supports it cleanly;
- expose method `KO_TKO` in structured form or the smallest compatible equivalent;
- emit one observer-visible finish consequence/outcome;
- stop scheduling further primary events immediately after terminal application;
- produce exactly one `FightFinished` lifecycle notification.

Do not create another clock.

If minimal structured winner/method fields are needed in `FightState` / `StateDelta`, add only those fields. Avoid building the judging/unified-results layer in this phase.

## RNG ownership

Use the existing stable `KNOCKDOWN_FINISH` stream for finish sampling unless the current RNG architecture already provides a separately locked finish stream.

Do not add order-dependent hidden RNGs.

Be explicit about draw ordering relative to the KD draw. Equal root seeds must remain deterministic.

## Config requirement

All new tunable finish coefficients must be added to:

`config/event_mc_v1.yaml`

under a dedicated section such as:

```yaml
finish:
  ...
```

The same resolved `EventMCCalibration` must reach the finish model so future weight-class overrides work automatically.

Do not put new calibration literals in Python except structural/numerical-safety constants.

The committed `weight_classes` mapping remains empty. No real weight-class tuning now.

## Absolute non-goals

Do NOT add or tune:

- KD calibration;
- strike-rate calibration;
- stamina calibration;
- KO/TKO population calibration;
- terminal submissions;
- judging/decisions;
- age transforms;
- tactical urgency;
- score effects;
- body-part damage;
- doctor stoppages;
- injury stoppages;
- trauma recovery;
- defender-stamina resistance penalties;
- broad phase-flow retuning;
- FSR rebuilds/ontology changes;
- legacy simulator changes.

## Tests required

At minimum add focused tests proving:

1. Missed strike cannot finish.
2. Non-landed/non-strike event cannot finish.
3. Higher impact ratio monotonically raises finish probability.
4. Higher baseline finish resistance lowers finish probability.
5. More cumulative trauma raises finish probability through lower resistance.
6. More acute vulnerability raises finish probability through lower resistance.
7. KD does not reduce finish probability and, if explicitly modeled, raises it monotonically.
8. Fresh one-shot finish remains possible.
9. No deterministic cumulative-trauma threshold exists.
10. `striking_power` and stamina are not multiplied directly in finish probability after impact is formed.
11. Finish sampling is deterministic for equal root seeds and stochastic across seeds.
12. Successful finish applies terminal state once and stops future primary events.
13. Exactly one `FightFinished` lifecycle event occurs.
14. Default config + same seeds preserve Phase 4B1 behavior whenever the finish model is disabled/not injected.
15. Synthetic resolved config override reaches finish coefficients.
16. Weight-class mapping remains empty/neutral in committed config.
17. Frozen FSR-32 checksum remains unchanged.

## Diagnostics

Run the five frozen fixtures:

- Lewis vs Daukaus
- Holloway vs Kattar
- Font vs Rosas
- Merab vs Yan
- Oliveira vs Poirier

Report at minimum:

- KO/TKO finish rate;
- average finish time among finishes;
- round distribution of finishes;
- fraction of finishes occurring on KD strikes;
- fraction of direct finishes without KD;
- impact ratio at finish;
- cumulative trauma at finish;
- acute vulnerability at finish;
- KDs/path before termination;
- scheduled-horizon rate;
- runtime/throughput.

Also build or reuse a descriptive historical KO/TKO anchor if current master fields support method and finish round/time.

Important: because KD calibration is intentionally deferred, these diagnostics are **mechanical exposure only**. Do not tune parameters to match the historical KO/TKO anchor in this phase and do not claim predictive performance.

## Scope protection

Must remain unchanged:

- Phase 2B wrestling-entry ontology;
- Phase 3 phase-flow formulas;
- Phase 4A stamina mechanics;
- Phase 4B1 impact/trauma/KD formulas and their current known calibration miss;
- config defaults already externalized;
- inheritance-based simulator;
- frozen FSR-32 artifact.

## Required final report

Return:

- starting SHA;
- final SHA;
- commit(s);
- legacy finish trace;
- exact new finish formula and parameter provenance;
- config keys added;
- tests run/results;
- five-fixture mechanics diagnostics;
- historical KO/TKO descriptive anchor if available;
- confirmation KD calibration was not changed;
- frozen FSR checksum;
- scope-protection statement.

End with exactly one of:

`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: PASS`

or

`PHASE 4B2 KO/TKO FINISH MECHANICS GATE: FAIL`

Stop there. Do not begin terminal submissions, judging, age, tactical urgency, calibration, or later phases.