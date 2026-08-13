# EVENT MC V1 — Phase 5A Deterministic Judging / Decisions

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Purpose

Implement deterministic round judging for EVENT MC V1 so every scheduled-horizon path resolves to a decision winner.

This is a mechanics/architecture phase, not population calibration.

## User locks

- NO draws.
- NO 10-8 rounds.
- NO 10-7 rounds.
- Every completed round must have exactly one winner and score exactly 10-9.
- No three-judge simulation yet.
- No split/majority decisions yet.
- No judge-noise model yet.

Because scheduled UFC bouts have an odd number of rounds, if every round has exactly one 10-9 winner the final fight result cannot tie.

## Judging hierarchy

Preserve the MMA judging hierarchy discussed with the user:

1. Effective striking + effective grappling are the PRIMARY criteria.
2. Effective aggression is considered only when primary effectiveness is effectively tied.
3. Fighting-area / Octagon control is considered only when primary effectiveness and effective aggression are effectively tied.
4. If all modeled criteria are exactly tied, use a deterministic final no-draw tiebreaker. Do not use an uncontrolled coin flip.

Do NOT add Octagon-control points directly to clearly superior effective offense.

Do NOT let raw volume, takedown count, or passive control automatically outweigh meaningful damaging/threatening offense.

## Design principle

The scorer should answer:

> Who produced the more meaningful effective offense in this round?

It should NOT behave like an arcade point system where every stat occurrence has a large fixed reward.

Damage/impact should matter more than raw strike count.

A takedown with no meaningful follow-up should carry much less primary judging value than damaging ground offense or a dangerous submission sequence.

Passive clinch/ground control belongs primarily in the late control tiebreak layer, not the main effectiveness layer.

## Phase 5A architecture target

Build two clean pieces:

### A. RoundJudgingAccumulator

Observer-only round bookkeeping. It must NOT mutate `FightState`.

Accumulate round-local evidence only. Never let Round 1 offense score Round 2.

At minimum track, per side and per round, from existing EVENT MC events/state:

### Effective striking evidence
- landed strike count;
- total strike impact inflicted;
- primary trauma inflicted during THIS round;
- knockdowns inflicted;
- strongest/high-impact strike evidence if useful and already available.

### Effective grappling evidence
- submission attempts;
- submission threat / conversion probability from `SubmissionFinishOutcome`;
- successful takedowns;
- reversals / meaningful positional changes;
- damaging ground offense via existing landed ground-strike impact;
- do not treat passive top time itself as primary effective grappling.

### Effective aggression evidence
Use a small, transparent set of already-generated offensive actions, such as:
- offensive attempts;
- successful phase entries / initiations;
- other existing action evidence that represents attempting to finish or meaningfully attack.

Do not invent a complex new tactical-aggression state.

### Fighting-area / Octagon control evidence
Use existing positional/phase information, primarily:
- ground control time;
- clinch control time;
- successful imposition of fight location/phase where existing observations support it.

Control is a tertiary tiebreak criterion only.

## B. DeterministicRoundJudge / DecisionModel

At each completed round:

1. compute RED and BLUE primary effectiveness;
2. if the effectiveness difference exceeds a configured close-round/tie band, higher side wins 10-9;
3. otherwise compare effective aggression;
4. if aggression is not effectively tied, higher side wins 10-9;
5. otherwise compare fighting-area control;
6. if control is not exactly/effectively tied, higher side wins 10-9;
7. if still tied, use a deterministic final tiebreak from already-observed round evidence.

The final fallback must be stable/reproducible and should not introduce systematic red/blue corner bias if reasonably avoidable.

Possible final fallback hierarchy:
- total successful offensive actions;
- total offensive attempts;
- another stable round-local observable;
- only if literally all round observables are identical, use the existing named JUDGING RNG stream in a seed-reproducible way OR another documented unbiased deterministic mechanism.

If JUDGING RNG is used, use only the existing `RNGStream.JUDGING`; no hidden RNG.

Every RoundScore must be exactly either:
- RED 10-9 BLUE
or
- BLUE 10-9 RED.

No even round representation.

## Scheduled-horizon terminal behavior

Current engine behavior marks horizon as `scheduled_horizon`.

Phase 5A should instead, when the scheduled fight horizon is reached without KO/TKO or SUB:

1. finalize the final completed round;
2. total round winners / 10-9 cards;
3. determine fight winner;
4. return/apply terminal state with:
   - `finished=True`
   - `winner=<red|blue>`
   - `finish_method="DEC"`
   - decision reason/method as appropriate without adding split/unanimous subtype yet;
5. emit exactly one `FightFinished` lifecycle event;
6. emit no later time advance or primary events.

Do not generate a DEC before the scheduled horizon.

KO/TKO and SUB terminal behavior must remain unchanged.

## Scoring formula requirements

Before choosing coefficients:

1. audit existing EVENT MC event payloads and FlowStats data;
2. trace any legacy simulator judging logic for useful concepts;
3. identify what can be measured causally and round-locally from current simulation events;
4. keep the formula compact and interpretable.

Preferred structure:

```text
primary_effectiveness = effective_striking + effective_grappling
```

But do NOT blindly sum raw incompatible units. Normalize/scale components transparently.

Recommended conceptual ownership:

### Effective striking
Primary signal should be strike impact / damaging effectiveness.
Secondary evidence may include:
- knockdown emphasis;
- landed-strike volume only as supporting evidence.

Do not count cumulative trauma from prior rounds. Use only trauma deposited in the current round.

### Effective grappling
Primary grappling effectiveness should emphasize:
- dangerous submission threat;
- damaging ground offense;
- meaningful successful positional offense.

Takedown completion may contribute, but a takedown alone should not overwhelm clearly superior damaging offense.

Passive control time is NOT primary effectiveness.

### Effective aggression
Use only after primary effectiveness is within the configured close-round band.

### Octagon control
Use only after both primary effectiveness and aggression fail to separate the round.

## Configuration

All new tunables go in:

`config/event_mc_v1.yaml`

under a dedicated section such as:

```yaml
judging:
  ...
```

Use the existing effective calibration resolver so future optional weight-class overrides remain structurally possible.

Do not add real weight-class judging overrides now.

Do not silently hardcode active behavior-changing coefficients in Python.

## Single-fight runner

Extend the Codespaces runner so a path that reaches decision prints transparent round cards.

Trace output should include, for each completed round:

```text
ROUND N JUDGING
RED effective striking=...
BLUE effective striking=...
RED effective grappling=...
BLUE effective grappling=...
primary effectiveness diff=...
criterion used=PRIMARY | AGGRESSION | CONTROL | FINAL_TIEBREAKER
round winner=RED|BLUE
score=10-9
```

At the end:

```text
FINAL DECISION
R1 RED 10-9
R2 BLUE 10-9
R3 RED 10-9
RED rounds=2 BLUE rounds=1
winner=RED method=DEC
```

Multi-path summary must separate:
- KO/TKO
- SUB
- DEC

and report red/blue decision wins.

## Diagnostics / sanity checks

Mechanics diagnostics only; do NOT calibrate in Phase 5A.

Required:

1. controlled synthetic round tests where obvious striking dominance wins;
2. controlled test where a knockdown/damaging striking edge beats passive control;
3. controlled grappling test where a dangerous submission/meaningful ground offense wins;
4. controlled test where primary effectiveness is tied and aggression decides;
5. controlled test where primary + aggression are tied and control decides;
6. exact-tie fallback proves no draw and seed/deterministic reproducibility;
7. three-round controlled fight produces correct 2-1 decision;
8. five-round controlled fight produces correct round-majority decision;
9. no 10-8 / 10-10 output anywhere;
10. KO/TKO and SUB paths never invoke decision judging as terminal method;
11. terminal lifecycle remains exactly once;
12. frozen FSR-32 checksum unchanged.

If a descriptive historical decision-rate anchor is easy to produce from authoritative master data, report it, but do not tune to it.

If historical scorecards are not cleanly available in repo data, report that rather than scraping/building a new scorecard dataset in this phase.

## Scope exclusions

Do NOT add:
- draws;
- 10-8 or 10-7 scoring;
- 10-10 rounds;
- point deductions/fouls;
- three simulated judges;
- unanimous/split/majority subtypes;
- judge personalities/noise;
- population judging calibration;
- KD/KO calibration;
- SUB calibration;
- submission-attempt calibration;
- age;
- tactical urgency;
- body-part injury systems;
- trauma recovery;
- FSR rebuild;
- phase/stamina retuning;
- real weight-class tuning.

## Validation

Run full EVENT MC tests plus focused judging tests.

Verify frozen FSR-32 SHA-256 remains:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Commit and push to:

`origin/feature/fsr-32-stamina-shadow`

Stop after implementation, tests, diagnostics, commit/push, and report.

Expected final line:

`PHASE 5A DETERMINISTIC JUDGING GATE: PASS`

or FAIL.
