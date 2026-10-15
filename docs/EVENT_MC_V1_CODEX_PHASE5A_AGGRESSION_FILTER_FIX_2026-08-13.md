# EVENT MC V1 — Phase 5A Aggression Filter Fix

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Purpose

Narrow correction to Phase 5A deterministic judging only.

Independent review found that `DeterministicJudgingModel.on_event()` currently increments aggression for every `ActionAttempt`.

That incorrectly gives effective-aggression credit to defensive/non-offensive action families such as:
- `ground_escape`
- `clinch_separation`
- `ground_reversal`

This conflicts with the judging hierarchy already locked for EVENT MC V1. Effective aggression is a secondary tiebreaker representing offensive initiative, not any attempted action.

## Required behavior

Keep all primary judging logic unchanged.

Aggression credit must be restricted to offensive initiative families only:
- `strike`
- `takedown`
- `clinch_entry`
- `clinch_strike`
- `clinch_takedown`
- `ground_strike`
- `submission_attempt`

Do NOT give aggression credit to:
- `ground_escape`
- `clinch_separation`
- `ground_reversal`

`ground_reversal` must retain its existing effective-grappling value when successful. This fix changes only secondary aggression accounting.

## Hard locks

Do not change:
- effective striking formula or weights;
- effective grappling formula or weights;
- primary close band;
- control handling;
- 10-9-only scoring;
- no-draw behavior;
- JUDGING RNG ownership;
- KO/TKO;
- KD;
- submissions;
- stamina;
- phase/action rates;
- FSR;
- weight-class overrides;
- population calibration.

## Tests

Add focused tests proving:
1. offensive strike/takedown/submission/clinch-entry attempts receive aggression credit;
2. `ground_escape` receives no aggression credit;
3. `clinch_separation` receives no aggression credit;
4. `ground_reversal` receives no aggression credit but still receives its effective-grappling credit on success;
5. aggression can still resolve a primary-close round;
6. passive control still cannot override a clear primary-effectiveness winner;
7. every scored round remains exactly 10-9 with one winner and no draw.

Run the EVENT MC suite and the existing historical fixture judging diagnostic.

Do not tune any values.

Preserve frozen FSR-32 checksum exactly:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Commit/push and report the exact commit SHA, tests, diagnostics, and checksum.

Expected final line:

`PHASE 5A AGGRESSION FILTER FIX GATE: PASS`

or FAIL.
