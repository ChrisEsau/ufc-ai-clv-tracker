# EVENT MC V1 — Phase 7F Submission Conversion Position Neutralization

Ontology correction only. Build on and preserve Phase 7D2 and Phase 7E history.

## Authorized change

Change only:

`defaults.submission_finish.top_position_bonus: 0.25 -> 0.00`

Keep `bottom_position_bonus = 0.00`. Legitimate otherwise-identical top and bottom submission attempts must therefore receive identical intrinsic position contribution to conversion.

## Frozen scope

- submission attempt generation, including `bottom_multiplier = 1.00`;
- submission finish intercept `-2.20`, rating scale, threat/resistance weights, stamina term, bottom bonus, and logit clip;
- KD midpoint 36 and finish midpoint 36;
- KD, KO/TKO, impact, phase/action/ground, stamina, judging, RNG, FSR, overrides, age, and urgency.

No global submission calibration is authorized. Rerun the Phase 7D 100-fight x 10-path cohort from 2020 with seed 20260813, report attempt stability, top/bottom conversion, methods, exposure/timing, and KD/KO guardrails, then establish the neutral conversion baseline before any later intercept fit.

Expected final line:

`PHASE 7F SUBMISSION POSITION NEUTRALIZATION GATE: PASS`
