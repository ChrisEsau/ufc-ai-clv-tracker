# EVENT MC V1 — Winner Discrimination / FSR Wiring Audit

Begin immediately when this prompt is received. Do not ask for confirmation or approval to start the task. Execute the authorized work, tests, diagnostics, commits, and push without waiting for another user message. Only stop if a required operation is technically blocked by the environment or would exceed the explicit scope of this prompt.

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

## Objective

The fresh 100-fight EVENT MC predictive replay produced:

- winner accuracy: 49/100 = 49.0%
- winner Brier: 0.3133
- winner log loss: 0.8687
- method accuracy: 61/100 = 61.0%
- joint winner-method accuracy: 31/100 = 31.0%
- method population shares were comparatively close to history: historical KO_TKO/SUB/DEC 27/13/60% versus mean MC 25.78/16.88/57.34%
- high-confidence winner discrimination was poor: 28 fights at >=80% confidence with only 46.43% accuracy

This task is a measurement-only root-cause audit of why fighter-to-fighter winner direction is poor while global fight-environment/method composition is much healthier.

Do NOT tune the simulator. Do NOT change YAML. Do NOT alter mechanics. Do NOT change FSR generation. Do NOT change age rules. Do NOT change stamina rules. Do NOT promote any candidate fix.

The goal is to determine whether the primary failure is one or more of:

1. wrong/stale prefight FSR snapshot selection;
2. missing fight-night age adjustment wiring;
3. wrong trait sign/direction in FSR -> FighterProfile -> formula use;
4. excessive transformation/amplification of FSR matchup differences;
5. incorrect stamina/fatigue trait mapping or direction;
6. a particular FSR family dominating wrong-way high-confidence predictions;
7. some other concrete wiring or semantic defect discoverable from code/data.

## Frozen inputs

Use the exact frozen FSR-32 artifact:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Required SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Never rebuild, rewrite, normalize, recompress, or commit the parquet.

Use the exact same fresh 100-fight cohort selected by:

`pipeline/simulation/event_mc_v1/diagnostics/fresh_100_fight_predictive_replay.py`

The cohort spans 2025-03-29 through 2025-08-16 and must be reproduced from the same selection rule, not manually reconstructed.

## Hard freeze

No changes are authorized to:

- `config/event_mc_v1.yaml`
- `config/fsr_age_modifiers.yaml`
- EVENT MC mechanics
- FSR calculations or stored values
- wrestling_entry ontology
- stamina/fatigue equations
- damage/trauma/KD/KO
- submissions
- judging
- RNG ownership/seed rules
- weight-class overrides
- round-specific calibration
- age coefficients

Diagnostic code/tests/docs only.

## Audit A — prove prefight snapshot identity and chronology

For every one of the 100 fresh fights, verify exactly which two FSR-32 rows are selected.

For each fighter print/report:

- event_date
- bout_id
- fighter_id
- fighter_name
- target fight date
- FSR snapshot fight_id
- FSR snapshot event_date/date field available in artifact
- prior_ufc_fights
- whether snapshot is exactly the target bout's leakage-safe prefight row
- whether any later/postfight row could have been selected
- duplicate candidate row count before final selection

The audit must establish that `_fight()` / population loading uses the intended target-fight prefight state rather than latest-career, stale, duplicate, or postfight state.

If the FSR artifact stores the target bout ID on a row that represents prefight state, state this explicitly so terminology is not confused with a prior-bout row.

FAIL loudly on any chronology/leakage ambiguity.

## Audit B — age wiring, end to end

The repository contains:

- `config/fsr_age_modifiers.yaml`
- `scripts/experimental/fsr_age_modifiers.py`

Current enabled/calibrated age rules include at least:

- `knockdown_resistance`: -2 FSR points/year after age 30
- `damage_durability`: -2 FSR points/year after age 30

Do not assume these are or are not already baked into FSR-32. Prove it.

Trace the entire production/replay path used by EVENT MC:

fresh cohort row
-> frozen FSR-32 row
-> any age-adjustment function
-> `FighterProfile.from_mapping`
-> runtime profile fields consumed by EVENT MC

For all 100 fights, determine fight-night age from the authoritative source already used by the repository. Do not use current age. Use age on event date.

For each fighter report at minimum:

- fight-night age
- stored `damage_durability`
- stored `knockdown_resistance`
- configured age modifier for each
- expected effective value if the age layer is applied at runtime
- actual value received by EVENT MC
- boolean `age_modifier_applied_runtime`
- boolean/assessment `appears_prebaked_in_fsr_artifact`

For the prebaked assessment, inspect the FSR-32 build path and age-related code/history rather than inferring from numerical coincidence alone.

Summarize:

- how many of 200 fighter-fight profiles are age 30+
- how many should receive a nonzero durability modifier
- how many should receive a nonzero KD-resistance modifier
- how many actually receive it in EVENT MC runtime
- exact source files/functions responsible

If age is not wired into EVENT MC, say exactly where the intended chain breaks. Do not fix it in this task.

## Audit C — all FSR-32 traits: semantic direction and runtime destination

Build an explicit mapping table for every FSR trait consumed by EVENT MC V1.

At minimum include all fields accepted by `FighterProfile` and populated from FSR-32.

Columns:

- source FSR trait
- source scale/range
- semantic meaning of HIGH value
- FighterProfile destination field
- every production formula/module that consumes it
- expected effect of increasing the trait
- actual mathematical effect in code
- direction status: CORRECT / INVERTED / AMBIGUOUS / UNUSED
- notes

This must be based on code tracing, not naming assumptions.

Pay special attention to defensive/resistance traits where higher should help the owner, including:

- distance/clinch/ground striking defense
- td_defense
- control_resistance
- submission_resistance / submission defense mapping
- knockdown_resistance
- damage_durability
- stamina depletion/accumulation resistance
- stamina performance resilience
- recovery/adversity traits if present in the EVENT MC profile

Also inspect offensive traits:

- distance/clinch/ground pressure
- precision
- wrestling_entry
- wrestling_conversion
- control_imposition
- submission_pressure
- submission_conversion
- striking_power
- reversal ability

Do not alter any mapping even if an inversion is found. Report exact file/line/function and a minimal demonstration test proving the direction.

## Audit D — stamina/fatigue implementation comparison: MC V2 versus EVENT MC V1

Trace the older RFS MC V2 shared-state fatigue architecture and compare it directly with EVENT MC V1.

MC V2 reference files include, as applicable:

- `pipeline/simulation/rfs_mc_v2_shared_state/dynamic_parameters.py`
- `dynamic_state_updater.py`
- `effective_phase_parameters.py`
- dynamic exposure/calibration modules

EVENT MC reference files include, as applicable:

- `pipeline/simulation/event_mc_v1/stamina.py`
- `modifiers.py`
- `components/actions.py`
- `engine.py`
- `components/profiles.py`

Produce a side-by-side table showing:

- source fighter traits
- workload sources
- accumulation formula
- resistance formula
- segment/continuous recovery
- round-break recovery
- state bounds
- downstream capabilities affected
- exact performance effects

Then verify for the fresh-100 runtime that the EVENT MC stamina-related FSR values are populated with the intended source traits and correct direction.

For every fighter in the fresh cohort report:

- stored source fatigue/stamina traits
- mapped EVENT MC `stamina_capacity`
- `stamina_depletion_resistance`
- `stamina_performance_resilience`
- any transformations used

Flag constant/default substitution, missing fields, NaN fallback, inverted scaling, or severe compression.

Do not change stamina in this task.

## Audit E — matchup transformation / sensitivity

The fresh replay produced numerous 80-94% predictions that were wrong. Determine how raw FSR differences become large outcome differences.

For every formula involving FSR deltas, odds/logits, exponentials, multipliers, clamps, or nonlinear transformations, document:

- raw trait delta
- scale divisor
- transformation
- clamp
- resulting rate/probability multiplier at representative deltas: +/-2, +/-5, +/-10, +/-15, +/-20 FSR points where mathematically applicable

Focus particularly on transformations that can generate large asymmetry in:

- strike attempt rates
- strike accuracy
- phase transitions
- takedown attempts
- takedown success
- control persistence/escape
- submission attempts/conversion
- impact/power/KD
- judging/evidence accumulation

Identify any transformation where ordinary observed FSR differences cause extreme multipliers or near-saturated probabilities.

Measurement only.

## Audit F — high-confidence wrong-way miss decomposition

Use the completed fresh-100 prediction results if available from the diagnostic output; otherwise deterministically reproduce only what is needed from the same code/seed. Do not run another 25,000 paths unless necessary.

Prioritize every incorrect winner prediction with predicted confidence >=75%.

At minimum include these known misses if they are in the canonical fresh cohort:

- Jeremy Stephens vs Mason Jones
- Stephen Thompson vs Gabriel Bonfim
- Jessica Andrade vs Jasmine Jasudavicius
- Viviane Araujo vs Tracy Cortez
- Merab Dvalishvili vs Sean O'Malley
- Terrance McKinney vs Viacheslav Borshchev
- Miles Johns vs Jean Matsumoto
- Michel Pereira vs Abus Magomedov
- Angela Hill vs Iasmin Lucindo
- Ketlen Souza vs Piera Rodriguez
- Kennedy Nzechukwu vs Valter Walker
- Giga Chikadze vs David Onama
- Amanda Ribas vs Tabatha Ricci
- Elizeu Zaleski dos Santos vs Neil Magny
- Istela Nunes vs Loma Lookboonmee
- Nikita Krylov vs Dominick Reyes
- Sean Woodson vs Dan Ige
- Kelvin Gastelum vs Joe Pyfer
- Vicente Luque vs Kevin Holland
- Calvin Kattar vs Steve Garcia
- Nora Cornolle vs Karol Rosa
- Julian Erosa vs Melquizael Costa

For each high-confidence miss create an interpretable pre-simulation matchup decomposition showing red-vs-blue differences and resulting effective initial parameters grouped into:

1. DISTANCE striking
2. CLINCH striking/control
3. wrestling/takedowns
4. GROUND control/striking
5. submissions
6. power/durability/KD resistance
7. stamina/fatigue/recovery
8. judging-relevant output/evidence inputs
9. age adjustment status

Do not invent a post-hoc single number if the simulator has no such score. Use the actual intermediate rates/probabilities/parameters entering the engine.

Then identify which family/families structurally favored the incorrectly predicted fighter.

## Audit G — correct high-confidence controls

To avoid diagnosing only misses, select a comparable set of at least 15 correct predictions at >=75% confidence from the same fresh cohort.

Run the identical decomposition.

Compare miss versus hit distributions for:

- age differential
- FSR trait deltas
- initial strike-rate edge
- accuracy edge
- TD attempt/success edge
- submission edge
- power/durability edge
- stamina edge
- any transformation saturation metrics

This is descriptive; do not fit a new predictive model in this task.

## Required output

Create a diagnostic module under:

`pipeline/simulation/event_mc_v1/diagnostics/`

Use a clear name such as:

`winner_discrimination_audit.py`

Machine-readable output default:

`/tmp/event_mc_v1_winner_discrimination_audit.json`

Optional flat CSVs may also be written under `/tmp` for profile-level and fight-level tables.

Terminal/report must contain these sections:

1. EXECUTIVE FINDING
2. PREFIGHT SNAPSHOT IDENTITY
3. AGE WIRING
4. FSR TRAIT DIRECTION MAP
5. MC V2 VS EVENT MC FATIGUE/STAMINA
6. MATCHUP TRANSFORMATION SENSITIVITY
7. HIGH-CONFIDENCE MISS DECOMPOSITION
8. HIGH-CONFIDENCE HIT CONTROLS
9. ROOT-CAUSE RANKING
10. EXACTLY ONE RECOMMENDED NEXT ACTION

## Root-cause ranking

At the end rank each candidate as:

- CONFIRMED DEFECT
- STRONG EVIDENCE
- POSSIBLE CONTRIBUTOR
- NOT SUPPORTED

Candidates:

- stale/wrong FSR snapshot selection
- leakage/postfight state
- age modifier missing at runtime
- age already correctly represented elsewhere
- inverted trait direction
- missing/unused important FSR trait
- wrong stamina mapping
- insufficient stamina effects
- over-amplified FSR transforms
- one dominant FSR family causing wrong-way predictions
- judging-direction issue
- finish mechanics direction issue
- other discovered cause

Support every ranking with evidence from the audit.

## Tests

Add focused tests proving diagnostic correctness, including:

- fresh cohort identity remains exactly 100 and unchanged
- target-fight FSR row selection is leakage-safe
- age computation uses event date
- configured enabled age rules are evaluated correctly
- audit detects whether runtime value equals stored or age-adjusted value
- trait-direction probes behave as reported
- stamina mapping is deterministic and finite
- no diagnostic function mutates source profiles
- no RNG changes occur merely from inspection
- `config/event_mc_v1.yaml` remains unchanged
- `config/fsr_age_modifiers.yaml` remains unchanged
- frozen FSR SHA remains exact

Run at minimum:

`python -m pytest -q tests/simulation/event_mc_v1 tests/experimental/test_fsr_static_mc_v0.py`

`python -m compileall pipeline scrapers tabs utils tests/simulation/event_mc_v1`

`sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

`git diff --check`

`git diff -- config/event_mc_v1.yaml config/fsr_age_modifiers.yaml`

The config diff must be empty.

## Commit / push

Commit diagnostic code, tests, and continuity update to:

`feature/fsr-32-stamina-shadow`

Push the branch.

Do not commit generated `/tmp` outputs.

## Continuity

Update:

`docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`

Record the audit findings, exact defects confirmed or rejected, and exactly one recommended next action. Do not describe any unimplemented fix as completed.

## Decision rule

This task PASSES if the audit executes end to end, preserves all frozen mechanics/config/artifacts, and produces an evidence-backed root-cause ranking.

It does NOT require finding a defect.

Expected final line:

`EVENT MC V1 WINNER DISCRIMINATION AUDIT: PASS`

or

`EVENT MC V1 WINNER DISCRIMINATION AUDIT: FAIL`
