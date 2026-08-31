# Codex Prompt — EVENT MC V1 Phase 4B1 Config Externalization Completion

Date: 2026-08-13

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Status

The first config-externalization implementation at commit `1a9f59c583408dc00aee5097d59e028ee3d0a2c3` is **not yet independently accepted**.

Default/global deterministic parity was good, but independent review found two incomplete requirements:

1. Some active calibration coefficients remain hard-coded in Python.
2. Resolved `EventMCCalibration` is not consistently threaded through the full fight-flow/rate/formula path, so future weight-class overrides would only affect some subsystems.

This is a completion/fix of the already-authorized behavior-neutral config phase. It is NOT a new calibration phase.

## Hard requirements

### 1. Externalize remaining active tunable coefficients

Audit all active EVENT MC V1 mechanics and move remaining behavior-changing calibration coefficients into `config/event_mc_v1.yaml`.

At minimum independently review and externalize where they are calibration rather than pure numerical safety mechanics:

- Dynamic modifier resilience normalization currently using `(trait - 10) / 80`.
- `style_preferences()` blend weights such as `0.5`, `0.75`, `0.25`.
- `_modifier()` behavioral clip magnitude currently `8.0` if it is an intentional calibration limit.
- ground-exit escape/reversal blend weights currently `0.60` / `0.40`.
- ground-exit edge clip currently `1.5` if intentional calibration.
- reversal sensitivity currently `0.75`.
- any other active numeric coefficient that meaningfully changes simulated rates, stamina, power, damage, trauma, KD risk, phase persistence, or submission attempts.

Do NOT externalize obvious mathematical identities, zero/one bounds, tiny numerical epsilons used only to avoid invalid log/probability operations, stable RNG IDs, state/action names, or unit arithmetic that is not a tunable model assumption.

Return a second inventory table:

`remaining old Python literal/constant -> config key -> unchanged value -> reason it is tunable`

### 2. Thread one resolved calibration through the complete simulator

A fight should be able to resolve once:

`calibration = DEFAULT_RESOLVER.for_weight_class(weight_class_key)`

and inject that same immutable object through all relevant EVENT MC consumers.

At minimum make sure weight-class overrides can reach:

- DISTANCE strike attempt rate and accuracy;
- DISTANCE TD initiation and success;
- DISTANCE clinch entry;
- CLINCH strike rate/accuracy;
- CLINCH TD initiation;
- CLINCH separation;
- GROUND strike rate/accuracy;
- submission-attempt generation;
- ground exit and reversal partition behavior;
- stamina action/positional costs and round recovery;
- DynamicModifiers output/power curves;
- impact/trauma/KD physiology and acute decay.

Do not leave phase/action formulas silently reading module-level global calibration when a resolved calibration object was supplied to the fight.

Prefer composition/injection. Do not create global mutable config state.

### 3. Weight-class override integration test

Add at least one synthetic end-to-end test that constructs a non-default weight-class calibration with overrides spanning **multiple subsystems** (for example DISTANCE rate + stamina + KD), injects it into a fight/provider/model stack, and proves each requested subsystem changes while unspecified values inherit defaults.

Also test an override to a CLINCH or GROUND flow parameter specifically, because independent review found those paths were not using runtime calibration consistently.

### 4. Preserve exact default behavior

This remains a pure refactor.

No values may be intentionally changed.
No KD calibration.
No weight-class-specific real tuning.
No KO/TKO.
No Phase 4B2.

Repeat deterministic before/after parity against commit `65a6f2d4e703af5c777f1943f134728b715d4c55` or the already-captured pre-externalization baseline. Discrete event outcomes and RNG draw order must remain identical under the default calibration. Tiny floating serialization differences only are acceptable.

### 5. Scope protection

Do not modify:
- FSR builders/ratings/ontology;
- frozen FSR-32 artifact;
- wrestling-entry semantics;
- Phase 3 mechanics numerically;
- Phase 4A stamina numerically;
- Phase 4B1 physiology numerically;
- legacy inheritance simulator.

## Validation

Run EVENT MC + relevant legacy tests, compile checks, deterministic parity, frozen checksum, and an explicit synthetic weight-class override integration test.

Stop after implementation, diagnostics/tests, commit/push, and report.

Expected final gate:

`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS`

Do not begin KD calibration or Phase 4B2.