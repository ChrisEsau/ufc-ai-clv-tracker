# Codex Prompt — EVENT MC V1 Phase 4B1 Config Externalization

Date: 2026-08-13

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Source branch: `feature/fsr-32-stamina-shadow`
Architecture revision: v0.3

## Status before this phase

- Phase 0 operational baseline: PASS.
- Phase 1 generic continuous-time kernel: PASS.
- Phase 2A DISTANCE temporal/mechanical parity: PASS.
- Phase 2B wrestling-entry ontology correction: PASS.
- Phase 3 CLINCH + GROUND flow: PASS.
- Phase 4A stamina + dynamic modifiers: PASS.
- Phase 4B1 impact + trauma + knockdown implementation: PASS after independent ChatGPT review.
- Phase 4B1 KD calibration: intentionally OPEN; user explicitly said do not calibrate yet.
- This config-externalization phase is explicitly authorized by the user.
- Phase 4B2 KO/TKO: NOT AUTHORIZED.

Codex cloud may use a local branch named `work`. That is acceptable. Verify ancestry/content, not local branch name.

Before implementation:

```bash
git fetch origin --prune
git merge-base --is-ancestor 65a6f2d4e703af5c777f1943f134728b715d4c55 HEAD
```

If stale, safely rebase onto:

`origin/feature/fsr-32-stamina-shadow`

Then verify:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE4B1_CONFIG_EXTERNALIZATION_2026-08-13.md
```

# Goal

Move EVENT MC V1 **tunable simulation calibration constants** out of Python implementation modules and into one clear external configuration file, while preserving current behavior exactly.

Preferred config path:

`config/event_mc_v1.yaml`

The refactor must also establish a simple future seam for **weight-class-specific overrides**, but no weight-class-specific tuning is authorized now.

This phase is a configuration refactor only.

# User intent

The user wants to be able to inspect and edit simulator calibration without modifying Python formulas.

Today we tune against the overall UFC population. In the future, the user expects different weight classes may require different simulator calibration because divisions plausibly differ in pace, power, finish behavior, grappling burden, durability, and fatigue.

Therefore build the config so current global defaults remain authoritative now, while future weight-class overrides can be introduced without another simulator refactor.

# Absolute behavior lock

**ZERO INTENTIONAL NUMERICAL OR STOCHASTIC BEHAVIOR CHANGE.**

For the default/global configuration:

- same formulas;
- same constants;
- same seeds;
- same event rates;
- same action costs;
- same stamina curves;
- same impact draws;
- same trauma;
- same KD probabilities;
- same path outcomes;
- same RNG stream ownership/order.

Do not use this phase to fix the currently high KD rate.

Do not tune any number.

# What belongs in external config

Move tunable calibration parameters that define simulator behavior, including the active constants used by the reviewed EVENT MC layers.

At minimum inspect and externalize active tunable parameters from:

- `pipeline/simulation/event_mc_v1/components/formulas.py`
- `pipeline/simulation/event_mc_v1/components/action_rates.py` where applicable
- `pipeline/simulation/event_mc_v1/stamina.py`
- `pipeline/simulation/event_mc_v1/modifiers.py`
- `pipeline/simulation/event_mc_v1/physiology.py`

Expected logical sections include, as applicable:

```text
distance
clinch
ground
submission_attempts
stamina
dynamic_modifiers
damage
knockdown
```

Examples of values that should be externalized if active/tunable:

- DISTANCE strike attempt base intensity;
- DISTANCE strike accuracy base;
- DISTANCE TD attempt base;
- TD success offset/scales;
- DISTANCE clinch-entry base/caps/scales;
- CLINCH separation base;
- CLINCH TD attempt base;
- CLINCH strike attempt base/accuracy;
- GROUND exit base;
- GROUND strike attempt base/accuracy;
- submission-attempt base;
- reversal share;
- bottom strike/submission multipliers;
- shared modifier/rating scales where they are genuine calibration values;
- stamina action costs;
- stamina depletion-resistance scale and clamps;
- positional stamina drain rates;
- round recovery fraction;
- output/power modifier floors and exponents/resilience mappings;
- impact Gamma base/tail shapes/scales;
- tail probability;
- power rating scale;
- impact scale;
- durability scale;
- trauma erosion scale;
- KD resistance scale;
- KD slope;
- KD midpoint impact ratio;
- KD acute increment;
- acute-vulnerability half-life.

Inspect actual active code rather than relying only on this list.

# What should remain code-level / structural

Do NOT externalize constants merely because they are numeric if they are architectural invariants rather than calibration knobs.

Examples that should generally remain in code unless there is a clear reason otherwise:

- RNG stable stream IDs;
- numerical epsilon used only for safe clipping;
- enum/string identifiers;
- state field names;
- phase names;
- engine ordering;
- event class names;
- authoritative clock semantics;
- pure unit-conversion helpers.

Keep the distinction clear: **configuration controls calibration; code controls mechanics and invariants.**

# Target configuration architecture

## 1. One obvious config file

Use one primary YAML file:

`config/event_mc_v1.yaml`

Do not scatter EVENT MC calibration across multiple unrelated config files in this phase.

Organize it by simulator subsystem with readable units in key names or comments where useful.

## 2. Typed immutable config object

Load YAML into a small typed/immutable configuration model, for example conceptually:

```text
EventMCConfig
  distance
  clinch
  ground
  stamina
  dynamic_modifiers
  damage
  knockdown
```

Exact class names are flexible.

Requirements:

- immutable after construction where practical;
- explicit defaults come from the YAML, not duplicated Python literals;
- validate obvious invalid ranges at load time without building an elaborate schema framework;
- do not introduce a new heavy dependency if the repository already has a suitable YAML/config dependency;
- no mutable global singleton that tests can silently contaminate.

## 3. Dependency injection

Pass the resolved simulator config into the relevant rate/formula/stamina/modifier/physiology components.

Avoid modules reaching into YAML repeatedly during simulation.

Preferred lifecycle:

```text
load config once
-> resolve effective config once per matchup/run
-> inject immutable config into components
-> simulate many paths
```

Do not perform filesystem reads per event or per Monte Carlo path when one preloaded config object can be reused.

## 4. Future weight-class override seam

Support the following conceptual resolution model now:

```text
global/default config
        +
optional weight-class override
        =
effective matchup config
```

The committed configuration should contain **no active numerical weight-class differences** in this phase.

A reasonable YAML shape is:

```yaml
defaults:
  ... current global calibration ...

weight_classes: {}
```

or empty named override sections if that is cleaner.

The resolver should allow a future call such as conceptually:

```python
config.for_weight_class("heavyweight")
```

with behavior:

- no override -> exact global defaults;
- partial override -> only specified fields change;
- unspecified fields inherit global defaults.

Do not hard-wire a complicated UFC division taxonomy in this phase. Keep override keys simple strings so later work can decide whether the correct segmentation is weight class, UFC division, sex+weight division, or another hierarchy.

Do not infer a weight class from fighter weight or fighter names in this phase.

Do not activate different configs for current frozen fixtures.

## 5. Reproducibility metadata

Make it easy for diagnostics/results to identify which config was used.

At minimum expose a stable config path and, if simple, a deterministic config fingerprint/hash of the resolved calibration mapping.

Do not let metadata alter simulation RNG behavior.

# Required parity validation

Because this phase is intended to be behavior-neutral, parity is the main gate.

Before modifying code, capture deterministic pre-refactor signatures from the current commit for representative runs.

At minimum include:

- Phase 3 flow behavior;
- Phase 4A stamina behavior;
- Phase 4B1 physiology behavior;
- the five frozen fixtures where practical.

After refactor, rerun the exact same seeds and inputs.

Compare deterministic physics outputs exactly or to floating precision appropriate for serialization. Exclude runtime/throughput timestamps from equality checks.

The goal is not merely similar aggregate rates. The default config should reproduce the current deterministic path behavior.

Add focused unit tests proving:

1. YAML current defaults equal the constants being removed from code.
2. Global/default config reproduces current formulas.
3. Empty/no weight-class override resolves identically to defaults.
4. A synthetic partial override changes only the requested field(s).
5. Config loading is deterministic and does not consume simulation RNG.
6. Different sinks still do not change physics/RNG.
7. Existing Phase 2B/3/4A/4B1 tests continue to pass.

# Current constants must not be silently lost

Before deleting or replacing Python constants, inventory them and map every active tunable constant to its config key.

Return a source-to-config table in the final report, for example:

| old module/constant | config key | unchanged value |
|---|---|---|

If a numeric constant is deliberately left in code, explain why it is structural rather than tunable.

# Scope protection

Do NOT in this phase:

- calibrate KD;
- change the current high KD output;
- change any numerical default;
- add KO/TKO;
- add terminal submissions;
- add judging;
- add age transforms;
- change FSR-32;
- change FSR builders/ratings/ontology;
- change wrestling semantics;
- change phase behavior;
- change stamina behavior;
- change damage/KD architecture;
- add active weight-class tuning;
- infer weight class automatically;
- rewrite the simulator architecture;
- modify the inheritance-based legacy simulator.

Phase 4B2 remains unauthorized.

# Frozen artifact

Continue to use the frozen FSR-32 artifact read-only:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rebuild or rewrite it.

# Performance

The config refactor must not add per-event YAML reads or other obvious hot-path I/O.

Do not spend time micro-optimizing beyond preventing clear config-loading regressions.

# Required final report

Return:

1. starting SHA and final SHA;
2. files changed;
3. config schema summary;
4. source-constant -> config-key mapping;
5. typed config/resolver architecture;
6. weight-class override behavior and proof that no overrides are active;
7. deterministic before/after parity results;
8. existing test results;
9. frozen FSR checksum;
10. confirmation that KD calibration and Phase 4B2 were not touched;
11. commit/push status.

Final gate must be exactly one of:

`PHASE 4B1 CONFIG EXTERNALIZATION GATE: PASS`

or

`PHASE 4B1 CONFIG EXTERNALIZATION GATE: FAIL`

Stop after this config-externalization implementation, tests, parity diagnostics, commit/push, and PASS/FAIL report.
