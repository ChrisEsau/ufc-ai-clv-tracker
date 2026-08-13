# Codex Prompt — EVENT MC V1 Phase 2B Wrestling-Entry Ontology Correction

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`
Architecture revision: **v0.3**

Status:
- Phase 0 operational baseline: **PASS**.
- Phase 1 generic continuous-time kernel: **PASS**.
- Phase 2A distance temporal/mechanical parity: **PASS** at commit `5b7574c7689ffa2e55821a49fca47a2c1c937991` after independent ChatGPT review.
- Phase 2B is now explicitly authorized by the user.
- Phase 3 and later mechanics remain **NOT AUTHORIZED**.

## Read first

Before touching code, fetch/pull the latest feature branch and read:

1. `AGENTS.md`
2. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`
6. the Phase 2A implementation under `pipeline/simulation/event_mc_v1/`
7. this prompt

Do not reconstruct this task from memory if the governing file is missing. Fetch the branch first.

# Objective

Make **one deliberate semantic change only**:

> DISTANCE takedown initiation must be driven intrinsically by `wrestling_entry`, rather than the Phase 2A legacy blended style score.

Everything else from Phase 2A remains fixed.

This is an ontology correction, not a broad calibration phase.

The experiment should answer:

> What happens when the simulator finally consumes `wrestling_entry` according to its intended FSR meaning, while keeping timing, strike mechanics, TD conversion, clinch behavior, RNG, and all other mechanics unchanged?

# Development standard

Keep this change small, explicit, measurable, and easy to reverse/iterate.

Do not over-engineer or add defensive machinery that is not required for correctness.

The ultimate project target is predictive moneyline/prop accuracy, not theoretical perfection.

# Locked ontology

```text
wrestling_entry      = intrinsic takedown initiation frequency
wrestling_conversion = ability/probability to complete a takedown attempt
td_defense           = opponent ability to prevent completion
control_imposition   = persistence/behavior after control is established
```

Therefore, in Phase 2B:

- `wrestling_entry` MAY affect TD attempt frequency;
- `wrestling_conversion` MUST affect TD success, not attempt frequency;
- opponent `td_defense` MUST affect TD success, not attempt frequency;
- `control_imposition` MUST NOT affect intrinsic TD attempt frequency;
- `distance_striking_pressure` MUST NOT directly suppress intrinsic TD attempt frequency;
- `clinch_striking_pressure` MUST NOT directly suppress intrinsic TD attempt frequency.

Those latter traits can eventually matter through fight context/phase opportunity, but they must no longer be baked into the fighter's intrinsic TD initiation rate.

# Phase 2A behavior being replaced

Phase 2A intentionally preserved the legacy consumer:

```text
legacy_wrestling_preference =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

and then used that blended score to modify the legacy DISTANCE TD base probability.

That was correct for parity testing and is now intentionally retired from the active Phase 2B TD initiation path.

Do not delete the Phase 2A legacy formula/helper if it remains useful for diagnostics and A/B comparison. Prefer to preserve it as an explicit legacy comparator rather than losing the evidence trail.

# New intrinsic TD initiation mapping

Use the existing DISTANCE TD calibration base already carried into Phase 2A:

```text
DISTANCE_TD_ATTEMPT_BASE_30S = 0.10
```

Keep the existing exact 30-second -> 10-second rescaling and exact interval-probability -> per-second hazard conversion.

For the Phase 2B intrinsic fighter modifier, center the FSR score at neutral 50 and reuse the already-existing modifier scale rather than introducing a newly tuned parameter:

```text
entry_delta = wrestling_entry - 50.0
entry_modifier = exp(clip(entry_delta, -8, 8) / MODIFIER_SCALE)

p_td_10s = DISTANCE_TD_ATTEMPT_BASE_10S * entry_modifier
p_td_10s = clip(p_td_10s, 0, 1 - epsilon)

lambda_td = -ln(1 - p_td_10s) / 10
```

Use the existing Phase 2A `MODIFIER_SCALE` value. Do not fit or retune it in this phase.

Important: verify the exact existing helper semantics before editing. Reuse the existing `_modifier` behavior if it already provides this centered/clipped exponential shape cleanly.

This formula is the Phase 2B implementation contract unless the current code makes an equivalent expression cleaner.

# Context multiplier architecture

Preserve a clean seam for future context effects, but **do not invent new context coefficients now**.

Conceptually:

```text
intrinsic_td_rate = f(wrestling_entry)
context_multiplier = 1.0   # Phase 2B default
final_td_rate = intrinsic_td_rate * context_multiplier
```

If useful, represent the context multiplier explicitly in diagnostics or a small typed helper. Do not build a strategy engine.

Future phases may let context alter opportunity based on phase, stamina, score state, cage position, recent attempts, etc. Phase 2B does not.

# What must remain unchanged

Do not change:

- continuous-time scheduler;
- one authoritative clock;
- Phase 1 RNG stream identities;
- strike attempt intensity;
- strike landing probability;
- clinch-entry formula/behavior;
- TD success formula;
- `wrestling_conversion` vs opponent `td_defense` mapping;
- DISTANCE/CLINCH/GROUND state transitions already in Phase 2A;
- FSR-32 values/builders/ontology code;
- frozen FSR artifact;
- current inheritance-based simulator;
- current simulator calibration constants;
- damage/KD/KO/submission/stamina/recovery/age/judging systems.

No Phase 3 clinch or ground internals.

# Keep Phase 2A as an A/B comparator

Add a clear diagnostic ability to compare for the same fighter/profile:

```text
Phase 2A legacy blended TD initiation
vs
Phase 2B ontology-correct intrinsic TD initiation
```

For each fighter show at minimum:

- wrestling_entry;
- control_imposition;
- distance_striking_pressure;
- clinch_striking_pressure;
- legacy blended wrestling preference;
- Phase 2A TD probability / 10 sec;
- Phase 2A hazard / sec;
- Phase 2B intrinsic TD probability / 10 sec;
- Phase 2B hazard / sec;
- ratio or percent change in initiation rate.

This diagnostic must not feed back into engine state.

# Required tests

Add focused Phase 2B tests proving:

1. `wrestling_entry` monotonically changes TD attempt probability/rate.
2. At equal `wrestling_entry`, changing `control_imposition` does NOT change Phase 2B TD initiation.
3. At equal `wrestling_entry`, changing `distance_striking_pressure` does NOT change Phase 2B TD initiation.
4. At equal `wrestling_entry`, changing `clinch_striking_pressure` does NOT change Phase 2B TD initiation.
5. Changing `wrestling_conversion` does NOT change TD initiation but DOES change TD success.
6. Changing opponent `td_defense` does NOT change TD initiation but DOES change TD success.
7. Phase 2A legacy helper still reproduces the prior blended result where retained for comparison.
8. The Phase 2B base-50 fighter maps to the unmodified existing base TD probability.
9. Exact probability -> hazard round trip remains correct.
10. Strike and clinch formulas/results are unchanged from Phase 2A.
11. Phase 1 kernel invariants/tests remain passing.

# Frozen fixture diagnostics

Run the Phase 2A vs Phase 2B TD initiation comparison for at least:

1. Rob Font vs Raul Rosas Jr. — required/non-replaceable
2. Merab Dvalishvili vs Petr Yan — high-wrestling tendency
3. Max Holloway vs Calvin Kattar — low-wrestling/high-volume striking contrast
4. Derrick Lewis vs Chris Daukaus — additional style contrast

For Font/Rosas specifically report the exact Phase 2A and Phase 2B values for both fighters.

Phase 2A reference from the accepted diagnostic:

```text
Rob Font
wrestling_entry = 48.59305
legacy blended preference = -4.44615
Phase 2A TD p/10s = 1.64486%
Phase 2A TD attempts / 15 min matched distance exposure = 1.5030 EVENT MC

Raul Rosas Jr.
wrestling_entry = 54.43824
legacy blended preference = 6.38876
Phase 2A TD p/10s = 10.00891%
Phase 2A TD attempts / 15 min matched distance exposure = 9.5456 EVENT MC
```

These are comparison anchors, **not tuning targets**.

Do not force Phase 2B to preserve them. The purpose of Phase 2B is to measure the semantic correction.

# Matched-distance-exposure diagnostic

Using deterministic seeds and enough paths for stable comparison, report Phase 2A vs Phase 2B TD attempts over equal DISTANCE exposure.

Do not require full-fight simulation because CLINCH/GROUND internals remain out of scope.

Report at minimum:

- TD attempts / 15 min distance exposure;
- TD success %;
- Phase 2A -> Phase 2B change;
- interpretation based on the fighter traits that were removed from initiation.

Strike volume and accuracy should remain statistically unchanged.

# Interpretation we care about

We expect the ontology correction to have effects such as:

- fighters with high `wrestling_entry` no longer being suppressed simply because they also have high striking-pressure traits;
- fighters with high control but ordinary entry no longer receiving extra intrinsic shot frequency from control ability;
- TD completion quality remaining separate from shot frequency.

Do not assume these changes improve predictions yet. Report what moves; historical validation comes later.

# Important non-goals

Do NOT:

- tune MODIFIER_SCALE or the base TD probability;
- optimize Font/Rosas;
- modify FSR scores;
- alter wrestling_entry builder semantics;
- add opponent TD defense to attempt initiation;
- add control_imposition back through a hidden multiplier;
- add strategic/context coefficients;
- add clinch/ground mechanics;
- add stamina;
- add damage/KD/KO;
- add submissions;
- add recovery;
- add age;
- add judging;
- add score urgency;
- add cooldown constants;
- add MatReturn;
- refactor unrelated Phase 1/2A code;
- modify the current simulator.

# Safety / modularity

Keep future damage/KD/KO systems replaceable behind clean interfaces. The parallel damage-system review must not block this task and no damage architecture changes are needed here.

# Validation commands

Run at minimum:

1. all `tests/simulation/event_mc_v1` tests;
2. the relevant legacy V0 regression tests used by Phase 2A;
3. compileall on touched EVENT MC source/tests;
4. the Phase 2B fixture/A-B diagnostic;
5. frozen FSR-32 SHA-256 verification;
6. `git diff --check`;
7. clean status after commit/push.

Do not turn this into a repository-wide cleanup.

# Required return report

Return:

1. starting SHA and final SHA;
2. exact files changed;
3. exact old Phase 2A TD initiation formula;
4. exact new Phase 2B intrinsic TD initiation formula;
5. confirmation of reused base probability and modifier scale, with no tuning;
6. confirmation TD success formula is unchanged;
7. unit tests demonstrating trait separation;
8. Phase 2A vs 2B matched-distance diagnostics;
9. detailed Font/Rosas comparison;
10. Merab/Yan comparison;
11. confirmation strike/clinch behavior remained unchanged;
12. all test/diagnostic command results;
13. FSR artifact checksum;
14. confirmation current simulator, FSR builders, ratings, ontology, and calibration untouched;
15. confirmation no Phase 3/later mechanics were added;
16. commit/push/PR status;
17. final line exactly one of:

`PHASE 2B WRESTLING ENTRY ONTOLOGY GATE: PASS`

or

`PHASE 2B WRESTLING ENTRY ONTOLOGY GATE: FAIL`

STOP after Phase 2B. Do not start Phase 3.