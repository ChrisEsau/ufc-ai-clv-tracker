# EVENT MC V1 Phase 0 Baseline Freeze Contract

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Architecture source of truth: `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md` revision **v0.3**

Phase 0 closure record: `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Codex materialization prompt: `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

Architecture review code snapshot: `7b98ac629dacc094342ba7f6668ffc77aed3b246`

Status: **Frozen reproducibility contract. No EVENT MC V1 implementation is authorized until this baseline is materialized and reviewed.**

## Purpose

This document is the canonical numerical/reproducibility contract for the current-simulator baseline used to compare EVENT MC V1.

It reconciles the architecture v0.3 closure record with the pre-implementation Codex task so future chats do not invent different seed sets, cohort sizes, fixtures, or output locations.

The baseline is a ruler, not a target. EVENT MC V1 is not expected to reproduce every old path exactly after the time architecture changes.

## Hard locks

- Do not modify current simulator mechanics, inheritance, constants, or FSR construction while capturing the baseline.
- Keep FSR-32 hooked up.
- Do not correct the legacy blended TD-attempt consumer during baseline capture.
- Do not retune KO, SUB, TD, stamina, recovery, damage, KD, judging, age, or any other calibration.
- Record unavailable data/fixtures honestly instead of changing rules to make them work.
- No `pipeline/simulation/event_mc_v1/` implementation begins until the materialization gate passes.

## Pinned current inputs

FSR-32 artifact contract:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

FSR-32 builder:

`scripts/experimental/build_fsr_32_database.py`

Historical cohort helper:

`scripts/experimental/fsr_32_historical_cohort.py`

Single historical diagnostic:

`scripts/experimental/run_single_historical_age_power_diagnostic.py`

Current full-fight entry class:

`scripts.experimental.fsr_static_mc_ko_sub_decision_v1.StaticFSRMCFullFightV1`

The materialized manifest must record the actual commit SHA, FSR-32 SHA-256, file shape, Python version, and all output checksums.

## Deterministic single-path trace seeds

For every fixed fixture, capture current-simulator traces with exactly:

```text
7
17
20260811
```

These are causal diagnostic fixtures, not probability estimates.

## Fixed matchup fixtures

### Required anchor

```text
Rob Font vs Raul Rosas Jr.
event date: 2026-03-07
bout_id: bed89a91da9d04c1
actual: Raul Rosas Jr. by decision
```

Purpose: high-entry wrestling / TD-volume stress case.

Known recorded current-research comparison evidence:

```text
Font win probability: 59.3%
Rosas win probability: 40.7%
Font TD attempts/path: 0.68
Rosas TD attempts/path: 4.49
Font TD landed/path: 0.29
Rosas TD landed/path: 2.37
Font control seconds/path: 32.04
Rosas control seconds/path: 284.15
Font significant attempts/path: 127.61
Rosas significant attempts/path: 60.81
```

These values are sanity anchors only. A reproduced mismatch must be explained by commit/data/config/seed differences; do not tune to force a match.

### Additional style fixtures

```text
Derrick Lewis vs Chris Daukaus — 2021-12-18
Max Holloway vs Calvin Kattar — 2021-01-16
Charles Oliveira vs Dustin Poirier — 2021-12-11
Merab Dvalishvili vs Petr Yan — 2023-03-11
```

Fixture intent:

```text
Font/Rosas           high-entry wrestling
Lewis/Daukaus        power / KO
Holloway/Kattar      high-volume striking
Oliveira/Poirier     submission / grappling
Dvalishvili/Yan      sustained wrestling / control
```

If a non-anchor fixture cannot resolve in the current mature aligned FSR-32 cohort, replace it with a documented same-purpose matchup selected from the cohort. Do not relax maturity/leakage rules.

## Single-path trace fields

Retain as much currently exposed chronological detail as possible, including:

```text
round / segment / clock
phase start/end
clinch controller
ground controller
significant offense
TD events
submission events
stamina state where exposed
damage/KD state where exposed
finish/outcome
```

Also store compact final stats per fighter:

```text
sig attempts / landed
TD attempts / landed
control seconds
clinch control seconds
ground control seconds
submission attempts
reversals
knockdowns
final outcome/method/finish round/time
```

Do not modify the simulator simply to expose a missing field; record omissions.

## Matchup Monte Carlo baseline

For every resolved fixture:

```text
paths = 1000
root seed = 20260811
```

Generate a deterministic path-seed vector from the root seed and record the method.

Capture:

```text
red/blue win probability
KO/TKO probability
SUB probability
DEC probability
finish round / duration where available
significant attempts / landed
TD attempts / landed / success rate
clinch control seconds
ground control seconds
total control seconds
DISTANCE / CLINCH / GROUND occupancy
submission attempts
reversals
knockdowns
```

For Font/Rosas additionally record the exact FSR-32 values used for:

```text
wrestling_entry
wrestling_conversion
td_defense
control_imposition
control_resistance
distance_striking_pressure
clinch_striking_pressure
```

and compute the legacy current-consumer value:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

This is observation only; do not correct it during baseline materialization.

## Compact historical parity cohort

Use the existing mature 2020+ aligned FSR-32 historical cohort and the same stable ordering used by the current calibration diagnostics.

Freeze a deterministic first-200-bout slice.

Simulation contract:

```text
bouts = 200
paths per bout = 10
root seed = 20260810
```

This 2,000-path compact cohort is the routine V0-to-V1 migration audit. It intentionally matches the scale already used by current calibration diagnostics rather than introducing a new heavier baseline definition during Phase 0.

Capture where available:

```text
winner accuracy / Brier
KO/TKO rate
SUB rate
DEC rate
finish-round distribution
significant attempts / landed
TD attempts / landed / success rate
control seconds
phase occupancy
submission attempts
reversals
knockdowns
```

## Full submission/method anchor

Preserve the existing mature 2020+ submission audit observations:

```text
cohort: 1,565 fights
paths: 10 per fight
historical SUB rate: 16.23%
current simulated SUB rate: 16.49%
neutral P(SUB | attempt): 34%
historical submission attempts/fight: 0.5655
simulated submission attempts/path: 0.4994
historical >=1 submission-attempt rate: 35.02%
simulated >=1 submission-attempt rate: 35.08%
```

Materialize/reproduce these where the existing diagnostics support it without modifying simulator behavior. Treat discrepancies as investigation items, not invitations to retune.

## Output location

Canonical baseline directory:

`data/experimental/event_mc_v1_baseline/`

At minimum produce compact reproducibility artifacts:

```text
manifest.json
single_path_traces.jsonl
matchup_summary.csv
cohort_200_summary.csv
```

Also produce `full_method_baseline.csv` if the current mature method/submission audit can be serialized without changing physics.

If larger generated artifacts are intentionally uncommitted, record exact paths and SHA-256 checksums in `manifest.json`.

## Manifest requirements

At minimum:

```text
repository
branch
commit_sha actually run
python version
architecture_revision
FSR-32 path
FSR-32 SHA-256
FSR-32 row/column counts
simulator entry module/class
cohort construction module/function
fixture names / bout IDs / dates / corner orientation / ages
single-path seeds
matchup root seed / path count
cohort root seed / bout count / paths per bout
age rule/config used
major named finish/recovery candidate description
metric definitions
output paths
output SHA-256 checksums
fixture substitutions/omissions
```

Pinning the exact commit is the authoritative constant freeze; do not manually duplicate every calibration constant in the manifest.

## Operational Phase 0 acceptance gate

The pre-implementation baseline materialization passes only when:

1. Current simulator files are confirmed unchanged.
2. FSR-32 input is checksummed and recorded.
3. Font/Rosas anchor resolves exactly to the specified bout.
4. Each additional style category resolves or has a documented replacement.
5. Three deterministic trace seeds are captured per fixture.
6. 1000-path matchup summaries are captured.
7. The first-200 mature 2020+ cohort × 10 paths is captured.
8. Existing full submission/method anchors are reproduced/materialized where supported.
9. Compact outputs and manifest are stored reproducibly.
10. No simulator mechanic/constant was changed to improve or force results.

Codex must report either:

```text
PHASE 0 OPERATIONAL BASELINE GATE: PASS
```

or:

```text
PHASE 0 OPERATIONAL BASELINE GATE: FAIL
```

with exact remaining blockers.

## Attribution rule for all later comparisons

Differences between current MC and EVENT MC V1 must first be assigned to:

```text
A. Phase 2A temporal/mechanical change
B. Phase 2B deliberate wrestling semantic correction
C. Phase 3-5 component port difference
D. Later explicitly approved calibration/ablation change
E. Bug / unintended regression
```

Do not tune away a difference until its source is understood.
