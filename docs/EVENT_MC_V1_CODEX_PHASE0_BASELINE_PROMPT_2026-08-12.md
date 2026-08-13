# Codex Prompt — EVENT MC V1 Phase 0 Baseline Materialization

Use this prompt verbatim or with only mechanical path corrections if the repository has moved. This is a **pre-implementation gate**. It is not permission to implement EVENT MC V1.

---

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Before doing anything, read:

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`

Treat those documents as the architecture and Phase 0 source of truth.

## Objective

Materialize the frozen **current-simulator comparison baseline** required before EVENT MC V1 implementation begins.

Do **not** implement `pipeline/simulation/event_mc_v1/` yet.

Do **not** change current simulator mechanics, constants, FSR ratings, age rules, wrestling ontology, KO/SUB/TD/stamina/judging logic, or calibration behavior.

This task may add baseline/diagnostic orchestration code only if required to run existing simulator behavior reproducibly. Prefer using existing diagnostics directly where possible.

## Hard locks

- Existing simulator behavior is read-only/frozen.
- FSR-32 remains the active simulator profile source.
- Preserve corrected rating ontology:
  - `wrestling_entry` = intrinsic TD initiation frequency as a rating definition.
  - `wrestling_conversion` = TD completion ability.
  - `td_defense` = opponent TD prevention.
  - `control_imposition` = post-position control ability.
- Do not "fix" the current V0 blended TD consumer in this task. Its blended behavior is part of the old-simulator baseline and will be separated later in EVENT MC V1 Phase 2A/2B.
- Do not retune any constants.
- Do not modify production paths.

## Frozen baseline references

Architecture-review code snapshot:

`7b98ac629dacc094342ba7f6668ffc77aed3b246`

FSR-32 artifact contract:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

FSR-32 builder:

`scripts/experimental/build_fsr_32_database.py`

Historical cohort helper:

`scripts/experimental/fsr_32_historical_cohort.py`

Current single historical diagnostic:

`scripts/experimental/run_single_historical_age_power_diagnostic.py`

Current full-fight entry class:

`StaticFSRMCFullFightV1`

## Required output directory

Create/use:

`data/experimental/event_mc_v1_baseline/`

At minimum produce:

```text
manifest.json
single_path_traces.jsonl
matchup_summary.csv
cohort_200_summary.csv
```

Also produce `full_method_baseline.csv` if the existing full mature 2020+ method audit can be materialized without changing simulator behavior.

If generated data is intentionally gitignored/too large, commit the manifest plus compact summaries and record exact generated paths and SHA-256 checksums in the manifest.

## Manifest requirements

`manifest.json` must record:

```text
repository
branch
commit_sha actually run
python version
FSR artifact path
FSR artifact SHA-256
simulator entry class/module
age rule/config used by the diagnostic
cohort construction function/module
fixture names and bout IDs/dates
root seeds
path counts
metric definitions
output file paths
output SHA-256 checksums
```

Record the actual commit SHA used when the baseline is run. Do not silently substitute another branch.

## Deterministic path traces

For each resolved matchup below, run single paths with root seeds:

```text
7
17
20260811
```

Required anchor:

```text
Rob Font vs Raul Rosas Jr.
2026-03-07
bout_id: bed89a91da9d04c1
```

Additional intended style fixtures:

```text
Derrick Lewis vs Chris Daukaus — 2021-12-18
Max Holloway vs Calvin Kattar — 2021-01-16
Charles Oliveira vs Dustin Poirier — 2021-12-11
Merab Dvalishvili vs Petr Yan — 2023-03-11
```

If one of these cannot resolve in the aligned mature FSR-32 cohort, replace it with a documented same-purpose matchup selected from the cohort and record the reason/replacement in `manifest.json`.

Intended stress categories:

```text
Font/Rosas       high-entry wrestling
Lewis/Daukaus    power/KO
Holloway/Kattar  high-volume striking
Oliveira/Poirier submission/grappling
Dvalishvili/Yan  sustained wrestling/control
```

For each single path retain enough event/segment detail from the current simulator to inspect:

- round/segment/time
- phase start/end
- clinch/ground ownership
- significant offense
- TD events
- submission events
- stamina where currently exposed
- damage/KD/finish where currently exposed
- final outcome

Do not invent fields the current engine cannot expose; document omissions.

## Matchup Monte Carlo baseline

For each resolved matchup run:

```text
paths = 1000
root seed = 20260811
```

Capture per-matchup means/probabilities for:

```text
red/blue win probability
KO/TKO probability
SUB probability
DEC probability
finish round / fight duration where available
significant attempts
significant landed
TD attempts
TD landed
TD success rate
clinch control seconds
ground control seconds
total control seconds
DISTANCE / CLINCH / GROUND occupancy
submission attempts
knockdowns
```

Known Font/Rosas comparison evidence from the latest recorded FSR-32 research run is:

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

Use these only as a sanity cross-check. If your reproduced output differs, do not tune anything. Instead identify whether the difference is due to commit/config/data/seed/path selection and report it.

## Cohort-200 baseline

Use the existing mature 2020+ aligned FSR-32 cohort.

Select a deterministic first-200-bout slice using the same stable ordering used by current calibration diagnostics.

Run:

```text
10 paths per bout
root seed = 20260810
```

Capture aggregate and per-bout outputs sufficient to compute:

```text
winner accuracy/Brier where actual outcomes exist
KO/TKO rate
SUB rate
DEC rate
finish-round distribution
significant attempts/landed
TD attempts/landed/success rate
control seconds
phase occupancy
submission attempts
knockdowns
```

Do not reorder the cohort opportunistically to improve metrics.

## Full method baseline

Where supported by existing diagnostics, materialize the mature 2020+ submission/method baseline without retuning.

Known frozen comparison observations:

```text
mature 2020+ submission cohort: 1,565 fights
10 paths/fight
historical SUB rate: 16.23%
current simulated SUB rate: 16.49%
neutral P(SUB | attempt): 34%
historical submission attempts/fight: 0.5655
simulated attempts/path: 0.4994
historical >=1 attempt rate: 35.02%
simulated >=1 attempt rate: 35.08%
```

Again: reproduce/record; do not force-match by changing constants.

## Tests / validation

Add tests only for baseline orchestration/manifest integrity if new orchestration code is necessary.

At minimum verify:

- same seed + same inputs reproduce identical current-simulator baseline rows/traces;
- manifest includes all required metadata;
- no existing simulator files are modified;
- no calibration constants are changed;
- FSR-32 path is the one recorded in the manifest;
- output checksums are stable for the exact same environment/input where deterministic behavior permits it.

Run the relevant existing test suite after any diagnostic-only code addition.

## Required report back

Stop after baseline materialization. Report:

1. files created/changed;
2. exact commit SHA run;
3. exact commands run;
4. test results;
5. baseline fixture resolution/replacements;
6. compact matchup summary;
7. compact cohort summary;
8. any mismatch against the known Font/Rosas or submission observations and the identified reason;
9. confirmation that no EVENT MC V1 simulator implementation was started and no current-simulator mechanics/constants were changed.

Do **not** begin Phase 1 in this task.