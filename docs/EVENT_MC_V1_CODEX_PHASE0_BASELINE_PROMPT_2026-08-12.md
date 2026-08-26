# Codex Prompt — EVENT MC V1 Phase 0 Baseline Materialization

Use this as the **first Codex task before any EVENT MC V1 implementation**.

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

## Read first — source-of-truth order

Read these completely before touching code:

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`

Interpretation hierarchy:

- architecture audit v0.3 = canonical architecture direction;
- Phase 0 closure = closure rationale and known frozen research anchors;
- interface decisions = exact implementation-interface locks for later Phase 1;
- baseline freeze = canonical numerical fixture/seed/cohort/output contract for **this task**.

If older wording conflicts with the baseline-freeze document on seeds, cohort size, fixture details, or output paths, follow the baseline-freeze document.

## Objective

Materialize the frozen **current-simulator comparison baseline** required before EVENT MC V1 coding begins.

This is an observational/reproducibility task. It is not permission to implement the new simulator.

## Absolute non-goals

Do **not**:

- implement anything under `pipeline/simulation/event_mc_v1/`;
- modify existing simulator mechanics or inheritance;
- change FSR-32 construction or ratings;
- correct the current blended TD-attempt consumer;
- retune KO, SUB, TD, stamina, recovery, damage, KD, judging, age, or other constants;
- modify production simulator paths;
- alter maturity/leakage cohort rules to make fixtures resolve;
- force current outputs to match previously recorded numbers.

If the baseline cannot be reproduced, report why. Do not tune it.

## Step 1 — repository/input verification

Before adding any harness, report:

```text
current branch
current commit SHA
git status
Python version
```

Verify current simulator sources have not been behaviorally changed from the architecture-review lineage.

Verify the FSR-32 artifact:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Record:

```text
exists
SHA-256
file size
row count
column count
latest fight/event date if available
```

Confirm current entry points still resolve:

```text
scripts.experimental.fsr_static_mc_ko_sub_decision_v1.StaticFSRMCFullFightV1
scripts.experimental.fsr_32_historical_cohort.build_aligned_cohort()
```

## Step 2 — observational capture harness only if needed

Prefer existing diagnostic scripts.

If serialization/orchestration is missing, you may add a new observational diagnostic script that only:

- imports current simulator/cohort code;
- resolves bouts;
- runs existing behavior with explicit seeds;
- reads exposed stats/state/events;
- writes baseline artifacts.

It must not subclass/override/monkey-patch simulator mechanics or constants.

If a requested metric is not currently observable, write `not_available` and document why.

## Step 3 — resolve frozen fixtures

Follow `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md` exactly.

Required anchor:

```text
Rob Font vs Raul Rosas Jr.
event date: 2026-03-07
bout_id: bed89a91da9d04c1
```

Additional intended style fixtures:

```text
Derrick Lewis vs Chris Daukaus — 2021-12-18
Max Holloway vs Calvin Kattar — 2021-01-16
Charles Oliveira vs Dustin Poirier — 2021-12-11
Merab Dvalishvili vs Petr Yan — 2023-03-11
```

Do not substitute the Font/Rosas anchor.

If another fixture is unavailable under existing aligned-mature rules, use a documented same-purpose replacement without weakening those rules.

Record bout ID/date/corners/fighter IDs/ages/scheduled rounds for every resolved fixture.

## Step 4 — deterministic single-path traces

For every resolved fixture, run exactly these seeds:

```text
7
17
20260811
```

Capture the chronological current-simulator path detail exposed by the engine plus compact final fighter/fight statistics specified in the baseline-freeze contract.

These traces are causal diagnostic fixtures, not population estimates.

## Step 5 — matchup 1000-path summaries

For every resolved fixture:

```text
paths = 1000
root seed = 20260811
```

Generate a deterministic path-seed vector and record the generation method.

Capture all required outcome/method/striking/TD/control/phase/submission/KD metrics from the baseline-freeze contract.

For Font/Rosas also capture the exact FSR-32 values used for:

```text
wrestling_entry
wrestling_conversion
td_defense
control_imposition
control_resistance
distance_striking_pressure
clinch_striking_pressure
```

and report the current legacy blended consumer value:

```text
wrestling_pref =
    0.75 * wrestling_entry
  + 0.25 * control_imposition
  - 0.50 * distance_striking_pressure
  - 0.50 * clinch_striking_pressure
```

Do not correct it.

Known prior Font/Rosas research anchors are recorded in the closure/baseline docs. Use them only as sanity checks; explain mismatches rather than tuning.

## Step 6 — compact historical parity cohort

Use the existing mature 2020+ aligned FSR-32 cohort and the same stable ordering used by current calibration diagnostics.

Run exactly:

```text
first 200 eligible bouts
10 paths per bout
root seed = 20260810
```

Capture enough per-bout/aggregate data to compute the required winner/method/striking/TD/control/phase/submission/KD metrics.

Do not opportunistically reorder or reselect the cohort.

## Step 7 — full method/submission anchor

Where existing diagnostics support it without changing physics, materialize the mature 2020+ submission/method baseline recorded in the baseline-freeze document, including the existing 1,565-fight / 10-path-per-fight comparison observations.

If exact reproduction differs, identify the data/commit/config reason. Do not change the 34% neutral candidate or any other parameter.

## Step 8 — write artifacts

Use exactly:

`data/experimental/event_mc_v1_baseline/`

Required compact artifacts:

```text
manifest.json
single_path_traces.jsonl
matchup_summary.csv
cohort_200_summary.csv
```

Also produce `full_method_baseline.csv` if supported observationally.

If large generated files are intentionally not committed, record exact paths and SHA-256 checksums in `manifest.json`.

The manifest requirements in the baseline-freeze document are mandatory.

## Step 9 — validate no simulator changes

Before reporting:

- show `git diff --stat`;
- inspect `git diff`;
- list every changed/new file;
- confirm no current simulator module changed;
- confirm no FSR builder/trait module changed;
- confirm no calibration constant changed;
- run relevant existing tests after any observational harness addition;
- add tests for baseline orchestration/manifest determinism only if new harness code was necessary.

## Required report back

Return:

1. exact commit SHA/data checksum/environment used;
2. commands run;
3. fixture-resolution table;
4. deterministic trace capture status;
5. compact 1000-path fixture summary;
6. compact 200×10 cohort summary;
7. full submission/method baseline status;
8. Font/Rosas FSR + legacy blended-consumer values;
9. any mismatches against prior recorded anchors and the identified reason;
10. files created/changed;
11. tests run/results;
12. explicit confirmation that no EVENT MC V1 implementation began and no existing simulator physics/constants changed.

End with exactly one gate line:

```text
PHASE 0 OPERATIONAL BASELINE GATE: PASS
```

or

```text
PHASE 0 OPERATIONAL BASELINE GATE: FAIL
```

If FAIL, list exact blockers.

## STOP CONDITION

**Stop immediately after the Phase 0 baseline report. Do not begin Phase 1 even if the gate passes.**
