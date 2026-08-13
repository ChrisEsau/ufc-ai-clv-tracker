# EVENT MC V1 Phase 0 Baseline Freeze Contract

Date: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Baseline source commit: `7b98ac629dacc094342ba7f6668ffc77aed3b246`

Architecture source: `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md` revision v0.2 at the baseline source commit.

Status: **Phase 0 reproducibility contract. No simulator implementation changes are authorized by this document.**

## Purpose

This document freezes the comparison contract that must be captured from the current simulator before `event_mc_v1` implementation changes begin.

The new event-driven simulator is **not** expected to reproduce every old path exactly. The purpose of this baseline is attribution: every later difference should be traceable to temporal architecture, the deliberate Phase 2B wrestling semantic correction, a component port, or a later approved calibration change.

## Hard locks

- Do not modify the current simulator while capturing this baseline.
- Keep FSR-32 hooked up.
- Use the current full-fight simulator inheritance stack as-is.
- Preserve all currently configured KO, SUB, TD, stamina, recovery, age, and judging constants.
- Do not alter the corrected FSR rating ontology.
- Do not introduce the future ontology-correct TD consumer during baseline capture.
- Record failures rather than changing code merely to make a fixture run.

## Pinned input contract

FSR-32 builder contract:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

The baseline capture must record for the actual file used:

- path;
- existence;
- row count;
- column count;
- file size;
- SHA-256 checksum;
- latest event/fight date present, when available.

The capture must also record the exact Git commit at execution time. It must begin from baseline source commit `7b98ac629dacc094342ba7f6668ffc77aed3b246` or a descendant containing documentation-only Phase 0 closure commits. If simulator code has changed, stop and report the mismatch.

## Current simulator entry points to preserve

Primary single historical diagnostic:

`scripts/experimental/run_single_historical_age_power_diagnostic.py`

Current full-fight class:

`scripts.experimental.fsr_static_mc_ko_sub_decision_v1.StaticFSRMCFullFightV1`

Historical cohort alignment:

`scripts.experimental.fsr_32_historical_cohort.build_aligned_cohort()`

The baseline capture may add a **new diagnostic/capture script only if necessary to serialize existing outputs**, but it may not modify simulator classes, formulas, constants, inheritance, FSR construction, or fight mechanics. If a new capture script is needed, it must be mechanically observational: import current simulator code, run it, and write results.

## Deterministic seed contract

Use explicit seeds everywhere.

### Single-path trace seeds

For every deterministic fixture, capture the following path seeds:

```text
2026081201
2026081202
2026081203
2026081204
2026081205
```

These five seeds are diagnostic fixtures, not probability estimates.

### Matchup Monte Carlo seed

For matchup-level distribution summaries:

```text
root seed: 20260811
paths per matchup: 1000
```

### Historical cohort seed

For aggregate historical replay:

```text
root seed: 20260810
```

Use a deterministic seed matrix derived from that root. Record the seed-generation method in the captured manifest.

## Fixed matchup fixture set

The capture must attempt these named historical fixtures using the current FSR-32 aligned cohort. Resolve and record the actual `bout_id`, event date, fighter IDs, corner orientation, ages, and scheduled rounds from repository data before simulation.

1. **Rob Font vs Raul Rosas Jr.**
   - Primary wrestling-volume / ontology diagnostic.
   - Do not substitute another Rosas matchup.

2. **Derrick Lewis vs Chris Daukaus**
   - Event date: 2021-12-18.
   - High KO / acute-power diagnostic.

3. **Andre Lima vs Kevin Borjas**
   - Striking-heavy diagnostic used in prior validation work.
   - Resolve exact repository bout/date and record it.

4. **Islam Makhachev vs Charles Oliveira**
   - Event date: 2022-10-22.
   - Grappling / submission / control diagnostic.

5. **Merab Dvalishvili vs Petr Yan**
   - Event date: 2023-03-11.
   - High-volume wrestling / control diagnostic.

6. **Max Holloway vs Calvin Kattar**
   - Event date: 2021-01-16.
   - High-volume striking diagnostic.

If any fixture is not present in the mature aligned FSR-32 cohort, do not loosen maturity/leakage rules. Mark it unavailable and select a replacement only by the deterministic replacement rule below.

### Replacement rule

A replacement must be selected from the existing aligned mature cohort by transparent trait quantiles, not subjective name-picking:

- striker replacement: top decile distance striking pressure and bottom half wrestling entry;
- wrestler replacement: top decile wrestling entry;
- control replacement: top decile control imposition;
- grappler replacement: top decile submission pressure;
- low-action replacement if needed: bottom decile combined distance pressure + wrestling entry.

Record the selection query, quantile thresholds, bout ID, and fighter names.

## Deterministic single-path capture fields

For each fixture × each of the five trace seeds, store at minimum:

```text
fixture_name
bout_id
event_date
red_fighter_id
red_name
blue_fighter_id
blue_name
red_age
blue_age
scheduled_rounds
seed
winner
method
finish_round
finish_segment
finish_clock_start
red_sig_att
blue_sig_att
red_sig_landed
blue_sig_landed
red_td_att
blue_td_att
red_td_landed
blue_td_landed
red_control_seconds
blue_control_seconds
red_ground_control_seconds
blue_ground_control_seconds
red_clinch_control_seconds
blue_clinch_control_seconds
red_sub_att
blue_sub_att
red_reversals
blue_reversals
red_knockdowns
blue_knockdowns
distance_segments
clinch_segments
ground_segments
red_final_stamina_fraction
blue_final_stamina_fraction
red_final_damage_fraction
blue_final_damage_fraction
```

Where the current class exposes full path events, retain the chronological trace for these five seeds.

## Matchup-level 1000-path summaries

For each fixed fixture, run 1000 paths with root seed `20260811` and store:

```text
p_red_win
p_blue_win
p_ko
p_sub
p_dec
mean_finish_round among finishes
mean_sig_att red/blue
mean_sig_landed red/blue
mean_td_att red/blue
mean_td_landed red/blue
TD success rate red/blue
mean_control_seconds red/blue
mean_ground_control_seconds red/blue
mean_clinch_control_seconds red/blue
mean_sub_att red/blue
mean_reversals red/blue
mean_knockdowns red/blue
mean distance/clinch/ground occupancy
```

For Font vs Rosas, additionally retain the exact prefight FSR-32 values used for:

```text
wrestling_entry
wrestling_conversion
td_defense
control_imposition
control_resistance
distance_striking_pressure
clinch_striking_pressure
```

and compute/report the **legacy blended `wrestling_pref`** consumed by V0. This makes Phase 2A vs Phase 2B attribution explicit.

## Aggregate historical cohort freeze

Use the existing leakage-safe aligned mature FSR-32 historical cohort.

Primary comparison cohort:

- event date >= 2020-01-01;
- both fighters satisfy the current mature/aligned cohort rule already encoded by `fsr_32_historical_cohort`;
- do not redefine maturity in the capture script;
- deterministic ordering by event date then `bout_id`;
- first 200 eligible bouts for the compact frozen audit cohort.

Paths:

```text
100 paths per bout
root seed 20260810
```

This produces 20,000 simulated paths and is the primary compact V0-to-V1 comparison cohort.

Record cohort-level historical observed values and simulator values where available for:

```text
winner accuracy
winner Brier score
KO rate
SUB rate
DEC rate
finish-round distribution
mean finish round
fight-duration proxy / scheduled-distance share
significant attempts
significant landed
TD attempts
TD landed
TD success rate
control seconds
clinch control seconds
ground control seconds
phase occupancy
submission attempts
reversals
knockdowns
```

Where an observed historical field is unavailable or not comparable, record `not_available` rather than fabricating a target.

## Stratified cohort views

Using the same frozen 200-bout cohort, report simulator and historical summaries where possible for:

```text
high wrestling_entry: top quartile
low wrestling_entry: bottom quartile
high control_imposition: top quartile
high submission_pressure: top quartile
3-round bouts
5-round bouts if present
fighter age <= 30 vs > 30
experience strata if already available in the aligned cohort
```

Do not tune against these strata during capture.

## Output location contract

The baseline capture should be stored under a dedicated immutable-style directory, for example:

```text
data/experimental/event_mc_v1_phase0_baseline/
```

Recommended outputs:

```text
manifest.json
fixture_resolution.csv
single_path_fixtures.parquet
single_path_traces.jsonl.gz
matchup_1000_path_summary.csv
historical_200x100_path_results.parquet
historical_200x100_summary.json
historical_200x100_strata.csv
```

If repository policy does not permit committing large generated artifacts, commit at minimum:

- `manifest.json`;
- fixture-resolution metadata;
- compact summary CSV/JSON files;
- checksums and exact filesystem paths for larger local artifacts.

Do not commit a giant trace artifact merely for convenience.

## Baseline manifest requirements

`manifest.json` must include:

```text
capture_timestamp
repository
branch
commit_sha
architecture_revision
fsr32_path
fsr32_sha256
fsr32_rows
fsr32_columns
single_path_seeds
matchup_root_seed
historical_root_seed
historical_cohort_definition
historical_bout_count
paths_per_historical_bout
simulator_entrypoint
full_fight_class
age_rule description
submission neutral candidate
recovery candidate description
notes on any unavailable fixtures
```

Do not duplicate every calibration constant manually in the manifest. Pinning the exact commit is the authoritative calibration freeze; record only the major named candidate configuration for human readability.

## Acceptance gate for closing Phase 0 operationally

Phase 0 baseline capture is complete only when:

1. The current simulator code is confirmed unchanged from the pinned baseline lineage.
2. FSR-32 input is checksummed and recorded.
3. Every fixed fixture is resolved or transparently replaced by the quantile rule.
4. Five deterministic single-path seeds are captured for each fixture.
5. 1000-path summaries are captured for each fixture.
6. The frozen 200-bout × 100-path historical audit is captured.
7. The compact summaries and manifest are stored reproducibly.
8. No simulator constants or mechanics were changed to make the capture pass.

Only after this gate may implementation of the generic `event_mc_v1` kernel begin.

## Interpretation rule

This baseline is a ruler, not a target.

Exact path equality is not expected after moving to continuous time. During migration, every observed difference should instead be assigned to one of these categories:

```text
A. Temporal/mechanical change (Phase 2A)
B. Deliberate wrestling semantic correction (Phase 2B)
C. Component port difference (Phases 3-5)
D. Later explicitly approved calibration/ablation change (Phase 7+)
E. Bug / unintended regression
```

Never tune away a difference until its category is understood.
