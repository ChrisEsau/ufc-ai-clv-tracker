# EVENT MC V1 — Phase 6 Population Historical Validation Harness

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Objective

Build a population-scale, measurement-only historical validation harness for the complete EVENT MC V1 engine now that terminal outcomes are result-complete:

- KO/TKO
- SUB
- DEC

This phase is **validation only**. Do not tune or change any mechanics or calibration values.

The purpose is to quantify where the complete simulator is wrong before Phase 7 calibration.

## Current single-fight sanity checkpoint

Derrick Lewis vs Chris Daukaus (`4b7ec02b39fc6f70`), 1000 paths, seed 70:

- blue win 68.2%
- red win 31.8%
- KO/TKO 99.0%
- SUB 0.4%
- DEC 0.6%
- R1 finishes 70.0%
- KDs/path 0.919
- zero-KD 36.5%
- multi-KD 19.3%
- average KO/TKO finish 233.9s
- runtime 12.39s for 1000 paths (~80.7 paths/s)

This is one diagnostic matchup only. Do NOT calibrate from it.

## Hard locks

Do NOT change:

- FSR-32
- action rates
- phase transition rates
- stamina costs or recovery
- dynamic output/power
- impact generation
- trauma
- knockdown mechanics
- KO/TKO mechanics
- submission attempt rates
- submission conversion
- judging
- RNG streams or draw ordering
- weight-class overrides
- age
- tactical urgency
- legacy simulator

Do not edit `config/event_mc_v1.yaml` except if strictly necessary to add non-mechanical harness configuration; preferably do not edit it at all.

Frozen FSR-32 checksum must remain:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Cohort

Use leakage-safe historical fights that can be resolved from the frozen FSR-32 prefight snapshot file.

Primary cohort:

- completed UFC fights
- both fighters resolvable in frozen FSR-32 at the historical prefight date
- both fighters have at least 3 completed prior UFC fights if that eligibility signal is already available without rebuilding FSR
- prefer 2020+ as the initial validation era if cohort construction supports it cleanly

If the mature-fighter filter cannot be implemented from already-authoritative leakage-safe data, report that clearly and use the broad resolvable cohort rather than inventing/rebuilding features.

## Runtime design

Do not run 1000 paths per historical fight initially.

Build the harness so paths are configurable.

Default smoke/audit run should use a practical path count such as 50 or 100 per fight.

The harness must report:

- number of fights
- paths per fight
- total simulated paths
- total runtime
- paths/sec
- fights/sec

Avoid full trace storage. Population validation should use compact observer/aggregate sinks only.

## Required fight-level outputs

Persist one row per historical fight with at least:

Identity:
- fight_id
- event_date
- red fighter
- blue fighter
- weight class/division
- scheduled rounds
- actual winner
- actual method normalized to KO_TKO / SUB / DEC where possible
- actual finish round/time if authoritative

Simulation:
- red win probability
- blue win probability
- KO/TKO probability
- SUB probability
- DEC probability
- red KO/TKO probability
- blue KO/TKO probability
- red SUB probability
- blue SUB probability
- red DEC probability
- blue DEC probability
- average simulated finish time
- simulated finish-round shares/counts
- KD/path
- zero-KD share
- multi-KD share
- submission attempts/path
- TD attempts/completions per side if cheap to aggregate
- strike attempts/landed per side if cheap to aggregate
- phase residence and control summaries if already available from FlowStatsSink

Do not create huge per-path datasets unless genuinely necessary.

## Required population metrics

### Winner prediction

Report:

- winner accuracy using 0.50 threshold
- Brier score
- log loss, with numerically safe clipping for metric computation only
- mean predicted probability assigned to actual winner
- calibration table by predicted red-win probability bins

Do not compare to sportsbook market yet unless existing odds data can be joined trivially without expanding scope. Market comparison belongs to a later predictive backtest phase.

### Method distribution

Compare historical vs simulated population shares for:

- KO/TKO
- SUB
- DEC

Also report by:

- scheduled 3-round vs 5-round fights
- weight class/division when sample sizes are adequate
- actual year or broad era buckets if easy

### Finish timing

Compare historical vs simulated:

- finish-round distribution
- average/median finish time for finishes
- R1/R2/R3/R4/R5 shares where applicable

Keep decision horizon paths separate from finish-time summaries when appropriate.

### Knockdowns

Compare historical descriptive anchors where available:

- KD/fight
- zero-KD share
- >=1 KD share
- multi-KD share

Historical KD data must respect observed fight duration/censoring. Do not treat a fight that ended in R1 as having exposure through later rounds.

### Submission exposure

Compare historical descriptive anchors where available:

- submission attempts/fight
- share of fights with >=1 recorded submission attempt
- SUB finish share

Do NOT interpret fight-level `P(SUB finish | fight had >=1 attempt)` as attempt-level conversion probability.

## Censoring / exposure requirements

This is critical.

Historical event counts must be interpreted over actual observed fight exposure, not scheduled maximum duration.

At minimum:

- actual completed fight duration must be retained
- per-time or exposure-normalized diagnostics should use observed duration
- 3-round and 5-round fights should not be mixed blindly
- finish mechanics must not be judged from raw counts alone when exposure differs

Do not overengineer a full survival-analysis framework in this phase. Just preserve correct denominators and report the exposure-aware summaries needed for diagnosis.

## Diagnostic breakdowns

At minimum provide population summaries by:

- weight class/division
- scheduled rounds (3 vs 5)
- actual method
- simulated dominant method if useful

If easy and authoritative, also include:

- fighter experience/maturity bucket
- event year

Do not add age analysis in Phase 6 unless it is purely a descriptive column already present and requires no model change. Age effects remain a future mechanics question.

## Output artifacts

Create a command-line module under a sensible path such as:

`pipeline/simulation/event_mc_v1/diagnostics/population_validation.py`

It should support arguments similar to:

```bash
python -m pipeline.simulation.event_mc_v1.diagnostics.population_validation \
  --paths 50 \
  --start-year 2020
```

Optional useful arguments:

- `--limit N`
- `--seed`
- `--output-dir`
- `--weight-class`

Persist compact machine-readable outputs, preferably parquet/CSV plus JSON summary, under a diagnostic output directory that does not interfere with production.

Also print a concise console summary.

## Smoke run before broad run

First prove the harness on a small deterministic cohort, e.g. 10-25 fights.

Then run a larger practical validation cohort using a path count that completes in reasonable Codespaces time.

Do not launch an hours-long run blindly. Estimate runtime after the smoke cohort and choose a reasonable first broad run.

If runtime becomes a blocker, profile and report the bottleneck; do not optimize simulator mechanics or change RNG semantics in this phase.

## Tests

Add tests for at least:

1. one historical fight produces one fight-level result row;
2. simulated method probabilities sum to 1 within floating tolerance;
3. red/blue win probabilities sum to 1;
4. no draws/unresolved scheduled horizons in EVENT MC V1 result rows;
5. deterministic repeatability for same seed/cohort/path count;
6. metric computation is correct on controlled synthetic rows;
7. historical observed-duration denominator is used for exposure-normalized diagnostics;
8. no full trace is required for population run;
9. frozen FSR checksum remains unchanged.

## Report back

Return:

- implementation commit SHA
- exact commands run
- smoke cohort size/paths/runtime
- broad cohort size/paths/runtime
- winner metrics
- historical vs simulated KO/TKO, SUB, DEC shares
- finish-round comparison
- KD comparison
- submission-attempt comparison
- notable weight-class / 3-vs-5-round breakdowns
- top 5 largest systematic discrepancies you observe
- top 5 fight-level probability misses if available
- frozen FSR checksum
- test count/status

Do not propose or commit calibration changes yet.

The purpose of your report is to give ChatGPT enough evidence to choose the Phase 7 calibration order.

Expected final line:

`PHASE 6 POPULATION HISTORICAL VALIDATION GATE: PASS`

or `FAIL` with the blocker.
