# EVENT MC V1 — PHASE 6 POPULATION METRICS CORRECTION

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Purpose

Correct population aggregation in the Phase 6 historical validation harness before using its outputs to choose Phase 7 calibration targets.

This is a **measurement-only correction**. Do not change simulator mechanics or calibration.

Phase 6 implementation commit under review:
`c4e750a0dfe23c2dd87df8b69c768aacd119b61f`

## Review findings to correct

### 1. Simulated finish-round shares must be pooled by simulated finishing paths

Current implementation computes a finish-round share within each fight and then averages those fight-level shares across fights.

That weights a fight with one simulated finish equally to a fight with many simulated finishes.

For population finish-round distribution, pool counts across **all non-decision simulated paths**:

`sim_finish_round_N_count / total_simulated_nondecision_paths`

Keep any useful fight-level columns if desired, but the authoritative population summary must use pooled path counts.

### 2. Simulated mean finish time must be pooled by simulated finishing paths

Current population summary averages each fight's mean finish time equally.

Instead compute the authoritative simulated population mean finish time from the sum of finish seconds across all non-decision paths divided by the total number of non-decision paths.

If convenient, persist compact per-fight fields such as:
- `simulated_nondecision_paths`
- `simulated_finish_time_sum_seconds`
- per-round simulated finish counts

Do not retain individual path traces.

### 3. Add true simulated exposure-normalized KD rate

Historical KD rate already uses actual observed duration:

`historical total KD / historical observed seconds * 900`

Add the equivalent simulated measure:

`simulated total KD / simulated total path fight-seconds * 900`

This must include every simulated path's actual elapsed duration, including decision paths at their full scheduled horizon.

Persist compact per-fight sufficient statistics as needed, e.g.:
- `simulated_total_kd`
- `simulated_total_exposure_seconds`

Then report:
- historical KD / 15 observed minutes
- simulated KD / 15 simulated minutes
- ratio / difference

Continue to report raw KD/path, zero-KD share, and multi-KD share separately.

### 4. Correct the semantics of simulated submission-attempt exposure

Current `simulated_share_with_submission_attempt` is derived from whether a fight-level mean attempts/path is greater than zero. That is effectively "share of historical matchups where at least one sampled path had an attempt", not "share of simulated paths with an attempt".

Add the true path-level pooled metric:

`number of simulated paths with >=1 submission attempt / total simulated paths`

Persist compact per-fight sufficient statistics such as:
- `simulated_paths_with_submission_attempt`
- `simulated_total_submission_attempts`

Report both:
- submission attempts per simulated path
- share of simulated paths with >=1 submission attempt

Historical comparison remains:
- attempts per historical fight
- share of historical fights with >=1 recorded attempt

Do not label matchup-level any-path exposure as path share.

## Invariants

Do NOT change:
- `config/event_mc_v1.yaml` mechanics values
- impact / KD / KO-TKO formulas
- submission generation or conversion
- stamina
- action rates
- phase rates
- judging
- FSR-32
- RNG ownership or draw order
- weight-class overrides
- age / tactical urgency

Frozen FSR-32 SHA-256 must remain:
`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Tests

Add controlled tests proving:
1. pooled finish-round shares weight simulated paths, not historical fights equally;
2. pooled mean finish time weights finishing paths correctly;
3. simulated KD/15min uses total simulated exposure seconds including decision horizons;
4. simulated share with submission attempt is path-level, not matchup-level;
5. deterministic replay remains unchanged.

Run focused tests and full EVENT MC test suite.

## Diagnostic rerun

After correction rerun the same broad cohort:

```bash
python -m pipeline.simulation.event_mc_v1.diagnostics.population_validation \
  --paths 10 \
  --start-year 2020 \
  --limit 100 \
  --seed 20260813 \
  --output-dir /tmp/event_mc_phase6_broad_corrected
```

Report old vs corrected values for:
- simulated finish-round shares
- simulated mean finish time
- simulated KD/path
- historical KD/15min
- simulated KD/15min
- submission attempts/path
- simulated path share with >=1 submission attempt

Method shares and winner probabilities should remain exactly unchanged for the same seeds.

## Stop condition

Commit and push only this metrics correction plus tests/continuity. Do not begin calibration.

Expected final line:

`PHASE 6 POPULATION METRICS FIX GATE: PASS`

or FAIL.