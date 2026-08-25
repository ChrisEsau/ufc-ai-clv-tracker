# Event Clock V2 Historical Calibration Contract

## Scope

This harness measures mechanics; it does not maximize winner accuracy, fit market
probabilities, change FSR traits, or choose tradeoffs through an aggregate loss.
Standing cadence, brain timing/policy, memory, timeline semantics, legality,
judging, historical translation, and RNG mechanics remain frozen.

## Frozen cohort V1

The manifest contains the most recent 1,000 chronologically ordered fights that
have UFCStats round rows and at least two strictly prior UFC fights for **both**
fighters. Contiguous splits are development 400, calibration 200, validation
200, and final holdout 200. The holdout membership is locked but ordinary tools
refuse to evaluate it. Two prior fights is a cohort definition, not a claim that
cold start is solved.

Prior-fight counts are updated only after every bout on an event date has been
evaluated, so another same-date bout cannot become prior evidence. Manifest
construction and every runner invocation also require exactly one canonical
prefight row for each `(event_date, fight_id, fighter_id)` corner. Missing,
wrong-date, and duplicate rows fail before any simulation begins.

Historical fighter inputs are exact `(event_date, fight_id, fighter_id)` prefight
FSR V3 rows. The old Stage 8 use of `load_latest_profiles()` was leakage: latest
future performance changed empirical percentile ranks. Calibration instead
constructs one fixed population reference from each fighter's last snapshot
strictly before the earliest cohort date. Thus both individual values and the
normalization population are chronology-safe.

## Frozen targets

`historical_targets_v1.json` is generated from the frozen calibration split and
committed before parameter search. Its target digest binds membership, values,
tolerances, and schema. Candidates only read it. Acceptance is per metric as
PASS/WARN/FAIL; no combined score exists. UFCStats action locations are valid
comparators, but authoritative historical phase-time exposure is unavailable;
simulator phase shares are therefore reported without fabricated targets.
Structure/rate bands use ±20% for PASS and ±40% before FAIL; method-share
bands use ±15% and ±30%, with small absolute floors. These tolerances are part
of the frozen target payload and are not recomputed by candidates.

## Reproducibility and overrides

Root seeds are the first unsigned 64 bits of
`SHA256(seed_set_version|bout_id|path_id)`. Config identity is excluded so arms
receive matched streams; the engine retains its five child streams.

The sole override seam is `MechanicsCalibrationConfig`. YAML must contain only
`mechanics`, and keys must be in the explicit allowlist in `calibration/config.py`.
Canonical defaults are resolved first and immutable replacement applies explicit
overrides second. The baseline uses `explicit_overrides: {}`.

## Outputs

Each ledger record includes stable experiment/config identity, SHA, cohort and
target versions, paths, seed version, resolved/default and explicit configs,
fingerprint, acceptance results, invariants, baseline comparison, CI metadata,
and artifact metadata when available. Detailed outputs stay as workflow artifacts.
The metrics fingerprint covers deterministic simulator measurements and invariant
counts, while excluding timestamps and CI metadata. A hard invariant failure or
an out-of-band metric makes the run `FAIL`; a tolerance warning makes it `WARN`.
