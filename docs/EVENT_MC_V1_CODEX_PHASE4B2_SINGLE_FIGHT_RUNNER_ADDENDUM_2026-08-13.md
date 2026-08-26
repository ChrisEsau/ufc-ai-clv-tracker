# EVENT MC V1 — Phase 4B2 Single-Fight Runner Addendum

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Purpose

Add a Codespaces-friendly single-fight diagnostic runner for EVENT MC V1. This is a sanity/inspection utility only. It must not become a calibration engine and must not change simulation mechanics.

This addendum is part of the already-authorized Phase 4B2 work. Phase 4B2 remains a mechanics/architecture phase, not a population-calibration phase.

## Required CLI

Implement one module with a simple command such as:

```bash
python -m pipeline.simulation.event_mc_v1.single_fight --fight-id <FIGHT_ID> --paths 1 --trace
```

and:

```bash
python -m pipeline.simulation.event_mc_v1.single_fight --fight-id <FIGHT_ID> --paths 1000
```

The command must work cleanly in GitHub Codespaces from repository root.

If the current canonical historical identifier is `bout_id` rather than `fight_id`, accept both flags if practical, or clearly normalize the user-provided ID to the canonical internal identifier. Do not create a second identity system.

## Fight lookup

Use existing project historical/master data and the frozen FSR-32 pre-fight snapshot source. Resolve the requested historical fight by ID and use the correct pre-fight fighter profiles for that event date.

The runner must print enough fight metadata to verify the lookup:
- supplied ID and resolved canonical ID;
- event date;
- red fighter;
- blue fighter;
- scheduled rounds / horizon;
- weight class when available;
- resolved EVENT MC calibration fingerprint;
- resolved weight-class config key, if any; currently real overrides remain inactive unless explicitly present in config.

Do not rebuild FSR-32.

## Mode A — single path full trace

When `--paths 1 --trace` is used, print a human-readable chronological trace to stdout.

Before the event ledger, print:
- fighter profile inputs relevant to active mechanics;
- starting dynamic modifiers;
- active calibration fingerprint;
- key active calibration values for stamina, damage, KD, and KO/TKO sections once Phase 4B2 is implemented.

For every event, print a compact row/line containing as much of the following as applies:
- timestamp and round / clock within round;
- phase before event;
- actor / defender;
- action family;
- attempt outcome;
- phase transition / controller change;
- stamina before -> after for both fighters when changed;
- output multiplier / power multiplier used by acting fighter;
- landed strike impact;
- primary trauma;
- defender cumulative trauma before -> after;
- defender acute vulnerability before -> after;
- current KD resistance;
- KD probability;
- KD result;
- current finish resistance once Phase 4B2 exists;
- KO/TKO probability once Phase 4B2 exists;
- terminal finish result / winner if finish occurs.

Round-start and round-end lifecycle events must also be visible.

At the end print a compact path summary:
- winner if terminal winner exists;
- finish method/reason;
- finish time and round;
- total attempts/landed strikes by fighter and phase;
- TD attempts/completions;
- submission attempts (still nonterminal unless separately authorized later);
- clinch/ground control time;
- knockdowns by fighter;
- final stamina;
- final cumulative trauma;
- final acute vulnerability;
- scheduled-horizon indicator.

Do not dump giant Python reprs. The goal is readable terminal diagnostics.

## Mode B — multiple paths summary only

When `--paths N` is used without `--trace`, do NOT print each event.

Print one compact matchup header and one aggregate summary containing the calibration/config context and simulation results.

At minimum report:
- paths requested/completed;
- root seed scheme / base seed;
- calibration fingerprint;
- weight-class config key;
- runtime and paths/sec;
- red/blue terminal win counts and percentages where terminal outcomes exist;
- scheduled-horizon/no-terminal counts while judging/submissions remain incomplete;
- KO/TKO counts/rates after Phase 4B2;
- average finish time for KO/TKO paths;
- finish-round distribution;
- average total KDs/path and fighter-specific KDs;
- zero-KD / >=1 KD / multi-KD path rates;
- average final cumulative trauma by fighter;
- average final stamina by fighter;
- average phase time (DISTANCE/CLINCH/GROUND);
- average attempts / landed strikes / TD attempts / TD completions / SUB attempts by fighter;
- average control time by fighter;
- optional selected quantiles for stamina/trauma/finish time if easy and useful.

This is intentionally a single-fight sanity runner, not the population calibration harness.

## Reproducibility

Support an explicit `--seed` option. Default to a documented deterministic seed.

For multi-path mode derive stable per-path seeds without order-dependent hashing or hidden RNGs. Re-running the same command with the same seed, fight ID, path count, and config must reproduce discrete outcomes.

## Architecture / scope locks

- Runner is observer/diagnostic infrastructure only.
- Do not modify simulation formulas to make this runner convenient.
- Do not calibrate KD or KO/TKO.
- Do not add judging or terminal submission logic.
- Do not rebuild or alter FSR-32.
- Do not create a second fight clock.
- Do not create a second state model.
- Reuse existing engine, sinks/events, profile loading, calibration resolver, and frozen data sources.
- If additional trace observer support is required, keep it observer-only.
- Default/global simulator behavior must remain unchanged.

## Tests

Add focused tests proving at least:
1. historical fight ID resolves to the expected fighters/date from project data;
2. one-path trace mode executes successfully;
3. multi-path summary executes successfully;
4. same seed produces deterministic summary/discrete outcome data;
5. trace/summary runner does not mutate calibration or FSR artifacts;
6. trace mode records lifecycle + action + physiology events in chronological order;
7. a terminal Phase 4B2 KO/TKO stops further events once that system exists.

## Delivery

Return:
- implementation commit SHA;
- exact Codespaces commands for one traced path and a 100/1000-path summary;
- one abbreviated sample output for each mode;
- tests run;
- frozen FSR checksum confirmation;
- explicit statement that no calibration values/mechanics changed.

This addendum does not change the Phase 4B2 gate. It adds the user-facing diagnostic runner required for manual sanity checks while population calibration remains separate.
