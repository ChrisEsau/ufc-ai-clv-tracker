# EVENT MC V1 — Phase 4B2 Single-Fight Runner Completion

Date: 2026-08-13

## Status

Phase 4B2 KO/TKO finish mechanics: PASS after independent review.
Single-fight runner implementation commit: `05485f0bd0460ae726fb2f0283373a727464cb38`.

This prompt completes the already-authorized Codespaces diagnostic runner only. Do not change simulator mechanics or calibration.

## Independent review findings

The runner is functional, deterministic, observer-only, and correctly uses the historical master plus frozen FSR-32. However several requirements from the governing addendum remain incomplete.

### 1. Trace detail

For each physiology event, clearly print the defender's post-event:
- cumulative trauma;
- acute vulnerability;
- current KD resistance;
- KD probability/result.

For finish checks print:
- current finish resistance;
- impact ratio;
- finish probability/result.

When a primary action changes phase/controller ownership, print the before -> after phase plus clinch/ground controller transition clearly enough for a human to follow the path.

Do not create new causal state or RNG. Use trace snapshots/events already produced by the engine.

### 2. Aggregate summary completeness

Add explicit:
- KO/TKO finish-round distribution;
- red TD attempts and completions separately;
- blue TD attempts and completions separately;
- submission attempts per side explicitly;
- scheduled-horizon count and rate;
- red/blue KO/TKO wins/counts and rates where available.

Keep compact output; do not print per-path traces in multi-path mode.

### 3. Tests required by original addendum

Add focused tests proving:
- trace event timestamps are nondecreasing;
- no trace event occurs after terminal `FightFinished`;
- same seed reproduces identical discrete trace/summary data;
- multi-path summary arithmetic is correct on a small deterministic fixture or controlled synthetic run;
- historical fight lookup resolves canonical ID and both frozen prefight profiles.

Do not weaken existing Phase 4B2 lifecycle tests.

### 4. Lewis/Daukaus sanity ID

The original addendum suggested bout/fight ID `4b7ec02b39fc6f70` for Derrick Lewis vs Chris Daukaus if it resolves cleanly.

Check it explicitly. If it resolves, include exact Codespaces commands for that fight in the final report. If it does not resolve in `data/master/ufc_master.parquet`, report the actual lookup result/blocker and retain the working Benoit Saint Denis vs Mauricio Ruffy example instead. Do not invent an ID mapping.

## Scope locks

Do NOT change:
- config values;
- simulation formulas;
- KD/KO behavior;
- RNG stream ownership/order;
- fight state;
- engine clock/lifecycle;
- FSR-32;
- submissions;
- judging;
- age;
- real weight-class tuning.

## Validation

Run at minimum:

```bash
python -m pytest -q tests/simulation/event_mc_v1/test_single_fight.py tests/simulation/event_mc_v1/test_phase4b2_finishes.py
python -m pytest -q tests/simulation/event_mc_v1 tests/experimental/test_fsr_static_mc_v0.py
python -m compileall pipeline/simulation/event_mc_v1 tests/simulation/event_mc_v1
sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet
```

Also run one traced path and one small multi-path summary from repo root.

## Return

Return:
1. implementation commit SHA;
2. tests/checksum;
3. exact one-path Codespaces command;
4. exact multi-path Codespaces command;
5. abbreviated trace output;
6. abbreviated aggregate output;
7. Lewis/Daukaus ID lookup result.

Final line:

`PHASE 4B2 SINGLE-FIGHT RUNNER GATE: PASS`

or FAIL.
