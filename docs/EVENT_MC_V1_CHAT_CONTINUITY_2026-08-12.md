# EVENT MC V1 Chat Continuity / Working Memory

Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current state
Phase 0 through 7H history is preserved in repository history. Phase 7H PASS: `submission_finish.intercept = -0.60`; submission attempt base remains 0.045; bottom submission attempt multiplier 1.0; top/bottom submission conversion bonuses 0.0; KD midpoint 36; finish midpoint 36. Frozen FSR-32 SHA-256 is `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`.

## FSR-32 recovery procedure
If the ignored local parquet is missing, do not rebuild it. Download GitHub Release tag `event-mc-v1-fsr32-handoff`, asset `fsr_32_prefight_snapshots.parquet`, verify SHA-256 `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a` before use, copy byte-for-byte to `data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`, verify the destination checksum again, and never commit or rewrite the parquet. Resume the current blocked phase, not the historical Phase 0 baseline.

## Phase 7I — Strike exposure definition + baseline audit
Phase 7I is measurement-only. No strike calibration is authorized yet. Determine whether EVENT MC modeled distance/clinch/ground strike actions are best compared with UFCStats total strikes, significant strikes, or neither. On the same train/holdout cohorts used in Phase 7G/7H, compare historical total and significant strike attempts/landed/rates/accuracy with current committed EVENT MC all-strike and phase-specific attempts/landed/rates/accuracy. Inspect code semantics end-to-end and report the recommended historical comparator for attempt generation and landing probability. Report KO/SUB/DEC, KD, and timing guardrails. Do not modify YAML, mechanics, FSR, RNG, or any calibration. Expected return: `PHASE 7I STRIKE EXPOSURE DEFINITION BASELINE AUDIT GATE: PASS` or FAIL.
