# Codex Prompt — EVENT MC V1 FSR-32 Release Ingest + Phase 0 Baseline Resume

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Purpose: download the exact frozen FSR-32 parquet from the temporary GitHub Release asset, verify its byte identity by SHA-256, place it into the expected ignored path without transformation, then resume the deferred Phase 0 operational baseline exactly as previously specified.

This prompt does **not** authorize rebuilding FSR-32, changing the current simulator, retuning calibration, or beginning Phase 2A/2B.

## Known frozen artifact identity

Release tag:

`event-mc-v1-fsr32-handoff`

Release asset:

`fsr_32_prefight_snapshots.parquet`

Repository:

`ChrisEsau/ufc-ai-clv-tracker`

Expected SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Observed original Codespace size:

approximately `3.3M` (`3.20 MiB` shown as release asset)

Original Codespace source path used to establish identity:

`/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Expected destination in Codex checkout:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

## Step 1 — verify repository / branch state

Before downloading anything, report:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git remote -v
```

The working tree must be clean or contain only explicitly expected Phase 1 work already in progress. Do not overwrite unrelated user changes.

If Phase 1 implementation is currently uncommitted/in progress in this same checkout, preserve it safely and do not discard it. The artifact destination is gitignored, so FSR-32 ingestion must not require changing tracked source files.

## Step 2 — download the exact GitHub Release asset

Use any available authenticated GitHub mechanism that retrieves the raw release asset bytes without transformation.

Preferred if `gh` is available:

```bash
mkdir -p /tmp/event_mc_v1_fsr32_handoff

gh release download event-mc-v1-fsr32-handoff \
  --repo ChrisEsau/ufc-ai-clv-tracker \
  --pattern 'fsr_32_prefight_snapshots.parquet' \
  --dir /tmp/event_mc_v1_fsr32_handoff
```

If `gh` is unavailable, use an authenticated GitHub release-asset download method available in the environment. Do not rebuild the parquet and do not substitute another artifact.

## Step 3 — verify SHA-256 BEFORE use

Compute:

```bash
sha256sum /tmp/event_mc_v1_fsr32_handoff/fsr_32_prefight_snapshots.parquet
```

It must equal exactly:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

If it differs by even one byte:

- STOP;
- do not copy it into the simulator path;
- do not run the baseline;
- report the observed checksum and download method.

## Step 4 — copy byte-for-byte into the expected ignored path

Only after checksum PASS:

```bash
mkdir -p data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow
cp /tmp/event_mc_v1_fsr32_handoff/fsr_32_prefight_snapshots.parquet \
  data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet
```

Then verify destination identity:

```bash
sha256sum data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet
```

The destination SHA must also equal:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Do not rewrite, normalize, recompress, reserialize, or otherwise transform the parquet.

Do not commit the parquet.

## Step 5 — inspect frozen artifact metadata

Without modifying the file, report at minimum:

- exact SHA-256;
- byte size;
- parquet row count;
- parquet column count;
- schema / column names sufficient to verify expected FSR-32 structure;
- latest fight/event date if available;
- whether the Font/Rosas records needed by the frozen anchor are present.

If the parquet is unreadable or structurally incompatible with the frozen FSR-32 contract, STOP and report. Do not rebuild it.

## Step 6 — resume the deferred Phase 0 baseline

Once source/destination checksum verification and parquet metadata checks pass, immediately re-read and execute:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

using the restored exact FSR-32 artifact.

The Phase 0 numerical gate was previously DEFERRED / NOT PASSED solely because this artifact was unavailable. Do not treat prior failure as a calibration problem.

Run the frozen baseline contract exactly, including the locked fixtures, seeds, path counts, cohort ordering, outputs, checksums, and report format.

## Hard locks

Do NOT:

- rebuild FSR-32 or any upstream FSR artifact;
- change FSR ratings, ontology, builders, maturity rules, or leakage rules;
- modify current simulator mechanics or inheritance;
- retune KO, SUB, TD, stamina, recovery, age, damage, KD, judging, or any calibration constant;
- correct the legacy blended takedown-attempt consumer during baseline capture;
- modify or commit the release parquet;
- begin Phase 2A or Phase 2B;
- claim Phase 0 PASS unless the full frozen operational baseline requirements actually pass.

If Phase 1 generic-kernel work already exists in the checkout, do not mix its behavior into the current-simulator baseline. The Phase 0 baseline must continue to exercise the frozen existing simulator entry point, not EVENT MC V1.

## Required return

Return:

1. release-download method and exact source path;
2. downloaded SHA-256;
3. destination SHA-256 and byte-identity confirmation;
4. artifact metadata/shape/date coverage;
5. exact branch and commit used for baseline execution;
6. complete Phase 0 baseline report required by the original prompt;
7. files created/changed;
8. tests/checks;
9. confirmation parquet was not committed and no simulator/FSR/calibration code was changed;
10. explicit final gate line:

`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

or

`PHASE 0 OPERATIONAL BASELINE GATE: FAIL`

with exact blockers.

Do not begin Phase 2A.