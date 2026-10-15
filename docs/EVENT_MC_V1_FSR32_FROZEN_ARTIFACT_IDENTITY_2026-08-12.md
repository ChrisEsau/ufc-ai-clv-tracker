# EVENT MC V1 — Frozen FSR-32 Artifact Identity

Date recorded: 2026-08-12

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch context: `feature/fsr-32-stamina-shadow`

Purpose: record the identity of the exact pre-existing FSR-32 parquet located in the user's development Codespace so the deferred Phase 0 numerical baseline can later be materialized against the same artifact rather than a rebuilt substitute.

## Artifact

Codespaces path:

`/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Repository-relative expected path:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Observed human-readable size:

`3.3M`

Observed filesystem metadata at discovery:

`-rw-rw-rw- 1 vscode vscode 3.3M Aug 13 01:45`

## Interpretation

This is the artifact that should be supplied to Codex for the deferred Phase 0 baseline unless later validation proves otherwise.

Do not silently replace it with a rebuilt FSR-32 parquet.

When transferred to Codex, verify the destination SHA-256 is exactly:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

before running any baseline simulations.

The artifact remains ignored/generated data and should not be committed to normal Git history merely to transfer it.

Phase 1 generic-kernel work may continue independently. Before Phase 2A parity work, revisit the deferred Phase 0 numerical baseline using this exact artifact unless the user explicitly authorizes another exception.
