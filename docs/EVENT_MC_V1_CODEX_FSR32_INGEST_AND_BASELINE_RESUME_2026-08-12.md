# Codex Prompt — Ingest Frozen FSR-32 Artifact and Resume Deferred Phase 0 Baseline

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Purpose: ingest the exact frozen FSR-32 parquet supplied by the user, verify byte identity against the recorded SHA-256, place it at the expected ignored path without transformation, and resume the previously deferred Phase 0 operational baseline. This does not authorize any FSR rebuild, simulator retuning, or Phase 2 work.

## Frozen artifact identity

The exact historical artifact has been located in the user's real Codespace at:

`/workspaces/ufc-ai-clv-tracker/data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Recorded SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Recorded size from `ls -lh`: approximately `3.3M`.

Canonical identity note:

`docs/EVENT_MC_V1_FSR32_FROZEN_ARTIFACT_IDENTITY_2026-08-12.md`

## User-supplied file

The user will attach/provide the parquet file to this Codex task. Treat that attachment as a candidate only until its SHA-256 is verified.

Do not rebuild, reserialize, repartition, rewrite, normalize, convert, or otherwise transform the parquet.

## Step 1 — verify repository state

Before using the attachment:

- confirm branch is `feature/fsr-32-stamina-shadow`;
- fetch/fast-forward from `origin` if needed so the current prompt and continuity docs are present;
- confirm the working tree is clean apart from any temporary uploaded file outside tracked paths.

Do not overwrite tracked work.

## Step 2 — locate the attached parquet

Identify the actual filesystem path where the uploaded attachment was mounted/staged by the Codex environment.

Confirm it is a readable parquet file.

Compute:

```bash
sha256sum <uploaded_file_path>
```

The result must be exactly:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

If the SHA differs by even one byte, STOP. Report the observed SHA and do not use the file.

## Step 3 — inspect without rewriting

Read metadata only and report:

- exact byte size;
- row count;
- column count;
- schema/column names;
- minimum and maximum fight/event date if identifiable;
- whether the Font/Rosas anchor rows can be resolved from the artifact.

Do not write the parquet back out during inspection.

## Step 4 — install the exact artifact at the expected ignored path

Expected destination:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Create the parent directory if absent, then copy the uploaded file byte-for-byte to the destination.

Do not use pandas/pyarrow to rewrite it. Use a filesystem copy.

After copying, compute SHA-256 on the destination.

Source and destination SHA-256 must both equal:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

The parquet remains ignored/generated data. Do not force-add or commit the parquet to Git.

## Step 5 — resume the deferred Phase 0 operational baseline

Once byte identity is verified, re-read current versions of:

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`

Then execute the Phase 0 baseline prompt exactly as written.

Use the exact frozen counts/seeds/fixtures from the baseline-freeze contract. Do not tune any parameter to prior anchors.

Required high-level work remains:

- deterministic traces for seeds `7`, `17`, `20260811` per resolved fixture;
- 5 fixture matchup summaries × 1000 paths with root seed `20260811`;
- first 200 mature aligned bouts × 10 paths with root seed `20260810`;
- full 1,565-fight method/submission baseline where supported observationally;
- manifest/checksums and compact outputs under `data/experimental/event_mc_v1_baseline/` as defined by the frozen contract.

## Phase 1 interaction

Phase 1 generic kernel work may already be in progress or complete on the branch. Do not mix that implementation into the baseline measurement.

The Phase 0 baseline must continue to use the untouched current simulator entry point specified by the freeze contract (`StaticFSRMCFullFightV1`) and the supplied frozen FSR-32 parquet.

Do not alter EVENT MC V1 kernel behavior to make the old baseline run.

## Hard non-goals

Do not:

- rebuild FSR-32 or any upstream FSR artifact;
- modify FSR-32 ratings/builders;
- modify the current simulator physics/inheritance/calibration;
- change KO/SUB/TD/stamina/recovery/damage/KD/judging/age constants;
- correct the legacy blended TD consumer during baseline capture;
- weaken maturity/leakage rules;
- commit the parquet;
- begin Phase 2A or Phase 2B.

## Required report

Return:

1. branch and commit actually used;
2. uploaded source path;
3. source SHA-256;
4. destination SHA-256;
5. exact byte size / rows / columns / date coverage;
6. fixture resolution table;
7. deterministic trace status;
8. 1000-path fixture summary;
9. 200 × 10 cohort summary;
10. full method/submission baseline status;
11. Font/Rosas FSR values + legacy blended consumer;
12. mismatches vs frozen research anchors and identified reasons;
13. files created/changed;
14. tests/checks run;
15. confirmation that the parquet was not committed;
16. confirmation that no current simulator/FSR/calibration changes were made;
17. confirmation that no Phase 2 work began.

End exactly with:

`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

or

`PHASE 0 OPERATIONAL BASELINE GATE: FAIL`

with exact blockers.