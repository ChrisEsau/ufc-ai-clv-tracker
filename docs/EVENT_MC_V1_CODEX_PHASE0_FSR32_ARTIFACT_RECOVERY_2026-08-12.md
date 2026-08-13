# Codex Prompt — EVENT MC V1 Phase 0 FSR-32 Artifact Recovery

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Branch: `feature/fsr-32-stamina-shadow`

Purpose: recover the exact pre-existing FSR-32 parquet (or a byte-identical backup) required by the frozen Phase 0 baseline. This is an artifact-discovery/recovery task only. It does **not** authorize rebuilding the FSR chain, modifying simulator code, or beginning EVENT MC V1.

## Read first

Re-read current versions of:

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`

## Current blocker

The required frozen input is absent from the isolated checkout:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

The Phase 0 operational baseline must not proceed until the exact existing artifact (or a byte-identical backup/copy) is recovered and checksummed.

Do **not** rebuild the FSR-32 chain in this task. A newly rebuilt file is not automatically equivalent to the historical frozen artifact and would defeat the baseline-freeze purpose unless separately approved.

## Objective

Search available local/mounted storage and recover the exact pre-existing FSR-32 artifact if it exists.

Search targets should include, where accessible:

- the current repository and all parent/sibling worktrees under `/workspace`;
- other cloned/worktree copies of `ufc-ai-clv-tracker`;
- `/workspaces`, `/workspace`, `/mnt`, `/tmp`, `/var/tmp`, and user-home locations that are readable;
- Codespace/worktree persistent mounts and caches visible to this environment;
- backup names such as:
  - `fsr_32_prefight_snapshots.parquet.bak`
  - `fsr_32_prefight_snapshots.bak.parquet`
  - `*.bak` beside an `fsr_32_shadow` directory;
- archived or copied `fsr_32_shadow` directories;
- existing GitHub Actions/downloaded artifacts already present locally.

Use filesystem metadata searches first. Avoid scanning system pseudo-filesystems or paths that would create excessive permission/noise.

Useful search patterns include:

```bash
find /workspace /workspaces /mnt /tmp /var/tmp "$HOME" \
  -type f \
  \( -name 'fsr_32_prefight_snapshots.parquet' \
     -o -name 'fsr_32_prefight_snapshots.parquet.bak' \
     -o -name '*fsr_32*prefight*snapshot*.parquet*' \) \
  2>/dev/null
```

Also search for directories:

```bash
find /workspace /workspaces /mnt /tmp /var/tmp "$HOME" \
  -type d -name 'fsr_32_shadow' 2>/dev/null
```

If GitHub CLI/API access is available in the environment, inspect repository Actions artifacts/releases only for evidence of previously uploaded/generated FSR snapshot artifacts. Do not launch workflows and do not rebuild data.

## Candidate validation

For every candidate file found, record:

```text
absolute path
file size
mtime
SHA-256
parquet row count
parquet column count
column names or schema summary
min/max event/fight date where available
```

Then verify it is structurally consistent with the current FSR-32 builder contract and simulator-facing usage.

If multiple candidates exist:

- do not choose by filename alone;
- compare checksums, shape, date coverage, and known FSR-32 trait columns;
- identify byte-identical copies;
- preserve all candidate evidence in the report.

For the best candidate, if Font/Rosas rows can be inspected without running the simulator, report the exact prefight FSR-32 values for the 2026-03-07 anchor where available. Use this only as an identity/sanity check, not as permission to alter data.

## Recovery into the expected path

If and only if a credible pre-existing candidate is found, copy (do not transform/rebuild) it into the required local path:

`data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/fsr_32_prefight_snapshots.parquet`

Preserve the source file. Do not move/delete the original.

After copying, recompute SHA-256 and confirm source and destination are byte-identical.

Because `*.parquet` is ignored, this recovered artifact should normally remain untracked. Do not force-add it to Git unless repository policy explicitly requires that, which it currently does not.

## If no exact artifact is found

Stop and report:

```text
FSR-32 ARTIFACT RECOVERY: NOT FOUND
```

Include all searched roots and any partial/older FSR candidates discovered.

Do not rebuild FSR-32, FSR-28, or earlier FSR generations in this task.

At that point the user/assistant will decide separately whether to:

- recover the artifact from another known Codespace/local machine;
- upload/copy it into this environment;
- or approve a controlled reconstruction with a new baseline identity.

## If recovery succeeds

Report:

```text
FSR-32 ARTIFACT RECOVERY: FOUND
source absolute path
destination path
SHA-256
row count
column count
latest date
source/destination byte-identical confirmation
```

Then immediately resume the already-approved Phase 0 baseline materialization prompt:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

No new implementation-plan approval is needed if the recovered artifact passes the checks above.

## Hard locks

Do not:

- rebuild any FSR database generation;
- modify FSR builders/traits;
- modify current simulator mechanics/inheritance;
- retune any calibration constants;
- correct the legacy blended TD consumer;
- create/modify `pipeline/simulation/event_mc_v1/`;
- alter maturity/leakage rules;
- fabricate baseline values from prior documentation anchors.

The two pre-existing age-contract test failures observed in the prior baseline attempt should be recorded as pre-existing test debt, not modified in this artifact-recovery task.

Do not begin Phase 1.