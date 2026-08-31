# Codex Prompt — Bootstrap Repository + Ingest Frozen FSR-32 Release + Resume Phase 0

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Target branch: `feature/fsr-32-stamina-shadow`

Purpose: recover from a fresh generic Codex workspace that is currently on local branch `work` with no Git remote, restore access to the correct repository/branch, then execute the existing frozen FSR-32 release-ingest and Phase 0 baseline instructions.

This prompt is deliberately self-contained for bootstrap. Do not assume the current local `docs/` directory is authoritative until the correct remote branch has been fetched and checked out.

## Important correction

The correct repository owner is:

`ChrisEsau`

NOT:

`ChrisEsasu`

A prior diagnostic used the misspelled owner in one unauthenticated GitHub API request. Do not repeat that typo.

## Step 1 — inspect current generic workspace

Report:

```bash
pwd
git branch --show-current || true
git rev-parse HEAD || true
git status --short --branch || true
git remote -v || true
```

Do not discard any nontrivial user changes. If the tree is clean as previously reported, continue.

## Step 2 — restore the correct Git remote

The exact remote URL is:

```text
https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
```

If `origin` is absent:

```bash
git remote add origin https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
```

If `origin` exists but points elsewhere, STOP and report it instead of overwriting it silently.

Then attempt:

```bash
git fetch origin --prune
```

If authentication is required, use only the GitHub credentials/credential helper already provisioned to the Codex environment. Do not print, log, or expose credentials/tokens.

If authenticated repository access is genuinely unavailable, STOP with:

`CODEX REPOSITORY BOOTSTRAP: AUTH BLOCKED`

and report only the non-secret credential mechanism checks attempted. Do not fabricate repository state.

## Step 3 — check out the exact feature branch

After a successful fetch:

```bash
git checkout -B feature/fsr-32-stamina-shadow origin/feature/fsr-32-stamina-shadow
```

Then report:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git remote -v
```

The target branch must be exactly:

`feature/fsr-32-stamina-shadow`

The working tree should be clean before baseline execution, aside from ignored runtime artifacts.

## Step 4 — verify governing documents now exist

Verify and read these files from the checked-out feature branch:

```text
docs/EVENT_MC_V1_CODEX_FSR32_RELEASE_INGEST_AND_BASELINE_RESUME_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md
docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md
docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md
docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md
```

If the release-ingest prompt is still absent after the correct branch is checked out, STOP and report the exact fetched branch SHA and directory listing.

## Step 5 — obtain the frozen release asset

Release tag:

`event-mc-v1-fsr32-handoff`

Asset:

`fsr_32_prefight_snapshots.parquet`

Expected SHA-256:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Preferred if `gh` is available and authenticated:

```bash
mkdir -p /tmp/event_mc_v1_fsr32_handoff

gh release download event-mc-v1-fsr32-handoff \
  --repo ChrisEsau/ufc-ai-clv-tracker \
  --pattern 'fsr_32_prefight_snapshots.parquet' \
  --dir /tmp/event_mc_v1_fsr32_handoff
```

If `gh` is unavailable, use the authenticated GitHub credential mechanism already available for the successful repository fetch to retrieve the release asset through GitHub's release API. Do not expose credential values in logs or output. Do not use an unauthenticated request for this private repository and interpret a 404 as evidence that the release is missing.

If the repository fetch succeeds but the environment provides no safe authenticated mechanism to download private release assets, STOP with:

`FSR-32 RELEASE DOWNLOAD: AUTH BLOCKED`

Do not rebuild or substitute the artifact.

## Step 6 — mandatory checksum gate

Before use:

```bash
sha256sum /tmp/event_mc_v1_fsr32_handoff/fsr_32_prefight_snapshots.parquet
```

It must equal exactly:

`621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

Any mismatch => STOP. Do not copy/use the file.

## Step 7 — execute the existing release-ingest prompt

Once repository/branch access is restored and the release asset can be downloaded, read and execute exactly:

`docs/EVENT_MC_V1_CODEX_FSR32_RELEASE_INGEST_AND_BASELINE_RESUME_2026-08-12.md`

That document governs:

- byte-for-byte copy to the ignored expected FSR-32 path;
- destination checksum verification;
- parquet metadata inspection;
- resumed Phase 0 baseline execution;
- frozen fixtures/seeds/path counts/cohort ordering;
- output/manifest contract;
- hard non-goals;
- final PASS/FAIL report.

## Hard locks

Do NOT:

- rebuild FSR-32 or upstream FSR artifacts;
- alter the parquet;
- commit the parquet;
- modify the existing simulator;
- modify FSR builders/ratings/ontology;
- retune any calibration;
- begin Phase 2A or Phase 2B;
- mix EVENT MC V1 Phase 1 kernel behavior into the old-simulator baseline;
- expose authentication secrets.

## Required return

Return one of these outcomes:

### Success path

Complete the full required Phase 0 report from the release-ingest/baseline prompt and end with exactly:

`PHASE 0 OPERATIONAL BASELINE GATE: PASS`

or the corresponding FAIL line with true baseline blockers.

### Bootstrap auth failure

End with:

`CODEX REPOSITORY BOOTSTRAP: AUTH BLOCKED`

### Release-only auth failure

End with:

`FSR-32 RELEASE DOWNLOAD: AUTH BLOCKED`

In all cases, report exact branch/SHA/remote state and checks attempted, but never secrets.