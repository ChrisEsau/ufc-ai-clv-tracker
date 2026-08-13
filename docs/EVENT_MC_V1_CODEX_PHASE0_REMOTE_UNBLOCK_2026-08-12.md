# Codex Prompt — EVENT MC V1 Phase 0 Remote Unblock

Repository: `ChrisEsau/ufc-ai-clv-tracker`

Target branch: `feature/fsr-32-stamina-shadow`

Purpose: resolve the environmental Git-remote blocker, verify the exact requested branch, then resume the already-approved Phase 0 baseline materialization task. This prompt does **not** authorize EVENT MC V1 implementation.

## Known GitHub truth

The repository exists at:

`https://github.com/ChrisEsau/ufc-ai-clv-tracker.git`

The remote branch `feature/fsr-32-stamina-shadow` has been independently verified on GitHub.

Expected remote head at issuance of this prompt:

`6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6`

Observed local checkout from your blocker report:

```text
local branch: work
local SHA: eebb8134fd2f7cb4eb68ffcb93464ab74883633f
working tree: clean
configured remotes: none
```

The observed local SHA is an ancestor of the expected remote feature-branch head, but do not rely on that fact alone; fetch and verify the exact remote branch.

## Step 1 — configure the known remote

If `git remote -v` still shows no remotes, add exactly:

```bash
git remote add origin https://github.com/ChrisEsau/ufc-ai-clv-tracker.git
```

If a remote named `origin` now exists, do not overwrite it blindly. Show its URL and verify it points to `ChrisEsau/ufc-ai-clv-tracker`.

If authentication or network access prevents fetching this repository, stop and report the exact failure. Do not create a substitute branch.

## Step 2 — fetch and verify the exact feature branch

Run:

```bash
git fetch origin feature/fsr-32-stamina-shadow
git rev-parse origin/feature/fsr-32-stamina-shadow
```

Expected SHA at issuance:

`6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6`

If the fetched branch points to a **newer descendant** because documentation-only commits were added after this prompt was issued, that is acceptable only if:

1. the branch name is exactly `feature/fsr-32-stamina-shadow`;
2. `6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6` is an ancestor of the fetched head;
3. inspection shows the intervening commits do not alter current simulator mechanics, FSR construction, or calibration.

If any of those checks fail, stop and report before baseline materialization.

Useful lineage check:

```bash
git merge-base --is-ancestor 6a4c594690243b8c0ee4b3b6f066d54e78cc7ad6 origin/feature/fsr-32-stamina-shadow
```

## Step 3 — check out the exact remote branch

With a clean working tree and verified remote branch, create/reset the local feature branch to track the verified remote branch:

```bash
git switch -C feature/fsr-32-stamina-shadow --track origin/feature/fsr-32-stamina-shadow
```

Then report:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
```

The current branch must now be exactly:

`feature/fsr-32-stamina-shadow`

## Step 4 — re-read current Phase 0 source-of-truth docs

After checkout, re-read completely:

1. `docs/EVENT_MC_V1_ARCHITECTURE_AUDIT_2026-08-12.md`
2. `docs/EVENT_MC_V1_PHASE0_CLOSURE_2026-08-12.md`
3. `docs/EVENT_MC_V1_PHASE0_INTERFACE_DECISIONS_2026-08-12.md`
4. `docs/EVENT_MC_V1_PHASE0_BASELINE_FREEZE_2026-08-12.md`
5. `docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`
6. `docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md`

Do not continue from stale copies read before the branch was fetched.

## Step 5 — resume the already-approved Phase 0 baseline task

Once branch verification succeeds, execute:

`docs/EVENT_MC_V1_CODEX_PHASE0_BASELINE_PROMPT_2026-08-12.md`

exactly as written.

The prior approval remains valid. You do **not** need to stop for another implementation-plan approval after successful branch verification unless you discover a new scope/architecture conflict.

## Hard locks remain unchanged

Do not:

- implement anything under `pipeline/simulation/event_mc_v1/`;
- modify existing simulator mechanics or inheritance;
- modify FSR-32 construction/ratings;
- retune KO, SUB, TD, stamina, recovery, damage, KD, judging, age, or any calibration constant;
- correct the legacy blended TD-attempt consumer during baseline capture;
- alter maturity/leakage rules;
- create an alternate feature branch from a different base.

## Stop conditions

Stop and report if:

- the GitHub remote cannot be added/fetched;
- authentication fails;
- the exact feature branch cannot be fetched;
- the fetched lineage fails the ancestor/integrity checks;
- the working tree is unexpectedly dirty before checkout;
- simulator/FSR/calibration changes are discovered in the branch lineage that violate the Phase 0 freeze.

Otherwise proceed directly through the Phase 0 baseline materialization prompt and return its required PASS/FAIL report.

Do not begin Phase 1.