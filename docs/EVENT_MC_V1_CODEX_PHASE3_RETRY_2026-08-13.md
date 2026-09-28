# Codex Retry — EVENT MC V1 Phase 3

The prior Phase 3 attempt stopped safely because the cloud sandbox contained Phase 2B history but did not contain the later Phase 3 governing prompt.

This is a stale sandbox snapshot, not a Phase 3 design failure.

Remote source branch:
`feature/fsr-32-stamina-shadow`

Required governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md`

Before implementation, update the sandbox from the remote feature branch without discarding existing history:

```bash
git fetch origin --prune
git log --oneline --decorate -12 origin/feature/fsr-32-stamina-shadow
git rebase origin/feature/fsr-32-stamina-shadow
```

A local cloud branch named `work` is acceptable.

Then verify:

```bash
git merge-base --is-ancestor 809389bdabe208e93536034bc795bbcf7e1ab038 HEAD
test -f docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
test -f docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md
```

If all succeed, read the latest continuity file and execute:

`docs/EVENT_MC_V1_CODEX_PHASE3_CLINCH_GROUND_FLOW_2026-08-13.md`

Phase 0 PASS. Phase 1 PASS. Phase 2A PASS. Phase 2B PASS. Phase 3 authorized. Phase 4 not authorized.

Do not begin any Phase 4 physiology/finish work.
