# Codex Retry Prompt — EVENT MC V1 Phase 2A Distance Temporal Parity

Date: 2026-08-13
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Why this retry exists

The prior Phase 2A run stopped safely at commit `1debecab69a141bf2f81179f3436af569733b750` because the governing Phase 2A prompt was not visible in that stale checkout.

The governing prompt now exists on the remote feature branch:

`docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md`

This is an environment synchronization issue only. No Phase 2A code was created in the failed attempt.

## Required bootstrap

1. Confirm the working tree is clean or contains no unrelated user work that would be overwritten.
2. Fetch the latest remote branch state:

```bash
git fetch origin --prune
git checkout feature/fsr-32-stamina-shadow
git pull --ff-only origin feature/fsr-32-stamina-shadow
```

If the local branch cannot fast-forward because of unrelated local changes, preserve them and stop rather than discarding them.

3. Verify the governing prompt is now present:

```bash
test -f docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md
git rev-parse HEAD
```

4. Read the latest continuity file and the governing Phase 2A prompt:

```text
docs/EVENT_MC_V1_CHAT_CONTINUITY_2026-08-12.md
docs/EVENT_MC_V1_CODEX_PHASE2A_DISTANCE_PARITY_2026-08-13.md
```

5. Execute the Phase 2A governing prompt exactly.

## Locks

- Phase 0: PASS.
- Phase 1: PASS.
- Phase 2A: explicitly authorized.
- Phase 2B: NOT authorized.
- Preserve the legacy blended wrestling consumer in Phase 2A.
- Preserve current effective mechanics and convert final legacy interval probabilities exactly to per-second hazards.
- Do not retune.
- Do not modify the current simulator, FSR builders/ratings/ontology, or calibration constants.
- Do not add stamina, damage, KD/KO, submissions, recovery, age, judging, clinch internals, or ground internals.
- Keep the implementation small, modular, observable, and easy to iterate.

Stop after Phase 2A implementation, diagnostics, tests, commit/push, and the required PASS/FAIL report from the governing prompt.
