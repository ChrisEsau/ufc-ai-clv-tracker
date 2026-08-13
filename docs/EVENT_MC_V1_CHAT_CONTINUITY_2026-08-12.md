# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 13:07 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6 population historical validation: IMPLEMENTED at `c4e750a0dfe23c2dd87df8b69c768aacd119b61f`; final PASS withheld pending narrow metrics correction
- Calibration, age, tactical urgency, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Phase 6
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE6_POPULATION_VALIDATION_2026-08-13.md`
Prompt commit: `4496d2f917628333ee148dccde55e51308c3561e`
Implementation commit: `c4e750a0dfe23c2dd87df8b69c768aacd119b61f`

Initial 100-fight x 10-path run reported KO/TKO 81.4% simulated vs 25.0% historical, SUB 2.7% vs 17.0%, DEC 15.9% vs 58.0%, KD/path 1.008 vs historical KD/fight 0.370, and submission attempts/path 0.178 vs 0.610 historical attempts/fight. Measurement only; no tuning.

Independent review found four aggregation issues to correct before Phase 7:
- pool simulated finish-round counts across finishing paths rather than averaging fight-level shares;
- pool simulated finish time across finishing paths rather than averaging per-fight means;
- add simulated KD per 15 minutes using actual simulated path exposure seconds;
- report true pooled share of simulated paths with >=1 submission attempt.

Correction prompt: `docs/EVENT_MC_V1_CODEX_PHASE6_POPULATION_METRICS_FIX_2026-08-13.md`
Prompt commit: `feb0eb43105f70b4991d5151b62203df9a2cefa2`

No mechanics/config/RNG/FSR/calibration changes are authorized. Same-seed winner and method probabilities must remain unchanged.

Expected return: `PHASE 6 POPULATION METRICS FIX GATE: PASS` or FAIL.
