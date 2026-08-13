# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 13:16 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6 population historical validation: PASS after pooled-metrics correction `0cc3ee58dae53b860a305f8bd78c64a907875967`
- Phase 7A strike/impact/KD/KO decomposition: AUTHORIZED / measurement only
- Calibration changes themselves: NOT YET AUTHORIZED
- Age, tactical urgency, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Phase 6 final
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE6_POPULATION_VALIDATION_2026-08-13.md`
Prompt commit: `4496d2f917628333ee148dccde55e51308c3561e`
Implementation: `c4e750a0dfe23c2dd87df8b69c768aacd119b61f`
Metrics correction prompt: `docs/EVENT_MC_V1_CODEX_PHASE6_POPULATION_METRICS_FIX_2026-08-13.md`
Metrics correction implementation: `0cc3ee58dae53b860a305f8bd78c64a907875967`

Corrected 100-fight x 10-path anchors:
- historical KO/TKO 25.0%, simulated 81.4%
- historical SUB 17.0%, simulated 2.7%
- historical DEC 58.0%, simulated 15.9%
- simulated non-decision R1 share 79.55%
- historical mean non-decision finish time 652.76s, simulated 171.65s
- historical KD/15 observed minutes 0.261
- simulated KD/15 simulated minutes 3.098 (~11.87x historical)
- historical submission attempts/fight 0.610; simulated 0.178/path
- true simulated path share with >=1 submission attempt 12.6%
- winner accuracy remained 59.0% on this low-path broad diagnostic

Phase 6 final gate: `PHASE 6 POPULATION HISTORICAL VALIDATION GATE: PASS`.

## Phase 7A
Before calibration, decompose the excessive KD/KO behavior into upstream stages so one parameter family does not compensate for another.

Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE7A_DECOMPOSITION_2026-08-13.md`
Prompt commit: `97ee0d388b90c584cce50ba219642b33048b3364`

Required measurement chain:
1. strike-attempt exposure per time;
2. landed-strike exposure per time;
3. KD per landed strike;
4. impact severity distribution;
5. finish conversion on KD strikes;
6. finish conversion on non-KD landed strikes;
7. repeated finish-check exposure and trauma relationship;
8. round/phase dependence.

Historical strike comparisons must use only genuinely comparable UFCStats fields and actual observed duration; simulated metrics use actual simulated elapsed path exposure. Same-seed win/method probabilities must remain exactly unchanged.

No simulator mechanics or calibration values may change in Phase 7A.

Expected return: `PHASE 7A DECOMPOSITION GATE: PASS` or FAIL.

Phase 7A measurement implemented: compact path sufficient statistics decompose attempts, landed strikes, impact tails, KD exposure, KD/non-KD finish conversion, repeated checks, trauma bins, and round/phase dependence. The 100-fight x 10-path rerun preserved every same-seed winner and method probability exactly; no mechanics, config, RNG, or FSR changed.
