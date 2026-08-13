# EVENT MC V1 Chat Continuity / Working Memory

Last updated: 2026-08-13 13:33 America/Chicago
Repository: `ChrisEsau/ufc-ai-clv-tracker`
Branch: `feature/fsr-32-stamina-shadow`

## Update rule
After every new Codex prompt, update this file. This file is continuity only, not architecture source of truth.

## Current gate state
- Phase 0 through Phase 5A: PASS
- Phase 6 population historical validation: PASS after pooled-metrics correction `0cc3ee58dae53b860a305f8bd78c64a907875967`
- Phase 7A strike/impact/KD/KO decomposition: PASS at `abd9bbbcff066f081b106b4c1b96c8e1626250e6`
- Phase 7B KD calibration: AUTHORIZED / current next phase
- KO/TKO conversion calibration: NOT YET AUTHORIZED; wait until Phase 7B KD environment is corrected
- Submission calibration: NOT YET AUTHORIZED
- Age, tactical urgency, real weight-class tuning: NOT AUTHORIZED

Frozen FSR-32 SHA-256: `621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a`

## Phase 6 final
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
- winner accuracy 59.0% on low-path diagnostic

Phase 6 gate: `PHASE 6 POPULATION HISTORICAL VALIDATION GATE: PASS`.

## Phase 7A final decomposition
Governing prompt: `docs/EVENT_MC_V1_CODEX_PHASE7A_DECOMPOSITION_2026-08-13.md`
Prompt commit: `97ee0d388b90c584cce50ba219642b33048b3364`
Implementation commit: `abd9bbbcff066f081b106b4c1b96c8e1626250e6`

100-fight x 10-path decomposition:
- historical strike attempts/15min 169.50 vs simulated 194.30 (1.15x)
- historical landed/15min 93.18 vs simulated 83.32 (0.89x)
- historical landing rate 54.97% vs simulated 42.88%
- historical KD/100 landed 0.280 vs simulated 3.718 (13.28x)
- historical KD/15min 0.261 vs simulated 3.098 (11.87x)
- simulated P(finish | KD strike) 46.53%
- simulated P(finish | non-KD landed strike) 1.322%
- 42.38% of KO/TKO finishing strikes were non-KDs
- 67.81% of KO/TKO paths had zero prior KDs
- finish checks/path 27.111
- impact median all 0.649; non-KD 0.615; KD 3.239; fight-ending 3.988
- same-seed red-win / KO_TKO / SUB / DEC probabilities preserved exactly; no mechanics changed.

Interpretation lock:
- strike volume is not the primary KD problem;
- KD probability conditional on a landed strike is the strongest demonstrated error;
- do not reduce strike output to hide the KD problem;
- impact-tail vs downstream KD shape cannot be separately identified from historical impact data, so first test the cleanest downstream global KD level parameter;
- KO conversion is also excessive but must remain frozen until KD mapping is calibrated, to preserve one-subsystem-at-a-time diagnosis.

Phase 7A gate: `PHASE 7A DECOMPOSITION GATE: PASS`.

## Phase 7B KD calibration
Governing prompt:
`docs/EVENT_MC_V1_CODEX_PHASE7B_KD_CALIBRATION_2026-08-13.md`
Prompt commit: `f63fe8390f9b32abcb1690dfda31b1411d8e9e1f`

Hard calibration scope:
- ONLY `defaults.knockdown.midpoint_impact_ratio` may move;
- all other KD parameters remain fixed;
- impact generation remains fixed;
- KO/TKO finish conversion remains fixed;
- no submission, stamina, judging, FSR, action-rate, phase-rate, age, urgency, or weight-class changes.

Calibration design:
- temporal split, preferably 2020-2024 train and 2025+ holdout;
- primary targets KD/100 landed and KD/15min;
- zero/multi-KD, round/phase, winner/method metrics are diagnostics/guardrails;
- coarse midpoint grid includes 8,12,16,24,32,48,64,96,128, then narrow refinement if bracketed;
- use common deterministic seeds;
- coarse low-path search then higher-path finalists/full train+holdout;
- do not optimize KO/TKO rate in Phase 7B;
- promote one YAML midpoint only if train and temporal holdout support it; otherwise stop without promotion.

Expected return: `PHASE 7B KD CALIBRATION GATE: PASS` or FAIL.

Next assistant action: independently review Phase 7B candidate search, temporal split, holdout support, exact config diff if promoted, and downstream method movement. If accepted, rerun/interpret decomposition in the corrected KD environment before authorizing KO/TKO conversion calibration.

Phase 7B result: common-seed coarse grid 8-128 and refined grid 28-40 on chronological 2020-2024 training and 2025+ holdout subsets supported midpoint 36 on both splits. Only `defaults.knockdown.midpoint_impact_ratio` was promoted from 8 to 36; all other mechanics and coefficients remain locked.
