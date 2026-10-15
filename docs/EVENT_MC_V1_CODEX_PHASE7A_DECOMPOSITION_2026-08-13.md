# EVENT MC V1 — Phase 7A Decomposition

Measurement only. Do not tune or change simulator mechanics.

Use the same leakage-safe mature-fighter frozen FSR-32 cohort as Phase 6.

Corrected Phase 6 anchors, 100 fights x 10 paths:
- historical KO/TKO 25.0%, simulated 81.4%
- historical SUB 17.0%, simulated 2.7%
- historical DEC 58.0%, simulated 15.9%
- historical KD/15 observed minutes 0.261
- simulated KD/15 simulated minutes 3.098
- KD exposure ratio about 11.87x
- simulated R1 share among finishes 79.55%
- simulated mean non-decision finish time 171.65s

Goal: determine where the excessive knockdown and KO/TKO behavior originates before calibration.

Build a compact diagnostic that separates:
1. strike attempts per unit exposure;
2. landed strikes per unit exposure;
3. knockdowns per landed strike;
4. landed-strike impact distribution;
5. finish conversion on KD strikes;
6. finish conversion on non-KD landed strikes;
7. repeated finish-check exposure and trauma relationship;
8. round and phase dependence.

Historical comparisons must use actual observed duration. Simulated comparisons must use actual simulated elapsed path time, including full horizon exposure for decisions.

Inspect the authoritative master schema and use only genuinely comparable historical strike fields. If historical UFCStats significant strikes are not directly comparable to EVENT MC total landed strikes, state that limitation instead of pretending equivalence.

Required simulated outputs:
- total/distance/clinch/ground strike attempts and landed strikes;
- attempts/15min, landed/15min, landing rate;
- impact count, mean, median, p75, p90, p95, p99, max;
- impact summaries for non-KD, KD, and fight-ending landed strikes;
- total KD, KD/100 landed, KD/15min, zero-KD and multi-KD path share;
- KD breakdown by round and phase;
- total landed-strike finish checks;
- P(finish | KD strike);
- P(finish | non-KD landed strike);
- share of KO/TKO finishes whose finishing strike was not a KD;
- share of KO/TKO paths with zero prior KDs;
- finish checks/path and finish checks/15min;
- compact trauma bins showing finish probability versus cumulative trauma/acute vulnerability using existing snapshots/events only.

Final summary must explicitly report, where comparable:
- simulated/historical strike-attempt exposure ratio;
- simulated/historical landed-strike exposure ratio;
- simulated/historical KD-per-landed ratio;
- simulated/historical KD/15min ratio;
- P(finish | KD strike);
- P(finish | non-KD strike);
- fraction of KO/TKO paths with zero KD before finish.

Rank likely root causes from strongest to weakest evidence.

Run:
1. focused tests;
2. small smoke cohort;
3. the same 100-fight x 10-path Phase 6 cohort.

Same-seed red/blue win and KO_TKO/SUB/DEC probabilities must remain exactly unchanged.

Do not modify config/event_mc_v1.yaml, action rates, impact, knockdown, finish, stamina, submission, judging, RNG, FSR, weight-class overrides, age, or tactical urgency.

FSR-32 checksum must remain 621cf4f389a150f8164678b4952b50d725b2be233c329448bb5dac0543230f3a.

Return implementation SHA, tests, broad-cohort decomposition, probability parity, and ranked diagnosis. Do not propose or commit tuned values.

Expected final line:
PHASE 7A DECOMPOSITION GATE: PASS
or FAIL.
