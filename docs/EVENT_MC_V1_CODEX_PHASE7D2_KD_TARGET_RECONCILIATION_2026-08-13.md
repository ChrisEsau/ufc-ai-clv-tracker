# EVENT MC V1 — Phase 7D2 KD Target Reconciliation

Measurement only. Do not change config or simulator mechanics.

Current committed values stay fixed:
- knockdown.midpoint_impact_ratio = 36.0
- finish.midpoint_impact_ratio = 36.0

The historical time-semantics correction changed exposure-normalized targets. We now have a conflict:
- historical KD/100 landed = 0.280
- simulated KD/100 landed ≈ 0.438
- historical KD/15min = 0.440
- simulated KD/15min ≈ 0.383

Because historical total-strike exposure is not guaranteed definition-identical to EVENT MC modeled landed strikes, do not select a new midpoint yet.

Rerun the same 100-fight x 10-path cohort, start-year 2020, seed 20260813, and report historical vs simulated:
- KD/fight or KD/path
- KD/100 landed
- KD/15min
- zero-KD share
- multi-KD share
- landed strikes/fight or path
- landed/15min
- KO/TKO share
- mean fight duration

Also report the same metrics for in-memory KD midpoint candidates 32, 36, 40, 44, and 48, with finish midpoint fixed at 36.

Do not rank candidates with a single combined objective. Instead show the tradeoff separately for:
1. KD/fight/path
2. KD/100 landed
3. KD/15min

State explicitly whether the corrected evidence justifies changing the committed KD midpoint or whether the target conflict is mainly an upstream strike-exposure/comparability issue.

No YAML promotion is authorized.

Also preserve this modeling lock for future submission work: top and bottom submission conversion should be treated 1:1 for now; do not introduce a top-position conversion bonus during later submission calibration unless UFC-specific evidence supports it.

Expected final line:
PHASE 7D2 KD TARGET RECONCILIATION GATE: PASS
