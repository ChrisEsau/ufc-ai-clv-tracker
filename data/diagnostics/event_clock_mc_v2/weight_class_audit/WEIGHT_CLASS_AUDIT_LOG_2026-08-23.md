# Event Clock MC V2 — Weight-Class Audit Log

Date: 2026-08-23

Research only. No weight-class tuning or simulator parameter overrides have been applied.
Historical mechanics below are exposure-normalized using canonical total fight elapsed seconds (`match_time_sec`).

## Men's Flyweight

Cohort: 46 eligible post-cutoff fights, 500 paths/fight.

Headline:
- ML accuracy: 67.4%
- ML Brier: 0.2234
- ML log loss: 0.6390
- Method accuracy: 52.2%
- Actual DEC / KO-TKO / SUB: 50.0% / 23.9% / 26.1%
- Simulated DEC / KO-TKO / SUB: 45.9% / 30.2% / 23.9%

Mechanics:
- Historical mean duration: 596.1 s
- Simulated mean duration: 679.8 s
- Duration bias: +14.0% (too long)
- Significant-strike attempt rate bias: -22.9%
- Significant-strike landed rate bias: -23.1%
- Takedown attempt rate bias: -14.4%
- Takedown landed rate bias: -13.2%
- Knockdown rate bias: -32.2%
- Submission-attempt rate bias: -31.6%
- Control-time rate bias: -21.4%

Interpretation: primary signal is low activity/pace per unit time despite simulated fights lasting too long. Do not treat this as a simple finish-share calibration problem.

## Men's Bantamweight

Cohort: 62 eligible post-cutoff fights, 500 paths/fight.

Headline:
- ML accuracy: 67.7%
- ML Brier: 0.2074
- ML log loss: 0.6025
- Method accuracy: 56.5%
- Actual DEC / KO-TKO / SUB: 59.7% / 21.0% / 19.4%
- Simulated DEC / KO-TKO / SUB: 52.2% / 30.4% / 17.4%
- Method biases: DEC -7.5 pp, KO/TKO +9.5 pp, SUB -1.9 pp

Mechanics:
- Historical mean duration: 634.2 s
- Simulated mean duration: 695.9 s
- Duration bias: +9.7% (too long)
- Significant-strike attempt rate bias: -16.5%
- Significant-strike landed rate bias: -14.7%
- Takedown attempt rate bias: -7.9%
- Takedown landed rate bias: -11.3%
- Knockdown rate bias: +34.6%
- Submission-attempt rate bias: -41.9%
- Control-time rate bias: -15.4%

Interpretation: pace is too low and duration too long, but unlike flyweight the knockdown rate is too high. The combination of less striking activity with more KDs and excess KO/TKO share suggests bantamweight damage/KD conversion per landed strike may be too aggressive. Submission generation and control activity are too low.
