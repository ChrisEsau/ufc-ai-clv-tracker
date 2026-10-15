# EVENT MC V1 — Phase 7G Global Submission Attempt Rate Calibration

Calibrate only `defaults.submission_attempts.base_30s` after position-neutral attempt generation and conversion. Search the common-seed grid `0.045, 0.050, 0.055, 0.060, 0.065, 0.070` on chronological 2020–2024 train and 2025+ holdout cohorts.

Primary split-specific targets are attempts/path, attempts/15 elapsed minutes, and paths with an attempt. Conversion and SUB share are diagnostics only. Freeze bottom multiplier 1.0, both position bonuses 0.0, conversion intercept -2.20, KD/finish midpoints 36, and every other mechanic/config value. Promote one base only if both temporal splits support the same region without material guardrail regression. Conversion calibration remains deferred.

Expected final line: `PHASE 7G GLOBAL SUBMISSION ATTEMPT RATE CALIBRATION GATE: PASS` or FAIL.
