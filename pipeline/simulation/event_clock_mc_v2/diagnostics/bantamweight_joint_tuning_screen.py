"""Research-only men's bantamweight joint lethality tuning screen.

Reuses the validated flyweight screening machinery but changes only the cohort
and compact consequence-side grid. Fatigue slope remains t/12; strike,
takedown, submission, timing, FSR, judging, and frozen V1 mechanics remain
unchanged.
"""
from __future__ import annotations

from pipeline.simulation.event_clock_mc_v2.diagnostics import flyweight_joint_tuning_screen as base

base.DIVISION = "bantamweight"
base.ARMS = []
for intercept in (5.0, 10.0, 15.0, 20.0):
    for bonus in (None, 1.0, 2.0):
        base.ARMS.append((f"i{int(intercept)}_b{'0' if bonus is None else int(bonus)}", intercept, bonus))
base.ARMS.append(("mens_ref_i35_b3", 35.0, 3.0))

if __name__ == "__main__":
    base.main()
