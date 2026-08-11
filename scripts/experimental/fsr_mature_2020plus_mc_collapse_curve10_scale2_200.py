"""Single 200-bout x 10-path audit at collapse scale=2.0, curvature=10.0.

This reuses the scale-2 curvature audit implementation but runs exactly one
candidate. All other working constants and terminal-collapse accounting remain
unchanged.
"""
from scripts.experimental import fsr_mature_2020plus_mc_collapse_curvature_sweep_scale2_200 as sweep


if __name__ == "__main__":
    sweep.CANDIDATES = (sweep.Candidate(10.0),)
    sweep.OUTPUT_PATH = sweep.Path("data/experimental/collapse_curve10_scale2_200.csv")
    sweep.main()
