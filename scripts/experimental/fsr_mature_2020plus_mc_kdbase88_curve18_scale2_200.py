"""Single research audit: KD base -8.80, collapse scale 2.0, curvature 18.0.

Everything matches the existing -8.80 / curve 20 comparison run except collapse curvature.
No production simulator or FSR artifact is modified.
"""
from pathlib import Path

from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as run87

run87.KD_BASE_LOGIT = -8.80
run87.COLLAPSE_CURVATURE = 18.0
run87.COLLAPSE = run87.collapse_mod.CollapseCandidate(
    "scale2.0_curve18.0",
    run87.COLLAPSE_SCALE,
    run87.COLLAPSE_CURVATURE,
)
run87.OUTPUT_PATH = Path("data/experimental/kdbase88_curve18_scale2_200.csv")

if __name__ == "__main__":
    run87.main()
