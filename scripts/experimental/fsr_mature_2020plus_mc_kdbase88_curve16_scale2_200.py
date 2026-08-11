"""Single research audit: KD base -8.80, collapse scale 2.0, curvature 16.0.

Everything matches the existing -8.80 / curve 18 comparison run except collapse curvature.
The same 200 historical bouts are freshly recalculated by the inherited audit.
No production simulator or FSR artifact is modified.
"""
from pathlib import Path

from scripts.experimental import fsr_mature_2020plus_mc_kdbase88_curve18_scale2_200 as run18

run18.run87.KD_BASE_LOGIT = -8.80
run18.run87.COLLAPSE_CURVATURE = 16.0
run18.run87.COLLAPSE = run18.run87.collapse_mod.CollapseCandidate(
    "scale2.0_curve16.0",
    run18.run87.COLLAPSE_SCALE,
    run18.run87.COLLAPSE_CURVATURE,
)
run18.run87.OUTPUT_PATH = Path("data/experimental/kdbase88_curve16_scale2_200.csv")

if __name__ == "__main__":
    run18.main()
