"""Single research audit: KD base -8.80, collapse scale 2.0, curvature 16.0.

Everything matches the existing -8.80 / curve 18 comparison run except collapse curvature.
The same 200 historical bouts are freshly recalculated by the inherited audit.
No production simulator or FSR artifact is modified.
"""
from pathlib import Path

from scripts.experimental import fsr_mature_2020plus_mc_kdbase88_curve18_scale2_200 as run18

run18.KD_BASE_LOGIT = -8.80
run18.COLLAPSE_CURVATURE = 16.0
run18.OUTPUT_PATH = Path("data/experimental/kdbase88_curve16_scale2_200.csv")
run18.SIM_ROUND_OUTPUT = Path("data/experimental/kdbase88_curve16_scale2_200_round_totals.csv")

if __name__ == "__main__":
    run18.main()
