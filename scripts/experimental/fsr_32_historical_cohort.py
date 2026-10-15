"""Historical mature-cohort loader for the explicit FSR-32 stamina contract.

This module deliberately reuses the established 2020+ mature cohort definition,
but reloads fighter profiles from the FSR-32 artifact and then strictly aligns
every two-fighter pair to master red/blue corners by fighter_id.
"""
from __future__ import annotations

import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_historical_corner_alignment as alignment
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as baseline
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


def build_aligned_cohort() -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    master = modern._load_master(modern.MASTER_PATH)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(fsr32.OUTPUT_PATH, cohort)
    cohort = cohort.merge(
        baseline._master_metadata(),
        on="bout_id",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    pairs = alignment.align_pair_dict_to_master_corners(cohort, pairs)
    return cohort, pairs
