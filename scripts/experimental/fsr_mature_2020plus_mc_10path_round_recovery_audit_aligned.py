"""Run the 2020+ mature recovery audit with strict master red/blue alignment."""
from __future__ import annotations

from scripts.experimental import fsr_historical_corner_alignment as alignment
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as population
from scripts.experimental import fsr_mature_2020plus_mc_10path_round_recovery_audit as audit


_original_build_cohort = population._build_cohort


def _aligned_build_cohort():
    cohort, pairs = _original_build_cohort()
    return cohort, alignment.align_pair_dict_to_master_corners(cohort, pairs)


def main() -> None:
    # recovery audit calls baseline._build_cohort(), where baseline is the same
    # imported population module object patched here.
    population._build_cohort = _aligned_build_cohort
    audit.main()


if __name__ == "__main__":
    main()
