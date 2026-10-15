"""Run the existing single-fight KO diagnostic with strict master-corner alignment.

This wrapper fixes a historical pairing bug without duplicating the diagnostic
implementation. The underlying FSR pair builder returns two rows in parquet
order; that order is not guaranteed to equal master red/blue corner order.
"""
from __future__ import annotations

from scripts.experimental import fsr_historical_corner_alignment as alignment
from scripts.experimental import fsr_single_fight_ko_failure_diagnostic as diagnostic


_original_build_cohort = diagnostic.population._build_cohort


def _aligned_build_cohort():
    cohort, pairs = _original_build_cohort()
    pairs = alignment.align_pair_dict_to_master_corners(cohort, pairs)
    return cohort, pairs


def main() -> None:
    diagnostic.population._build_cohort = _aligned_build_cohort
    diagnostic.main()


if __name__ == "__main__":
    main()
