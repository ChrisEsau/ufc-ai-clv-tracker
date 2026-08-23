"""Run the historical stamina/lethality study with the round-stats metadata collision fixed.

The shared round-stats standardizer temporarily joins finish_round to calculate
round exposure.  The study subsequently joins canonical fight metadata itself.
Drop that temporary copy before the study join so pandas does not create
finish_round_x / finish_round_y.
"""
from pipeline.simulation.event_clock_mc_v2.diagnostics import stamina_lethality_interaction as study

_original_standardize = study.standardize_round_stats_input


def _standardize_without_duplicate_finish_round(frame):
    out = _original_standardize(frame)
    return out.drop(columns=["finish_round"], errors="ignore")


study.standardize_round_stats_input = _standardize_without_duplicate_finish_round


if __name__ == "__main__":
    study.main()
