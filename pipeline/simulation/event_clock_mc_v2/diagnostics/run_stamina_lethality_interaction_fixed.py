"""Run the historical stamina/lethality study with metadata collisions removed.

The shared round-stats standardizer preserves some fight-level fields that the
study subsequently joins from the canonical master table. Drop any overlapping
fight-level metadata first so pandas cannot create *_x / *_y columns.
"""
from pipeline.simulation.event_clock_mc_v2.diagnostics import stamina_lethality_interaction as study

_original_standardize = study.standardize_round_stats_input


def _standardize_without_duplicate_fight_metadata(frame):
    out = _original_standardize(frame)
    return out.drop(
        columns=["finish_round", "division", "method_norm", "winner_id"],
        errors="ignore",
    )


study.standardize_round_stats_input = _standardize_without_duplicate_fight_metadata


if __name__ == "__main__":
    study.main()
