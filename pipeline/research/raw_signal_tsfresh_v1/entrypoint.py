from __future__ import annotations

"""Actions entrypoint for the tsfresh challenger.

The shared raw-signal observation builder attaches canonical fight dates from
master data. The raw round parquet also carries an event_date column, so remove
that duplicate input column before delegating to the shared builder.
"""

from pipeline.research.raw_signal_tsfresh_v1 import run as challenger


_original_build_fight_observations = challenger._build_fight_observations


def _build_fight_observations_without_duplicate_date(rounds, master):
    clean_rounds = rounds.drop(columns=["event_date"], errors="ignore")
    return _original_build_fight_observations(clean_rounds, master)


def main() -> None:
    challenger._build_fight_observations = _build_fight_observations_without_duplicate_date
    challenger.main()


if __name__ == "__main__":
    main()
