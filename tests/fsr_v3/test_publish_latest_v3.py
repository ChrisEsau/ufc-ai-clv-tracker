from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.publish import _final_global_value, _latest_post_rating


def test_latest_post_rating_uses_post_most_recent_fight_not_prefight():
    history = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fight_id": "f1",
                "fighter_id": "A",
                "trait": "takedown_tendency",
                "pre_rating": 4.0,
                "post_rating": 5.0,
            },
            {
                "event_date": pd.Timestamp("2025-06-01"),
                "fight_id": "f2",
                "fighter_id": "A",
                "trait": "takedown_tendency",
                "pre_rating": 5.0,
                "post_rating": 6.25,
            },
        ]
    )

    latest = _latest_post_rating(history, "takedown_tendency")
    assert len(latest) == 1
    assert latest.iloc[0]["fighter_id"] == "A"
    assert np.isclose(latest.iloc[0]["takedown_tendency"], 6.25)
    assert not np.isclose(latest.iloc[0]["takedown_tendency"], 5.0)


def test_final_global_value_uses_final_chronological_population_baseline():
    history = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2025-01-01"),
                "fight_id": "f1",
                "population_baseline": 0.45,
            },
            {
                "event_date": pd.Timestamp("2025-06-01"),
                "fight_id": "f2",
                "population_baseline": 0.52,
            },
        ]
    )
    assert np.isclose(_final_global_value(history, "population_baseline"), 0.52)
