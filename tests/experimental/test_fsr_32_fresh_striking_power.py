from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32


def _evidence_row(
    fighter_id: str,
    fight_id: str,
    fight_order: int,
    *,
    sig: float,
    event: bool,
    event_evidence: float,
) -> dict[str, object]:
    return {
        "fighter_id": fighter_id,
        "fight_id": fight_id,
        "fight_order": fight_order,
        "sig_str_landed": sig,
        "fight_power_evidence_v8": event_evidence,
        "power_event": event,
        "opportunity": 1.0 - np.exp(-sig / fsr32.LOW_END_SIG_TAU),
    }


def test_prefight_power_is_leakage_safe_and_non_degrading_after_demonstration() -> None:
    fighter = "fighter-a"
    order = pd.DataFrame(
        {
            "fight_id": ["f1", "f2", "f3", "f4"],
            "fight_order": [0, 1, 2, 3],
        }
    )
    snapshots = pd.DataFrame(
        {
            "fight_id": ["f1", "f2", "f3", "f4"],
            "fighter_id": [fighter] * 4,
        }
    )
    evidence = pd.DataFrame(
        [
            _evidence_row(fighter, "f1", 0, sig=20.0, event=False, event_evidence=0.0),
            _evidence_row(fighter, "f2", 1, sig=8.0, event=True, event_evidence=2.0),
            _evidence_row(fighter, "f3", 2, sig=25.0, event=False, event_evidence=0.0),
        ]
    )

    power = fsr32.build_prefight_striking_power(snapshots, evidence, order)
    values = dict(zip(power["fight_id"], power["striking_power"]))

    # No history is visible before the first fight.
    assert values["f1"] == 50.0

    # Fight 1 was a real clean-strike opportunity with no power event, so the
    # next prefight snapshot moves below neutral.
    assert values["f2"] < 50.0

    # The power event in fight 2 is NOT visible in fight 2 itself; it first
    # appears in the next prefight snapshot.
    assert values["f3"] > 50.0

    # A later quiet fight cannot erase demonstrated fresh power.
    assert values["f4"] >= values["f3"]


def test_low_end_curve_matches_selected_v9_candidate() -> None:
    opportunity = 3.25
    expected = 50.0 - 15.0 * (1.0 - np.exp(-opportunity / 6.0))
    assert np.isclose(fsr32._low_end_power_rating(opportunity), expected)


def test_positive_rating_stays_in_established_range() -> None:
    low = fsr32._positive_power_rating(
        prior_fights=1,
        prior_power_events=1,
        peak_single_fight_evidence=0.2,
    )
    high = fsr32._positive_power_rating(
        prior_fights=20,
        prior_power_events=15,
        peak_single_fight_evidence=5.0,
    )

    assert 50.0 <= low <= 90.0
    assert 50.0 <= high <= 90.0
    assert high > low
