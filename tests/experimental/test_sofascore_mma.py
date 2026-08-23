from scrapers.sofascore_mma import (
    event_summary,
    flatten_statistics,
    matches_promotion,
    stat_keys,
    stat_periods,
)


def sample_event():
    return {
        "id": 123456,
        "startTimestamp": 1757215033,
        "status": {"type": "finished"},
        "winnerCode": 2,
        "tournament": {
            "name": "LFA 216",
            "slug": "lfa-216",
            "uniqueTournament": {
                "name": "Legacy Fighting Alliance",
                "slug": "legacy-fighting-alliance",
            },
            "category": {"name": "MMA"},
        },
        "homeTeam": {"id": 10, "name": "Jonathan Elias"},
        "awayTeam": {"id": 20, "name": "Douglas Da Lapa"},
    }


def sample_stats():
    return {
        "statistics": [
            {
                "period": "ALL",
                "groups": [
                    {
                        "groupName": "Striking",
                        "statisticsItems": [
                            {
                                "name": "Significant strikes",
                                "key": "significantStrikes",
                                "home": "23/55",
                                "away": "31/58",
                                "homeValue": 23,
                                "awayValue": 31,
                            },
                            {
                                "name": "Takedowns",
                                "key": "takedowns",
                                "home": "1/3",
                                "away": "2/4",
                                "homeValue": 1,
                                "awayValue": 2,
                            },
                        ],
                    }
                ],
            },
            {
                "period": "ROUND_1",
                "groups": [
                    {
                        "groupName": "Striking",
                        "statisticsItems": [
                            {
                                "name": "Significant strikes",
                                "key": "significantStrikes",
                                "homeValue": 10,
                                "awayValue": 12,
                            }
                        ],
                    }
                ],
            },
        ]
    }


def test_promotion_matching_checks_unique_tournament_and_event_name():
    event = sample_event()
    assert matches_promotion(event, "LFA")
    assert matches_promotion(event, "legacy fighting")
    assert not matches_promotion(event, "Fury")


def test_event_summary_keeps_ids_and_fighter_names():
    row = event_summary(sample_event())
    assert row["event_id"] == 123456
    assert row["promotion"] == "Legacy Fighting Alliance"
    assert row["home_fighter"] == "Jonathan Elias"
    assert row["away_fighter"] == "Douglas Da Lapa"


def test_flatten_statistics_is_period_agnostic_and_preserves_values():
    rows = flatten_statistics(sample_event(), sample_stats())
    assert len(rows) == 3
    assert {row["period"] for row in rows} == {"ALL", "ROUND_1"}
    td = next(row for row in rows if row["stat_key"] == "takedowns")
    assert td["home_display"] == "1/3"
    assert td["away_value"] == 2


def test_stat_inventory_reports_actual_payload_keys_and_periods():
    payload = sample_stats()
    assert stat_keys(payload) == ["significantStrikes", "takedowns"]
    assert stat_periods(payload) == ["ALL", "ROUND_1"]
