import pandas as pd

from scripts.experimental import fsr_clinch_striking_v1 as clinch


def _frame():
    rows = []
    for date, fight_id, a, b, a_vals, b_vals in [
        ("2024-01-01", "f1", "a", "b", (12, 0.6, 8, 0.5, 4, 0.25), (4, 0.2, 4, 0.25, 8, 0.5)),
        ("2024-02-01", "f2", "a", "b", (14, 0.7, 10, 0.6, 5, 0.3), (3, 0.15, 5, 0.3, 10, 0.6)),
        ("2024-03-01", "f3", "a", "b", (15, 0.72, 11, 0.62, 5, 0.3), (3, 0.14, 5, 0.3, 11, 0.62)),
    ]:
        for fighter_id, name, vals in [(a, "A", a_vals), (b, "B", b_vals)]:
            apr, share, attempts, acc, opp_attempts, allowed = vals
            rows.append({
                "fight_id": fight_id,
                "date": date,
                "fighter_id": fighter_id,
                "fighter_name": name,
                clinch.C["clinch_attempts_per_round"]: apr,
                clinch.C["clinch_attempt_share"]: share,
                clinch.C["clinch_attempts"]: attempts,
                clinch.C["clinch_accuracy"]: acc,
                clinch.C["opp_clinch_attempts"]: opp_attempts,
                clinch.C["clinch_accuracy_allowed"]: allowed,
            })
    return pd.DataFrame(rows)


def test_build_prefight_snapshots_preserves_grain():
    out = clinch.build_prefight_snapshots(_frame())
    assert len(out) == 6
    assert not out.duplicated(["fight_id", "fighter_id"]).any()


def test_first_fight_starts_at_50():
    out = clinch.build_prefight_snapshots(_frame())
    first = out[out["fight_id"] == "f1"]
    for skill in clinch.SKILLS:
        assert (first[skill] == 50.0).all()


def test_pressure_moves_after_prior_date_pool_exists():
    out = clinch.build_prefight_snapshots(_frame())

    # The first date only bootstraps prior-date population pools. The second
    # fight can then generate an informative update, visible in the third
    # pre-fight snapshot.
    third_a = out[(out["fight_id"] == "f3") & (out["fighter_id"] == "a")].iloc[0]
    assert third_a["clinch_striking_pressure"] > 50.0


def test_ratings_stay_in_bounds():
    out = clinch.build_prefight_snapshots(_frame())
    for skill in clinch.SKILLS:
        assert out[skill].between(clinch.MIN_RATING, clinch.MAX_RATING).all()
