import pandas as pd

from scripts.experimental import fsr_distance_striking_pressure_v1 as distance


def _frame():
    rows = []
    fights = [
        ("2024-01-01", "f1", (40, 0.70, 40), (20, 0.40, 20)),
        ("2024-02-01", "f2", (45, 0.75, 45), (18, 0.35, 18)),
        ("2024-03-01", "f3", (50, 0.80, 50), (15, 0.30, 15)),
    ]
    for date, fight_id, a_vals, b_vals in fights:
        for fighter_id, name, vals in [
            ("a", "A", a_vals),
            ("b", "B", b_vals),
        ]:
            attempts_per_round, share, attempts = vals
            rows.append({
                "fight_id": fight_id,
                "date": date,
                "fighter_id": fighter_id,
                "fighter_name": name,
                distance.C["distance_attempts_per_round"]: attempts_per_round,
                distance.C["distance_attempt_share"]: share,
                distance.C["distance_attempts"]: attempts,
            })
    return pd.DataFrame(rows)


def test_build_prefight_snapshots_preserves_grain():
    out = distance.build_prefight_snapshots(_frame())
    assert len(out) == 6
    assert not out.duplicated(["fight_id", "fighter_id"]).any()


def test_first_fight_starts_at_50():
    out = distance.build_prefight_snapshots(_frame())
    first = out[out["fight_id"] == "f1"]
    assert (first[distance.SKILL] == 50.0).all()


def test_pressure_moves_after_prior_date_pool_exists():
    out = distance.build_prefight_snapshots(_frame())
    third_a = out[(out["fight_id"] == "f3") & (out["fighter_id"] == "a")].iloc[0]
    assert third_a[distance.SKILL] > 50.0


def test_zero_distance_attempts_do_not_update():
    frame = _frame()
    mask = (frame["fight_id"] == "f2") & (frame["fighter_id"] == "a")
    frame.loc[mask, distance.C["distance_attempts"]] = 0
    out = distance.build_prefight_snapshots(frame)
    third_a = out[(out["fight_id"] == "f3") & (out["fighter_id"] == "a")].iloc[0]
    assert third_a[f"{distance.SKILL}_updates"] == 1


def test_ratings_stay_in_bounds():
    out = distance.build_prefight_snapshots(_frame())
    assert out[distance.SKILL].between(distance.MIN_RATING, distance.MAX_RATING).all()
