from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the repository root is importable when this test is run directly with
# ``pytest tests/experimental/test_fsr_finish_reservoir_traits_v1.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir


def _row(
    fight_id: str,
    fighter_id: str,
    date: str,
    *,
    kd: float = 0.0,
    sig: float = 20.0,
    head: float = 10.0,
    ground: float = 1.0,
    ctrl: float = 0.0,
    rounds: float = 3.0,
    ko_loss: float = 0.0,
) -> dict[str, object]:
    return {
        "fight_id": fight_id,
        "fighter_id": fighter_id,
        "date": date,
        reservoir.KD_COL: kd,
        reservoir.SIG_ABS_COL: sig,
        reservoir.HEAD_ABS_COL: head,
        reservoir.GROUND_ABS_COL: ground,
        reservoir.OPP_CTRL_COL: ctrl,
        reservoir.ROUNDS_COL: rounds,
        reservoir.KO_LOSS_COL: ko_loss,
    }


def test_first_fight_is_neutral_and_all_rows_are_bounded() -> None:
    rfs = pd.DataFrame(
        [
            _row("f1", "a", "2025-01-01"),
            _row("f1", "b", "2025-01-01", kd=1, ko_loss=1),
            _row("f2", "a", "2025-02-01", sig=45, head=25),
            _row("f2", "c", "2025-02-01"),
            _row("f3", "a", "2025-03-01", kd=1, sig=35),
            _row("f3", "b", "2025-03-01", sig=50, head=30),
        ]
    )

    out = reservoir.build_prefight_snapshots(rfs)

    assert len(out) == len(rfs)
    assert not out.duplicated(["fight_id", "fighter_id"]).any()

    first = out[out["fight_id"] == "f1"].sort_values("fighter_id")
    assert (first["knockdown_resistance"] == 50.0).all()
    assert (first["damage_durability"] == 50.0).all()
    assert (first["knockdown_resistance_updates"] == 0).all()
    assert (first["damage_durability_updates"] == 0).all()

    for skill in reservoir.SKILLS:
        assert out[skill].between(reservoir.MIN_RATING, reservoir.MAX_RATING).all()
        assert out[skill].notna().all()


def test_same_date_rows_snapshot_before_same_date_updates() -> None:
    # Fighter a has one prior fight, then appears twice on the same date. Both
    # same-date snapshots must be identical because neither fight can update the
    # state used by the other.
    rfs = pd.DataFrame(
        [
            _row("prior", "a", "2025-01-01", kd=0, sig=35, head=18),
            _row("same1", "a", "2025-02-01", kd=0, sig=15),
            _row("same2", "a", "2025-02-01", kd=2, sig=45, ko_loss=1),
        ]
    )

    out = reservoir.build_prefight_snapshots(rfs)
    same = out[out["fight_id"].isin(["same1", "same2"])].sort_values("fight_id")

    assert len(same) == 2
    assert same.iloc[0]["knockdown_resistance"] == same.iloc[1]["knockdown_resistance"]
    assert same.iloc[0]["damage_durability"] == same.iloc[1]["damage_durability"]
    assert same.iloc[0]["knockdown_resistance_updates"] == 1
    assert same.iloc[1]["knockdown_resistance_updates"] == 1
    assert same.iloc[0]["damage_durability_updates"] == 1
    assert same.iloc[1]["damage_durability_updates"] == 1


def test_missing_required_columns_fail_fast() -> None:
    bad = pd.DataFrame([{"fight_id": "f1", "fighter_id": "a", "date": "2025-01-01"}])

    try:
        reservoir.build_prefight_snapshots(bad)
    except ValueError as exc:
        assert "missing required reservoir-trait columns" in str(exc)
    else:
        raise AssertionError("expected missing-column validation to fail")
