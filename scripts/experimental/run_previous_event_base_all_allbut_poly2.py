"""Replay the mature event immediately before a cutoff with BASE vs ALL vs ALL-BUT poly2."""
from __future__ import annotations

import argparse
import sys
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import run_next_event_base_all_allbut_poly2 as replay


def _resolve_actual(row: pd.Series, red_name: str, blue_name: str) -> str:
    # First preserve the original resolver behavior.
    try:
        return replay._actual_winner_name(row, red_name, blue_name)
    except RuntimeError:
        pass

    # Historical cohort winner fields may be numeric rather than names/corners.
    for col in ("winner", "red_win", "r_win", "target", "label", "y"):
        if col in row.index and pd.notna(row[col]):
            value = row[col]
            try:
                num = int(float(value))
            except (TypeError, ValueError):
                continue
            if num in (0, 1):
                return red_name if num == 1 else blue_name

    # Some master artifacts store the winning fighter id.
    red_id = None
    blue_id = None
    for col in ("r_fighter_id", "red_fighter_id", "r_id"):
        if col in row.index and pd.notna(row[col]):
            red_id = str(row[col])
            break
    for col in ("b_fighter_id", "blue_fighter_id", "b_id"):
        if col in row.index and pd.notna(row[col]):
            blue_id = str(row[col])
            break
    for col in ("winner", "winner_id", "winner_fighter_id"):
        if col in row.index and pd.notna(row[col]):
            value = str(row[col])
            if red_id is not None and value == red_id:
                return red_name
            if blue_id is not None and value == blue_id:
                return blue_name

    raise RuntimeError(
        f"could not resolve actual winner for {red_name} vs {blue_name}; "
        f"available outcome fields: "
        + ", ".join(f"{c}={row[c]!r}" for c in row.index if "win" in c.lower() or c.lower() in {"y","target","label","outcome"})
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="2026-06-14", help="Select latest mature event strictly before YYYY-MM-DD")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    cohort, _ = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    dcol = replay._event_date_col(cohort)
    cohort["_event_date"] = pd.to_datetime(cohort[dcol], errors="raise").dt.normalize()
    cutoff = pd.Timestamp(args.before).normalize()
    prior = cohort.loc[cohort["_event_date"].lt(cutoff)].copy()
    if prior.empty:
        raise RuntimeError(f"no mature event found before {cutoff.date()}")
    selected = prior["_event_date"].max()

    # Existing replay runner selects the first event strictly after --after.
    # Passing the day before the selected event makes it select exactly that card.
    replay._actual_winner_name = _resolve_actual
    after = (selected - pd.Timedelta(days=1)).date().isoformat()
    sys.argv = [sys.argv[0], "--after", after, "--paths", str(args.paths), "--seed", str(args.seed)]
    print(f"[previous-event] cutoff={cutoff.date()} | selected={selected.date()}", flush=True)
    replay.main()


if __name__ == "__main__":
    main()
