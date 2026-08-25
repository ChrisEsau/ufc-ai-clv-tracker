"""Generic full-card runner for the Event Clock V2 / Brain MC.

The public selector is UFC event id + paths per fight.  The implementation
resolves the event id to the canonical master event name, then delegates to
the deterministic calibration runner so event runs use the same prefight
FSR state, Brain MC engine, empirical Event2 KO/KD, submissions, judging,
and matched path-seed semantics as validation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v2.calibration.runner import run
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture


_EVENT_ID_COLUMNS = ("event_id", "ufcstats_event_id")
_EVENT_URL_COLUMNS = ("event_url", "ufcstats_event_url")
_EVENT_NAME_COLUMNS = ("event_name", "ufcstats_event_name")
_EVENT_DATE_COLUMNS = ("date", "event_date", "ufcstats_event_date")
_EVENT_ID_RE = re.compile(r"/event-details/([A-Za-z0-9]+)")


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _master_events() -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).copy()
    name_col = _first_column(master, _EVENT_NAME_COLUMNS)
    date_col = _first_column(master, _EVENT_DATE_COLUMNS)
    if name_col is None or date_col is None:
        raise RuntimeError("canonical master is missing event name/date columns")

    id_col = _first_column(master, _EVENT_ID_COLUMNS)
    url_col = _first_column(master, _EVENT_URL_COLUMNS)

    events = pd.DataFrame(
        {
            "event_name": master[name_col].astype(str).str.strip(),
            "event_date": pd.to_datetime(master[date_col], errors="coerce"),
        }
    )
    if id_col is not None:
        events["event_id"] = master[id_col].astype(str).str.strip()
    elif url_col is not None:
        events["event_id"] = (
            master[url_col]
            .astype(str)
            .str.extract(_EVENT_ID_RE, expand=False)
            .fillna("")
            .str.strip()
        )
    else:
        # Older master snapshots may not persist UFCStats event ids.  Keeping
        # the field explicit lets next-event discovery still run and makes an
        # --event-id request fail loudly rather than matching the wrong card.
        events["event_id"] = ""

    events = (
        events.dropna(subset=["event_date"])
        .drop_duplicates(["event_name", "event_date", "event_id"])
        .sort_values(["event_date", "event_name"])
        .reset_index(drop=True)
    )
    return events


def resolve_event_id(event_id: str) -> pd.Series:
    target = str(event_id).strip()
    if not target:
        raise ValueError("event_id must be non-empty")
    events = _master_events()
    if not events.event_id.astype(str).str.len().gt(0).any():
        raise RuntimeError(
            "canonical master does not expose UFCStats event ids or event URLs; "
            "cannot resolve --event-id safely"
        )
    matches = events[events.event_id.astype(str).eq(target)]
    if len(matches) != 1:
        raise ValueError(f"event_id {target!r} matched {len(matches)} canonical events")
    return matches.iloc[0]


def resolve_next_event(after_event_name: str) -> pd.Series:
    """Resolve the first canonical UFC event strictly after a named event."""
    target = str(after_event_name).strip()
    events = _master_events()
    anchors = events[events.event_name.eq(target)]
    if anchors.empty:
        raise ValueError(f"anchor event not found: {target}")
    anchor_date = anchors.event_date.max()
    later = events[events.event_date.gt(anchor_date)].sort_values(
        ["event_date", "event_name"]
    )
    if later.empty:
        raise ValueError(f"no canonical event exists after {target}")
    return later.iloc[0]


def run_event(
    *,
    event_id: str | None,
    paths_per_fight: int,
    output: Path,
    next_after_event_name: str | None = None,
) -> dict:
    if paths_per_fight < 1:
        raise ValueError("paths_per_fight must be positive")
    if bool(event_id) == bool(next_after_event_name):
        raise ValueError("provide exactly one of event_id or next_after_event_name")

    selected = (
        resolve_event_id(str(event_id))
        if event_id
        else resolve_next_event(str(next_after_event_name))
    )
    resolved_id = str(selected.event_id).strip()
    resolved_name = str(selected.event_name).strip()
    resolved_date = pd.Timestamp(selected.event_date).date().isoformat()

    record = run(
        split="calibration",
        paths_per_fight=paths_per_fight,
        config_path=Path("configs/event_clock_v2/calibration/default.yaml"),
        output=output,
        ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
        event_name=resolved_name,
    )
    record["event_runner"] = {
        "requested_event_id": None if event_id is None else str(event_id),
        "resolved_event_id": resolved_id or None,
        "event_name": resolved_name,
        "event_date": resolved_date,
        "paths_per_fight": int(paths_per_fight),
    }
    # Persist the augmented record, not only the delegated calibration record.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a complete UFC event through the Brain MC."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--event-id", help="Canonical UFCStats event id")
    selector.add_argument(
        "--next-after-event-name",
        help="Discovery helper: run the first canonical event after this event name",
    )
    parser.add_argument("--paths", type=int, default=100, help="Paths per fight")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_event(
        event_id=args.event_id,
        next_after_event_name=args.next_after_event_name,
        paths_per_fight=args.paths,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
