"""Adapter for a broad longitudinal MMA fight database.

The first supported source is the public ``MMAStats and fights Complete
Database`` DuckDB dataset. Only dated fight facts are consumed here. Current
profile fields (current age, current record, gym, etc.) are deliberately not
used for historical validation.

Cold-start opponent quality is intentionally computed from non-UFC fights only.
That prevents prior UFC results from being re-imported through the external
layer and counted twice when a fighter leaves and later returns to the UFC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import normalize_method, normalize_name, validate_external_bouts

DEFAULT_ELO = 1500.0
DEFAULT_ELO_K = 24.0


def _clean_source_fights(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce source-level fight identity and reject unusable fighter rows."""
    x = frame.copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="raise").dt.normalize()
    x["fight_id"] = x["fight_id"].astype(str)
    if x["fight_id"].duplicated().any():
        duplicate_count = int(x["fight_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"MMA Global source violated unique fight_id contract: {duplicate_count} duplicate rows"
        )

    f1 = x["fighter_1"].map(normalize_name)
    f2 = x["fighter_2"].map(normalize_name)
    invalid = (f1 == "") | (f2 == "") | (f1 == f2)
    if invalid.any():
        print(f"MMA Global: excluding {int(invalid.sum())} invalid/self-match fight rows")
        x = x.loc[~invalid].copy()
    return x.reset_index(drop=True)


def load_mma_global_wide(path: str | Path) -> pd.DataFrame:
    """Load the dated longitudinal fight fact table from a local DuckDB file."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("duckdb is required to read MMA Global data") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    con = duckdb.connect(str(path), read_only=True)
    try:
        columns = [
            "fight_id", "organization", "event_name", "event_date", "weight_class",
            "is_major_org", "fighter_1", "fighter_2", "winner", "method_normalized",
            "round_num", "time_finish_seconds", "f1_height_cm", "f1_weight_kg",
            "f2_height_cm", "f2_weight_kg",
        ]
        query = "SELECT " + ", ".join(columns) + " FROM fights_career_longitudinal"
        frame = con.execute(query).fetchdf()
    finally:
        con.close()
    frame = _clean_source_fights(frame)
    return frame.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def _expected_score(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def add_leakage_safe_elo(
    wide: pd.DataFrame,
    *,
    initial: float = DEFAULT_ELO,
    k_factor: float = DEFAULT_ELO_K,
) -> pd.DataFrame:
    """Add pre/post Elo using only prior-date outcomes in the supplied frame.

    Same-date changes are delayed to remove arbitrary row-order dependence.
    Callers control the evidence universe; ``load_mma_global_fighter_bouts``
    supplies non-UFC fights only so the cold-start quality signal remains
    independent of UFC observations.
    """
    x = wide.copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="raise").dt.normalize()
    x["f1_key"] = x["fighter_1"].map(normalize_name)
    x["f2_key"] = x["fighter_2"].map(normalize_name)
    x["winner_key"] = x["winner"].map(normalize_name)
    ratings: dict[str, float] = {}
    pieces: list[pd.DataFrame] = []

    for _, day in x.groupby("event_date", sort=True):
        day = day.copy()
        f1_pre = day["f1_key"].map(lambda key: ratings.get(key, initial)).astype(float)
        f2_pre = day["f2_key"].map(lambda key: ratings.get(key, initial)).astype(float)
        day["f1_pre_elo"] = f1_pre
        day["f2_pre_elo"] = f2_pre
        pending: dict[str, float] = {}
        for row in day.itertuples(index=False):
            r1 = float(row.f1_pre_elo)
            r2 = float(row.f2_pre_elo)
            if row.winner_key == row.f1_key:
                s1 = 1.0
            elif row.winner_key == row.f2_key:
                s1 = 0.0
            else:
                s1 = 0.5
            e1 = _expected_score(r1, r2)
            delta = float(k_factor) * (s1 - e1)
            pending[row.f1_key] = pending.get(row.f1_key, 0.0) + delta
            pending[row.f2_key] = pending.get(row.f2_key, 0.0) - delta
        for key, delta in pending.items():
            ratings[key] = ratings.get(key, initial) + delta
        day["f1_post_elo"] = day["f1_key"].map(lambda key: ratings.get(key, initial))
        day["f2_post_elo"] = day["f2_key"].map(lambda key: ratings.get(key, initial))
        pieces.append(day)
    return pd.concat(pieces, ignore_index=True) if pieces else x


def to_fighter_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert one-row-per-fight data into the canonical fighter-bout schema."""
    x = wide.copy()
    if "f1_pre_elo" not in x.columns:
        x = add_leakage_safe_elo(x)
    x["winner_key"] = x["winner"].map(normalize_name)
    rows: list[dict[str, object]] = []
    for row in x.itertuples(index=False):
        for side, opp in (("1", "2"), ("2", "1")):
            fighter_name = getattr(row, f"fighter_{side}")
            opponent_name = getattr(row, f"fighter_{opp}")
            fighter_key = normalize_name(fighter_name)
            opponent_key = normalize_name(opponent_name)
            if row.winner_key == fighter_key:
                result = "W"
            elif row.winner_key == opponent_key:
                result = "L"
            else:
                result = "D" if "draw" in str(getattr(row, "method_normalized", "")).lower() else "NC"
            rows.append(
                {
                    "fight_id": str(row.fight_id),
                    "event_date": pd.Timestamp(row.event_date),
                    "event_name": getattr(row, "event_name", None),
                    "organization": str(getattr(row, "organization", "unknown") or "unknown").lower(),
                    "weight_class": getattr(row, "weight_class", None),
                    "is_major_org": bool(getattr(row, "is_major_org", False)),
                    "fighter_name": fighter_name,
                    "opponent_name": opponent_name,
                    "result": result,
                    "method_class": normalize_method(getattr(row, "method_normalized", None)),
                    "round_num": getattr(row, "round_num", np.nan),
                    "time_finish_seconds": getattr(row, "time_finish_seconds", np.nan),
                    "fighter_height_cm": getattr(row, f"f{side}_height_cm", np.nan),
                    "fighter_weight_kg": getattr(row, f"f{side}_weight_kg", np.nan),
                    "opponent_height_cm": getattr(row, f"f{opp}_height_cm", np.nan),
                    "opponent_weight_kg": getattr(row, f"f{opp}_weight_kg", np.nan),
                    "fighter_pre_elo": getattr(row, f"f{side}_pre_elo"),
                    "opponent_pre_elo": getattr(row, f"f{opp}_pre_elo"),
                    "fighter_post_elo": getattr(row, f"f{side}_post_elo"),
                }
            )
    return validate_external_bouts(pd.DataFrame(rows))


def load_mma_global_fighter_bouts(path: str | Path) -> pd.DataFrame:
    """Load canonical cold-start evidence with UFC rows excluded before Elo."""
    wide = load_mma_global_wide(path)
    org = wide["organization"].fillna("").astype(str).str.lower().str.strip()
    non_ufc = wide[~org.map(lambda value: value == "ufc" or "ultimate fighting championship" in value)].copy()
    print(
        f"MMA Global cold-start source: {len(non_ufc):,} non-UFC fights "
        f"({len(wide) - len(non_ufc):,} UFC rows excluded before Elo)"
    )
    return to_fighter_long(add_leakage_safe_elo(non_ufc))
