from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.simulation.event_clock_mc_v1.compare_event_clock_to_market import (
    GIT_MARKET_PATH,
    HISTORY_PATH,
    _match_git_fight_rows,
    _read_git_market_outcomes,
    dedupe_snapshot,
    fighter_side,
    implied_probability,
    load_market_history,
    no_vig_probability,
    norm,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/market_comparisons")
EPS = 1e-9


def _edge_band(edge: float) -> str:
    if not np.isfinite(edge):
        return "unknown"
    if edge < 0.05:
        return "<5pp"
    if edge < 0.10:
        return "5-10pp"
    if edge < 0.15:
        return "10-15pp"
    return ">=15pp"


def _price_class(odds: float) -> str:
    return "UNDERDOG" if odds > 0 else "FAVORITE"


def _summarize(label: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"{label:28s} bets=  0")
        return
    clv = pd.to_numeric(frame["clv_probability_points"], errors="coerce")
    beat = clv > EPS
    print(
        f"{label:28s} bets={len(frame):3d}  "
        f"beat-close={beat.mean():6.1%}  "
        f"mean CLV={clv.mean()*100:+6.2f}pp  "
        f"median={clv.median()*100:+6.2f}pp  "
        f"mean model edge={frame['model_edge_vs_entry_novig'].mean()*100:+6.2f}pp"
    )


def _prepare_master() -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="coerce")
    return master


def _git_commits_for_event_day(event_date: pd.Timestamp, max_commits: int = 160) -> list[str]:
    """Return market-outcomes commits through the end of the target event day.

    We intentionally inspect the whole available pre-fight sequence rather than
    reusing compare_event_clock_to_market's single latest snapshot. The market
    rows themselves are later filtered by their embedded commence_time.
    """
    cutoff = pd.Timestamp(event_date)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    cutoff = cutoff.normalize() + pd.Timedelta(days=1)
    cutoff_text = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    proc = subprocess.run(
        [
            "git", "log", "--all", f"--before={cutoff_text}",
            "--format=%H", "--", GIT_MARKET_PATH,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()][:max_commits]


def _history_moneyline_snapshots(
    fight: pd.Series,
    bookmaker: str,
    max_commits: int = 160,
) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    """Collect every distinct valid pre-fight moneyline snapshot.

    Primary source is the append-only market intelligence history. Git market
    snapshots are used only when the history parquet has fewer than two valid
    observations for the fight.
    """
    history = load_market_history(HISTORY_PATH, bookmaker)

    if not history.empty:
        rows = history[
            history["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])
        ].copy()

        # Prefer stable fight_id, but historical market ids can differ from the
        # UFC master id. Fall back to normalized displayed matchup names.
        id_match = rows["fight_id"].astype(str) == str(fight["fight_id"])

        display = rows.get("fight_display", pd.Series("", index=rows.index)).map(norm)
        red = norm(fight["red"])
        blue = norm(fight["blue"])
        name_match = display.isin({
            norm(f"{fight['red']} vs {fight['blue']}"),
            norm(f"{fight['blue']} vs {fight['red']}"),
        })

        rows = rows[id_match | name_match].copy()

        if not rows.empty:
            # Keep only observations before the actual scheduled commence time
            # when that timestamp is available in the history.
            if "commence_time" in rows.columns:
                commence = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
                valid = commence.isna() | (rows["refresh_timestamp"] < commence)
                rows = rows[valid].copy()

            snapshots = []

            for ts, group in rows.groupby("refresh_timestamp", sort=True):
                group = dedupe_snapshot(group.copy())

                # fighter_side()/norm() cannot evaluate pandas.NA as boolean.
                # Historical market rows legitimately contain missing optional
                # identity fields, so normalize those to empty strings here.
                for col in (
                    "fighter_name",
                    "side",
                    "outcome_display",
                    "comparison_key",
                ):
                    if col in group.columns:
                        group[col] = group[col].fillna("")

                # There can occasionally be more than one provider moneyline
                # represented at the same refresh. Select a complete two-sided
                # market rather than mixing prices across market ids.
                candidates = []

                if "provider_market_id" in group.columns:
                    for _, market_group in group.groupby(
                        group["provider_market_id"].astype(str),
                        dropna=False,
                    ):
                        sides = {
                            fighter_side(row, fight)
                            for _, row in market_group.iterrows()
                        }
                        if {"red", "blue"}.issubset(sides):
                            candidates.append(market_group.copy())
                else:
                    sides = {
                        fighter_side(row, fight)
                        for _, row in group.iterrows()
                    }
                    if {"red", "blue"}.issubset(sides):
                        candidates.append(group.copy())

                if not candidates:
                    continue

                # Deterministic selection. DraftKings normally contributes one
                # complete moneyline pair per refresh.
                chosen = candidates[0].reset_index(drop=True)
                snapshots.append(
                    (
                        pd.Timestamp(ts),
                        str(chosen["refresh_id"].iloc[0]),
                        chosen,
                    )
                )

            if len(snapshots) >= 2:
                return snapshots

    # Historical parquet unavailable/incomplete: retain the original Git
    # reconstruction as a fallback.
    snapshots: dict[pd.Timestamp, tuple[str, pd.DataFrame]] = {}

    for commit in _git_commits_for_event_day(
        pd.Timestamp(fight["event_date"]),
        max_commits=max_commits,
    ):
        frame = _read_git_market_outcomes(commit)
        rows = _match_git_fight_rows(frame, fight, bookmaker)

        if rows.empty:
            continue

        ml = rows[
            rows["market_key"].astype(str).str.lower().isin(["moneyline", "h2h"])
        ].copy()

        if ml.empty:
            continue

        ml["snapshot_timestamp"] = pd.to_datetime(
            ml["snapshot_timestamp"],
            utc=True,
            errors="coerce",
        )

        for ts, group in ml.dropna(subset=["snapshot_timestamp"]).groupby(
            "snapshot_timestamp",
            sort=True,
        ):
            group = dedupe_snapshot(group.copy())
            sides = {
                fighter_side(row, fight)
                for _, row in group.iterrows()
            }

            if not {"red", "blue"}.issubset(sides):
                continue

            snapshots[pd.Timestamp(ts)] = (
                f"git_{commit[:12]}",
                group.reset_index(drop=True),
            )

    return [
        (ts, snapshots[ts][0], snapshots[ts][1])
        for ts in sorted(snapshots)
    ]

def _find_side_row(snapshot: pd.DataFrame, fight: pd.Series, side: str) -> pd.Series | None:
    if snapshot.empty:
        return None
    matches = [idx for idx, row in snapshot.iterrows() if fighter_side(row, fight) == side]
    if not matches:
        return None
    return snapshot.loc[matches[-1]]


def _snapshot_side_values(
    snapshot: pd.DataFrame,
    fight: pd.Series,
    side: str,
) -> tuple[float, float, float] | None:
    row = _find_side_row(snapshot, fight, side)
    if row is None:
        return None
    odds = pd.to_numeric(pd.Series([row.get("american_odds")]), errors="coerce").iloc[0]
    if pd.isna(odds):
        return None
    raw = implied_probability(float(odds))
    novig = no_vig_probability(snapshot, row)
    if novig is None or not np.isfinite(novig):
        return None
    return float(odds), float(raw), float(novig)


def audit_file(path: Path, master: pd.DataFrame, bookmaker: str) -> pd.DataFrame:
    comp = pd.read_csv(path).copy()
    required = {
        "fight_id", "red", "blue", "bet_key", "american_odds",
        "model_probability", "raw_implied_probability", "no_vig_probability",
    }
    missing = sorted(required - set(comp.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {', '.join(missing)}")

    ml = comp[comp["bet_key"].astype(str).str.endswith("_ML")].copy()
    rows = []
    master_lookup = master.set_index("fight_id", drop=False)

    # The comparison CSV was built from the latest pre-fight snapshot, so it is
    # a closing-line comparison, not a true entry snapshot. For CLV we rebuild
    # the observed market sequence and define entry=first observed, close=last
    # observed. The Event Clock model probability stays fixed throughout.
    for fight_id, group in ml.groupby(ml["fight_id"].astype(str), sort=False):
        if fight_id not in master_lookup.index:
            continue
        m = master_lookup.loc[fight_id]
        first = group.iloc[0]
        fight = pd.Series({
            "fight_id": fight_id,
            "red": str(first["red"]),
            "blue": str(first["blue"]),
            "event_date": m["event_date"],
        })
        sequence = _history_moneyline_snapshots(fight, bookmaker)
        if len(sequence) < 2:
            continue
        entry_ts, entry_commit, entry_snapshot = sequence[0]
        close_ts, close_commit, close_snapshot = sequence[-1]
        if close_ts <= entry_ts:
            continue

        for _, model_row in group.iterrows():
            side = str(model_row["bet_key"]).split("_")[0].lower()
            if side not in {"red", "blue"}:
                continue
            entry_vals = _snapshot_side_values(entry_snapshot, fight, side)
            close_vals = _snapshot_side_values(close_snapshot, fight, side)
            if entry_vals is None or close_vals is None:
                continue
            entry_odds, entry_raw, entry_novig = entry_vals
            close_odds, close_raw, close_novig = close_vals
            model_p = float(model_row["model_probability"])
            model_edge_raw = model_p - entry_raw
            model_edge_novig = model_p - entry_novig
            expected_roi = model_p * (entry_odds / 100.0 if entry_odds > 0 else 100.0 / abs(entry_odds)) - (1.0 - model_p)
            clv = close_novig - entry_novig
            qualifies_strict = bool(model_edge_raw >= 0.05 - EPS or expected_roi >= 0.10 - EPS)

            rows.append({
                "source_file": str(path),
                "fight_id": fight_id,
                "red": model_row["red"],
                "blue": model_row["blue"],
                "bet_key": model_row["bet_key"],
                "side": side,
                "american_odds_entry": entry_odds,
                "american_odds_close": close_odds,
                "model_probability": model_p,
                "entry_raw_implied_probability": entry_raw,
                "entry_no_vig_probability": entry_novig,
                "closing_raw_implied_probability": close_raw,
                "closing_no_vig_probability": close_novig,
                "model_edge_vs_entry_raw": model_edge_raw,
                "model_edge_vs_entry_novig": model_edge_novig,
                "clv_probability_points": clv,
                "residual_model_edge_at_close": model_p - close_novig,
                "market_closed_toward_model": bool(clv > EPS) if model_edge_novig > EPS else bool(clv < -EPS),
                "price_class": _price_class(entry_odds),
                "edge_band": _edge_band(model_edge_raw),
                "expected_roi_entry": expected_roi,
                "positive_ev": bool(expected_roi > EPS),
                "qualifies_strict": qualifies_strict,
                "won": bool(model_row.get("won", False)),
                "entry_refresh_timestamp": entry_ts,
                "entry_refresh_id": entry_commit,
                "closing_refresh_timestamp": close_ts,
                "closing_refresh_id": close_commit,
                "observed_snapshot_count": len(sequence),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Event Clock model edge at first observed DraftKings moneyline with movement to latest valid pre-fight line."
    )
    parser.add_argument("--comparison", nargs="+", required=True, type=Path)
    parser.add_argument("--bookmaker", default="DraftKings")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "event_clock_moneyline_edge_vs_clv.csv")
    args = parser.parse_args()

    master = _prepare_master()
    frames = [audit_file(path, master, args.bookmaker) for path in args.comparison]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if out.empty:
        raise RuntimeError("No moneyline rows had at least two valid historical pre-fight DraftKings snapshots.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("=" * 130)
    print("EVENT CLOCK MC — MODEL EDGE VS CLOSING-LINE VALUE")
    print("=" * 130)
    print(f"input cards: {len(args.comparison)}")
    print(f"matched moneyline sides: {len(out)}")
    print("ENTRY: first observed valid pre-fight DraftKings moneyline snapshot")
    print("CLOSE: latest observed valid pre-fight DraftKings moneyline snapshot")
    print("CLV definition: closing no-vig probability - entry no-vig probability for the same fighter")
    print("positive CLV = market moved toward that fighter")
    print("prediction probabilities changed: NO")
    print()

    positive = out[out["positive_ev"]].copy()
    strict = out[out["qualifies_strict"]].copy()

    print("POSITIVE-EV MONEYLINES")
    _summarize("ALL", positive)
    _summarize("UNDERDOG", positive[positive["price_class"] == "UNDERDOG"])
    _summarize("FAVORITE", positive[positive["price_class"] == "FAVORITE"])
    print()

    print("STRICT MONEYLINES")
    _summarize("ALL", strict)
    _summarize("UNDERDOG", strict[strict["price_class"] == "UNDERDOG"])
    _summarize("FAVORITE", strict[strict["price_class"] == "FAVORITE"])
    print()

    print("STRICT BY RAW EDGE BAND")
    for band in ["5-10pp", "10-15pp", ">=15pp"]:
        _summarize(band, strict[strict["edge_band"] == band])
    print()

    print("STRICT: WINNERS VS LOSERS")
    _summarize("WON", strict[strict["won"]])
    _summarize("LOST", strict[~strict["won"]])
    print()

    show_cols = [
        "red", "blue", "bet_key", "american_odds_entry", "american_odds_close",
        "model_probability", "entry_no_vig_probability", "closing_no_vig_probability",
        "model_edge_vs_entry_novig", "clv_probability_points", "residual_model_edge_at_close",
        "price_class", "edge_band", "won", "observed_snapshot_count",
    ]
    print("STRICT BETS — EDGE VS CLV")
    if strict.empty:
        print("none")
    else:
        print(strict.sort_values("model_edge_vs_entry_novig", ascending=False)[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"edge-vs-CLV CSV: {args.output}")


if __name__ == "__main__":
    main()
