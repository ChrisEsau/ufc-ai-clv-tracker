"""Evaluate historical UFC style-cluster matchups.

Research-only runner. Joins style cluster assignments to master fight rows,
optionally attaches historical moneyline odds, and writes matchup win-rate and
ROI reports under data/research/style_matchups/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_ODDS_PATH = "data/market/historical_moneyline_odds.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UFC style matchup history.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-fights", type=int, default=50)
    parser.add_argument("--odds-path", default=DEFAULT_ODDS_PATH)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style research config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def american_return(odds: float, stake: float = 100.0) -> float:
    if pd.isna(odds) or odds == 0:
        return float("nan")
    return stake * odds / 100.0 if odds > 0 else stake * 100.0 / abs(odds)


def style_side(assignments: pd.DataFrame, side: str) -> pd.DataFrame:
    out = assignments[["fight_id", "fighter_id", "style_cluster", "style_cluster_label"]].copy()
    return out.rename(columns={
        "fighter_id": f"{side}_id",
        "style_cluster": f"{side}_style_cluster",
        "style_cluster_label": f"{side}_style_label",
    })


def derive_red_win(df: pd.DataFrame) -> pd.Series:
    target = first_col(df, ["target", "red_win", "r_win"])
    if target:
        s = pd.to_numeric(df[target], errors="coerce")
        return s.where(s.isin([0, 1]))
    if "winner_id" in df.columns:
        winner_id = df["winner_id"].fillna("").astype(str).str.strip()
        r_id = df["r_id"].fillna("").astype(str).str.strip()
        b_id = df["b_id"].fillna("").astype(str).str.strip()
        valid = winner_id.ne("") & (winner_id.eq(r_id) | winner_id.eq(b_id))
        return winner_id.eq(r_id).astype(float).where(valid)
    if "winner" in df.columns:
        winner = df["winner"].fillna("").astype(str).str.strip().str.casefold()
        r_name = df["r_name"].fillna("").astype(str).str.strip().str.casefold()
        b_name = df["b_name"].fillna("").astype(str).str.strip().str.casefold()
        valid = winner.ne("") & (winner.eq(r_name) | winner.eq(b_name))
        return winner.eq(r_name).astype(float).where(valid)
    raise ValueError("No usable result columns found.")


def build_fights(master: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in [
        "event_id", "event_name", "fight_id", "date", "r_id", "b_id", "r_name", "b_name",
        "division", "title_fight", "method", "winner", "winner_id", "target", "red_win", "r_win",
    ] if c in master.columns]
    df = master[keep].copy()
    df = df.merge(style_side(assignments, "r"), on=["fight_id", "r_id"], how="left")
    df = df.merge(style_side(assignments, "b"), on=["fight_id", "b_id"], how="left")
    df = df.dropna(subset=["r_style_label", "b_style_label"]).copy()
    df["red_win"] = derive_red_win(df)
    df = df.dropna(subset=["red_win"]).copy()
    df["red_win"] = df["red_win"].astype(int)
    df["blue_win"] = 1 - df["red_win"]
    df["style_matchup_key"] = df["r_style_label"] + "_vs_" + df["b_style_label"]
    df["same_style_matchup"] = df["r_style_label"] == df["b_style_label"]
    return df


def normalize_odds(odds: pd.DataFrame) -> pd.DataFrame:
    if "fight_id" not in odds.columns:
        raise ValueError("Odds file must include fight_id.")
    r_col = first_col(odds, ["r_odds", "red_odds", "r_moneyline", "red_moneyline", "r_american_odds"])
    b_col = first_col(odds, ["b_odds", "blue_odds", "b_moneyline", "blue_moneyline", "b_american_odds"])
    if r_col and b_col:
        out = odds[["fight_id", r_col, b_col]].copy()
        out["r_odds"] = pd.to_numeric(out[r_col], errors="coerce")
        out["b_odds"] = pd.to_numeric(out[b_col], errors="coerce")
        return out.dropna(subset=["r_odds", "b_odds"]).drop_duplicates("fight_id", keep="last")[["fight_id", "r_odds", "b_odds"]]

    price_col = first_col(odds, ["odds", "moneyline", "american_odds", "price", "american_price"])
    side_col = first_col(odds, ["side", "corner", "fighter_side"])
    fighter_col = first_col(odds, ["fighter_id", "selection_id", "participant_id"])
    if price_col is None:
        raise ValueError(f"Could not find odds/price column. Columns: {list(odds.columns)}")
    long = odds.copy()
    long["odds_numeric"] = pd.to_numeric(long[price_col], errors="coerce")
    long = long.dropna(subset=["fight_id", "odds_numeric"])
    if side_col:
        side = long[side_col].fillna("").astype(str).str.strip().str.casefold().replace({"red": "r", "blue": "b"})
        long["side_norm"] = side
        wide = long[long["side_norm"].isin(["r", "b"])].pivot_table(index="fight_id", columns="side_norm", values="odds_numeric", aggfunc="last").reset_index()
        if "r" in wide.columns and "b" in wide.columns:
            return wide.rename(columns={"r": "r_odds", "b": "b_odds"})[["fight_id", "r_odds", "b_odds"]]
    if fighter_col:
        return long[["fight_id", fighter_col, "odds_numeric"]].rename(columns={fighter_col: "odds_fighter_id"})
    raise ValueError("Could not normalize odds schema.")


def attach_odds(fights: pd.DataFrame, odds_path: Path) -> tuple[pd.DataFrame, str]:
    if not odds_path.exists():
        return fights, "odds_file_missing"
    odds = normalize_odds(pd.read_parquet(odds_path))
    if {"r_odds", "b_odds"}.issubset(odds.columns):
        return fights.merge(odds, on="fight_id", how="left"), "fight_id_wide_join"
    odds["odds_fighter_id"] = odds["odds_fighter_id"].fillna("").astype(str).str.strip()
    red = odds.rename(columns={"odds_fighter_id": "r_id", "odds_numeric": "r_odds"})
    blue = odds.rename(columns={"odds_fighter_id": "b_id", "odds_numeric": "b_odds"})
    out = fights.merge(red[["fight_id", "r_id", "r_odds"]], on=["fight_id", "r_id"], how="left")
    out = out.merge(blue[["fight_id", "b_id", "b_odds"]], on=["fight_id", "b_id"], how="left")
    return out, "fighter_id_long_join"


def matchup_matrix(fights: pd.DataFrame) -> pd.DataFrame:
    g = fights.groupby(["r_style_label", "b_style_label", "style_matchup_key"], dropna=False)
    out = g.agg(fights=("fight_id", "count"), red_wins=("red_win", "sum"), blue_wins=("blue_win", "sum"), red_win_rate=("red_win", "mean"), same_style_matchup=("same_style_matchup", "max")).reset_index()
    out["blue_win_rate"] = 1 - out["red_win_rate"]
    return out.sort_values(["fights", "red_win_rate"], ascending=[False, False])


def roi_report(fights: pd.DataFrame, min_fights: int) -> pd.DataFrame:
    if "r_odds" not in fights.columns or "b_odds" not in fights.columns:
        return pd.DataFrame([{"status": "not_available", "reason": "No r_odds/b_odds after odds join."}])
    df = fights.copy()
    df["r_odds"] = pd.to_numeric(df["r_odds"], errors="coerce")
    df["b_odds"] = pd.to_numeric(df["b_odds"], errors="coerce")
    df = df.dropna(subset=["r_odds", "b_odds"])
    if df.empty:
        return pd.DataFrame([{"status": "not_available", "reason": "No matched fights with numeric odds."}])
    stake = 100.0
    df["red_profit"] = [american_return(o, stake) if w == 1 else -stake for o, w in zip(df["r_odds"], df["red_win"])]
    df["blue_profit"] = [american_return(o, stake) if w == 1 else -stake for o, w in zip(df["b_odds"], df["blue_win"])]
    df["red_is_underdog"] = df["r_odds"] > df["b_odds"]
    df["underdog_won"] = df["red_win"].where(df["red_is_underdog"], df["blue_win"])
    dog_odds = df["r_odds"].where(df["red_is_underdog"], df["b_odds"])
    df["underdog_profit"] = [american_return(o, stake) if w == 1 else -stake for o, w in zip(dog_odds, df["underdog_won"])]
    out = df.groupby("style_matchup_key").agg(fights=("fight_id", "count"), red_win_rate=("red_win", "mean"), avg_r_odds=("r_odds", "mean"), avg_b_odds=("b_odds", "mean"), flat_bet_red_profit=("red_profit", "sum"), flat_bet_blue_profit=("blue_profit", "sum"), underdog_win_rate=("underdog_won", "mean"), underdog_profit=("underdog_profit", "sum"), red_underdog_rate=("red_is_underdog", "mean")).reset_index()
    out["flat_bet_red_roi"] = out["flat_bet_red_profit"] / (out["fights"] * stake)
    out["flat_bet_blue_roi"] = out["flat_bet_blue_profit"] / (out["fights"] * stake)
    out["underdog_roi"] = out["underdog_profit"] / (out["fights"] * stake)
    return out[out["fights"] >= min_fights].sort_values(["flat_bet_red_roi", "fights"], ascending=[False, False])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    research_dir = Path(str((config.get("outputs") or {}).get("research_dir", "data/research/style_matchups")))
    master_path = Path(str((config.get("inputs") or {}).get("master_path", "data/master/ufc_master.parquet")))
    assignments_path = research_dir / "style_cluster_assignments.parquet"
    odds_path = Path(args.odds_path)

    master = pd.read_parquet(master_path)
    assignments = pd.read_parquet(assignments_path)
    fights = build_fights(master, assignments)
    fights, odds_join_status = attach_odds(fights, odds_path)
    odds_matched = int(fights[["r_odds", "b_odds"]].dropna().shape[0]) if {"r_odds", "b_odds"}.issubset(fights.columns) else 0

    matrix = matchup_matrix(fights)
    baseline = float(fights["red_win"].mean()) if not fights.empty else 0.0
    edge = matrix[matrix["fights"] >= args.min_fights].copy()
    edge["baseline_red_win_rate"] = baseline
    edge["red_win_rate_edge"] = edge["red_win_rate"] - baseline
    edge["abs_red_win_rate_edge"] = edge["red_win_rate_edge"].abs()
    edge = edge.sort_values(["abs_red_win_rate_edge", "fights"], ascending=[False, False])
    roi = roi_report(fights, args.min_fights)
    underdogs = roi[[c for c in ["style_matchup_key", "fights", "underdog_win_rate", "underdog_profit", "underdog_roi", "red_underdog_rate"] if c in roi.columns]].copy()

    research_dir.mkdir(parents=True, exist_ok=True)
    fights.to_parquet(research_dir / "style_matchup_fights.parquet", index=False)
    matrix.to_csv(research_dir / "style_matchup_matrix.csv", index=False)
    edge.to_csv(research_dir / "style_matchup_edge_report.csv", index=False)
    roi.to_csv(research_dir / "style_matchup_roi_report.csv", index=False)
    underdogs.to_csv(research_dir / "style_matchup_underdogs.csv", index=False)
    (research_dir / "style_matchup_summary.json").write_text(json.dumps({"fight_count": int(len(fights)), "baseline_red_win_rate": baseline, "matchup_count": int(len(matrix)), "edge_report_rows": int(len(edge)), "minimum_fights": int(args.min_fights), "odds_path": str(odds_path), "odds_join_status": odds_join_status, "odds_matched_fights": odds_matched, "roi_report_rows": int(len(roi))}, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
