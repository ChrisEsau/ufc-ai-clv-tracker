from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import _slug
from pipeline.market.providers.draftkings_public import (
    DraftKingsSnapshot,
    flatten_market_diagnostics,
)

RAW_ROOT = Path("data/market/raw/draftkings")
OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/market_comparisons")


def norm(value) -> str:
    text = str(value or "").lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def american_profit_per_1(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def snapshot_meta(path: Path) -> tuple[str, str] | None:
    m = re.search(r"draftkings_(\d{8})_(\d{6})", path.name)
    if not m:
        return None
    stamp = pd.Timestamp(f"{m.group(1)} {m.group(2)}", tz="UTC")
    return f"draftkings_{m.group(1)}_{m.group(2)}", stamp.isoformat()


def load_raw_history(raw_root: Path) -> pd.DataFrame:
    frames = []
    files = sorted(raw_root.rglob("snapshot_*.json"))
    for i, path in enumerate(files, start=1):
        meta = snapshot_meta(path)
        if meta is None:
            continue
        run_id, ts = meta
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            snap = DraftKingsSnapshot(run_id, ts, path)
            frame = flatten_market_diagnostics(payload, snapshot=snap)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"WARN skipped {path}: {exc}")
    if not frames:
        raise RuntimeError(f"No readable DraftKings raw snapshots under {raw_root}")
    out = pd.concat(frames, ignore_index=True)
    out["snapshot_timestamp"] = pd.to_datetime(out["snapshot_timestamp"], utc=True, errors="coerce")
    out["event_start_timestamp"] = pd.to_datetime(out["event_start_timestamp"], utc=True, errors="coerce")
    out["price_american"] = pd.to_numeric(out["price_american"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    return out


def fighter_match_text(row: pd.Series) -> str:
    return norm(" ".join([
        str(row.get("event_name", "")),
        str(row.get("raw_market_name", "")),
        str(row.get("raw_selection_name", "")),
        str(row.get("selection_participant_name", "")),
    ]))


def name_tokens(name: str) -> tuple[str, str]:
    n = norm(name)
    parts = n.split()
    return n, parts[-1] if parts else n


def event_match_score(group: pd.DataFrame, red: str, blue: str) -> int:
    text = " ".join(fighter_match_text(r) for _, r in group.iterrows())
    rfull, rlast = name_tokens(red)
    bfull, blast = name_tokens(blue)
    return (
        4 * int(rfull and rfull in text)
        + 4 * int(bfull and bfull in text)
        + int(rlast and rlast in text)
        + int(blast and blast in text)
    )


def select_prefight_market(raw: pd.DataFrame, fight: pd.Series) -> pd.DataFrame:
    candidates = []
    for event_id, g in raw.groupby("provider_event_id", dropna=True, sort=False):
        score = event_match_score(g, fight["red"], fight["blue"])
        if score >= 8:
            candidates.append((score, str(event_id)))
    if not candidates:
        return pd.DataFrame()
    best_score = max(x[0] for x in candidates)
    event_ids = [x[1] for x in candidates if x[0] == best_score]
    g = raw[raw["provider_event_id"].astype(str).isin(event_ids)].copy()
    start = g["event_start_timestamp"].dropna()
    if not start.empty:
        g = g[g["snapshot_timestamp"] < start.max()]
    if g.empty:
        return g
    latest = g["snapshot_timestamp"].max()
    return g[g["snapshot_timestamp"] == latest].copy()


def selection_side(row: pd.Series, red: str, blue: str) -> str | None:
    text = fighter_match_text(row)
    rfull, rlast = name_tokens(red)
    bfull, blast = name_tokens(blue)
    rs = 2 * int(rfull in text) + int(rlast in text)
    bs = 2 * int(bfull in text) + int(blast in text)
    if rs > bs and rs > 0:
        return "red"
    if bs > rs and bs > 0:
        return "blue"
    role = norm(row.get("selection_participant_venue_role", ""))
    if role in {"home", "red"}:
        return "red"
    if role in {"away", "blue"}:
        return "blue"
    return None


def no_vig_probability(market_rows: pd.DataFrame, selection_id) -> float | None:
    vals = market_rows.dropna(subset=["price_american"]).copy()
    if len(vals) != 2:
        return None
    implied = vals["price_american"].map(
        lambda o: 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)
    )
    total = implied.sum()
    if total <= 0:
        return None
    target = vals[vals["provider_selection_id"] == selection_id]
    if target.empty:
        return None
    idx = target.index[0]
    return float(implied.loc[idx] / total)


def model_probability(row: pd.Series, fight: pd.Series, paths: pd.DataFrame | None) -> tuple[float | None, str | None]:
    fam = str(row.get("supported_market_family") or "")
    label = norm(row.get("raw_selection_name", ""))
    side = selection_side(row, fight["red"], fight["blue"])

    if fam == "moneyline" and side:
        return float(fight[f"p_{side}_win"]), f"{side}_ML"

    if fam in {"ko_tko", "submission", "decision"} and side:
        key = {"ko_tko": "ko_tko", "submission": "sub", "decision": "dec"}[fam]
        return float(fight[f"p_{side}_{key}"]), f"{side}_{fam.upper()}"

    if fam == "goes_distance":
        if any(x in label.split() for x in ["yes", "y"]):
            return float(fight["p_fight_dec"]), "GOES_DISTANCE_YES"
        if any(x in label.split() for x in ["no", "n"]):
            return float(1.0 - fight["p_fight_dec"]), "GOES_DISTANCE_NO"

    if fam == "fighter_sig_strikes_total" and side and paths is not None and pd.notna(row.get("line")):
        line = float(row["line"])
        vals = paths.loc[paths["fight_id"].astype(str) == str(fight["fight_id"]), f"{side}_sig_landed"].astype(float)
        if vals.empty:
            return None, None
        if "over" in label:
            return float((vals > line).mean()), f"{side}_SIG_OVER_{line:g}"
        if "under" in label:
            return float((vals < line).mean()), f"{side}_SIG_UNDER_{line:g}"

    return None, None


def actual_result(key: str, fight: pd.Series, market_row: pd.Series) -> bool | None:
    if key.endswith("_ML"):
        return key.split("_")[0] == str(fight["actual_winner"])
    if key.endswith("_KO_TKO"):
        return key.split("_")[0] == str(fight["actual_winner"]) and str(fight["actual_method"]) == "KO_TKO"
    if key.endswith("_SUBMISSION"):
        return key.split("_")[0] == str(fight["actual_winner"]) and str(fight["actual_method"]) == "SUB"
    if key.endswith("_DECISION"):
        return key.split("_")[0] == str(fight["actual_winner"]) and str(fight["actual_method"]) == "DEC"
    if key == "GOES_DISTANCE_YES":
        return str(fight["actual_method"]) == "DEC"
    if key == "GOES_DISTANCE_NO":
        return str(fight["actual_method"]) != "DEC"
    if "_SIG_OVER_" in key or "_SIG_UNDER_" in key:
        side = key.split("_")[0]
        line = float(market_row["line"])
        actual = float(fight[f"hist_{side}_sig_landed"])
        return actual > line if "_SIG_OVER_" in key else actual < line
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Event Clock MC output CSVs to saved DraftKings pre-fight raw snapshots.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--min-edge-pp", type=float, default=5.0)
    parser.add_argument("--min-ev", type=float, default=0.10)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    summary["fight_id"] = summary["fight_id"].astype(str)
    paths = pd.read_csv(args.paths) if args.paths and args.paths.exists() else None
    if paths is not None:
        paths["fight_id"] = paths["fight_id"].astype(str)

    raw = load_raw_history(args.raw_root)
    print(f"raw snapshot selection rows loaded: {len(raw):,}")
    print(f"snapshot runs: {raw['snapshot_run_id'].nunique():,}")

    records = []
    missing = []
    for _, fight in summary.iterrows():
        market = select_prefight_market(raw, fight)
        if market.empty:
            missing.append(f"{fight['red']} vs {fight['blue']}")
            continue
        market = market[
            ~market["is_parlay"].fillna(False)
            & ~market["is_boost"].fillna(False)
            & ~market["is_promo"].fillna(False)
            & market["price_american"].notna()
        ].copy()
        for _, mr in market.iterrows():
            p_model, key = model_probability(mr, fight, paths)
            if p_model is None or key is None:
                continue
            odds = float(mr["price_american"])
            p_break = 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)
            same_market = market[market["provider_market_id"] == mr["provider_market_id"]]
            p_no_vig = no_vig_probability(same_market, mr["provider_selection_id"])
            edge_raw = p_model - p_break
            edge_novig = p_model - p_no_vig if p_no_vig is not None else np.nan
            profit1 = american_profit_per_1(odds)
            ev = p_model * profit1 - (1.0 - p_model)
            won = actual_result(key, fight, mr)
            pnl100 = np.nan if won is None else (100.0 * profit1 if won else -100.0)
            records.append({
                "fight_id": fight["fight_id"],
                "event_name": fight.get("event_name", ""),
                "event_date": fight.get("event_date", ""),
                "red": fight["red"],
                "blue": fight["blue"],
                "snapshot_timestamp": mr["snapshot_timestamp"],
                "provider_event_id": mr["provider_event_id"],
                "market_family": mr["supported_market_family"],
                "market_name": mr["raw_market_name"],
                "selection": mr["raw_selection_name"],
                "bet_key": key,
                "line": mr["line"],
                "american_odds": odds,
                "model_probability": p_model,
                "raw_implied_probability": p_break,
                "no_vig_probability": p_no_vig,
                "edge_vs_raw": edge_raw,
                "edge_vs_no_vig": edge_novig,
                "expected_roi": ev,
                "positive_ev": ev > 0,
                "qualifies_strict": (edge_raw >= args.min_edge_pp / 100.0) or (ev >= args.min_ev),
                "won": won,
                "flat_100_pnl": pnl100,
            })

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No comparable DraftKings markets found for this Event Clock output.")
    out = out.drop_duplicates(subset=["fight_id", "provider_market_id"] if "provider_market_id" in out.columns else ["fight_id", "bet_key", "american_odds"])

    prefix = args.output_prefix or _slug(str(summary["event_name"].iloc[0]) if "event_name" in summary else args.summary.stem)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{prefix}_market_comparison.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 150)
    print("EVENT CLOCK MC — DRAFTKINGS PRE-FIGHT MARKET COMPARISON")
    print("=" * 150)
    print(f"matched fights: {out['fight_id'].nunique()} / {summary['fight_id'].nunique()}")
    if missing:
        print("missing fights:")
        for name in missing:
            print("  -", name)

    display = out.sort_values(["qualifies_strict", "expected_roi"], ascending=[False, False])
    cols = ["red", "blue", "bet_key", "american_odds", "model_probability", "raw_implied_probability", "no_vig_probability", "edge_vs_raw", "expected_roi", "qualifies_strict", "won", "flat_100_pnl"]
    print("\n", display[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for label, mask in [
        ("ALL POSITIVE EV", out["positive_ev"]),
        ("STRICT QUALIFIERS", out["qualifies_strict"]),
    ]:
        bets = out[mask & out["won"].notna()].copy()
        risk = 100.0 * len(bets)
        pnl = float(bets["flat_100_pnl"].sum()) if len(bets) else 0.0
        print(f"\n{label}")
        print(f"bets: {len(bets)} | wins: {int(bets['won'].sum()) if len(bets) else 0} | losses: {int((~bets['won'].astype(bool)).sum()) if len(bets) else 0}")
        print(f"risked: ${risk:,.2f} | P/L: ${pnl:+,.2f} | ROI: {(pnl/risk if risk else 0):+.2%}")

    print(f"\ncomparison CSV: {out_path}")


if __name__ == "__main__":
    main()
