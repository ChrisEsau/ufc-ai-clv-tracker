from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.market.providers.draftkings_public import (
    DEFAULT_USER_AGENT,
    build_event_subcategory_markets_url,
    fetch_public_json,
)
from pipeline.research import xgboost_method_market_offset as method
from pipeline.research import xgboost_method_hierarchical_v5_oof as hier
from ufc_feature_engineering import add_v5_engineered_features

OUT = Path("data/research/prop_mispricing")
ML_PATH = OUT / "ufc_paris_v5_market_offset_current_20260831.csv"
OUTPUT = OUT / "ufc_paris_hierarchical_v5_methods_20260831.csv"
RAW = OUT / "ufc_paris_draftkings_method_prices_20260831.csv"
LEAGUE_NAV = "https://sportsbook-nash.draftkings.com/sites/US-KS-SB/api/sportscontent/navigation/dkusks/v2/nav/leagues/9034"
METHOD_SUBCATEGORY = "18911"

FIGHTS = [
    ("Dan Hooker", "Salahdine Parnasse"),
    ("Fares Ziam", "Axel Sola"),
    ("Michael Page", "Nursulton Ruziboev"),
    ("Daniil Donchenko", "Punahele Soriano"),
    ("Morgan Charriere", "Felipe Lima"),
    ("Losene Keita", "Muhammad Naimov"),
    ("Mario Pinto", "Ryan Spann"),
    ("Kurtis Campbell", "Trevor Peek"),
    ("Oumar Sy", "Modestas Bukauskas"),
    ("Nathaniel Wood", "Mairon Santos"),
    ("Michael Aljarouj", "Fabia Sintes"),
    ("Nora Cornolle", "Klaudia Sygula"),
    ("Matthieu Letho Duclos", "Luis Felipe Dias"),
    ("Delphine Benouaich", "Sofia Montenegro"),
]


def norm(x: object) -> str:
    s = "".join(ch for ch in unicodedata.normalize("NFKD", str(x)) if not unicodedata.combining(ch))
    return s.replace("’", "'").lower().strip()


def last(x: str) -> str:
    return norm(x).split()[-1]


def american_to_implied(v: object) -> float:
    s = str(v).replace("−", "-").replace("+", "").strip()
    ao = float(s)
    return 100.0 / (ao + 100.0) if ao > 0 else (-ao) / ((-ao) + 100.0)


def fetch_dk_events() -> list[dict]:
    headers = {
        "Accept": "*/*",
        "User-Agent": DEFAULT_USER_AGENT,
        "Origin": "https://sportsbook.draftkings.com",
        "Referer": "https://sportsbook.draftkings.com/",
        "X-Client-Name": "web",
    }
    r = requests.get(LEAGUE_NAV, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("events", [])


def match_event(events: list[dict], a: str, b: str) -> dict:
    la, lb = last(a), last(b)
    hits = []
    for ev in events:
        names = [norm(p.get("name", "")) for p in ev.get("participants", [])]
        event_name = norm(ev.get("name", ""))
        if (any(la in n.split() for n in names) and any(lb in n.split() for n in names)) or (la in event_name and lb in event_name):
            hits.append(ev)
    if len(hits) != 1:
        raise RuntimeError(f"DraftKings event match for {a} vs {b}: {[(x.get('id'), x.get('name')) for x in hits]}")
    return hits[0]


def fetch_method_prices(events: list[dict]) -> tuple[pd.DataFrame, dict[str, dict]]:
    raw_rows = []
    fight_markets: dict[str, dict] = {}
    for i, (a, b) in enumerate(FIGHTS, 1):
        fid = f"paris_v5_20260905_{i:02d}"
        ev = match_event(events, a, b)
        payload = fetch_public_json(build_event_subcategory_markets_url(str(ev["id"]), METHOD_SUBCATEGORY))
        market_name_by_id = {str(m["id"]): str(m.get("name", "")) for m in payload.get("markets", [])}
        six = {}
        six_odds = {}
        for s in payload.get("selections", []):
            mname = market_name_by_id.get(str(s.get("marketId")), "").lower()
            if "ko/tko" in mname:
                meth = "ko"
            elif "submission" in mname:
                meth = "sub"
            elif "decision" in mname:
                meth = "dec"
            else:
                continue
            ot = str(s.get("outcomeType", "")).lower()
            if ot == "home":
                side = "red"
            elif ot == "away":
                side = "blue"
            else:
                continue
            ao = s.get("displayOdds", {}).get("american")
            if ao is None:
                continue
            slug = f"{side}_{meth}"
            imp = american_to_implied(ao)
            six[slug] = imp
            six_odds[slug] = str(ao).replace("−", "-")
            raw_rows.append({
                "fight_id": fid,
                "provider_event_id": str(ev["id"]),
                "event_name": ev.get("name"),
                "red_fighter": a,
                "blue_fighter": b,
                "class_slug": slug,
                "american_odds": six_odds[slug],
                "implied_probability": imp,
            })
        missing = [s for s in method.SLUGS if s not in six]
        if missing:
            raise RuntimeError(f"Incomplete DraftKings method market for {a} vs {b} event={ev.get('id')}: missing={missing}; markets={list(market_name_by_id.values())}")
        total = sum(six.values())
        fair = {f"market_{slug}": six[slug] / total for slug in method.SLUGS}
        fight_markets[fid] = {**fair, **{f"odds_{slug}": six_odds[slug] for slug in method.SLUGS}, "method_overround": total, "provider_event_id": str(ev["id"]), "dk_event_name": ev.get("name")}
    return pd.DataFrame(raw_rows), fight_markets


def build_live_features() -> pd.DataFrame:
    latest = pd.read_parquet("data/features/latest_fighter_state.parquet").copy()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    if "fighter_name" not in latest.columns:
        raise RuntimeError("latest_fighter_state lacks fighter_name")
    latest["_norm_name"] = latest["fighter_name"].map(norm)
    prep_rows, state_rows = [], []
    for i, (a, b) in enumerate(FIGHTS, 1):
        fid = f"paris_v5_20260905_{i:02d}"
        ids = []
        for nm in (a, b):
            hit = latest[latest["_norm_name"].eq(norm(nm))]
            if len(hit):
                rec = hit.iloc[-1].drop(labels=["_norm_name"]).to_dict()
                fighter_id = str(rec["fighter_id"])
            else:
                rec = {c: np.nan for c in latest.columns if c != "_norm_name"}
                fighter_id = f"missing::{norm(nm)}"
                rec["fighter_id"] = fighter_id
                rec["fighter_name"] = nm
            rec["fight_id"] = fid
            state_rows.append(rec)
            ids.append(fighter_id)
        prep_rows.append({
            "fight_id": fid,
            "r_id": ids[0],
            "b_id": ids[1],
            "r_name": a,
            "b_name": b,
            "date": pd.Timestamp("2026-09-05"),
            "title_fight": False,
            "total_rounds": 5 if i == 1 else 3,
        })
    live = build_moneyline_feature_view(
        prepared_fights_df=pd.DataFrame(prep_rows),
        fighter_state_history_df=pd.DataFrame(state_rows),
    )
    return add_v5_engineered_features(live)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not ML_PATH.exists():
        raise RuntimeError(f"Missing frozen Paris V5 moneyline output: {ML_PATH}")

    events = fetch_dk_events()
    raw, markets = fetch_method_prices(events)
    raw.to_csv(RAW, index=False)

    live = build_live_features()
    train, features, _ = method._build_rows(True, True)
    missing_features = [c for c in features if c not in live.columns]
    if missing_features:
        raise RuntimeError(f"Live feature view missing frozen hierarchical method features: {missing_features}")

    score = live[["fight_id"] + features].copy()
    for c in method.MARKET_COLS:
        score[c] = score["fight_id"].map(lambda fid: markets[str(fid)][c])
    for c in ["method_overround", "provider_event_id", "dk_event_name"]:
        score[c] = score["fight_id"].map(lambda fid: markets[str(fid)][c])

    red_cond, _, red_fc = hier._fit_conditional(train, score, features, "red")
    blue_cond, _, blue_fc = hier._fit_conditional(train, score, features, "blue")

    ml = pd.read_csv(ML_PATH)
    ml_red = ml[ml["side"].astype(str).eq("red")].set_index("fight_id")
    p_red = score["fight_id"].map(lambda fid: float(ml_red.loc[str(fid), "v5_model_p"])).to_numpy(float)
    six_p = np.concatenate([p_red[:, None] * red_cond, (1.0 - p_red)[:, None] * blue_cond], axis=1)
    six_p = six_p / six_p.sum(axis=1, keepdims=True)

    rows = []
    for i, ((a, b), (_, sr)) in enumerate(zip(FIGHTS, score.iterrows()), 1):
        fid = str(sr["fight_id"])
        probs = {slug: float(six_p[i - 1, j]) for j, slug in enumerate(method.SLUGS)}
        winner_side = "red" if p_red[i - 1] >= 0.5 else "blue"
        winner = a if winner_side == "red" else b
        loser = b if winner_side == "red" else a
        candidates = [f"{winner_side}_ko", f"{winner_side}_sub", f"{winner_side}_dec"]
        top_slug = max(candidates, key=lambda s: probs[s])
        top_method = top_slug.split("_", 1)[1]
        mkt = markets[fid]
        top_market = float(mkt[f"market_{top_slug}"])
        top_model = probs[top_slug]
        rows.append({
            "fight_id": fid,
            "red_fighter": a,
            "blue_fighter": b,
            "v5_projected_winner": winner,
            "opponent": loser,
            "v5_winner_probability": float(p_red[i - 1] if winner_side == "red" else 1.0 - p_red[i - 1]),
            "winner_side": winner_side,
            "hier_winner_ko": probs[f"{winner_side}_ko"],
            "hier_winner_sub": probs[f"{winner_side}_sub"],
            "hier_winner_dec": probs[f"{winner_side}_dec"],
            "selected_top_method": top_method,
            "selected_method_probability": top_model,
            "dk_selected_method_odds": mkt[f"odds_{top_slug}"],
            "market_sixway_fair_probability": top_market,
            "method_probability_edge": top_model - top_market,
            "method_overround": float(mkt["method_overround"]),
            "provider_event_id": mkt["provider_event_id"],
            "dk_event_name": mkt["dk_event_name"],
            "hier_red_ko": probs["red_ko"],
            "hier_red_sub": probs["red_sub"],
            "hier_red_dec": probs["red_dec"],
            "hier_blue_ko": probs["blue_ko"],
            "hier_blue_sub": probs["blue_sub"],
            "hier_blue_dec": probs["blue_dec"],
            "red_feature_count": red_fc,
            "blue_feature_count": blue_fc,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(out[["red_fighter", "blue_fighter", "v5_projected_winner", "v5_winner_probability", "hier_winner_ko", "hier_winner_sub", "hier_winner_dec", "selected_top_method", "selected_method_probability", "dk_selected_method_odds"]].to_string(index=False))
    print(json.dumps({"rows": len(out), "output": str(OUTPUT), "raw_prices": str(RAW)}, indent=2))


if __name__ == "__main__":
    main()
