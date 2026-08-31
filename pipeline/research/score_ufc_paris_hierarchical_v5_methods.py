from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.market.normalizers.draftkings import normalize_draftkings_diagnostic_rows
from pipeline.market.providers.draftkings_public import (
    build_event_subcategory_markets_url,
    fetch_public_json,
    flatten_market_diagnostics,
    save_raw_snapshot,
)
from pipeline.market.run_draftkings_event_index import (
    DEFAULT_DRAFTKINGS_UFC_LEAGUE_NAV_URL,
    build_event_index,
)
from pipeline.research import xgboost_method_market_offset as method
from pipeline.research.xgboost_method_hierarchical_v5_oof import _fit_conditional

OUT = Path("data/research/prop_mispricing")
ML_PATH = OUT / "ufc_paris_v5_market_offset_current_20260831.csv"
OUTPUT = OUT / "ufc_paris_hierarchical_v5_methods_current_20260831.csv"
DK_RAW = Path("/tmp/dk_paris_methods")
SUBCATEGORY_ID = "18911"
REGISTRY_ENTRY = {
    "subcategory_id": SUBCATEGORY_ID,
    "name": "Fighter Method Props",
    "family": "fighter_method_props",
    "outcome_type": "fighter",
}
METHOD_KEYS = {
    "win_by_ko_tko_dq": "ko",
    "win_by_submission": "sub",
    "win_by_decision": "dec",
}


def norm(value: object) -> str:
    text = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", str(value))
        if not unicodedata.combining(ch)
    )
    text = text.replace("’", "'").lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce")


def _rate(df: pd.DataFrame, c: str) -> pd.Series:
    s = _num(df, c)
    return pd.Series(np.where(s > 1, s / 100.0, s), index=df.index)


def add_v5_engineered(livefv: pd.DataFrame) -> pd.DataFrame:
    livefv = livefv.copy()
    livefv["chin_risk_diff"] = _num(livefv, "r_pre_sapm") * (1 - _rate(livefv, "r_pre_str_def")) - _num(livefv, "b_pre_sapm") * (1 - _rate(livefv, "b_pre_str_def"))
    livefv["aggression_index_diff"] = (_num(livefv, "r_pre_splm") + _num(livefv, "r_pre_td_avg")) - (_num(livefv, "b_pre_splm") + _num(livefv, "b_pre_td_avg"))
    livefv["age_squared_diff"] = _num(livefv, "r_pre_age") ** 2 - _num(livefv, "b_pre_age") ** 2
    livefv["wrestling_mismatch_diff"] = _num(livefv, "r_pre_td_avg") * (1 - _rate(livefv, "b_pre_td_def")) - _num(livefv, "b_pre_td_avg") * (1 - _rate(livefv, "r_pre_td_def"))
    livefv["pressure_striking_adv_diff"] = _num(livefv, "r_pre_splm") * (1 - _rate(livefv, "b_pre_str_def")) - _num(livefv, "b_pre_splm") * (1 - _rate(livefv, "r_pre_str_def"))
    livefv["age_x_career_ko_losses_diff"] = _num(livefv, "r_pre_age") * _num(livefv, "r_pre_career_ko_losses") - _num(livefv, "b_pre_age") * _num(livefv, "b_pre_career_ko_losses")
    return livefv


def load_paris_card() -> pd.DataFrame:
    ml = pd.read_csv(ML_PATH)
    red = ml[ml["side"].astype(str).eq("red")][["fight_id", "fighter", "opponent", "v5_model_p", "fight_cold_start"]].copy()
    red = red.rename(columns={"fighter": "red_fighter", "opponent": "blue_fighter", "v5_model_p": "v5_p_red"})
    return red.sort_values("fight_id").reset_index(drop=True)


def discover_dk_method_markets(card: pd.DataFrame) -> pd.DataFrame:
    nav_url = DEFAULT_DRAFTKINGS_UFC_LEAGUE_NAV_URL
    nav_payload = fetch_public_json(nav_url)
    event_index = build_event_index(nav_payload, request_url=nav_url)
    if event_index.empty:
        raise RuntimeError("DraftKings UFC event index is empty")

    event_index = event_index.copy()
    event_index["pair_key"] = event_index.apply(
        lambda r: "|".join(sorted([norm(r.get("participant_home")), norm(r.get("participant_away"))])), axis=1
    )
    wanted = card.copy()
    wanted["pair_key"] = wanted.apply(
        lambda r: "|".join(sorted([norm(r["red_fighter"]), norm(r["blue_fighter"])])), axis=1
    )

    all_rows = []
    for _, fight in wanted.iterrows():
        hit = event_index[event_index["pair_key"].eq(fight["pair_key"])]
        if len(hit) != 1:
            print(f"DK_EVENT_MATCH fight_id={fight['fight_id']} count={len(hit)} pair={fight['pair_key']}")
            continue
        event_id = str(hit.iloc[0]["provider_event_id"])
        url = build_event_subcategory_markets_url(event_id, SUBCATEGORY_ID)
        payload = fetch_public_json(url)
        snap = save_raw_snapshot(
            payload,
            raw_root=DK_RAW,
            event_id=event_id,
            subcategory_id=SUBCATEGORY_ID,
        )
        diag = flatten_market_diagnostics(
            payload,
            snapshot=snap,
            request_url=url,
            registry_entry=REGISTRY_ENTRY,
        )
        canonical = normalize_draftkings_diagnostic_rows(diag)
        canonical = canonical[canonical["market_key"].isin(METHOD_KEYS)].copy()
        canonical["fight_id"] = fight["fight_id"]
        canonical["dk_event_id"] = event_id
        all_rows.append(canonical)

    if not all_rows:
        raise RuntimeError("No Paris DraftKings fighter-method markets were discovered")
    return pd.concat(all_rows, ignore_index=True)


def build_live_features(card: pd.DataFrame) -> pd.DataFrame:
    latest = pd.read_parquet("data/features/latest_fighter_state.parquet").copy()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    if "fighter_name" not in latest.columns:
        raise RuntimeError("latest_fighter_state lacks fighter_name")
    latest["_norm_name"] = latest["fighter_name"].map(norm)

    prep_rows = []
    state_rows = []
    for i, fight in card.reset_index(drop=True).iterrows():
        ids = []
        for nm in [fight["red_fighter"], fight["blue_fighter"]]:
            hit = latest[latest["_norm_name"].eq(norm(nm))]
            if len(hit):
                rec = hit.iloc[-1].drop(labels=["_norm_name"]).to_dict()
                fid = str(rec["fighter_id"])
            else:
                rec = {c: np.nan for c in latest.columns if c != "_norm_name"}
                fid = f"missing::{norm(nm)}"
                rec["fighter_id"] = fid
                rec["fighter_name"] = nm
            rec["fight_id"] = fight["fight_id"]
            state_rows.append(rec)
            ids.append(fid)
        prep_rows.append(
            {
                "fight_id": fight["fight_id"],
                "r_id": ids[0],
                "b_id": ids[1],
                "r_name": fight["red_fighter"],
                "b_name": fight["blue_fighter"],
                "date": pd.Timestamp("2026-09-05"),
                "title_fight": False,
                "total_rounds": 5 if i == 0 else 3,
            }
        )
    livefv = build_moneyline_feature_view(
        prepared_fights_df=pd.DataFrame(prep_rows),
        fighter_state_history_df=pd.DataFrame(state_rows),
    )
    return add_v5_engineered(livefv)


def build_score_frame(card: pd.DataFrame, dk: pd.DataFrame, livefv: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    priced = dk.copy()
    priced["fighter_norm"] = priced["fighter_name"].map(norm)
    priced["method_slug"] = priced["market_key"].map(METHOD_KEYS)
    priced["implied_probability"] = pd.to_numeric(priced["implied_probability"], errors="coerce")
    priced["american_odds"] = pd.to_numeric(priced["american_odds"], errors="coerce")
    priced = priced.dropna(subset=["fighter_norm", "method_slug", "implied_probability"])

    rows = []
    diagnostics = []
    live_by_id = livefv.set_index("fight_id", drop=False)
    for _, fight in card.iterrows():
        fid = fight["fight_id"]
        if fid not in live_by_id.index:
            diagnostics.append({"fight_id": fid, "status": "missing_live_features"})
            continue
        r = live_by_id.loc[fid]
        vals = {}
        odds = {}
        complete = True
        for side, fighter in [("red", fight["red_fighter"]), ("blue", fight["blue_fighter"])]:
            for meth in ["ko", "sub", "dec"]:
                hit = priced[
                    priced["fight_id"].astype(str).eq(str(fid))
                    & priced["fighter_norm"].eq(norm(fighter))
                    & priced["method_slug"].eq(meth)
                ]
                if len(hit) == 0:
                    complete = False
                    diagnostics.append({"fight_id": fid, "status": f"missing_{side}_{meth}", "fighter": fighter})
                    continue
                # Prefer the first standard non-promo row if duplicates are returned.
                hit = hit.sort_values(["is_promo", "is_boost", "provider_market_id"], na_position="last")
                rec = hit.iloc[0]
                vals[f"market_{side}_{meth}"] = float(rec["implied_probability"])
                odds[f"dk_{side}_{meth}_odds"] = float(rec["american_odds"]) if pd.notna(rec["american_odds"]) else np.nan
        if not complete:
            continue
        row = {
            "fight_id": fid,
            "date": pd.Timestamp("2026-09-05"),
            "red_fighter": fight["red_fighter"],
            "blue_fighter": fight["blue_fighter"],
            "v5_p_red": float(fight["v5_p_red"]),
            "fight_cold_start": bool(fight["fight_cold_start"]),
            **vals,
            **odds,
        }
        missing_features = [c for c in features if c not in livefv.columns]
        if missing_features:
            raise RuntimeError(f"live feature builder missing hierarchical method features: {missing_features}")
        for c in features:
            row[c] = r[c]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def american_to_decimal(odds: float) -> float:
    return 1.0 + (100.0 / -odds if odds < 0 else odds / 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    card = load_paris_card()
    print("Paris fights:", len(card))

    dk = discover_dk_method_markets(card)
    print("DraftKings fighter-method rows:", len(dk))

    train, features, _ = method._build_rows(True, True)
    livefv = build_live_features(card)
    score, diagnostic = build_score_frame(card, dk, livefv, features)
    if score.empty:
        print(diagnostic.to_string(index=False))
        raise RuntimeError("No Paris fights had a complete six-price DK fighter-method market")

    red_cond, red_train_n, red_fc = _fit_conditional(train, score, features, "red")
    blue_cond, blue_train_n, blue_fc = _fit_conditional(train, score, features, "blue")
    p_red = np.clip(score["v5_p_red"].to_numpy(float), 1e-12, 1 - 1e-12)
    p = np.concatenate([p_red[:, None] * red_cond, (1 - p_red)[:, None] * blue_cond], axis=1)
    p = p / p.sum(axis=1, keepdims=True)

    slugs = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]
    for j, slug in enumerate(slugs):
        score[f"hier_{slug}"] = p[:, j]

    result_rows = []
    for _, r in score.iterrows():
        winner_side = "red" if r["v5_p_red"] >= 0.5 else "blue"
        winner = r[f"{winner_side}_fighter"]
        candidate = {
            meth: float(r[f"hier_{winner_side}_{meth}"])
            for meth in ["ko", "sub", "dec"]
        }
        selected_method = max(candidate, key=candidate.get)
        selected_p = candidate[selected_method]
        selected_odds = float(r[f"dk_{winner_side}_{selected_method}_odds"])
        dec = american_to_decimal(selected_odds)
        ev = selected_p * dec - 1.0
        b = dec - 1.0
        full_kelly = max(0.0, (b * selected_p - (1.0 - selected_p)) / b) if b > 0 else 0.0
        result_rows.append(
            {
                "fight_id": r["fight_id"],
                "red_fighter": r["red_fighter"],
                "blue_fighter": r["blue_fighter"],
                "fight_cold_start": bool(r["fight_cold_start"]),
                "v5_p_red": float(r["v5_p_red"]),
                "v5_projected_winner": winner,
                "winner_side": winner_side,
                "winner_ko_p": candidate["ko"],
                "winner_sub_p": candidate["sub"],
                "winner_dec_p": candidate["dec"],
                "selected_method": selected_method.upper(),
                "selected_method_p": selected_p,
                "draftkings_odds": selected_odds,
                "ev_per_dollar": ev,
                "full_kelly_fraction": full_kelly,
                "quarter_kelly_fraction": full_kelly / 4.0,
                "quarter_kelly_stake_400": 400.0 * full_kelly / 4.0,
                "red_conditional_train_n": red_train_n,
                "blue_conditional_train_n": blue_train_n,
                "red_feature_count": red_fc,
                "blue_feature_count": blue_fc,
            }
        )

    out = pd.DataFrame(result_rows).sort_values("fight_id").reset_index(drop=True)
    out.to_csv(OUTPUT, index=False)
    print("\n=== UFC PARIS HIERARCHICAL V5 METHOD RUN ===")
    print(out.to_string(index=False))
    if not diagnostic.empty:
        print("\n=== INCOMPLETE DK METHOD MARKETS ===")
        print(diagnostic.to_string(index=False))
    print("\noutput=", OUTPUT)


if __name__ == "__main__":
    main()
