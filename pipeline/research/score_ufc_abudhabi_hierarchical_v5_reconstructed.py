from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features.views.moneyline import build_moneyline_feature_view
from pipeline.research import xgboost_method_market_offset as method
from pipeline.research.xgboost_method_hierarchical_v5_oof import _fit_conditional

OUT = Path("data/research/prop_mispricing")
OUTPUT = OUT / "ufc_abudhabi_hierarchical_v5_reconstructed_20260725.csv"
BET_OUTPUT = OUT / "ufc_abudhabi_hierarchical_v5_reconstructed_bets_logit030_20260725.csv"
FIGHT_DATE = pd.Timestamp("2026-07-25")
THRESHOLD = 0.30
EPS = 1e-12

# Reconstructed historical method prices. These are NOT an exact DraftKings snapshot.
# Each fight uses a complete six-way pre-fight method market recovered from archived
# sportsbook pages; source is recorded per outcome/fight below.
# American odds may be floats when converted exactly from fractional/decimal prices.
CARD = [
    dict(fid="abu_01", red="Magomed Ankalaev", blue="Bogdan Guskov", v5=.815315,
         r=( -187.5, 750, 400), b=(500, 2000, 1400), source="Coral archived 7-way", target="red_ko"),
    dict(fid="abu_02", red="Steve Erceg", blue="Ramazan Temirov", v5=.493466,
         r=(800, 800, 160), b=(225, 2250, 325), source="Covers/Bet99 archived", target="blue_ko"),
    dict(fid="abu_03", red="Magomed Zaynukov", blue="Damian Rzepecki", v5=.717040,
         r=(215, 900, 149), b=(1000, 750, 650), source="PhantomStat archived", target="red_dec"),
    dict(fid="abu_04", red="Rizvan Kuniev", blue="Tyrell Fortune", v5=.765695,
         r=(210, 900, 150), b=(550, 2000, 400), source="Coral archived 7-way", target="red_ko"),
    dict(fid="abu_05", red="Abubakar Vagaev", blue="Saygid Izagakhmaev", v5=.698073,
         r=(510, 1150, -118), b=(1400, 500, 650), source="MMA Index archived", target="red_dec"),
    dict(fid="abu_06", red="Ismael Bonfim", blue="Axel Sola", v5=.340293,
         r=(425, 900, 500), b=(175, 750, 250), source="Coral archived 7-way", target="blue_sub"),
    dict(fid="abu_07", red="Valter Walker", blue="Thomas Petersen", v5=.663077,
         r=(750, 130, 375), b=(375, 2000, 320), source="Coral archived 7-way", target="red_sub"),
    dict(fid="abu_08", red="Dustin Jacoby", blue="Muhammad Saidov", v5=.633140,
         r=(175, 1400, 240), b=(375, 800, 400), source="Coral archived 7-way", target="blue_ko"),
    dict(fid="abu_09", red="Santiago Ponzinibbio", blue="Sam Patterson", v5=.229243,
         r=(600, 1650, 700), b=(102, 280, 350), source="Betfair Exchange archived", target="blue_ko"),
    dict(fid="abu_10", red="Magomed Tuchalov", blue="Brendson Ribeiro", v5=.863915,
         r=(-300, 550, 750), b=(900, 1400, 1600), source="Coral archived 7-way", target="red_dec"),
    dict(fid="abu_11", red="Nurullo Aliev", blue="Mike Davis", v5=.729535,
         r=(400, 1000, 100), b=(600, 750, 500), source="Betfair/BetMGM archived", target="red_dec"),
    dict(fid="abu_12", red="Cody Gibson", blue="Abdul Hussein", v5=.202050,
         r=(1200, 1400, 900), b=(175, 175, 250), source="bwin archived 7-way", target="blue_sub"),
]

SLUGS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]


def norm(value: object) -> str:
    text = "".join(ch for ch in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def implied_from_american(a: float) -> float:
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def decimal_from_american(a: float) -> float:
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(np.log(p / (1.0 - p)))


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    if c not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce")


def _rate(df: pd.DataFrame, c: str) -> pd.Series:
    s = _num(df, c)
    return pd.Series(np.where(s > 1, s / 100.0, s), index=df.index)


def add_v5_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chin_risk_diff"] = _num(df, "r_pre_sapm") * (1 - _rate(df, "r_pre_str_def")) - _num(df, "b_pre_sapm") * (1 - _rate(df, "b_pre_str_def"))
    df["aggression_index_diff"] = (_num(df, "r_pre_splm") + _num(df, "r_pre_td_avg")) - (_num(df, "b_pre_splm") + _num(df, "b_pre_td_avg"))
    df["age_squared_diff"] = _num(df, "r_pre_age") ** 2 - _num(df, "b_pre_age") ** 2
    df["wrestling_mismatch_diff"] = _num(df, "r_pre_td_avg") * (1 - _rate(df, "b_pre_td_def")) - _num(df, "b_pre_td_avg") * (1 - _rate(df, "r_pre_td_def"))
    df["pressure_striking_adv_diff"] = _num(df, "r_pre_splm") * (1 - _rate(df, "b_pre_str_def")) - _num(df, "b_pre_splm") * (1 - _rate(df, "r_pre_str_def"))
    df["age_x_career_ko_losses_diff"] = _num(df, "r_pre_age") * _num(df, "r_pre_career_ko_losses") - _num(df, "b_pre_age") * _num(df, "b_pre_career_ko_losses")
    return df


def build_prefight_features(card: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    hist = pd.read_parquet("data/features/fighter_state_history.parquet").copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    if "fighter_name" not in hist.columns:
        raise RuntimeError("fighter_state_history lacks fighter_name")
    hist["_norm"] = hist["fighter_name"].map(norm)
    hist = hist[hist["date"] < FIGHT_DATE].copy()

    prep_rows = []
    state_rows = []
    found_map: dict[str, bool] = {}
    all_state_cols = [c for c in hist.columns if c != "_norm"]

    for i, fight in card.reset_index(drop=True).iterrows():
        ids = []
        found_both = True
        for nm in [fight["red_fighter"], fight["blue_fighter"]]:
            hit = hist[hist["_norm"].eq(norm(nm))].sort_values("date")
            if len(hit):
                rec = hit.iloc[-1].drop(labels=["_norm"]).to_dict()
                fid = str(rec["fighter_id"])
                found = True
            else:
                rec = {c: np.nan for c in all_state_cols}
                fid = f"missing::{norm(nm)}"
                rec["fighter_id"] = fid
                rec["fighter_name"] = nm
                rec["date"] = FIGHT_DATE - pd.Timedelta(days=1)
                found = False
            found_both = found_both and found
            rec["fight_id"] = fight["fight_id"]
            state_rows.append(rec)
            ids.append(fid)
        found_map[str(fight["fight_id"])] = found_both
        prep_rows.append({
            "fight_id": fight["fight_id"],
            "r_id": ids[0], "b_id": ids[1],
            "r_name": fight["red_fighter"], "b_name": fight["blue_fighter"],
            "date": FIGHT_DATE,
            "title_fight": False,
            "total_rounds": 5 if i == 0 else 3,
        })

    live = build_moneyline_feature_view(
        prepared_fights_df=pd.DataFrame(prep_rows),
        fighter_state_history_df=pd.DataFrame(state_rows),
    )
    return add_v5_engineered(live), found_map


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    card = pd.DataFrame([{ "fight_id": x["fid"], "red_fighter": x["red"], "blue_fighter": x["blue"], "v5_p_red": x["v5"], "source": x["source"], "target_slug": x["target"], "r_ko": x["r"][0], "r_sub": x["r"][1], "r_dec": x["r"][2], "b_ko": x["b"][0], "b_sub": x["b"][1], "b_dec": x["b"][2]} for x in CARD])

    train, features, _ = method._build_rows(True, True)
    live, found_map = build_prefight_features(card)
    live_by_id = live.set_index("fight_id", drop=False)

    score_rows = []
    for _, f in card.iterrows():
        fid = str(f["fight_id"])
        if fid not in live_by_id.index:
            raise RuntimeError(f"missing feature row for {fid}")
        lr = live_by_id.loc[fid]
        odds = [float(f[c]) for c in ["r_ko", "r_sub", "r_dec", "b_ko", "b_sub", "b_dec"]]
        raw = np.array([implied_from_american(x) for x in odds], float)
        six_norm = raw / raw.sum()
        row = {
            "fight_id": fid, "date": FIGHT_DATE,
            "red_fighter": f["red_fighter"], "blue_fighter": f["blue_fighter"],
            "v5_p_red": float(f["v5_p_red"]), "market_source": f["source"],
            "target_slug": f["target_slug"], "fight_cold_start": not found_map[fid],
        }
        for j, slug in enumerate(SLUGS):
            row[f"raw_{slug}_implied"] = raw[j]
            row[f"market_{slug}"] = six_norm[j]
            row[f"odds_{slug}"] = odds[j]
        for c in features:
            row[c] = lr[c] if c in lr.index else np.nan
        score_rows.append(row)
    score = pd.DataFrame(score_rows)

    # Conditional model base margins need within-side 3-way method shares. Scaling the
    # six-way normalized probabilities back within side is handled inside _fit_conditional.
    red_cond, red_n, red_fc = _fit_conditional(train, score, features, "red")
    blue_cond, blue_n, blue_fc = _fit_conditional(train, score, features, "blue")
    p_red = np.clip(score["v5_p_red"].to_numpy(float), EPS, 1-EPS)
    hier = np.concatenate([p_red[:, None] * red_cond, (1-p_red)[:, None] * blue_cond], axis=1)
    hier = hier / hier.sum(axis=1, keepdims=True)
    for j, slug in enumerate(SLUGS):
        score[f"hier_{slug}"] = hier[:, j]
        score[f"residual_{slug}"] = [logit(hier[i, j]) - logit(score.iloc[i][f"market_{slug}"]) for i in range(len(score))]

    bet_rows = []
    target_idx = {s:i for i,s in enumerate(SLUGS)}
    for _, r in score.iterrows():
        predicted_side = "red" if float(r["v5_p_red"]) >= 0.5 else "blue"
        for meth in ["ko", "sub", "dec"]:
            slug = f"{predicted_side}_{meth}"
            resid = float(r[f"residual_{slug}"])
            if resid < THRESHOLD:
                continue
            odds = float(r[f"odds_{slug}"])
            dec = decimal_from_american(odds)
            won = int(str(r["target_slug"]) == slug)
            profit = dec - 1.0 if won else -1.0
            bet_rows.append({
                "fight_id": r["fight_id"], "date": FIGHT_DATE,
                "red_fighter": r["red_fighter"], "blue_fighter": r["blue_fighter"],
                "fight_cold_start": bool(r["fight_cold_start"]),
                "v5_projected_winner": r[f"{predicted_side}_fighter"],
                "bet_slug": slug, "bet_fighter": r[f"{predicted_side}_fighter"], "bet_method": meth.upper(),
                "model_probability": float(r[f"hier_{slug}"]),
                "normalized_market_probability": float(r[f"market_{slug}"]),
                "signed_logit_residual": resid,
                "american_odds": odds, "decimal_odds": dec,
                "market_source": r["market_source"],
                "actual_slug": r["target_slug"], "won": won,
                "stake_units": 1.0, "profit_units": profit,
            })

    # Keep a compact all-fights diagnostic plus all six probabilities/residuals.
    keep = ["fight_id", "date", "red_fighter", "blue_fighter", "fight_cold_start", "v5_p_red", "market_source", "target_slug"]
    for slug in SLUGS:
        keep += [f"odds_{slug}", f"market_{slug}", f"hier_{slug}", f"residual_{slug}"]
    score[keep].to_csv(OUTPUT, index=False)
    bets = pd.DataFrame(bet_rows)
    bets.to_csv(BET_OUTPUT, index=False)

    print(f"FIGHTS={len(score)} RED_TRAIN_N={red_n} BLUE_TRAIN_N={blue_n} RED_FC={red_fc} BLUE_FC={blue_fc}")
    print("\n=== QUALIFYING METHOD BETS >= +0.30 LOGIT ===")
    if bets.empty:
        print("NONE")
    else:
        print(bets[["fight_id","bet_fighter","bet_method","american_odds","model_probability","normalized_market_probability","signed_logit_residual","actual_slug","won","profit_units","market_source"]].to_string(index=False))
        print(f"BETS={len(bets)} WINS={int(bets['won'].sum())} LOSSES={int((1-bets['won']).sum())} PROFIT_UNITS={bets['profit_units'].sum():.6f} ROI={bets['profit_units'].sum()/len(bets):.6f}")
    print("OUTPUT=", OUTPUT)
    print("BET_OUTPUT=", BET_OUTPUT)

if __name__ == "__main__":
    main()
