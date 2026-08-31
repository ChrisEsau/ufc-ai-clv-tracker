from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research.xgboost_method_market_offset import (
    ROOT,
    OOF_PATH,
    CLASS_ORDER,
    SLUGS,
    _read_market,
)

OUT_GRID = ROOT / "xgboost_method_bet_gate_oof__grid.csv"
OUT_FOLD = ROOT / "xgboost_method_bet_gate_oof__by_fold.csv"
OUT_LEDGER = ROOT / "xgboost_method_bet_gate_oof__selected_ledger.csv"
OUT_SUMMARY = ROOT / "xgboost_method_bet_gate_oof__summary.json"

LOGIT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
PRICE_EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
EPS = 1e-12


def _logit(x: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _raw_price_wide() -> pd.DataFrame:
    m = _read_market(True).copy()
    p = (
        m.pivot(index="fight_id", columns="class_slug", values="implied_probability")
        .reindex(columns=SLUGS)
        .dropna()
    )
    out = pd.DataFrame({"fight_id": p.index.astype(str)})
    for slug in SLUGS:
        out[f"raw_{slug}"] = p[slug].to_numpy(float)
    return out


def _long_ledger() -> pd.DataFrame:
    oof = pd.read_csv(OOF_PATH).copy()
    oof["date"] = pd.to_datetime(oof["date"], errors="coerce")
    oof["fight_id"] = oof["fight_id"].astype(str)
    oof = oof.merge(_raw_price_wide(), on="fight_id", how="inner", validate="one_to_one")

    rows: list[dict] = []
    for r in oof.itertuples(index=False):
        for j, slug in enumerate(SLUGS):
            raw = float(getattr(r, f"raw_{slug}"))
            fair = float(getattr(r, f"market_{slug}"))
            model = float(getattr(r, f"model_{slug}"))
            if not (0 < raw < 1 and 0 < fair < 1 and 0 < model < 1):
                continue
            won = int(int(r.target) == j)
            signed_logit = float(_logit([model])[0] - _logit([fair])[0])
            price_edge = model - raw
            rows.append({
                "fight_id": r.fight_id,
                "date": r.date,
                "event_name": r.event_name,
                "fold": int(r.fold),
                "red_fighter": r.red_fighter,
                "blue_fighter": r.blue_fighter,
                "class_slug": slug,
                "class_name": CLASS_ORDER[j],
                "won": won,
                "raw_implied_prob": raw,
                "fair_market_prob": fair,
                "model_prob": model,
                "price_edge": price_edge,
                "signed_logit_residual": signed_logit,
                "model_implied_ev_margin": model / raw - 1.0,
            })
    return pd.DataFrame(rows)


def _select_one(df: pd.DataFrame, logit_min: float, price_edge_min: float) -> pd.DataFrame:
    g = df[
        (df["signed_logit_residual"] >= float(logit_min))
        & (df["price_edge"] >= float(price_edge_min))
    ].copy()
    if g.empty:
        return g
    g = g.sort_values(
        ["date", "fight_id", "signed_logit_residual", "price_edge", "class_slug"],
        ascending=[True, True, False, False, True],
    )
    return g.drop_duplicates("fight_id", keep="first").reset_index(drop=True)


def _metrics(g: pd.DataFrame, total_fights: int) -> dict:
    n = len(g)
    if not n:
        return {
            "bets": 0,
            "fight_coverage": 0.0,
            "hit_rate": None,
            "mean_model_p": None,
            "mean_raw_p": None,
            "calibration_gap": None,
            "mean_price_edge": None,
            "mean_signed_logit_residual": None,
            "mean_model_implied_ev_margin": None,
            "cards": 0,
            "bets_per_card": None,
        }
    cards = int(g[["date", "event_name"]].drop_duplicates().shape[0])
    hit = float(g["won"].mean())
    model = float(g["model_prob"].mean())
    return {
        "bets": int(n),
        "fight_coverage": float(n / total_fights),
        "hit_rate": hit,
        "mean_model_p": model,
        "mean_raw_p": float(g["raw_implied_prob"].mean()),
        "calibration_gap": float(hit - model),
        "mean_price_edge": float(g["price_edge"].mean()),
        "mean_signed_logit_residual": float(g["signed_logit_residual"].mean()),
        "mean_model_implied_ev_margin": float(g["model_implied_ev_margin"].mean()),
        "cards": cards,
        "bets_per_card": float(n / cards) if cards else None,
    }


def _candidate_score(row: pd.Series, fold_df: pd.DataFrame) -> tuple:
    """Non-ROI ranking: calibration, fold stability, selectivity, then sample size.

    Prefer candidates with enough data in every fold, lower worst-fold absolute
    calibration error, lower pooled absolute calibration error, and fewer bets/card.
    This deliberately ignores profit/ROI.
    """
    f = fold_df[
        (fold_df["logit_threshold"] == row["logit_threshold"])
        & (fold_df["price_edge_threshold"] == row["price_edge_threshold"])
    ]
    min_fold_n = int(f["bets"].min()) if len(f) else 0
    worst_fold_cal = float(f["calibration_gap"].abs().max()) if len(f) else float("inf")
    sufficient = min_fold_n >= 20 and int(row["bets"]) >= 120
    return (
        0 if sufficient else 1,
        worst_fold_cal,
        abs(float(row["calibration_gap"])),
        float(row["bets_per_card"]),
        -int(row["bets"]),
    )


def main() -> None:
    ledger = _long_ledger()
    total_fights = int(ledger["fight_id"].nunique())
    if total_fights != 1604:
        raise RuntimeError(f"expected 1604 frozen OOF fights, got {total_fights}")

    grid_rows: list[dict] = []
    fold_rows: list[dict] = []
    selections: dict[tuple[float, float], pd.DataFrame] = {}

    for logit_min in LOGIT_THRESHOLDS:
        for edge_min in PRICE_EDGE_THRESHOLDS:
            g = _select_one(ledger, logit_min, edge_min)
            selections[(logit_min, edge_min)] = g
            grid_rows.append({
                "logit_threshold": logit_min,
                "price_edge_threshold": edge_min,
                **_metrics(g, total_fights),
            })
            for fold in [2021, 2022, 2023, 2024]:
                gf = g[g["fold"] == fold]
                fold_total = int(ledger.loc[ledger["fold"] == fold, "fight_id"].nunique())
                fold_rows.append({
                    "logit_threshold": logit_min,
                    "price_edge_threshold": edge_min,
                    "fold": fold,
                    **_metrics(gf, fold_total),
                })

    grid = pd.DataFrame(grid_rows)
    by_fold = pd.DataFrame(fold_rows)
    grid["selection_score"] = [
        str(_candidate_score(r, by_fold)) for _, r in grid.iterrows()
    ]
    ranked_idx = sorted(
        grid.index,
        key=lambda i: _candidate_score(grid.loc[i], by_fold),
    )
    chosen = grid.loc[ranked_idx[0]].copy()
    key = (float(chosen["logit_threshold"]), float(chosen["price_edge_threshold"]))
    selected = selections[key].copy()

    grid.to_csv(OUT_GRID, index=False)
    by_fold.to_csv(OUT_FOLD, index=False)
    selected.to_csv(OUT_LEDGER, index=False)

    chosen_folds = by_fold[
        (by_fold["logit_threshold"] == key[0])
        & (by_fold["price_edge_threshold"] == key[1])
    ].to_dict(orient="records")

    summary = {
        "experiment": "six_way_method_bet_gate_oof_diagnostic_v1",
        "development_period": "chronological 2021-2024 OOF only",
        "source": str(OOF_PATH),
        "frozen_oof_fights": total_fights,
        "cold_start_filter": False,
        "one_bet_per_fight": "hard maximum one; choose largest signed logit residual, then price edge",
        "selection_policy": "NO ROI OR PROFIT USED. Prefer >=20 bets in every fold and >=120 pooled, then minimize worst-fold absolute calibration gap, pooled absolute calibration gap, bets/card, and finally prefer larger sample.",
        "grid": {
            "logit_thresholds": LOGIT_THRESHOLDS,
            "price_edge_thresholds": PRICE_EDGE_THRESHOLDS,
        },
        "selected_gate": {
            "logit_threshold": key[0],
            "price_edge_threshold": key[1],
            **{k: (None if pd.isna(v) else v) for k, v in _metrics(selected, total_fights).items()},
            "per_fold": chosen_folds,
        },
        "note": "2025+ data is not read by this diagnostic and cannot influence gate selection.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
