"""Study whether UFC layoff length is associated with KO/TKO winner vs loser.

Scope
-----
- UFC bouts dated 2020-01-01+
- both fighters had >=3 prior UFC fights before the bout
- actual KO/TKO bouts only for winner-vs-loser directional analysis
- layoff = days since that fighter's immediately previous UFC fight

Questions
---------
1. Do KO winners have systematically shorter/longer layoffs than KO losers?
2. How often does the shorter-layoff fighter win the KO?
3. Does KO win rate change across layoff-difference buckets?
4. Does layoff add out-of-fold directional signal after fighter age?
5. Is there evidence for an age x layoff interaction?

This is a research diagnostic only. It changes no FSR values or simulator constants.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

MASTER_PATH = Path("data/master/ufc_master.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "ko_winner_loser_layoff_study_2020plus_mature.parquet"
)
CV_SEED = 20260810
N_SPLITS = 5
N_REPEATS = 5


def _resolve_age(frame: pd.DataFrame, corner: str) -> pd.Series:
    for col in (f"{corner}_age", f"{corner}_fighter_age", f"{corner}_age_years"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
    for col in (
        f"{corner}_dob", f"{corner}_date_of_birth",
        f"{corner}_fighter_dob", f"{corner}_fighter_date_of_birth",
    ):
        if col in frame.columns:
            dob = pd.to_datetime(frame[col], errors="coerce")
            return (frame["event_date"] - dob).dt.days / 365.25
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _attach_history(master: pd.DataFrame) -> pd.DataFrame:
    """Attach strict prior-fight count and previous UFC fight date for each corner."""
    red = master[["fight_id", "event_date", "r_id"]].rename(columns={"r_id": "fighter_id"})
    blue = master[["fight_id", "event_date", "b_id"]].rename(columns={"b_id": "fighter_id"})
    appearances = pd.concat([red, blue], ignore_index=True)

    # One row per fighter/date prevents same-event duplicates from leaking into
    # the previous-fight date. Multiple appearances on one date are still counted
    # correctly for the strict prior-appearance count below.
    daily = (
        appearances.groupby(["fighter_id", "event_date"], as_index=False)
        .size()
        .rename(columns={"size": "fights_on_date"})
        .sort_values(["fighter_id", "event_date"])
    )
    daily["prior_ufc_fights"] = (
        daily.groupby("fighter_id")["fights_on_date"].cumsum() - daily["fights_on_date"]
    )
    daily["previous_ufc_fight_date"] = daily.groupby("fighter_id")["event_date"].shift(1)

    lookup = daily.set_index(["fighter_id", "event_date"])
    out = master.copy()
    for corner in ("r", "b"):
        ids = out[f"{corner}_id"].astype(str)
        keys = list(zip(ids, out["event_date"]))
        out[f"{corner}_prior_ufc_fights"] = [
            int(lookup.loc[key, "prior_ufc_fights"]) for key in keys
        ]
        previous = pd.to_datetime(
            [lookup.loc[key, "previous_ufc_fight_date"] for key in keys], errors="coerce"
        )
        out[f"{corner}_layoff_days"] = (out["event_date"] - previous).dt.days.astype(float)
    return out


def _build_frame() -> pd.DataFrame:
    raw = pd.read_parquet(MASTER_PATH).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    for col in ("fight_id", "r_id", "b_id", "winner_id"):
        if col not in raw.columns:
            raise ValueError(f"UFC master missing required column: {col}")
        raw[col] = raw[col].astype(str)
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")

    raw["r_age"] = _resolve_age(raw, "r")
    raw["b_age"] = _resolve_age(raw, "b")
    raw = _attach_history(raw)
    raw["actual_ko_tko"] = raw["method"].map(modern._is_ko_tko).astype(int)

    frame = raw.loc[
        raw["event_date"].ge(modern.START_DATE)
        & raw["r_prior_ufc_fights"].ge(modern.MIN_PRIOR_UFC_FIGHTS)
        & raw["b_prior_ufc_fights"].ge(modern.MIN_PRIOR_UFC_FIGHTS)
        & raw["actual_ko_tko"].eq(1)
    ].copy()

    r_win = frame["winner_id"].eq(frame["r_id"])
    b_win = frame["winner_id"].eq(frame["b_id"])
    frame = frame.loc[r_win | b_win].copy()
    r_win = frame["winner_id"].eq(frame["r_id"])

    frame["winner_layoff_days"] = np.where(r_win, frame["r_layoff_days"], frame["b_layoff_days"])
    frame["loser_layoff_days"] = np.where(r_win, frame["b_layoff_days"], frame["r_layoff_days"])
    frame["winner_age"] = np.where(r_win, frame["r_age"], frame["b_age"])
    frame["loser_age"] = np.where(r_win, frame["b_age"], frame["r_age"])
    frame["winner_minus_loser_layoff"] = frame["winner_layoff_days"] - frame["loser_layoff_days"]
    frame["winner_minus_loser_age"] = frame["winner_age"] - frame["loser_age"]
    frame["shorter_layoff_winner"] = (frame["winner_layoff_days"] < frame["loser_layoff_days"]).astype(int)
    frame["same_layoff"] = frame["winner_layoff_days"].eq(frame["loser_layoff_days"]).astype(int)
    return frame.reset_index(drop=True)


def _side_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Two fighter-side rows per KO bout for directional OOF classification."""
    rows = []
    for _, bout in frame.iterrows():
        for corner, opp in (("r", "b"), ("b", "r")):
            rows.append(
                {
                    "bout_id": str(bout["fight_id"]),
                    "fighter_id": str(bout[f"{corner}_id"]),
                    "is_ko_winner": int(str(bout["winner_id"]) == str(bout[f"{corner}_id"])),
                    "layoff_days": bout[f"{corner}_layoff_days"],
                    "opponent_layoff_days": bout[f"{opp}_layoff_days"],
                    "layoff_edge_days": bout[f"{opp}_layoff_days"] - bout[f"{corner}_layoff_days"],
                    "age": bout[f"{corner}_age"],
                    "opponent_age": bout[f"{opp}_age"],
                    "age_edge": bout[f"{opp}_age"] - bout[f"{corner}_age"],
                }
            )
    side = pd.DataFrame(rows)
    side["layoff_over_365"] = np.maximum(side["layoff_days"] - 365.0, 0.0)
    side["age_over_35"] = np.maximum(side["age"] - 35.0, 0.0)
    side["age_x_layoff"] = side["age_over_35"] * np.log1p(side["layoff_days"].clip(lower=0.0))
    return side


def _paired_oof(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """OOF metrics with bout-grouped folds so both sides stay together."""
    work = frame[["bout_id", "is_ko_winner"] + features].copy()
    bouts = work[["bout_id"]].drop_duplicates().reset_index(drop=True)
    # Every bout has one positive and one negative, so ordinary K-fold partitioning
    # over bout IDs is already class balanced at the side-row level.
    rng = np.random.default_rng(CV_SEED)
    pred_sum = np.zeros(len(work), dtype=float)
    pred_count = np.zeros(len(work), dtype=int)

    for repeat in range(N_REPEATS):
        shuffled = bouts.sample(frac=1.0, random_state=CV_SEED + repeat).reset_index(drop=True)
        folds = np.array_split(shuffled["bout_id"].to_numpy(), N_SPLITS)
        for test_bouts in folds:
            test_mask = work["bout_id"].isin(set(test_bouts))
            train_mask = ~test_mask
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("lr", LogisticRegression(C=0.5, max_iter=5000, solver="liblinear")),
                ]
            )
            model.fit(work.loc[train_mask, features], work.loc[train_mask, "is_ko_winner"])
            p = model.predict_proba(work.loc[test_mask, features])[:, 1]
            idx = np.flatnonzero(test_mask.to_numpy())
            pred_sum[idx] += p
            pred_count[idx] += 1

    p = pred_sum / pred_count
    y = work["is_ko_winner"].to_numpy(dtype=int)

    # Directional hit rate: within each bout, did the actual winner receive the
    # larger probability? Ties count separately rather than as wins.
    scored = work[["bout_id", "is_ko_winner"]].copy()
    scored["p"] = p
    hits = []
    ties = []
    for _, g in scored.groupby("bout_id"):
        winner_p = float(g.loc[g["is_ko_winner"].eq(1), "p"].iloc[0])
        loser_p = float(g.loc[g["is_ko_winner"].eq(0), "p"].iloc[0])
        hits.append(int(winner_p > loser_p))
        ties.append(int(winner_p == loser_p))

    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "direction_hit_rate": float(np.mean(hits)),
        "direction_tie_rate": float(np.mean(ties)),
    }


def main() -> None:
    frame = _build_frame()
    valid = frame.dropna(subset=["winner_layoff_days", "loser_layoff_days"]).copy()

    print("=" * 118)
    print("KO/TKO WINNER vs LOSER LAYOFF STUDY — 2020+ MATURE COHORT")
    print("=" * 118)
    print(f"KO/TKO bouts: {len(frame):,}")
    print(f"both layoffs available: {len(valid):,} ({len(valid) / len(frame):.2%})")

    print("\nWINNER / LOSER LAYOFF SUMMARY")
    summary = pd.DataFrame(
        [
            {
                "side": "KO winner",
                "n": valid["winner_layoff_days"].notna().sum(),
                "mean_days": valid["winner_layoff_days"].mean(),
                "median_days": valid["winner_layoff_days"].median(),
                "p75_days": valid["winner_layoff_days"].quantile(0.75),
                "p90_days": valid["winner_layoff_days"].quantile(0.90),
            },
            {
                "side": "KO loser",
                "n": valid["loser_layoff_days"].notna().sum(),
                "mean_days": valid["loser_layoff_days"].mean(),
                "median_days": valid["loser_layoff_days"].median(),
                "p75_days": valid["loser_layoff_days"].quantile(0.75),
                "p90_days": valid["loser_layoff_days"].quantile(0.90),
            },
        ]
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nmean winner-minus-loser layoff: {valid['winner_minus_loser_layoff'].mean():+.2f} days")
    print(f"median winner-minus-loser layoff: {valid['winner_minus_loser_layoff'].median():+.2f} days")
    non_ties = valid.loc[valid["same_layoff"].eq(0)]
    print(f"shorter-layoff fighter won KO: {non_ties['shorter_layoff_winner'].mean():.2%} of non-tied bouts")

    bins = [-np.inf, -365, -180, -90, 90, 180, 365, np.inf]
    labels = [
        "winner 365+d shorter", "winner 180-364d shorter", "winner 90-179d shorter",
        "within 90d", "winner 90-179d longer", "winner 180-364d longer", "winner 365+d longer",
    ]
    valid["layoff_gap_band"] = pd.cut(valid["winner_minus_loser_layoff"], bins=bins, labels=labels)
    print("\nKO OUTCOMES BY WINNER-vs-LOSER LAYOFF GAP")
    gap = valid.groupby("layoff_gap_band", observed=True).agg(
        bouts=("fight_id", "size"),
        mean_gap_days=("winner_minus_loser_layoff", "mean"),
        mean_winner_age=("winner_age", "mean"),
        mean_loser_age=("loser_age", "mean"),
    ).reset_index()
    print(gap.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    side = _side_frame(valid)
    models = [
        ("layoff edge only", ["layoff_edge_days"]),
        ("age edge only", ["age_edge"]),
        ("age + layoff edge", ["age_edge", "layoff_edge_days"]),
        ("age + layoff nonlinear", ["age_edge", "layoff_edge_days", "layoff_over_365"]),
        ("age + layoff + interaction", ["age_edge", "layoff_edge_days", "layoff_over_365", "age_x_layoff"]),
    ]
    rows = []
    for name, features in models:
        rows.append({"model": name, "features": ", ".join(features), **_paired_oof(side, features)})

    print("\nOUT-OF-FOLD KO WINNER DIRECTION — LAYOFF vs AGE")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(frame):,} KO/TKO bouts to {OUTPUT_PATH}")
    print("No FSR values or simulator constants were changed.")


if __name__ == "__main__":
    main()
