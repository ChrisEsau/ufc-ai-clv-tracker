from __future__ import annotations

"""Measurement-only FSR V3 striking-power structural study.

This does not publish or modify any FSR rating.  It asks whether the frozen V2
power construction can be improved before a V3 power trait is implemented.

Training signal candidates:
    damaging_events = knockdowns + lambda_ko * KO_win
    opportunities    = landed significant strikes

The validation target is ALWAYS future knockdowns per landed significant strike,
so lambda_ko is selected only if adding KO information helps predict independent
future KD production.  Candidate models are population-only, attacker-only, and
paired attacker-minus-defender penalized logistic models.

Age is deliberately excluded.  Age translation remains a later simulator-side
study.  Same-date leakage is avoided by calendar train/test boundaries.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pipeline.common.paths import MASTER_PATH, ROUND_STATS_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v2.physical import build_physical_observations


KO_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
C_GRID = (0.01, 0.03, 0.10, 0.30, 1.00)
STRUCTURES = ("attacker", "paired")
DEV_FOLDS = (
    ("2020", "2020-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
)
OUTER_START = pd.Timestamp("2024-01-01")
EPS = 1e-9


@dataclass
class FitResult:
    model: LogisticRegression
    fighter_to_col: dict[str, int]
    structure: str


def _is_ko(method: object) -> bool:
    s = str(method).upper()
    return "KO" in s or "TKO" in s


def _prepare() -> pd.DataFrame:
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id")
    obs = build_physical_observations(rounds, master).copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs["fighter_id"] = obs["fighter_id"].astype(str)
    obs["opponent_id"] = obs["opponent_id"].astype(str)
    obs["fight_id"] = obs["fight_id"].astype(str)
    for c in ("sig_landed", "kd_scored", "ko_win"):
        obs[c] = pd.to_numeric(obs[c], errors="coerce").fillna(0.0)
    obs = obs[obs["sig_landed"] > 0].copy()
    obs["kd_scored"] = np.minimum(obs["kd_scored"], obs["sig_landed"])
    return obs.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _design(frame: pd.DataFrame, mapping: dict[str, int], structure: str) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for r, x in enumerate(frame.itertuples(index=False)):
        f = mapping.get(str(x.fighter_id))
        if f is not None:
            rows.append(r); cols.append(f); vals.append(1.0)
        if structure == "paired":
            o = mapping.get(str(x.opponent_id))
            if o is not None:
                rows.append(r); cols.append(o); vals.append(-1.0)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(frame), len(mapping)))


def _fit(train: pd.DataFrame, structure: str, c_value: float, ko_weight: float) -> FitResult:
    fighters = sorted(set(train["fighter_id"]) | set(train["opponent_id"]))
    mapping = {f: i for i, f in enumerate(fighters)}
    X = _design(train, mapping, structure)
    n = train["sig_landed"].to_numpy(float)
    k = train["kd_scored"].to_numpy(float) + ko_weight * train["ko_win"].to_numpy(float)
    k = np.minimum(k, n)

    # Aggregated binomial likelihood represented by weighted success/failure rows.
    X2 = sparse.vstack([X, X], format="csr")
    y2 = np.concatenate([np.ones(len(train), dtype=int), np.zeros(len(train), dtype=int)])
    w2 = np.concatenate([k, n - k])
    keep = w2 > 0

    model = LogisticRegression(
        penalty="l2",
        C=float(c_value),
        fit_intercept=True,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-9,
    )
    model.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    return FitResult(model=model, fighter_to_col=mapping, structure=structure)


def _predict(fit: FitResult, frame: pd.DataFrame) -> np.ndarray:
    X = _design(frame, fit.fighter_to_col, fit.structure)
    return np.clip(fit.model.predict_proba(X)[:, 1], EPS, 1 - EPS)


def _population_prob(train: pd.DataFrame) -> float:
    n = float(train["sig_landed"].sum())
    return float((train["kd_scored"].sum() + 0.5) / (n + 1.0))


def _kd_ll(frame: pd.DataFrame, p: np.ndarray) -> tuple[float, float]:
    n = frame["sig_landed"].to_numpy(float)
    k = frame["kd_scored"].to_numpy(float)
    ll = float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))
    return ll, ll / float(n.sum())


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def _development(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, start_s, end_s in DEV_FOLDS:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        train = obs[obs["date"] < start].copy()
        test = obs[(obs["date"] >= start) & (obs["date"] < end)].copy()
        if train.empty or test.empty:
            continue
        pop_p = _population_prob(train)
        pop_ll, pop_rate = _kd_ll(test, np.full(len(test), pop_p))
        rows.append({
            "fold": fold, "structure": "population", "ko_weight": 0.0, "C": np.nan,
            "fighter_fights": len(test), "landed_sig": float(test["sig_landed"].sum()),
            "kd_ll": pop_ll, "kd_ll_per_landed": pop_rate,
            "ko_winner_auc": _safe_auc(test["ko_win"].to_numpy(int), np.full(len(test), pop_p)),
        })
        for ko_weight in KO_WEIGHTS:
            for structure in STRUCTURES:
                for c_value in C_GRID:
                    fit = _fit(train, structure, c_value, ko_weight)
                    p = _predict(fit, test)
                    ll, rate = _kd_ll(test, p)
                    rows.append({
                        "fold": fold, "structure": structure, "ko_weight": ko_weight, "C": c_value,
                        "fighter_fights": len(test), "landed_sig": float(test["sig_landed"].sum()),
                        "kd_ll": ll, "kd_ll_per_landed": rate,
                        "ko_winner_auc": _safe_auc(test["ko_win"].to_numpy(int), p),
                    })
    return pd.DataFrame(rows)


def _select(dev: pd.DataFrame) -> pd.DataFrame:
    candidates = dev[dev["structure"] != "population"].copy()
    agg = candidates.groupby(["structure", "ko_weight", "C"], as_index=False).agg(
        folds=("fold", "nunique"),
        kd_ll=("kd_ll", "sum"),
        landed_sig=("landed_sig", "sum"),
        mean_ko_winner_auc=("ko_winner_auc", "mean"),
        worst_fold_ll=("kd_ll", "min"),
    )
    agg["kd_ll_per_landed"] = agg["kd_ll"] / agg["landed_sig"]

    pop = dev[dev["structure"] == "population"].groupby("fold")["kd_ll"].first()
    gains = []
    for key, g in candidates.groupby(["structure", "ko_weight", "C"]):
        by_fold = g.set_index("fold")["kd_ll"]
        aligned = by_fold.index.intersection(pop.index)
        delta = by_fold.loc[aligned] - pop.loc[aligned]
        gains.append((*key, float(delta.sum()), float(delta.min()), int((delta > 0).sum())))
    gain_df = pd.DataFrame(gains, columns=["structure", "ko_weight", "C", "ll_gain_vs_population", "worst_fold_gain", "folds_beating_population"])
    agg = agg.merge(gain_df, on=["structure", "ko_weight", "C"], how="left")
    return agg.sort_values(["kd_ll", "mean_ko_winner_auc"], ascending=[False, False]).reset_index(drop=True)


def _outer(obs: pd.DataFrame, selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = obs[obs["date"] < OUTER_START].copy()
    test = obs[obs["date"] >= OUTER_START].copy()

    pop_p = _population_prob(train)
    pop_ll, pop_rate = _kd_ll(test, np.full(len(test), pop_p))

    fit = _fit(train, str(selected.structure), float(selected.C), float(selected.ko_weight))
    p = _predict(fit, test)
    model_ll, model_rate = _kd_ll(test, p)

    metrics = pd.DataFrame([
        {
            "model": "population", "fighter_fights": len(test), "fights": test["fight_id"].nunique(),
            "landed_sig": float(test["sig_landed"].sum()), "kd_ll": pop_ll,
            "kd_ll_per_landed": pop_rate, "ll_gain_vs_population": 0.0,
            "ko_winner_auc": _safe_auc(test["ko_win"].to_numpy(int), np.full(len(test), pop_p)),
        },
        {
            "model": "selected_v3_candidate", "fighter_fights": len(test), "fights": test["fight_id"].nunique(),
            "landed_sig": float(test["sig_landed"].sum()), "kd_ll": model_ll,
            "kd_ll_per_landed": model_rate, "ll_gain_vs_population": model_ll - pop_ll,
            "ko_winner_auc": _safe_auc(test["ko_win"].to_numpy(int), p),
        },
    ])

    detail = test[["fight_id", "date", "fighter_id", "opponent_id", "sig_landed", "kd_scored", "ko_win"]].copy()
    detail["population_damage_probability"] = pop_p
    detail["candidate_damage_probability"] = p

    # V2 published power discrimination on exactly the same outer fighter-fights.
    try:
        fsr2 = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()
        fsr2["fight_id"] = fsr2["fight_id"].astype(str)
        fsr2["fighter_id"] = fsr2["fighter_id"].astype(str)
        detail = detail.merge(
            fsr2[["fight_id", "fighter_id", "striking_power"]],
            on=["fight_id", "fighter_id"], how="left", validate="one_to_one",
        )
        if detail["striking_power"].notna().all():
            v2_auc = _safe_auc(detail["ko_win"].to_numpy(int), detail["striking_power"].to_numpy(float))
            metrics.loc[len(metrics)] = {
                "model": "published_fsr_v2_power_auc_only", "fighter_fights": len(detail),
                "fights": detail["fight_id"].nunique(), "landed_sig": float(detail["sig_landed"].sum()),
                "kd_ll": np.nan, "kd_ll_per_landed": np.nan, "ll_gain_vs_population": np.nan,
                "ko_winner_auc": v2_auc,
            }
    except FileNotFoundError:
        pass

    return metrics, detail


def main(out_dir: str = "data/diagnostics/fsr_v3_power") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs = _prepare()
    print("=" * 120)
    print("FSR V3 POWER STRUCTURAL STUDY")
    print("=" * 120)
    print(f"fighter-fight observations with >=1 landed sig strike: {len(obs):,}")
    print(f"date range: {obs.date.min().date()} to {obs.date.max().date()}")
    print("Validation target: future knockdowns per landed significant strike")
    print("Age: excluded from persisted trait study")

    dev = _development(obs)
    selection = _select(dev)
    selected = selection.iloc[0]

    print("\nTOP DEVELOPMENT CANDIDATES")
    print(selection.head(15).to_string(index=False))
    print("\nSELECTED")
    print(selected.to_string())

    outer, detail = _outer(obs, selected)
    print("\nRESERVED OUTER 2024+ RESULTS")
    print(outer.to_string(index=False))

    dev.to_csv(out / "power_dev_folds.csv", index=False)
    selection.to_csv(out / "power_candidate_selection.csv", index=False)
    outer.to_csv(out / "power_outer_metrics.csv", index=False)
    detail.to_csv(out / "power_outer_detail.csv", index=False)
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
