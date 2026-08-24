"""Measurement-only next-round takedown policy learnability diagnostic.

Research question
-----------------
Can we predict whether a UFC fighter will attempt at least one takedown in the
NEXT round using only information that would have been available before that
round?

This is deliberately not a simulator change.  It separates stable prefight
fighter/opponent history from within-fight context and compares nested models:

A. population baseline
B. fighter prefight TD tendency
C. fighter + opponent prefight TD profile
D. fighter + opponent prefight profile + immediately previous-round context

The diagnostic uses ``data/fight_details/ufc_round_stats.parquet`` in its native
long format (one fighter/corner row per round), creates targets only for observed
consecutive rounds, builds all historical features strictly from PRIOR fights,
and performs a whole-fight chronological 70/30 train/test split.

No FSR, Event Clock mechanics, or source parquet are modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
OUT = Path("data/diagnostics/event_clock_mc_v2/policy_learnability")
TRAIN_FRACTION = 0.70
EPS = 1e-6

CORE_REQUIRED = {
    "fight_id",
    "event_date",
    "corner",
    "fighter_id",
    "round",
    "td_attempted",
}

# These are used when available.  Missing optional context fields never change
# the target and are simply omitted from the corresponding feature block.
OPTIONAL_NUMERIC = (
    "td_landed",
    "distance_attempted",
    "distance_landed",
    "clinch_attempted",
    "clinch_landed",
    "ground_attempted",
    "ground_landed",
    "sig_str_attempted",
    "sig_str_landed",
    "total_str_attempted",
    "total_str_landed",
    "ctrl_sec",
    "kd",
    "sub_attempted",
)


@dataclass
class FighterHistory:
    """Strictly prefight UFC history accumulated after each completed fight."""

    rounds: int = 0
    rounds_with_td_attempt: int = 0
    td_attempted: float = 0.0
    td_landed: float = 0.0
    opponent_rounds: int = 0
    opponent_rounds_with_td_attempt: int = 0
    opponent_td_attempted: float = 0.0
    opponent_td_landed: float = 0.0
    fights: int = 0
    extras: dict[str, float] = field(default_factory=dict)

    def own_features(self) -> dict[str, float]:
        r = max(self.rounds, 1)
        return {
            "fighter_prior_fights": float(self.fights),
            "fighter_prior_rounds": float(self.rounds),
            "fighter_td_round_rate": self.rounds_with_td_attempt / r if self.rounds else np.nan,
            "fighter_td_attempts_per_round": self.td_attempted / r if self.rounds else np.nan,
            "fighter_td_land_rate": self.td_landed / self.td_attempted if self.td_attempted > 0 else np.nan,
        }

    def opponent_features(self) -> dict[str, float]:
        r = max(self.opponent_rounds, 1)
        faced = self.opponent_td_attempted
        return {
            "opponent_prior_fights": float(self.fights),
            "opponent_prior_rounds": float(self.rounds),
            "opponent_td_attempt_round_exposure": self.opponent_rounds_with_td_attempt / r if self.opponent_rounds else np.nan,
            "opponent_td_attempts_faced_per_round": faced / r if self.opponent_rounds else np.nan,
            "opponent_td_allowed_rate": self.opponent_td_landed / faced if faced > 0 else np.nan,
            "opponent_td_denial_rate": 1.0 - (self.opponent_td_landed / faced) if faced > 0 else np.nan,
        }


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def load_rounds() -> tuple[pd.DataFrame, list[str]]:
    rd = pd.read_parquet(ROUND_PATH).copy()
    missing = sorted(CORE_REQUIRED - set(rd.columns))
    if missing:
        raise RuntimeError(f"round stats missing required columns: {missing}")

    rd["fight_id"] = rd["fight_id"].astype(str)
    rd["fighter_id"] = rd["fighter_id"].astype(str)
    rd["event_date"] = pd.to_datetime(rd["event_date"], errors="coerce").dt.normalize()
    rd["round"] = pd.to_numeric(rd["round"], errors="coerce")
    rd["corner_norm"] = rd["corner"].astype(str).str.strip().str.lower()

    bad_corner = ~rd["corner_norm"].isin(["red", "blue"])
    if bad_corner.any():
        vals = sorted(rd.loc[bad_corner, "corner"].astype(str).unique())
        raise RuntimeError(f"unexpected corner values: {vals}")

    rd = rd[rd["event_date"].notna() & rd["round"].notna()].copy()
    rd["round"] = rd["round"].astype(int)

    numeric = ["td_attempted"] + [c for c in OPTIONAL_NUMERIC if c in rd.columns]
    numeric = list(dict.fromkeys(numeric))
    for c in numeric:
        rd[c] = _num(rd[c])

    dup = rd.duplicated(["fight_id", "fighter_id", "round"], keep=False)
    if dup.any():
        raise RuntimeError(
            "round stats are not unique by fight_id/fighter_id/round; "
            f"duplicate rows={int(dup.sum())}"
        )

    return rd.sort_values(["event_date", "fight_id", "round", "corner_norm"]).reset_index(drop=True), numeric


def _history_snapshot(histories: dict[str, FighterHistory], fighter_id: str) -> FighterHistory:
    return histories.get(str(fighter_id), FighterHistory())


def build_prefight_snapshots(rd: pd.DataFrame, numeric: Iterable[str]) -> pd.DataFrame:
    """Create one leakage-safe prefight history snapshot per fighter-fight."""
    histories: dict[str, FighterHistory] = {}
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    fight_order = (
        rd[["fight_id", "event_date"]]
        .drop_duplicates()
        .sort_values(["event_date", "fight_id"])
    )

    for _, fr in fight_order.iterrows():
        fid = str(fr.fight_id)
        fight = rd[rd["fight_id"].eq(fid)].copy()
        fighters = fight[["fighter_id", "corner_norm"]].drop_duplicates()
        if len(fighters) != 2 or fighters["corner_norm"].nunique() != 2:
            continue

        red_id = str(fighters.loc[fighters.corner_norm.eq("red"), "fighter_id"].iloc[0])
        blue_id = str(fighters.loc[fighters.corner_norm.eq("blue"), "fighter_id"].iloc[0])

        # Snapshot both fighters BEFORE this fight contributes to either history.
        for fighter_id, opponent_id, corner in (
            (red_id, blue_id, "red"),
            (blue_id, red_id, "blue"),
        ):
            own = _history_snapshot(histories, fighter_id)
            opp = _history_snapshot(histories, opponent_id)
            rec: dict[str, float | str | pd.Timestamp] = {
                "fight_id": fid,
                "event_date": fr.event_date,
                "fighter_id": fighter_id,
                "opponent_id": opponent_id,
                "corner_norm": corner,
            }
            rec.update(own.own_features())
            # Opponent matchup features describe what prior opponents were able
            # to attempt/land against the CURRENT opponent.
            rec.update(opp.opponent_features())
            rows.append(rec)

        # Update histories only after both prefight snapshots are frozen.
        by_fighter = {str(k): g for k, g in fight.groupby("fighter_id")}
        for fighter_id, opponent_id in ((red_id, blue_id), (blue_id, red_id)):
            own_rounds = by_fighter[fighter_id]
            opp_rounds = by_fighter[opponent_id]
            h = histories.setdefault(fighter_id, FighterHistory())
            h.fights += 1
            h.rounds += int(len(own_rounds))
            h.rounds_with_td_attempt += int((own_rounds["td_attempted"] > 0).sum())
            h.td_attempted += float(own_rounds["td_attempted"].sum())
            if "td_landed" in own_rounds:
                h.td_landed += float(own_rounds["td_landed"].sum())

            h.opponent_rounds += int(len(opp_rounds))
            h.opponent_rounds_with_td_attempt += int((opp_rounds["td_attempted"] > 0).sum())
            h.opponent_td_attempted += float(opp_rounds["td_attempted"].sum())
            if "td_landed" in opp_rounds:
                h.opponent_td_landed += float(opp_rounds["td_landed"].sum())

    return pd.DataFrame(rows)


def _pair_opponents(rd: pd.DataFrame) -> pd.DataFrame:
    """Attach same-round opponent values with an explicit self-join."""
    value_cols = [c for c in ["td_attempted", *OPTIONAL_NUMERIC] if c in rd.columns]
    value_cols = list(dict.fromkeys(value_cols))
    left = rd.copy()
    opp = rd[["fight_id", "round", "fighter_id", *value_cols]].copy()
    opp = opp.rename(columns={
        "fighter_id": "opponent_id_round",
        **{c: f"opp__{c}" for c in value_cols},
    })
    z = left.merge(opp, on=["fight_id", "round"], how="inner")
    z = z[z["fighter_id"].ne(z["opponent_id_round"])].copy()
    counts = z.groupby(["fight_id", "fighter_id", "round"]).size()
    if not counts.eq(1).all():
        raise RuntimeError("could not uniquely pair fighter and opponent round rows")
    return z


def build_examples(rd: pd.DataFrame, prefight: pd.DataFrame) -> pd.DataFrame:
    """Create target rows: previous round context -> next-round TD choice."""
    z = _pair_opponents(rd)
    z = z.merge(
        prefight,
        on=["fight_id", "event_date", "fighter_id", "corner_norm"],
        how="inner",
        validate="many_to_one",
    )

    # We predict round r+1 from the observed state through round r.  Shift only
    # within the same fighter/fight, then require true consecutive rounds.
    z = z.sort_values(["fight_id", "fighter_id", "round"]).copy()
    g = z.groupby(["fight_id", "fighter_id"], sort=False)
    z["next_round"] = g["round"].shift(-1)
    z["next_td_attempted"] = g["td_attempted"].shift(-1)
    z = z[z["next_round"].eq(z["round"] + 1)].copy()
    z["attempted_td_next_round"] = (z["next_td_attempted"] > 0).astype(int)
    z["target_round"] = z["next_round"].astype(int)

    # Previous-round context.  All values below are from round r, never r+1.
    z["prev_own_td_attempted"] = z["td_attempted"]
    z["prev_own_attempted_td"] = (z["td_attempted"] > 0).astype(float)
    z["prev_opp_td_attempted"] = z["opp__td_attempted"]
    z["prev_opp_attempted_td"] = (z["opp__td_attempted"] > 0).astype(float)

    for c in OPTIONAL_NUMERIC:
        if c in z.columns:
            z[f"prev_own__{c}"] = z[c]
        oc = f"opp__{c}"
        if oc in z.columns:
            z[f"prev_opp__{c}"] = z[oc]

    if "td_landed" in z.columns:
        z["prev_own_td_failed"] = (z["td_attempted"] - z["td_landed"]).clip(lower=0)
        z["prev_own_td_success_rate"] = np.where(
            z["td_attempted"] > 0,
            z["td_landed"] / z["td_attempted"],
            0.0,
        )
    if "opp__td_landed" in z.columns:
        z["prev_opp_td_failed"] = (z["opp__td_attempted"] - z["opp__td_landed"]).clip(lower=0)

    # Differential proxies are behavioral context, not claimed judging state.
    for base in ("distance_landed", "sig_str_landed", "total_str_landed", "ctrl_sec", "kd"):
        own, opp = base, f"opp__{base}"
        if own in z.columns and opp in z.columns:
            z[f"prev_diff__{base}"] = z[own] - z[opp]

    return z.sort_values(["event_date", "fight_id", "fighter_id", "round"]).reset_index(drop=True)


def whole_fight_split(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fights = (
        examples[["fight_id", "event_date"]]
        .drop_duplicates()
        .sort_values(["event_date", "fight_id"])
        .reset_index(drop=True)
    )
    if len(fights) < 10:
        raise RuntimeError(f"too few fights for chronological validation: {len(fights)}")
    cut = max(1, min(len(fights) - 1, int(TRAIN_FRACTION * len(fights))))
    train_ids = set(fights.iloc[:cut].fight_id)
    test_ids = set(fights.iloc[cut:].fight_id)
    tr = examples[examples.fight_id.isin(train_ids)].copy()
    te = examples[examples.fight_id.isin(test_ids)].copy()
    meta = {
        "fights": int(len(fights)),
        "train_fights": int(len(train_ids)),
        "test_fights": int(len(test_ids)),
        "train_end_date": str(fights.iloc[cut - 1].event_date.date()),
        "test_start_date": str(fights.iloc[cut].event_date.date()),
    }
    return tr, te, meta


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_prediction": float(np.mean(p)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=4000),
    )
    model.fit(tr[cols], tr["attempted_td_next_round"])
    return model.predict_proba(te[cols])[:, 1]


def evaluate(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    tr, te, split_meta = whole_fight_split(examples)
    ytr = tr["attempted_td_next_round"].to_numpy(dtype=int)
    yte = te["attempted_td_next_round"].to_numpy(dtype=int)

    fighter_cols = [
        "fighter_prior_fights",
        "fighter_prior_rounds",
        "fighter_td_round_rate",
        "fighter_td_attempts_per_round",
        "fighter_td_land_rate",
        "target_round",
    ]
    matchup_cols = fighter_cols + [
        "opponent_prior_fights",
        "opponent_prior_rounds",
        "opponent_td_attempt_round_exposure",
        "opponent_td_attempts_faced_per_round",
        "opponent_td_allowed_rate",
        "opponent_td_denial_rate",
    ]
    context_candidates = [
        "prev_own_td_attempted",
        "prev_own_attempted_td",
        "prev_opp_td_attempted",
        "prev_opp_attempted_td",
        "prev_own_td_failed",
        "prev_own_td_success_rate",
        "prev_opp_td_failed",
        "prev_own__td_landed",
        "prev_opp__td_landed",
        "prev_own__distance_attempted",
        "prev_opp__distance_attempted",
        "prev_own__distance_landed",
        "prev_opp__distance_landed",
        "prev_own__clinch_attempted",
        "prev_opp__clinch_attempted",
        "prev_own__ground_attempted",
        "prev_opp__ground_attempted",
        "prev_own__ctrl_sec",
        "prev_opp__ctrl_sec",
        "prev_own__kd",
        "prev_opp__kd",
        "prev_diff__distance_landed",
        "prev_diff__sig_str_landed",
        "prev_diff__total_str_landed",
        "prev_diff__ctrl_sec",
        "prev_diff__kd",
    ]
    context_cols = [c for c in context_candidates if c in examples.columns]
    dynamic_cols = matchup_cols + context_cols

    p0 = np.full(len(te), float(np.mean(ytr)))
    specs = [
        ("A_population", None, p0),
        ("B_fighter_history", fighter_cols, None),
        ("C_fighter_plus_opponent", matchup_cols, None),
        ("D_plus_previous_round_context", dynamic_cols, None),
    ]

    rows = []
    prediction_rows = te[[
        "fight_id", "event_date", "fighter_id", "opponent_id", "corner_norm",
        "round", "target_round", "attempted_td_next_round",
    ]].copy()
    for label, cols, fixed in specs:
        p = fixed if fixed is not None else _fit_predict(tr, te, cols or [])
        rows.append({"model": label, "feature_count": 0 if cols is None else len(cols), **_score(yte, p)})
        prediction_rows[f"p__{label}"] = p

    metrics = pd.DataFrame(rows)
    c = metrics.loc[metrics.model.eq("C_fighter_plus_opponent")].iloc[0]
    metrics["delta_auc_vs_C"] = metrics["auc"] - float(c.auc)
    metrics["delta_brier_vs_C"] = metrics["brier"] - float(c.brier)
    metrics["delta_log_loss_vs_C"] = metrics["log_loss"] - float(c.log_loss)

    split_meta.update({
        "train_examples": int(len(tr)),
        "test_examples": int(len(te)),
        "train_positive_rate": float(np.mean(ytr)),
        "test_positive_rate": float(np.mean(yte)),
        "context_features": context_cols,
    })
    return metrics, prediction_rows, split_meta


def persistence_table(examples: pd.DataFrame) -> pd.DataFrame:
    """Simple OOS-agnostic descriptive adaptation checks, not model scores."""
    rows = []
    conditions = {
        "all_examples": np.ones(len(examples), dtype=bool),
        "attempted_td_previous_round": examples["prev_own_td_attempted"].gt(0).to_numpy(),
        "no_td_attempt_previous_round": examples["prev_own_td_attempted"].eq(0).to_numpy(),
    }
    if "prev_own_td_failed" in examples:
        conditions["attempted_and_failed_all_previous_tds"] = (
            examples["prev_own_td_attempted"].gt(0)
            & examples.get("prev_own__td_landed", pd.Series(0, index=examples.index)).eq(0)
        ).to_numpy()
        conditions["landed_td_previous_round"] = examples.get(
            "prev_own__td_landed", pd.Series(0, index=examples.index)
        ).gt(0).to_numpy()
    for label, mask in conditions.items():
        sub = examples.loc[mask]
        rows.append({
            "condition": label,
            "n": int(len(sub)),
            "next_round_td_attempt_rate": float(sub["attempted_td_next_round"].mean()) if len(sub) else np.nan,
            "mean_next_round_td_attempts": float(sub["next_td_attempted"].mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    rd, numeric = load_rounds()
    prefight = build_prefight_snapshots(rd, numeric)
    examples = build_examples(rd, prefight)
    metrics, predictions, split_meta = evaluate(examples)
    persistence = persistence_table(examples)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    persistence.to_csv(OUT / "persistence_adaptation.csv", index=False)
    pd.DataFrame([split_meta]).to_json(OUT / "split_summary.json", orient="records", indent=2)

    print("TD POLICY LEARNABILITY — NEXT-ROUND TD ATTEMPT")
    print(
        f"examples={len(examples)} fights={split_meta['fights']} "
        f"train={split_meta['train_examples']} test={split_meta['test_examples']}"
    )
    print(
        f"chronological split: train through {split_meta['train_end_date']} | "
        f"test from {split_meta['test_start_date']}"
    )
    print("\nMODEL COMPARISON")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPREVIOUS-ROUND PERSISTENCE / ADAPTATION DESCRIPTIVES")
    print(persistence.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDYNAMIC CONTEXT FEATURES")
    print("\n".join(f"- {c}" for c in split_meta["context_features"]))

    d = metrics.loc[metrics.model.eq("D_plus_previous_round_context")].iloc[0]
    c = metrics.loc[metrics.model.eq("C_fighter_plus_opponent")].iloc[0]
    print("\nPRIMARY INCREMENT D -> C")
    print(f"delta_auc={float(d.auc-c.auc):+.5f}")
    print(f"delta_brier={float(d.brier-c.brier):+.5f}  (negative is better)")
    print(f"delta_log_loss={float(d.log_loss-c.log_loss):+.5f}  (negative is better)")


if __name__ == "__main__":
    main()
