"""Measurement-only next-round takedown policy learnability diagnostic.

Question: can we predict whether a UFC fighter attempts a takedown in the NEXT
round from leakage-safe prefight history, matchup information, and the immediately
preceding round?  This file does not modify FSR or Event Clock mechanics.

Nested chronological OOS comparison:
A population baseline
B fighter prefight TD tendency
C fighter + opponent prefight TD profile
D C + previous-round context
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
REQUIRED = {"fight_id", "event_date", "corner", "fighter_id", "round", "td_attempted"}
OPTIONAL = (
    "td_landed", "distance_attempted", "distance_landed", "clinch_attempted",
    "clinch_landed", "ground_attempted", "ground_landed", "sig_str_attempted",
    "sig_str_landed", "total_str_attempted", "total_str_landed", "ctrl_sec",
    "kd", "sub_attempted",
)


@dataclass
class History:
    fights: int = 0
    rounds: int = 0
    td_rounds: int = 0
    td_att: float = 0.0
    td_land: float = 0.0
    opp_rounds: int = 0
    opp_td_rounds: int = 0
    opp_td_att: float = 0.0
    opp_td_land: float = 0.0

    def own(self) -> dict[str, float]:
        return {
            "fighter_prior_fights": float(self.fights),
            "fighter_prior_rounds": float(self.rounds),
            "fighter_td_round_rate": self.td_rounds / self.rounds if self.rounds else np.nan,
            "fighter_td_attempts_per_round": self.td_att / self.rounds if self.rounds else np.nan,
            "fighter_td_land_rate": self.td_land / self.td_att if self.td_att > 0 else np.nan,
        }

    def faced(self) -> dict[str, float]:
        return {
            "opponent_prior_fights": float(self.fights),
            "opponent_prior_rounds": float(self.rounds),
            "opponent_td_attempt_round_exposure": self.opp_td_rounds / self.opp_rounds if self.opp_rounds else np.nan,
            "opponent_td_attempts_faced_per_round": self.opp_td_att / self.opp_rounds if self.opp_rounds else np.nan,
            "opponent_td_allowed_rate": self.opp_td_land / self.opp_td_att if self.opp_td_att > 0 else np.nan,
            "opponent_td_denial_rate": 1.0 - self.opp_td_land / self.opp_td_att if self.opp_td_att > 0 else np.nan,
        }


def load_rounds() -> pd.DataFrame:
    rd = pd.read_parquet(ROUND_PATH).copy()
    missing = sorted(REQUIRED - set(rd.columns))
    if missing:
        raise RuntimeError(f"round stats missing required columns: {missing}")
    rd["fight_id"] = rd.fight_id.astype(str)
    rd["fighter_id"] = rd.fighter_id.astype(str)
    rd["event_date"] = pd.to_datetime(rd.event_date, errors="coerce").dt.normalize()
    rd["round"] = pd.to_numeric(rd["round"], errors="coerce")
    rd["corner_norm"] = rd.corner.astype(str).str.strip().str.lower()
    bad = ~rd.corner_norm.isin(["red", "blue"])
    if bad.any():
        raise RuntimeError(f"unexpected corner values: {sorted(rd.loc[bad, 'corner'].astype(str).unique())}")
    rd = rd[rd.event_date.notna() & rd["round"].notna()].copy()
    rd["round"] = rd["round"].astype(int)
    for c in ["td_attempted", *[x for x in OPTIONAL if x in rd.columns]]:
        rd[c] = pd.to_numeric(rd[c], errors="coerce").fillna(0.0)
    dup = rd.duplicated(["fight_id", "fighter_id", "round"], keep=False)
    if dup.any():
        raise RuntimeError(f"duplicate fighter-round rows: {int(dup.sum())}")
    return rd.sort_values(["event_date", "fight_id", "round", "corner_norm"]).reset_index(drop=True)


def build_prefight(rd: pd.DataFrame) -> pd.DataFrame:
    histories: dict[str, History] = {}
    out: list[dict[str, object]] = []
    groups = {str(fid): g.copy() for fid, g in rd.groupby("fight_id", sort=False)}
    order = rd[["fight_id", "event_date"]].drop_duplicates().sort_values(["event_date", "fight_id"])

    for _, key in order.iterrows():
        fid = str(key.fight_id)
        fight = groups[fid]
        fighters = fight[["fighter_id", "corner_norm"]].drop_duplicates()
        if len(fighters) != 2 or fighters.corner_norm.nunique() != 2:
            continue
        red = str(fighters.loc[fighters.corner_norm.eq("red"), "fighter_id"].iloc[0])
        blue = str(fighters.loc[fighters.corner_norm.eq("blue"), "fighter_id"].iloc[0])

        for fighter, opponent, corner in ((red, blue, "red"), (blue, red, "blue")):
            h = histories.get(fighter, History())
            oh = histories.get(opponent, History())
            rec: dict[str, object] = {
                "fight_id": fid, "event_date": key.event_date,
                "fighter_id": fighter, "opponent_id": opponent, "corner_norm": corner,
            }
            rec.update(h.own())
            # These describe how prior opponents attacked the current opponent.
            rec.update(oh.faced())
            out.append(rec)

        by = {str(k): g for k, g in fight.groupby("fighter_id")}
        for fighter, opponent in ((red, blue), (blue, red)):
            own, opp = by[fighter], by[opponent]
            h = histories.setdefault(fighter, History())
            h.fights += 1
            h.rounds += len(own)
            h.td_rounds += int((own.td_attempted > 0).sum())
            h.td_att += float(own.td_attempted.sum())
            if "td_landed" in own.columns:
                h.td_land += float(own.td_landed.sum())
            h.opp_rounds += len(opp)
            h.opp_td_rounds += int((opp.td_attempted > 0).sum())
            h.opp_td_att += float(opp.td_attempted.sum())
            if "td_landed" in opp.columns:
                h.opp_td_land += float(opp.td_landed.sum())
    return pd.DataFrame(out)


def pair_rounds(rd: pd.DataFrame) -> pd.DataFrame:
    vals = list(dict.fromkeys(["td_attempted", *[c for c in OPTIONAL if c in rd.columns]]))
    # The raw source may already contain an opponent_id.  Drop it here so the
    # canonical prefight opponent_id can merge without suffixing/collision.
    left = rd.drop(columns=["opponent_id"], errors="ignore").copy()
    opp = rd[["fight_id", "round", "fighter_id", *vals]].rename(columns={
        "fighter_id": "opponent_id_round", **{c: f"opp__{c}" for c in vals}
    })
    z = left.merge(opp, on=["fight_id", "round"], how="inner")
    z = z[z.fighter_id.ne(z.opponent_id_round)].copy()
    counts = z.groupby(["fight_id", "fighter_id", "round"]).size()
    if not counts.eq(1).all():
        raise RuntimeError("could not uniquely pair same-round opponents")
    return z


def build_examples(rd: pd.DataFrame, prefight: pd.DataFrame) -> pd.DataFrame:
    z = pair_rounds(rd).merge(
        prefight,
        on=["fight_id", "event_date", "fighter_id", "corner_norm"],
        how="inner", validate="many_to_one",
    )
    if not z.opponent_id.astype(str).eq(z.opponent_id_round.astype(str)).all():
        raise RuntimeError("prefight opponent mapping disagrees with same-round pairing")

    z = z.sort_values(["fight_id", "fighter_id", "round"]).copy()
    g = z.groupby(["fight_id", "fighter_id"], sort=False)
    z["next_round"] = g["round"].shift(-1)
    z["next_td_attempted"] = g.td_attempted.shift(-1)
    z = z[z.next_round.eq(z["round"] + 1)].copy()
    z["target_round"] = z.next_round.astype(int)
    z["attempted_td_next_round"] = (z.next_td_attempted > 0).astype(int)

    z["prev_own_td_attempted"] = z.td_attempted
    z["prev_own_attempted_td"] = (z.td_attempted > 0).astype(float)
    z["prev_opp_td_attempted"] = z.opp__td_attempted
    z["prev_opp_attempted_td"] = (z.opp__td_attempted > 0).astype(float)
    for c in OPTIONAL:
        if c in z.columns:
            z[f"prev_own__{c}"] = z[c]
        if f"opp__{c}" in z.columns:
            z[f"prev_opp__{c}"] = z[f"opp__{c}"]
    if "td_landed" in z.columns:
        z["prev_own_td_failed"] = (z.td_attempted - z.td_landed).clip(lower=0)
        z["prev_own_td_success_rate"] = np.where(z.td_attempted > 0, z.td_landed / z.td_attempted, 0.0)
    if "opp__td_landed" in z.columns:
        z["prev_opp_td_failed"] = (z.opp__td_attempted - z.opp__td_landed).clip(lower=0)
    for c in ("distance_landed", "sig_str_landed", "total_str_landed", "ctrl_sec", "kd"):
        if c in z.columns and f"opp__{c}" in z.columns:
            z[f"prev_diff__{c}"] = z[c] - z[f"opp__{c}"]
    return z.sort_values(["event_date", "fight_id", "fighter_id", "round"]).reset_index(drop=True)


def split(ex: pd.DataFrame):
    # Split on unique dates so no same-event fight can appear on both sides.
    dates = np.array(sorted(ex.event_date.dropna().unique()))
    if len(dates) < 10:
        raise RuntimeError(f"too few event dates: {len(dates)}")
    cut = max(1, min(len(dates) - 1, int(TRAIN_FRACTION * len(dates))))
    boundary = dates[cut]
    tr = ex[ex.event_date < boundary].copy()
    te = ex[ex.event_date >= boundary].copy()
    meta = {
        "fights": int(ex.fight_id.nunique()),
        "train_fights": int(tr.fight_id.nunique()), "test_fights": int(te.fight_id.nunique()),
        "train_end_date": str(pd.Timestamp(dates[cut - 1]).date()),
        "test_start_date": str(pd.Timestamp(boundary).date()),
    }
    return tr, te, meta


def score(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {
        "n": len(y), "positive_rate": float(np.mean(y)), "mean_prediction": float(np.mean(p)),
        "accuracy": float(accuracy_score(y, p >= .5)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def fit_predict(tr, te, cols):
    m = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(C=1.0, max_iter=4000),
    )
    m.fit(tr[cols], tr.attempted_td_next_round)
    return m.predict_proba(te[cols])[:, 1]


def evaluate(ex: pd.DataFrame):
    tr, te, meta = split(ex)
    ytr = tr.attempted_td_next_round.to_numpy(int)
    yte = te.attempted_td_next_round.to_numpy(int)
    fighter = [
        "fighter_prior_fights", "fighter_prior_rounds", "fighter_td_round_rate",
        "fighter_td_attempts_per_round", "fighter_td_land_rate", "target_round",
    ]
    matchup = fighter + [
        "opponent_prior_fights", "opponent_prior_rounds", "opponent_td_attempt_round_exposure",
        "opponent_td_attempts_faced_per_round", "opponent_td_allowed_rate", "opponent_td_denial_rate",
    ]
    context_candidates = [
        "prev_own_td_attempted", "prev_own_attempted_td", "prev_opp_td_attempted",
        "prev_opp_attempted_td", "prev_own_td_failed", "prev_own_td_success_rate",
        "prev_opp_td_failed", "prev_own__td_landed", "prev_opp__td_landed",
        "prev_own__distance_attempted", "prev_opp__distance_attempted",
        "prev_own__distance_landed", "prev_opp__distance_landed",
        "prev_own__clinch_attempted", "prev_opp__clinch_attempted",
        "prev_own__ground_attempted", "prev_opp__ground_attempted",
        "prev_own__ctrl_sec", "prev_opp__ctrl_sec", "prev_own__kd", "prev_opp__kd",
        "prev_diff__distance_landed", "prev_diff__sig_str_landed",
        "prev_diff__total_str_landed", "prev_diff__ctrl_sec", "prev_diff__kd",
    ]
    context = [c for c in context_candidates if c in ex.columns]
    specs = [
        ("A_population", None),
        ("B_fighter_history", fighter),
        ("C_fighter_plus_opponent", matchup),
        ("D_plus_previous_round_context", matchup + context),
    ]
    rows = []
    pred = te[["fight_id", "event_date", "fighter_id", "opponent_id", "corner_norm", "round", "target_round", "attempted_td_next_round"]].copy()
    for label, cols in specs:
        p = np.full(len(te), float(ytr.mean())) if cols is None else fit_predict(tr, te, cols)
        rows.append({"model": label, "feature_count": 0 if cols is None else len(cols), **score(yte, p)})
        pred[f"p__{label}"] = p
    metrics = pd.DataFrame(rows)
    c = metrics[metrics.model.eq("C_fighter_plus_opponent")].iloc[0]
    metrics["delta_auc_vs_C"] = metrics.auc - c.auc
    metrics["delta_brier_vs_C"] = metrics.brier - c.brier
    metrics["delta_log_loss_vs_C"] = metrics.log_loss - c.log_loss
    meta.update({
        "train_examples": len(tr), "test_examples": len(te),
        "train_positive_rate": float(ytr.mean()), "test_positive_rate": float(yte.mean()),
        "context_features": context,
    })
    return metrics, pred, meta


def persistence(ex: pd.DataFrame) -> pd.DataFrame:
    landed = ex.get("prev_own__td_landed", pd.Series(0.0, index=ex.index))
    conditions = {
        "all_examples": pd.Series(True, index=ex.index),
        "attempted_td_previous_round": ex.prev_own_td_attempted.gt(0),
        "no_td_attempt_previous_round": ex.prev_own_td_attempted.eq(0),
        "attempted_and_failed_all_previous_tds": ex.prev_own_td_attempted.gt(0) & landed.eq(0),
        "landed_td_previous_round": landed.gt(0),
    }
    rows = []
    for name, mask in conditions.items():
        q = ex[mask]
        rows.append({
            "condition": name, "n": len(q),
            "next_round_td_attempt_rate": float(q.attempted_td_next_round.mean()) if len(q) else np.nan,
            "mean_next_round_td_attempts": float(q.next_td_attempted.mean()) if len(q) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    rd = load_rounds()
    prefight = build_prefight(rd)
    ex = build_examples(rd, prefight)
    metrics, predictions, meta = evaluate(ex)
    desc = persistence(ex)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    desc.to_csv(OUT / "persistence_adaptation.csv", index=False)
    pd.DataFrame([meta]).to_json(OUT / "split_summary.json", orient="records", indent=2)

    print("TD POLICY LEARNABILITY — NEXT-ROUND TD ATTEMPT")
    print(f"examples={len(ex)} fights={meta['fights']} train={meta['train_examples']} test={meta['test_examples']}")
    print(f"chronological split: train through {meta['train_end_date']} | test from {meta['test_start_date']}")
    print("\nMODEL COMPARISON")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPERSISTENCE / ADAPTATION DESCRIPTIVES")
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDYNAMIC CONTEXT FEATURES")
    print("\n".join(f"- {c}" for c in meta["context_features"]))
    c = metrics[metrics.model.eq("C_fighter_plus_opponent")].iloc[0]
    d = metrics[metrics.model.eq("D_plus_previous_round_context")].iloc[0]
    print("\nPRIMARY INCREMENT D VS C")
    print(f"delta_auc={d.auc-c.auc:+.5f}")
    print(f"delta_brier={d.brier-c.brier:+.5f}  (negative is better)")
    print(f"delta_log_loss={d.log_loss-c.log_loss:+.5f}  (negative is better)")


if __name__ == "__main__":
    main()
