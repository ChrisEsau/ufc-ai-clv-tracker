"""Research-only OOS validation of a clinch-equivalent entry proxy.

UFCStats does not record literal clinch entries. This study therefore does NOT
claim to recover true entry counts. It uses clean fighter-rounds with no landed
takedowns and no ground significant-strike attempts by either fighter, where
clinch significant-strike attempts and credited CTRL seconds are the best
available observables of clinch involvement.

A dimensionless 'clinch-equivalent episode' proxy is defined from pre-2025
positive clean rounds by normalizing clinch strike attempts and control seconds
to their pre-2025 positive medians and taking the larger signal. Any positive
clinch signal has a floor of one equivalent episode. This is intended only as a
relative timing target for research, not as observed entry count.

Point-in-time fighter tendency and opponent allowance histories are evaluated on
untouched 2025-2026 clean rounds. Production is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

DATA = Path("data/fight_details/ufc_round_stats.parquet")
OUTDIR = Path("data/research/clinch_entry_proxy_oos")
CUTOFF = pd.Timestamp("2025-01-01")
ALPHA = 0.50
SHRINK_N = 10.0
EPS = 1e-12


def _col(df: pd.DataFrame, *names: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise RuntimeError(f"missing required column; tried {names}; available={list(df.columns)}")


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    fight = _col(raw, "fight_id", "bout_id")
    rnd = _col(raw, "round", "round_number")
    fighter = _col(raw, "fighter_name")
    opponent = _col(raw, "opponent_name")
    date = _col(raw, "event_date", "date")
    td_land = _col(raw, "td_landed")
    ctrl = _col(raw, "ctrl_sec", "control_seconds")
    clinch_att = _col(raw, "clinch_attempted", "clinch_att")
    ground_att = _col(raw, "ground_attempted", "ground_att")

    x = raw[[fight, rnd, fighter, opponent, date, td_land, ctrl, clinch_att, ground_att]].copy()
    x.columns = ["fight_id", "round", "fighter_name", "opponent_name", "event_date", "td_landed", "ctrl_sec", "clinch_att", "ground_att"]
    x["event_date"] = pd.to_datetime(x["event_date"], errors="coerce").dt.normalize()
    for c in ["td_landed", "ctrl_sec", "clinch_att", "ground_att"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["event_date"]).sort_values(["event_date", "fight_id", "round", "fighter_name"]).reset_index(drop=True)

    # Attach opponent same-round ground/TD activity so the clean restriction is bout-round level.
    opp = x[["fight_id", "round", "fighter_name", "td_landed", "ground_att"]].rename(columns={
        "fighter_name": "opponent_name_join", "td_landed": "opp_td_landed", "ground_att": "opp_ground_att"
    })
    x = x.merge(
        opp,
        left_on=["fight_id", "round", "opponent_name"],
        right_on=["fight_id", "round", "opponent_name_join"],
        how="left",
    )
    x["opp_td_landed"] = x["opp_td_landed"].fillna(0.0)
    x["opp_ground_att"] = x["opp_ground_att"].fillna(0.0)
    x["clean_zero_ground_round"] = (
        (x["td_landed"] <= 0) & (x["opp_td_landed"] <= 0) &
        (x["ground_att"] <= 0) & (x["opp_ground_att"] <= 0)
    )
    return x[x["clean_zero_ground_round"]].copy()


def _ewm(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = float(vals[0])
    for v in vals[1:]:
        s = ALPHA * float(v) + (1.0 - ALPHA) * s
    return s


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def main() -> None:
    clean = _prepare(pd.read_parquet(DATA))
    train = clean[clean.event_date < CUTOFF].copy()
    if train.empty:
        raise RuntimeError("empty pre-2025 clean set")

    positive = train[(train.clinch_att > 0) | (train.ctrl_sec > 0)]
    strike_scale = float(positive.loc[positive.clinch_att > 0, "clinch_att"].median())
    ctrl_scale = float(positive.loc[positive.ctrl_sec > 0, "ctrl_sec"].median())
    strike_scale = max(strike_scale, 1.0)
    ctrl_scale = max(ctrl_scale, 1.0)

    # Relative clinch-equivalent episodes, explicitly not literal entries.
    clean["actual_any_clinch"] = ((clean.clinch_att > 0) | (clean.ctrl_sec > 0)).astype(int)
    raw_proxy = np.maximum(clean.clinch_att / strike_scale, clean.ctrl_sec / ctrl_scale)
    clean["actual_clinch_equiv"] = np.where(clean.actual_any_clinch > 0, np.maximum(1.0, raw_proxy), 0.0)

    global_any = float(train.assign(any=((train.clinch_att > 0) | (train.ctrl_sec > 0)).astype(float))["any"].mean())
    # Recompute global proxy on train with the frozen pre-2025 normalization.
    train_proxy = np.where(
        ((train.clinch_att > 0) | (train.ctrl_sec > 0)),
        np.maximum(1.0, np.maximum(train.clinch_att / strike_scale, train.ctrl_sec / ctrl_scale)),
        0.0,
    )
    global_equiv = float(np.mean(train_proxy))

    tendency_any: dict[str, list[float]] = {}
    allowance_any: dict[str, list[float]] = {}
    tendency_eq: dict[str, list[float]] = {}
    allowance_eq: dict[str, list[float]] = {}
    out = []

    for r in clean.itertuples(index=False):
        f, o = str(r.fighter_name), str(r.opponent_name)
        any_y = float(r.actual_any_clinch)
        eq_y = float(r.actual_clinch_equiv)

        fa = tendency_any.get(f, [])
        oa = allowance_any.get(o, [])
        fe = tendency_eq.get(f, [])
        oe = allowance_eq.get(o, [])

        fa_ewm = global_any if not fa else float(_ewm(fa))
        oa_ewm = global_any if not oa else float(_ewm(oa))
        fe_ewm = global_equiv if not fe else float(_ewm(fe))
        oe_ewm = global_equiv if not oe else float(_ewm(oe))

        fa_shr = (len(fa) * (np.mean(fa) if fa else global_any) + SHRINK_N * global_any) / (len(fa) + SHRINK_N)
        oa_shr = (len(oa) * (np.mean(oa) if oa else global_any) + SHRINK_N * global_any) / (len(oa) + SHRINK_N)
        fe_shr = (len(fe) * (np.mean(fe) if fe else global_equiv) + SHRINK_N * global_equiv) / (len(fe) + SHRINK_N)
        oe_shr = (len(oe) * (np.mean(oe) if oe else global_equiv) + SHRINK_N * global_equiv) / (len(oe) + SHRINK_N)

        out.append({
            "event_date": r.event_date, "fight_id": r.fight_id, "round": int(r.round),
            "fighter_name": f, "opponent_name": o,
            "clinch_att": float(r.clinch_att), "ctrl_sec": float(r.ctrl_sec),
            "actual_any_clinch": int(any_y), "actual_clinch_equiv": eq_y,
            "pred_any_global": global_any,
            "pred_any_ewm_geom": float(np.sqrt(max(fa_ewm, EPS) * max(oa_ewm, EPS))),
            "pred_any_shrunk_geom": float(np.sqrt(max(fa_shr, EPS) * max(oa_shr, EPS))),
            "pred_equiv_global": global_equiv,
            "pred_equiv_ewm_geom": float(np.sqrt(max(fe_ewm, EPS) * max(oe_ewm, EPS))),
            "pred_equiv_shrunk_geom": float(np.sqrt(max(fe_shr, EPS) * max(oe_shr, EPS))),
            "fighter_prior_n": len(fe), "opponent_prior_n": len(oe),
        })

        tendency_any.setdefault(f, []).append(any_y)
        allowance_any.setdefault(o, []).append(any_y)
        tendency_eq.setdefault(f, []).append(eq_y)
        allowance_eq.setdefault(o, []).append(eq_y)

    pit = pd.DataFrame(out)
    hold = pit[pit.event_date >= CUTOFF].copy()
    if hold.empty:
        raise RuntimeError("empty 2025-26 holdout")

    yb = hold.actual_any_clinch.to_numpy(int)
    yc = hold.actual_clinch_equiv.to_numpy(float)
    binary = {}
    for c in ["pred_any_global", "pred_any_ewm_geom", "pred_any_shrunk_geom"]:
        p = np.clip(hold[c].to_numpy(float), 1e-6, 1 - 1e-6)
        binary[c] = {
            "auc": _safe_auc(yb, p),
            "brier": float(brier_score_loss(yb, p)),
            "log_loss": float(log_loss(yb, p, labels=[0, 1])),
            "mean_pred": float(np.mean(p)), "actual_rate": float(np.mean(yb)),
        }
    continuous = {}
    for c in ["pred_equiv_global", "pred_equiv_ewm_geom", "pred_equiv_shrunk_geom"]:
        p = hold[c].to_numpy(float)
        continuous[c] = {
            "mae": float(mean_absolute_error(yc, p)),
            "rmse": float(mean_squared_error(yc, p) ** 0.5),
            "corr": float(np.corrcoef(yc, p)[0, 1]) if np.std(p) > 0 and np.std(yc) > 0 else None,
            "mean_pred": float(np.mean(p)), "mean_actual": float(np.mean(yc)),
        }

    report = {
        "study": "clean-round clinch-equivalent entry proxy OOS validation",
        "production_changed": False,
        "source": str(DATA), "cutoff": str(CUTOFF.date()),
        "clean_definition": "both fighters td_landed=0 and ground significant-strike attempts=0 in the round",
        "important_limitation": "proxy is NOT literal clinch-entry count; UFCStats has no entry count or phase-time field",
        "proxy_definition": "max(1, clinch_att/pre2025_positive_median_clinch_att, ctrl_sec/pre2025_positive_median_ctrl_sec) when either signal >0; otherwise 0",
        "pre2025_positive_median_clinch_att": strike_scale,
        "pre2025_positive_median_ctrl_sec": ctrl_scale,
        "global_pre2025_any_clinch_rate": global_any,
        "global_pre2025_clinch_equiv_mean": global_equiv,
        "train_clean_fighter_rounds": int((pit.event_date < CUTOFF).sum()),
        "holdout_clean_fighter_rounds": int(len(hold)),
        "binary_any_clinch_models": binary,
        "continuous_equiv_models": continuous,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pit.to_csv(OUTDIR / "fighter_round_predictions.csv", index=False)
    (OUTDIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
