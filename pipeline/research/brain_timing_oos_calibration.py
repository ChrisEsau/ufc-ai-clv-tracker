"""Research-only OOS calibration of observable fighter action cadence.

UFCStats does not expose true phase-time denominators, so this study does NOT
fit separate standing/clinch/ground clocks.  It calibrates the observable
fighter offensive-event cadence from round-level counts using only information
available at/after the round for the target and point-in-time history for
fighter-level prediction.

Observable offensive events = significant-strike attempts + takedown attempts
+ submission attempts.  This is intentionally a lower-bound comparator for the
Brain event clock because control/escape/reversal actions are not observed as
counts in UFCStats.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/fight_details/ufc_round_stats.parquet")
OUT = Path("data/research/brain_timing_oos_calibration/results.json")
CUTOFF = pd.Timestamp("2025-01-01")
ALPHA = 0.50
EPS = 1e-9


def col(df, *names):
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"missing any of {names}; columns={list(df.columns)}")


def num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def ewm_prior(vals):
    if not vals:
        return None
    x = float(vals[0])
    for v in vals[1:]:
        x = ALPHA * float(v) + (1.0 - ALPHA) * x
    return x


def build():
    d = pd.read_parquet(DATA).copy()
    date = col(d, "event_date", "date", "fight_date")
    fighter = col(d, "fighter_name", "fighter")
    elapsed = None
    for candidate in ("round_elapsed_seconds", "round_time_seconds", "elapsed_seconds", "round_duration_seconds"):
        if candidate in {str(c).lower(): c for c in d.columns}:
            elapsed = col(d, candidate)
            break
    if elapsed is None:
        # UFC completed rounds are 300s; partial final rounds can be recovered
        # from common round-time fields when available.
        round_num = col(d, "round", "round_number")
        d["_elapsed"] = 300.0
        # Try mm:ss clock fields if present.
        for candidate in ("round_time", "time", "match_time"):
            try:
                tc = col(d, candidate)
            except KeyError:
                continue
            def parse(v):
                if pd.isna(v): return np.nan
                s = str(v)
                if ":" in s:
                    a,b = s.split(":",1)
                    try: return 60*float(a)+float(b)
                    except: return np.nan
                try: return float(s)
                except: return np.nan
            parsed = d[tc].map(parse)
            # Only use plausible within-round elapsed values.
            mask = parsed.between(0,300)
            d.loc[mask,"_elapsed"] = parsed[mask]
            break
        elapsed = "_elapsed"

    sig_att = col(d, "sig_str_att", "sig_str_attempted", "sig_str_attempts", "significant_strikes_attempted")
    td_att = col(d, "td_att", "td_attempted", "td_attempts", "takedowns_attempted")
    try:
        sub_att = col(d, "sub_att", "sub_attempts", "submission_attempts")
        sub = num(d[sub_att])
    except KeyError:
        sub = pd.Series(0.0, index=d.index)

    f = pd.DataFrame({
        "event_date": pd.to_datetime(d[date], errors="coerce"),
        "fighter": d[fighter].astype(str),
        "elapsed": num(d[elapsed]).clip(lower=1.0, upper=300.0),
        "events": (num(d[sig_att]) + num(d[td_att]) + sub).clip(lower=0.0),
    }).dropna(subset=["event_date"])
    f["rate_per_sec"] = f.events / f.elapsed
    return f.sort_values(["event_date","fighter"]).reset_index(drop=True)


def main():
    f = build()
    train = f[f.event_date < CUTOFF].copy()
    test = f[f.event_date >= CUTOFF].copy()

    global_rate = float(train.events.sum() / max(train.elapsed.sum(), EPS))
    global_delay = 1.0 / max(global_rate, EPS)

    # Point-in-time fighter EWM of observed offensive rate; fall back global.
    history = {}
    preds = []
    for r in f.itertuples(index=False):
        hist = history.get(r.fighter, [])
        pr = ewm_prior(hist)
        preds.append(global_rate if pr is None else pr)
        history.setdefault(r.fighter, []).append(float(r.rate_per_sec))
    f["pred_rate"] = preds
    test = f[f.event_date >= CUTOFF].copy()

    actual_events = float(test.events.sum())
    pred_global = float((global_rate * test.elapsed).sum())
    pred_ewm = float((test.pred_rate * test.elapsed).sum())

    # Event-rate MAE at fighter-round level and timing interpretation.
    mae_global = float(np.mean(np.abs(test.rate_per_sec - global_rate)))
    mae_ewm = float(np.mean(np.abs(test.rate_per_sec - test.pred_rate)))

    # Quartile calibration of point-in-time predicted rate.
    test["q"] = pd.qcut(test.pred_rate.rank(method="first"), 4, labels=False)
    buckets = []
    for q,g in test.groupby("q"):
        buckets.append({
            "quartile": int(q)+1,
            "rows": int(len(g)),
            "pred_events_per_15": float(g.pred_rate.mean()*900),
            "actual_events_per_15": float(g.events.sum()/g.elapsed.sum()*900),
        })

    out = {
        "study":"Brain observable offensive timing OOS calibration",
        "production_changed":False,
        "target_semantics":"sig strike attempts + TD attempts + submission attempts per fighter-round",
        "important_limitation":"No true phase-time denominators; cannot empirically fit separate standing/clinch/ground delays from UFCStats round data.",
        "cutoff":str(CUTOFF.date()),
        "train_rows":int(len(train)),
        "holdout_rows":int(len(test)),
        "global_train_events_per_15":float(global_rate*900),
        "global_train_mean_seconds_per_observable_offensive_event":float(global_delay),
        "holdout_actual_events_per_15":float(actual_events/test.elapsed.sum()*900),
        "holdout_global_E_over_O":pred_global/actual_events if actual_events else None,
        "holdout_fighter_ewm_E_over_O":pred_ewm/actual_events if actual_events else None,
        "fighter_round_rate_mae_global_per_sec":mae_global,
        "fighter_round_rate_mae_ewm_per_sec":mae_ewm,
        "fighter_ewm_quartiles":buckets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__ == "__main__": main()
