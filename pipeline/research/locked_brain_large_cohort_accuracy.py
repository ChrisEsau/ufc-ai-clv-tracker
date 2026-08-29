#!/usr/bin/env python3
"""Research-only broad locked Brain accuracy diagnostic.

Runs the immutable bundle-backed locked Brain harness at low path count over the
most recent clean historical fights supported by all locked bundle inputs.
Scores winner, 3-way method, exact six-way outcome, Brier and log loss.
Production mechanics and the locked bundle are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, FILES

OUT = Path("data/diagnostics/locked_brain_large_cohort_accuracy")


def norm_method(x: object) -> str | None:
    s = str(x or "").upper()
    if "SUB" in s:
        return "SUB"
    if "KO" in s or "TKO" in s:
        return "KO/TKO"
    if "DEC" in s or "DECISION" in s:
        return "DEC"
    return None


def supported_fights(bundle: Path, n: int) -> pd.DataFrame:
    ewm = pd.read_parquet(bundle / FILES["ewm_fsr"])
    ko = pd.read_parquet(bundle / FILES["ko_prefight"])
    sub = pd.read_parquet(bundle / FILES["sub_prefight"])
    for x in (ewm, ko, sub):
        x["fight_id"] = x["fight_id"].astype(str)
    e_ok = set(ewm.groupby("fight_id").size().loc[lambda s: s.eq(2)].index)
    k_ok = set(ko.groupby("fight_id").size().loc[lambda s: s.eq(2)].index)
    s_ok = set(sub.groupby("fight_id").size().loc[lambda s: s.eq(2)].index)
    ids = e_ok & k_ok & s_ok

    m = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m = m[m.fight_id.isin(ids)].copy()
    date_col = "date" if "date" in m.columns else "event_date"
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
    m["actual_method"] = m["method"].map(norm_method)
    m = m[m.actual_method.notna()].copy()
    m = m[m["winner_id"].notna()].copy()
    # Exclude draws/NCs implicitly because no recognized winner/method combination remains.
    m = m.sort_values([date_col, "fight_id"], ascending=[False, False]).head(n).copy()
    return m


def run_one(fid: str, paths: int, bundle: Path) -> dict:
    subprocess.run([
        "python", "-m", "pipeline.research.locked_brain_mc",
        "--fight-id", fid, "--paths", str(paths), "--bundle-dir", str(bundle),
    ], check=True)
    src = Path("data/research/locked_brain_mc") / fid / "run" / "sim" / "results.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    return json.loads(src.read_text())


def get_summary(res: dict) -> list[dict]:
    if isinstance(res.get("summary"), list):
        return res["summary"]
    if isinstance(res.get("adjusted"), dict) and isinstance(res["adjusted"].get("summary"), list):
        return res["adjusted"]["summary"]
    for v in res.values():
        if isinstance(v, dict) and isinstance(v.get("summary"), list):
            return v["summary"]
    raise KeyError("could not locate fighter summary in locked results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", type=int, default=200)
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle_dir)
    cohort = supported_fights(bundle, args.fights)
    cohort.to_csv(OUT / "cohort.csv", index=False)

    rows, failures = [], []
    for i, r in enumerate(cohort.itertuples(index=False), 1):
        fid = str(r.fight_id)
        try:
            res = run_one(fid, args.paths, bundle)
            sim = get_summary(res)
            total = sum(int(x.get("decision_wins", 0)) + int(x.get("ko_tko_wins", 0)) + int(x.get("submission_wins", 0)) for x in sim)
            if total != args.paths:
                raise RuntimeError(f"terminal total {total} != paths {args.paths}")
            for x in sim:
                rows.append({
                    "fight_id": fid,
                    "fighter": str(x["fighter"]),
                    "ml": float(x["ml_probability"]),
                    "ko": float(x["ko_tko_probability"]),
                    "sub": float(x["submission_probability"]),
                    "dec": float(x["decision_probability"]),
                })
            print(f"[{i}/{len(cohort)}] OK {fid}", flush=True)
        except Exception as exc:
            failures.append({"fight_id": fid, "error": repr(exc)})
            print(f"[{i}/{len(cohort)}] FAIL {fid}: {exc}", flush=True)
        finally:
            p = Path("data/research/locked_brain_mc") / fid
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("no successful locked Brain results")

    # Map actual winner by fighter id through bundle target rows, which expose fighter_name/id.
    ewm = pd.read_parquet(bundle / FILES["ewm_fsr"])
    ewm["fight_id"] = ewm["fight_id"].astype(str)
    name_col = next(c for c in ("fighter_name", "name", "fighter") if c in ewm.columns)
    lookup = ewm[["fight_id", "fighter_id", name_col]].drop_duplicates().rename(columns={name_col:"fighter"})
    lookup["fighter_id"] = lookup["fighter_id"].astype(str)
    meta = cohort[["fight_id", "winner_id", "actual_method", "method"]].copy()
    meta["fight_id"] = meta["fight_id"].astype(str)
    meta["winner_id"] = meta["winner_id"].astype(str)
    meta = meta.merge(lookup, left_on=["fight_id","winner_id"], right_on=["fight_id","fighter_id"], how="left")
    meta = meta.rename(columns={"fighter":"actual_winner"})

    fight_rows = []
    for fid, g in pred.groupby("fight_id", sort=False):
        mm = meta[meta.fight_id.eq(fid)]
        if len(mm) != 1 or pd.isna(mm.iloc[0].actual_winner):
            failures.append({"fight_id":fid,"error":"actual winner name mapping failed"})
            continue
        actual_winner = str(mm.iloc[0].actual_winner)
        actual_method = str(mm.iloc[0].actual_method)
        fav = str(g.loc[g.ml.idxmax(), "fighter"])
        method_probs = {"KO/TKO":float(g.ko.sum()), "SUB":float(g.sub.sum()), "DEC":float(g.dec.sum())}
        pred_method = max(method_probs, key=method_probs.get)
        six = []
        for rr in g.itertuples(index=False):
            six += [(rr.ko, rr.fighter, "KO/TKO"), (rr.sub, rr.fighter, "SUB"), (rr.dec, rr.fighter, "DEC")]
        _, six_fighter, six_method = max(six, key=lambda z:z[0])
        actual_row = g[g.fighter.eq(actual_winner)]
        if len(actual_row) != 1:
            failures.append({"fight_id":fid,"error":"actual winner absent from sim summary"})
            continue
        p_actual = float(actual_row.iloc[0].ml)
        p_actual_method = method_probs[actual_method]
        p_exact = float(actual_row.iloc[0][{"KO/TKO":"ko","SUB":"sub","DEC":"dec"}[actual_method]])
        fight_rows.append({
            "fight_id":fid, "actual_winner":actual_winner, "actual_method":actual_method,
            "brain_favorite":fav, "winner_correct":fav==actual_winner,
            "p_actual_winner":p_actual,
            "pred_method":pred_method, "method_correct":pred_method==actual_method,
            "p_actual_method":p_actual_method,
            "pred_six_fighter":six_fighter, "pred_six_method":six_method,
            "six_way_correct":six_fighter==actual_winner and six_method==actual_method,
            "p_exact_outcome":p_exact,
            **{f"fight_{k.lower().replace('/','_')}_prob":v for k,v in method_probs.items()},
        })

    scored = pd.DataFrame(fight_rows)
    scored.to_csv(OUT / "fight_predictions.csv", index=False)
    pred.to_csv(OUT / "fighter_probabilities.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)

    p = np.clip(scored.p_actual_winner.to_numpy(float), 1e-12, 1-1e-12)
    metrics = {
        "requested_fights": int(args.fights), "paths_per_fight": int(args.paths),
        "successful_fights": int(len(scored)), "failures": int(len(failures)),
        "winner_accuracy": float(scored.winner_correct.mean()),
        "winner_brier": float(np.mean((1-p)**2)),
        "winner_log_loss": float(np.mean(-np.log(p))),
        "mean_p_actual_winner": float(p.mean()),
        "method_accuracy": float(scored.method_correct.mean()),
        "mean_p_actual_method": float(scored.p_actual_method.mean()),
        "six_way_accuracy": float(scored.six_way_correct.mean()),
        "mean_p_exact_outcome": float(scored.p_exact_outcome.mean()),
        "actual_method_share": scored.actual_method.value_counts(normalize=True).to_dict(),
        "pred_method_share": scored.pred_method.value_counts(normalize=True).to_dict(),
        "production_changed": False, "bundle_rebuilt": False,
    }
    (OUT / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
