from __future__ import annotations

"""Build a research-only, leakage-safe prefight feature bank from UFC round stats.

No current-fight statistics are exposed as model features. Fighter history is
updated only after every fight on a calendar date has been snapshotted, so two
same-date fights always share the same prior state.
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = Path("pipeline/research/raw_signal_discovery_v1/config.yaml")
COUNT_COLUMNS = (
    "kd", "sig_str_landed", "sig_str_attempted", "total_str_landed",
    "total_str_attempted", "td_landed", "td_attempted", "sub_att", "rev",
    "ctrl_sec", "head_landed", "head_attempted", "body_landed",
    "body_attempted", "leg_landed", "leg_attempted", "distance_landed",
    "distance_attempted", "clinch_landed", "clinch_attempted",
    "ground_landed", "ground_attempted",
)
ACCURACY_PAIRS = (
    ("sig_acc", "sig_str_landed", "sig_str_attempted"),
    ("total_acc", "total_str_landed", "total_str_attempted"),
    ("td_acc", "td_landed", "td_attempted"),
    ("head_acc", "head_landed", "head_attempted"),
    ("body_acc", "body_landed", "body_attempted"),
    ("leg_acc", "leg_landed", "leg_attempted"),
    ("distance_acc", "distance_landed", "distance_attempted"),
    ("clinch_acc", "clinch_landed", "clinch_attempted"),
    ("ground_acc", "ground_landed", "ground_attempted"),
)
ROUND_PATTERN_COLUMNS = (
    "sig_str_attempted", "sig_str_landed", "td_attempted", "ctrl_sec",
    "distance_attempted", "clinch_attempted", "ground_attempted",
)
IDENTITY_COLUMNS = {
    "fight_id", "event_date", "fighter_id", "opponent_id", "fighter_win",
    "history_max_date", "opponent_history_max_date",
}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("raw signal discovery config must be a mapping")
    return payload


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if np.isfinite(n) and np.isfinite(d) and d > 0 else np.nan


def _parse_measure(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in {"--", "nan", "none"}:
        return np.nan
    if "'" in text:
        cleaned = text.replace('"', "").replace("in", "")
        feet_text, _, inches_text = cleaned.partition("'")
        try:
            return 12.0 * float(feet_text.strip()) + float(inches_text.strip() or 0)
        except ValueError:
            return np.nan
    numeric = "".join(ch if ch.isdigit() or ch in ".-" else " " for ch in text)
    parts = [p for p in numeric.split() if p]
    try:
        return float(parts[0]) if parts else np.nan
    except ValueError:
        return np.nan


def _age_years(dob: object, event_date: pd.Timestamp) -> float:
    dob_ts = pd.to_datetime(dob, errors="coerce")
    return float((event_date - dob_ts).days / 365.25) if pd.notna(dob_ts) else np.nan


def _elapsed_seconds(master: pd.DataFrame) -> pd.Series:
    finish_round = pd.to_numeric(master["finish_round"], errors="coerce")
    match_time = pd.to_numeric(master["match_time_sec"], errors="coerce")
    elapsed = (finish_round - 1.0) * 300.0 + match_time
    scheduled = pd.to_numeric(master["total_rounds"], errors="coerce") * 300.0
    elapsed = elapsed.where(elapsed.gt(0), scheduled)
    return elapsed.clip(lower=1.0)


def _build_fight_observations(rounds: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    required_round = {"fight_id", "fighter_id", "opponent_id", "round", *COUNT_COLUMNS}
    missing = sorted(required_round - set(rounds.columns))
    if missing:
        raise ValueError(f"round stats missing required discovery columns: {missing}")
    required_master = {
        "fight_id", "date", "winner_id", "finish_round", "match_time_sec",
        "total_rounds", "r_id", "b_id",
    }
    missing = sorted(required_master - set(master.columns))
    if missing:
        raise ValueError(f"master missing required discovery columns: {missing}")

    r = rounds.copy()
    r["fight_id"] = r["fight_id"].astype(str)
    r["fighter_id"] = r["fighter_id"].astype(str)
    r["opponent_id"] = r["opponent_id"].astype(str)
    r["round"] = pd.to_numeric(r["round"], errors="coerce")
    for c in COUNT_COLUMNS:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["event_date"] = pd.to_datetime(m["date"], errors="raise").dt.normalize()
    m["elapsed_seconds"] = _elapsed_seconds(m)
    meta = m[["fight_id", "event_date", "elapsed_seconds"]].drop_duplicates("fight_id")
    r = r.merge(meta, on="fight_id", how="inner", validate="many_to_one")

    output: list[dict[str, Any]] = []
    for (fight_id, fighter_id), g in r.groupby(["fight_id", "fighter_id"], sort=False):
        g = g.sort_values("round")
        first = g.iloc[0]
        elapsed = float(first["elapsed_seconds"])
        row: dict[str, Any] = {
            "fight_id": str(fight_id),
            "fighter_id": str(fighter_id),
            "opponent_id": str(first["opponent_id"]),
            "event_date": pd.Timestamp(first["event_date"]),
            "elapsed_seconds": elapsed,
            "rounds_observed": float(g["round"].nunique()),
        }
        totals: dict[str, float] = {}
        for c in COUNT_COLUMNS:
            totals[c] = float(g[c].fillna(0).sum())
            row[f"{c}_per15"] = totals[c] / elapsed * 900.0
        for name, landed, attempted in ACCURACY_PAIRS:
            row[name] = _safe_div(totals[landed], totals[attempted])
        sig_attempts = totals["sig_str_attempted"]
        for loc in ("head", "body", "leg", "distance", "clinch", "ground"):
            row[f"{loc}_attempt_share"] = _safe_div(totals[f"{loc}_attempted"], sig_attempts)
        row["kd_per100_sig_landed"] = 100.0 * _safe_div(totals["kd"], totals["sig_str_landed"])
        row["control_share"] = float(np.clip(totals["ctrl_sec"] / elapsed, 0.0, 1.0))
        row["ground_attempts_per_control_min"] = 60.0 * _safe_div(totals["ground_attempted"], totals["ctrl_sec"])
        row["sub_attempts_per_control_min"] = 60.0 * _safe_div(totals["sub_att"], totals["ctrl_sec"])

        indexed = g.set_index("round")
        for c in ROUND_PATTERN_COLUMNS:
            for rnd in (1, 2, 3):
                row[f"r{rnd}_{c}"] = float(indexed.loc[rnd, c]) if rnd in indexed.index else np.nan
            if np.isfinite(row[f"r1_{c}"]):
                row[f"r2_minus_r1_{c}"] = (
                    row[f"r2_{c}"] - row[f"r1_{c}"] if np.isfinite(row[f"r2_{c}"]) else np.nan
                )
                row[f"r3_minus_r1_{c}"] = (
                    row[f"r3_{c}"] - row[f"r1_{c}"] if np.isfinite(row[f"r3_{c}"]) else np.nan
                )
            else:
                row[f"r2_minus_r1_{c}"] = np.nan
                row[f"r3_minus_r1_{c}"] = np.nan
        output.append(row)

    obs = pd.DataFrame(output)
    return obs.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _vector_summary(arr: np.ndarray, metrics: list[str], prefix: str, aggs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if arr.size == 0:
        for agg in aggs:
            for metric in metrics:
                out[f"hist_{prefix}_{metric}_{agg}"] = np.nan
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if "mean" in aggs:
            vals = np.nanmean(arr, axis=0)
            out.update({f"hist_{prefix}_{m}_mean": float(v) for m, v in zip(metrics, vals)})
        if "std" in aggs:
            vals = np.nanstd(arr, axis=0)
            out.update({f"hist_{prefix}_{m}_std": float(v) for m, v in zip(metrics, vals)})
        if "median" in aggs:
            vals = np.nanmedian(arr, axis=0)
            out.update({f"hist_{prefix}_{m}_median": float(v) for m, v in zip(metrics, vals)})
        if "min" in aggs:
            vals = np.nanmin(arr, axis=0)
            out.update({f"hist_{prefix}_{m}_min": float(v) for m, v in zip(metrics, vals)})
        if "max" in aggs:
            vals = np.nanmax(arr, axis=0)
            out.update({f"hist_{prefix}_{m}_max": float(v) for m, v in zip(metrics, vals)})
    if "slope" in aggs:
        x = np.arange(arr.shape[0], dtype=float)[:, None]
        mask = np.isfinite(arr)
        count = mask.sum(axis=0).astype(float)
        safe_count = np.where(count > 0, count, 1.0)
        x_mean = (mask * x).sum(axis=0) / safe_count
        y0 = np.where(mask, arr, 0.0)
        y_mean = y0.sum(axis=0) / safe_count
        dx = x - x_mean
        num = np.where(mask, dx * (arr - y_mean), 0.0).sum(axis=0)
        den = np.where(mask, dx * dx, 0.0).sum(axis=0)
        vals = np.where((count >= 2) & (den > 0), num / den, np.nan)
        out.update({f"hist_{prefix}_{m}_slope": float(v) for m, v in zip(metrics, vals)})
    return out


def _profile_features(master_row: pd.Series, fighter_id: str, event_date: pd.Timestamp) -> dict[str, float]:
    side = "r" if str(master_row["r_id"]) == fighter_id else "b"
    stance = str(master_row.get(f"{side}_stance", "")).strip().lower()
    return {
        "profile_age": _age_years(master_row.get(f"{side}_dob"), event_date),
        "profile_height": _parse_measure(master_row.get(f"{side}_height")),
        "profile_reach": _parse_measure(master_row.get(f"{side}_reach")),
        "profile_weight": _parse_measure(master_row.get(f"{side}_weight")),
        "profile_stance_orthodox": float("orthodox" in stance),
        "profile_stance_southpaw": float("southpaw" in stance),
        "profile_stance_switch": float("switch" in stance),
        "profile_stance_unknown": float(not stance or stance in {"--", "nan", "none"}),
    }


def _build_prefight_snapshots(obs: pd.DataFrame, master: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fcfg = config["feature_factory"]
    fight_windows = [int(x) for x in fcfg["fight_windows"]]
    day_windows = [int(x) for x in fcfg["day_windows"]]
    aggs = [str(x) for x in fcfg["aggregations"]]

    metric_columns = [
        c for c in obs.columns
        if c not in {"fight_id", "fighter_id", "opponent_id", "event_date"}
    ]
    vectors = {
        str(row.fight_id) + "|" + str(row.fighter_id): row[metric_columns].to_numpy(dtype=float)
        for _, row in obs.iterrows()
    }
    obs_lookup = {
        pd.Timestamp(d): g.copy()
        for d, g in obs.groupby("event_date", sort=True)
    }

    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["event_date"] = pd.to_datetime(m["date"], errors="raise").dt.normalize()
    m = m[m["winner_id"].notna()].copy()
    m = m[m.apply(lambda x: str(x["winner_id"]) in {str(x["r_id"]), str(x["b_id"])}, axis=1)]
    m = m.sort_values(["event_date", "fight_id"]).reset_index(drop=True)

    history: dict[str, list[tuple[pd.Timestamp, np.ndarray]]] = defaultdict(list)
    snapshots: list[dict[str, Any]] = []

    for event_date, date_fights in m.groupby("event_date", sort=True):
        event_date = pd.Timestamp(event_date)
        for _, fight in date_fights.iterrows():
            fight_id = str(fight["fight_id"])
            for fighter_id, opponent_id in ((str(fight["r_id"]), str(fight["b_id"])), (str(fight["b_id"]), str(fight["r_id"]))):
                hist = history.get(fighter_id, [])
                row: dict[str, Any] = {
                    "fight_id": fight_id,
                    "event_date": event_date,
                    "fighter_id": fighter_id,
                    "opponent_id": opponent_id,
                    "fighter_win": int(str(fight["winner_id"]) == fighter_id),
                    "history_fights": float(len(hist)),
                    "history_max_date": hist[-1][0] if hist else pd.NaT,
                    "scheduled_rounds": float(pd.to_numeric(fight.get("total_rounds"), errors="coerce")),
                    "title_fight": float(pd.to_numeric(fight.get("title_fight", 0), errors="coerce") or 0),
                }
                row.update(_profile_features(fight, fighter_id, event_date))
                if hist:
                    all_arr = np.vstack([v for _, v in hist])
                else:
                    all_arr = np.empty((0, len(metric_columns)), dtype=float)
                row.update(_vector_summary(all_arr, metric_columns, "career", aggs))
                for n in fight_windows:
                    row.update(_vector_summary(all_arr[-n:], metric_columns, f"f{n}", aggs))
                for days in day_windows:
                    cutoff = event_date - pd.Timedelta(days=days)
                    selected = [v for d, v in hist if d >= cutoff]
                    arr = np.vstack(selected) if selected else np.empty((0, len(metric_columns)), dtype=float)
                    row.update(_vector_summary(arr, metric_columns, f"d{days}", aggs))
                snapshots.append(row)

        # Update only after every target fight on this calendar date has been snapshotted.
        current = obs_lookup.get(event_date)
        if current is not None:
            for _, current_row in current.iterrows():
                key = str(current_row["fight_id"]) + "|" + str(current_row["fighter_id"])
                history[str(current_row["fighter_id"])].append((event_date, vectors[key]))

    return pd.DataFrame(snapshots).sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _build_directional_matchups(snapshots: pd.DataFrame) -> pd.DataFrame:
    meta = [
        "fight_id", "event_date", "fighter_id", "opponent_id", "fighter_win",
        "history_max_date",
    ]
    feature_cols = [c for c in snapshots.columns if c not in meta]
    by_key = snapshots.set_index(["fight_id", "fighter_id"], drop=False)
    rows: list[dict[str, Any]] = []
    for s in snapshots.itertuples(index=False):
        opp = by_key.loc[(str(s.fight_id), str(s.opponent_id))]
        row: dict[str, Any] = {
            "fight_id": str(s.fight_id),
            "event_date": pd.Timestamp(s.event_date),
            "fighter_id": str(s.fighter_id),
            "opponent_id": str(s.opponent_id),
            "fighter_win": int(s.fighter_win),
            "history_max_date": s.history_max_date,
            "opponent_history_max_date": opp["history_max_date"],
            "sample_weight": 0.5,
        }
        for c in feature_cols:
            a = float(getattr(s, c)) if np.isscalar(getattr(s, c)) else np.nan
            b = pd.to_numeric(pd.Series([opp[c]]), errors="coerce").iloc[0]
            b = float(b) if pd.notna(b) else np.nan
            row[f"self_{c}"] = a
            row[f"opp_{c}"] = b
            row[f"diff_{c}"] = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_manifest(bank: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in bank.columns if c not in IDENTITY_COLUMNS | {"sample_weight"}]
    rows = []
    for c in feature_cols:
        perspective = c.split("_", 1)[0]
        base = c.split("_", 1)[1] if "_" in c else c
        family = "profile" if "profile_" in base else ("history" if "hist_" in base else "context")
        rows.append({"feature": c, "perspective": perspective, "family": family})
    return pd.DataFrame(rows)


def _leakage_audit(bank: pd.DataFrame, outer_start: pd.Timestamp) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "value": str(value)})

    current = pd.to_datetime(bank["event_date"])
    self_max = pd.to_datetime(bank["history_max_date"], errors="coerce")
    opp_max = pd.to_datetime(bank["opponent_history_max_date"], errors="coerce")
    self_bad = self_max.notna() & self_max.ge(current)
    opp_bad = opp_max.notna() & opp_max.ge(current)
    add("self_history_strictly_prior_date", not self_bad.any(), int(self_bad.sum()))
    add("opponent_history_strictly_prior_date", not opp_bad.any(), int(opp_bad.sum()))
    add("exactly_two_directional_rows_per_fight", bank.groupby("fight_id").size().eq(2).all(), bank.groupby("fight_id").size().value_counts().to_dict())
    add("targets_complement_within_fight", bank.groupby("fight_id")["fighter_win"].sum().eq(1).all(), int((~bank.groupby("fight_id")["fighter_win"].sum().eq(1)).sum()))
    feature_cols = [c for c in bank.columns if c not in IDENTITY_COLUMNS | {"sample_weight"}]
    forbidden = ("winner", "method", "finish_round", "match_time", "actual_", "postfight", "corner", "red_", "blue_")
    bad_names = [c for c in feature_cols if any(token in c.lower() for token in forbidden)]
    add("no_forbidden_feature_names", not bad_names, bad_names[:30])
    add("outer_rows_reserved_present", current.ge(outer_start).any(), int(current.ge(outer_start).sum()))
    return pd.DataFrame(checks)


def build(config_path: Path = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_config(config_path)
    round_path = Path(config["inputs"]["round_stats_path"])
    master_path = Path(config["inputs"]["master_path"])
    outputs = config["outputs"]
    out_root = Path(outputs["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    rounds = pd.read_parquet(round_path)
    master = pd.read_parquet(master_path)
    print(f"round rows: {len(rounds):,} | master fights: {len(master):,}")
    obs = _build_fight_observations(rounds, master)
    print(f"historical fighter-fight observations built in memory: {len(obs):,}")
    snapshots = _build_prefight_snapshots(obs, master, config)
    bank = _build_directional_matchups(snapshots)

    outer_start = pd.Timestamp(config["validation"]["outer_start"])
    audit = _leakage_audit(bank, outer_start)
    if not audit["passed"].all():
        print(audit.to_string(index=False))
        raise RuntimeError("raw signal feature-bank leakage audit failed")

    bank.to_parquet(outputs["prefight_feature_bank"], index=False)
    _feature_manifest(bank).to_csv(outputs["feature_manifest"], index=False)
    audit.to_csv(outputs["leakage_audit"], index=False)
    print(audit.to_string(index=False))
    print(f"feature bank: {bank.shape} -> {outputs['prefight_feature_bank']}")
    print(f"features: {len(_feature_manifest(bank)):,}")
    return bank, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(Path(args.config))


if __name__ == "__main__":
    main()
