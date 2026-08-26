from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.fsr_v3.replay.math import (
    beta_binomial_log_likelihood,
    nb2_log_likelihood,
    normalize_log_weights,
    weighted_mean_sd,
)
from pipeline.fsr_v3.replay.paired_effectiveness import (
    _fit_population_beta,
    _normal_prior,
    build_effectiveness_fighter_fights,
    standing_effectiveness_spec,
)
from pipeline.fsr_v3.replay.rate_families import (
    _fit_suppression_population,
    _fit_tendency_population,
    _log_gamma_prior,
    build_rate_fighter_fights,
    standing_spec,
)
from pipeline.simulation.event_clock_mc_v2.calibration.runner import run
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
    translate_capability,
)

WINDOW = 3
TARGET_NAMES = {"Jan Blachowicz", "Navajo Stirling"}
SHADOW_EVENT_NAME = "SHADOW_STIRLING_LAST3_STANDING_FSR"


def _sum_history(items: deque[np.ndarray], size: int) -> np.ndarray:
    if not items:
        return np.zeros(size, dtype=float)
    return np.sum(np.stack(list(items), axis=0), axis=0)


def rolling_tendency(fights: pd.DataFrame, spec) -> pd.DataFrame:
    grid = np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)
    states: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW))
    pop_y: list[float] = []
    pop_e: list[float] = []
    params = None
    rows = []
    for event_date, batch in fights.groupby("event_date", sort=True):
        params = _fit_tendency_population(pop_y, pop_e, params, spec)
        q_pop, alpha = params
        shape = max(q_pop * spec.tendency_prior_seconds / 900.0, 1e-9)
        prior_lp = _log_gamma_prior(grid, q_pop, shape)
        pending = []
        for rec in batch.to_dict("records"):
            fid = str(rec["fighter_id"])
            y = float(rec["numerator"])
            e = float(rec["exposure_seconds"])
            lp = prior_lp + _sum_history(states[fid], len(grid))
            w = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(grid, w)
            ll = None
            if e > 0:
                ll = nb2_log_likelihood(y, e / 900.0 * grid, alpha)
                post = normalize_log_weights(lp + ll)
                post_mean, post_sd = weighted_mean_sd(grid, post)
            else:
                post_mean, post_sd = pre_mean, pre_sd
            rows.append({**rec, "pre_rating": pre_mean, "pre_posterior_sd": pre_sd,
                         "post_rating": post_mean, "post_posterior_sd": post_sd,
                         "population_rate_15m": q_pop, "observation_alpha": alpha})
            pending.append((fid, y, e, ll))
        for fid, y, e, ll in pending:
            if e > 0 and ll is not None:
                ll = ll - np.max(ll)
                states[fid].append(ll)
                pop_y.append(y)
                pop_e.append(e)
    return pd.DataFrame(rows)


def rolling_suppression(tendency: pd.DataFrame, spec) -> pd.DataFrame:
    grid = np.linspace(spec.suppression_grid_min, spec.suppression_grid_max, spec.suppression_grid_points)
    source = tendency.copy()
    source["expected_attempts"] = source["exposure_seconds"].astype(float) / 900.0 * source["pre_rating"].astype(float)
    states: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW))
    pop_y: list[float] = []
    pop_exp: list[float] = []
    params = None
    rows = []
    for event_date, batch in source.groupby("event_date", sort=True):
        params = _fit_suppression_population(pop_y, pop_exp, params, spec)
        s_pop, alpha = params
        prior_lp = _log_gamma_prior(grid, s_pop, spec.suppression_prior_shape)
        pending = []
        for rec in batch.to_dict("records"):
            defender = str(rec["opponent_id"])
            y = float(rec["numerator"])
            expected = float(rec["expected_attempts"])
            lp = prior_lp + _sum_history(states[defender], len(grid))
            w = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(grid, w)
            ll = None
            if expected > 0:
                ll = nb2_log_likelihood(y, expected * grid, alpha)
                post = normalize_log_weights(lp + ll)
                post_mean, post_sd = weighted_mean_sd(grid, post)
            else:
                post_mean, post_sd = pre_mean, pre_sd
            rows.append({
                "event_date": rec["event_date"], "fight_id": str(rec["fight_id"]),
                "fighter_id": defender, "fighter_name": rec["opponent_name"],
                "opponent_id": str(rec["fighter_id"]), "opponent_name": rec["fighter_name"],
                "pre_rating": pre_mean, "pre_posterior_sd": pre_sd,
                "post_rating": post_mean, "post_posterior_sd": post_sd,
                "population_multiplier": s_pop, "observation_alpha": alpha,
            })
            pending.append((defender, y, expected, ll))
        for defender, y, expected, ll in pending:
            if expected > 0 and ll is not None:
                ll = ll - np.max(ll)
                states[defender].append(ll)
                pop_y.append(y)
                pop_exp.append(expected)
    return pd.DataFrame(rows)


def rolling_effectiveness(fights: pd.DataFrame, spec) -> pd.DataFrame:
    grid = np.linspace(spec.grid_min, spec.grid_max, spec.grid_points)
    off_prior = _normal_prior(grid, spec.sigma_offense)
    def_prior = _normal_prior(grid, spec.sigma_defense)
    off_states: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW))
    def_states: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW))
    pop_y: list[float] = []
    pop_n: list[float] = []
    beta = None
    rows = []
    for event_date, batch in fights.groupby("event_date", sort=True):
        beta = _fit_population_beta(pop_y, pop_n, spec.rho, beta)
        pending = []
        for rec in batch.to_dict("records"):
            attacker = str(rec["fighter_id"])
            defender = str(rec["opponent_id"])
            y = float(rec["landed"])
            n = float(rec["attempted"])
            off_lp = off_prior + _sum_history(off_states[attacker], len(grid))
            def_lp = def_prior + _sum_history(def_states[defender], len(grid))
            off_w = normalize_log_weights(off_lp)
            def_w = normalize_log_weights(def_lp)
            off_pre, off_sd = weighted_mean_sd(grid, off_w)
            def_pre, def_sd = weighted_mean_sd(grid, def_w)
            off_ll = def_ll = None
            if n > 0:
                off_ll = beta_binomial_log_likelihood(y, n, expit(beta + grid - def_pre), spec.rho)
                def_ll = beta_binomial_log_likelihood(y, n, expit(beta + off_pre - grid), spec.rho)
                off_post = normalize_log_weights(off_lp + off_ll)
                def_post = normalize_log_weights(def_lp + def_ll)
                off_post_mean, off_post_sd = weighted_mean_sd(grid, off_post)
                def_post_mean, def_post_sd = weighted_mean_sd(grid, def_post)
            else:
                off_post_mean, off_post_sd = off_pre, off_sd
                def_post_mean, def_post_sd = def_pre, def_sd
            common = {"event_date": rec["event_date"], "fight_id": str(rec["fight_id"]),
                      "population_baseline": float(expit(beta))}
            rows.append({**common, "fighter_id": attacker, "trait": spec.offense_trait,
                         "pre_rating": off_pre, "post_rating": off_post_mean})
            rows.append({**common, "fighter_id": defender, "trait": spec.defense_trait,
                         "pre_rating": def_pre, "post_rating": def_post_mean})
            pending.append((attacker, defender, y, n, off_ll, def_ll))
        for attacker, defender, y, n, off_ll, def_ll in pending:
            if n > 0 and off_ll is not None and def_ll is not None:
                off_states[attacker].append(off_ll - np.max(off_ll))
                def_states[defender].append(def_ll - np.max(def_ll))
                pop_y.append(y)
                pop_n.append(n)
    out = pd.DataFrame(rows)
    return out.sort_values(["event_date", "fight_id", "fighter_id", "trait"]).reset_index(drop=True)


def _fight_probability(record: dict, fight_id: str) -> dict:
    return record["metrics"]["fight_probabilities"][str(fight_id)]


def main() -> None:
    outdir = Path("data/diagnostics/physical_profile/stirling_last3_shadow")
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(MASTER_PATH).copy()
    name_mask = master.apply(lambda r: {str(r.get("r_name")), str(r.get("b_name"))} == TARGET_NAMES, axis=1)
    target = master[name_mask].copy()
    if len(target) != 1:
        raise RuntimeError(f"expected one Jan/Stirling fight, found {len(target)}")
    fight = target.iloc[0]
    fight_id = str(fight["fight_id"])
    date = pd.Timestamp(fight["date"]).normalize()
    red_id, blue_id = str(fight["r_id"]), str(fight["b_id"])

    # Isolate this fight in the generic runner without changing mechanics.
    master.loc[master["fight_id"].astype(str).eq(fight_id), "event_name"] = SHADOW_EVENT_NAME
    master.to_parquet(MASTER_PATH, index=False)

    baseline_path = outdir / "baseline.json"
    baseline = run(split="calibration", paths_per_fight=1000,
                   config_path=Path("configs/event_clock_v2/calibration/default.yaml"),
                   output=baseline_path, ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
                   event_name=SHADOW_EVENT_NAME)

    paired = build_paired_rounds()
    cfg = FSRV3Config()
    rspec = standing_spec(cfg)
    espec = standing_effectiveness_spec(cfg)
    rate_fights = build_rate_fighter_fights(rspec, paired_rounds=paired)
    tendency = rolling_tendency(rate_fights, rspec)
    suppression = rolling_suppression(tendency, rspec)
    eff_fights = build_effectiveness_fighter_fights(espec, paired_rounds=paired)
    effectiveness = rolling_effectiveness(eff_fights, espec)

    snapshots = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    snapshots["event_date"] = pd.to_datetime(snapshots["event_date"]).dt.normalize()
    snapshots["fight_id"] = snapshots["fight_id"].astype(str)
    snapshots["fighter_id"] = snapshots["fighter_id"].astype(str)
    mask = snapshots["event_date"].eq(date) & snapshots["fight_id"].eq(fight_id) & snapshots["fighter_id"].isin([red_id, blue_id])
    if mask.sum() != 2:
        raise RuntimeError("target snapshot rows missing")

    audit_rows = []
    for fid in [red_id, blue_id]:
        tend_row = tendency[(tendency["event_date"].eq(date)) & (tendency["fight_id"].astype(str).eq(fight_id)) & (tendency["fighter_id"].astype(str).eq(fid))]
        supp_row = suppression[(suppression["event_date"].eq(date)) & (suppression["fight_id"].astype(str).eq(fight_id)) & (suppression["fighter_id"].astype(str).eq(fid))]
        eff_rows = effectiveness[(effectiveness["event_date"].eq(date)) & (effectiveness["fight_id"].astype(str).eq(fight_id)) & (effectiveness["fighter_id"].astype(str).eq(fid))]
        if len(tend_row) != 1 or len(supp_row) != 1 or set(eff_rows["trait"]) != {espec.offense_trait, espec.defense_trait}:
            raise RuntimeError(f"last3 standing state incomplete for {fid}")
        vals = {
            "standing_striking_tendency": float(tend_row.iloc[0]["pre_rating"]),
            "standing_striking_suppression": float(supp_row.iloc[0]["pre_rating"]),
            "standing_striking_offense": float(eff_rows[eff_rows["trait"].eq(espec.offense_trait)].iloc[0]["pre_rating"]),
            "standing_striking_defense": float(eff_rows[eff_rows["trait"].eq(espec.defense_trait)].iloc[0]["pre_rating"]),
        }
        idx = snapshots.index[mask & snapshots["fighter_id"].eq(fid)][0]
        before = {k: float(snapshots.loc[idx, k]) for k in vals}
        for k, v in vals.items():
            snapshots.loc[idx, k] = v
        audit_rows.append({"fighter_id": fid, "before": before, "last3": vals})

    # Compare capabilities before/after using the same population reference.
    original = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    original["event_date"] = pd.to_datetime(original["event_date"]).dt.normalize()
    original["fight_id"] = original["fight_id"].astype(str)
    original["fighter_id"] = original["fighter_id"].astype(str)
    cutoff = original["event_date"].min()
    # Match runner reference semantics: calibration manifest minimum is 2025-03-22.
    reference = CapabilityReference.from_prefight_before(original, pd.Timestamp("2025-03-22"))
    def rows(frame):
        s = frame[frame["event_date"].eq(date) & frame["fight_id"].eq(fight_id)].set_index("fighter_id")
        return s.loc[red_id], s.loc[blue_id]
    or_red, or_blue = rows(original)
    sh_red, sh_blue = rows(snapshots)
    cap_audit = {}
    for label, rr, bb in [("baseline", or_red, or_blue), ("last3", sh_red, sh_blue)]:
        rc = translate_capability(rr, bb, reference, prior_ufc_fights=99)
        bc = translate_capability(bb, rr, reference, prior_ufc_fights=99)
        cap_audit[label] = {
            "red": {"fighter_id": red_id, "standing_rate_15m": rc.standing_rate_15m,
                    "standing_accuracy": rc.standing_accuracy, "standing": rc.capability.standing,
                    "counter": rc.capability.counter, "pressure": rc.capability.pressure},
            "blue": {"fighter_id": blue_id, "standing_rate_15m": bc.standing_rate_15m,
                     "standing_accuracy": bc.standing_accuracy, "standing": bc.capability.standing,
                     "counter": bc.capability.counter, "pressure": bc.capability.pressure},
        }

    snapshots.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    shadow_path = outdir / "last3.json"
    shadow = run(split="calibration", paths_per_fight=1000,
                 config_path=Path("configs/event_clock_v2/calibration/default.yaml"),
                 output=shadow_path, ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
                 event_name=SHADOW_EVENT_NAME)

    summary = {
        "fight_id": fight_id,
        "date": date.date().isoformat(),
        "red_fighter": str(fight["r_name"]),
        "blue_fighter": str(fight["b_name"]),
        "window": WINDOW,
        "scope": "standing FSR only; all other FSR/mechanics unchanged",
        "trait_audit": audit_rows,
        "capabilities": cap_audit,
        "baseline_probability": _fight_probability(baseline, fight_id),
        "last3_probability": _fight_probability(shadow, fight_id),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
