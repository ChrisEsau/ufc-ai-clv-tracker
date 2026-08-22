"""Measurement-only test of fighter-specific submission offense/defense variance."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

SEED = 20260822
VALIDATION_START = pd.Timestamp("2022-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
DEFAULT_OUT = Path("data/diagnostics/fsr_v3/active_trait_audit/submission_conversion")
SIGMAS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.80)
CS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
GRID = np.linspace(-2.5, 2.5, 501)
GH_X, GH_W = hermgauss(15)
GH_W = GH_W / np.sqrt(np.pi)


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _bern_ll(y, q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return y * np.log(q) + (1 - y) * np.log1p(-q)


def _fight_q(p, n):
    return 1.0 - np.power(1.0 - np.clip(p, 1e-12, 1 - 1e-12), n)


def _prior(s):
    if s <= 0:
        z = np.full_like(GRID, -np.inf)
        z[np.argmin(np.abs(GRID))] = 0
        return z
    z = -0.5 * (GRID / s) ** 2
    return z - logsumexp(z)


def _mom(logw):
    w = np.exp(logw - logsumexp(logw))
    m = float((w * GRID).sum())
    v = float((w * (GRID - m) ** 2).sum())
    return m, np.sqrt(max(v, 0))


def _bucket(n):
    n = int(n)
    return "0" if n <= 0 else "1" if n == 1 else "2" if n == 2 else "3plus"


def build_obs():
    paired = build_paired_rounds()
    keys = ["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "opponent_name"]
    frame = paired.groupby(keys, as_index=False).agg(
        attempts=("effective_submission_attempts", "sum"),
        finish=("submission_finish", "max"),
    )
    frame.event_date = pd.to_datetime(frame.event_date, errors="raise").dt.normalize()
    frame.fight_id = frame.fight_id.astype(str)
    frame.fighter_id = frame.fighter_id.astype(str)
    frame.opponent_id = frame.opponent_id.astype(str)
    frame = frame[frame.attempts > 0].copy()
    frame.finish = frame.finish.astype(float)

    fsr = pd.read_parquet(
        FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
        columns=["event_date", "fight_id", "fighter_id", "submission_offense", "submission_defense", "submission_conversion_baseline"],
    ).copy()
    fsr.event_date = pd.to_datetime(fsr.event_date, errors="raise").dt.normalize()
    fsr.fight_id = fsr.fight_id.astype(str)
    fsr.fighter_id = fsr.fighter_id.astype(str)
    own = fsr.rename(columns={"submission_offense": "legacy_offense"})[
        ["event_date", "fight_id", "fighter_id", "legacy_offense", "submission_conversion_baseline"]
    ]
    opp = fsr.rename(columns={"fighter_id": "opponent_id", "submission_defense": "legacy_defense"})[
        ["event_date", "fight_id", "opponent_id", "legacy_defense"]
    ]
    frame = frame.merge(own, on=["event_date", "fight_id", "fighter_id"], how="inner", validate="one_to_one")
    frame = frame.merge(opp, on=["event_date", "fight_id", "opponent_id"], how="inner", validate="one_to_one")

    apps = frame[["event_date", "fighter_id"]].drop_duplicates()
    counts = apps.groupby(["fighter_id", "event_date"], as_index=False).size().sort_values(["fighter_id", "event_date"])
    counts["prior_ufc_fights"] = counts.groupby("fighter_id")["size"].cumsum() - counts["size"]
    frame = frame.merge(counts[["fighter_id", "event_date", "prior_ufc_fights"]], on=["fighter_id", "event_date"], how="left")
    frame["prior_bucket"] = frame.prior_ufc_fights.map(_bucket)
    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def pred_ll(y, n, base_logit, mean, variance, c):
    if c <= 0 or variance <= 1e-14:
        return float(_bern_ll(y, _fight_q(expit(base_logit + mean), n)))
    eta = mean + np.sqrt(2 * c * variance) * GH_X
    q = _fight_q(expit(base_logit + eta), n)
    ll = _bern_ll(y, q)
    peak = float(np.max(ll))
    return float(peak + np.log(np.sum(GH_W * np.exp(ll - peak))))


def replay(obs, sigma_offense, sigma_defense):
    prior_o, prior_d = _prior(sigma_offense), _prior(sigma_defense)
    offense_states, defense_states, rows = {}, {}, []
    for _, batch in obs.groupby("event_date", sort=True):
        pending_o = defaultdict(lambda: np.zeros_like(GRID))
        pending_d = defaultdict(lambda: np.zeros_like(GRID))
        for record in batch.to_dict("records"):
            attacker = str(record["fighter_id"])
            defender = str(record["opponent_id"])
            ow = offense_states.get(attacker, prior_o)
            dw = defense_states.get(defender, prior_d)
            om, osd = _mom(ow)
            dm, dsd = _mom(dw)
            y = float(record["finish"])
            n = float(record["attempts"])
            base = float(_logit(record["submission_conversion_baseline"]))
            mean = om - dm
            variance = osd ** 2 + dsd ** 2
            population_q = float(_fight_q(expit(base), n))
            legacy_q = float(_fight_q(expit(base + float(record["legacy_offense"]) - float(record["legacy_defense"])), n))
            plugin_q = float(_fight_q(expit(base + mean), n))
            row = dict(record)
            row.update({
                "offense_mean": om, "offense_sd": osd,
                "defense_mean": dm, "defense_sd": dsd,
                "plugin_q": plugin_q, "population_q": population_q, "legacy_q": legacy_q,
                "plugin_ll": float(_bern_ll(y, plugin_q)),
                "population_ll": float(_bern_ll(y, population_q)),
                "legacy_ll": float(_bern_ll(y, legacy_q)),
            })
            for c in CS:
                row[f"predictive_ll_c_{c:g}"] = pred_ll(y, n, base, mean, variance, c)
            rows.append(row)
            pending_o[attacker] += _bern_ll(y, _fight_q(expit(base + GRID - dm), n))
            pending_d[defender] += _bern_ll(y, _fight_q(expit(base + om - GRID), n))
        for attacker, ll in pending_o.items():
            current = offense_states.get(attacker, prior_o)
            updated = current + ll
            offense_states[attacker] = updated - logsumexp(updated)
        for defender, ll in pending_d.items():
            current = defense_states.get(defender, prior_d)
            updated = current + ll
            defense_states[defender] = updated - logsumexp(updated)
    return pd.DataFrame(rows)


def window(frame, start, end, column):
    x = frame[frame.event_date >= pd.Timestamp(start)]
    if end is not None:
        x = x[x.event_date < pd.Timestamp(end)]
    return {"rows": len(x), "fights": x.fight_id.nunique(), "total_ll": x[column].sum(), "mean_ll": x[column].mean()}


def bootstrap(frame, a, b, draws, seed):
    grouped = frame.groupby("fight_id")[[a, b]].sum()
    diff = (grouped[a] - grouped[b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    sims = np.array([diff[rng.integers(0, len(diff), len(diff))].sum() for _ in range(draws)])
    return diff.sum(), np.quantile(sims, .025), np.quantile(sims, .975), np.mean(sims > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs = build_obs()
    sweep, cache = [], {}
    for sigma_offense in SIGMAS:
        for sigma_defense in SIGMAS:
            frame = replay(obs, sigma_offense, sigma_defense)
            cache[(sigma_offense, sigma_defense)] = frame
            sweep.append({
                "sigma_offense": sigma_offense,
                "sigma_defense": sigma_defense,
                **window(frame, VALIDATION_START, HOLDOUT_START, "predictive_ll_c_1"),
            })
    sweep = pd.DataFrame(sweep)
    best = sweep.sort_values(["total_ll", "sigma_offense", "sigma_defense"], ascending=[False, True, True]).iloc[0]
    sigma_offense = float(best.sigma_offense)
    sigma_defense = float(best.sigma_defense)
    selected = cache[(sigma_offense, sigma_defense)]
    c_rows = []
    for c in CS:
        for label, start, end in (
            ("validation_2022_2023", VALIDATION_START, HOLDOUT_START),
            ("holdout_2024plus", HOLDOUT_START, None),
        ):
            c_rows.append({"c": c, "window": label, **window(selected, start, end, f"predictive_ll_c_{c:g}")})
    c_summary = pd.DataFrame(c_rows)
    best_c = float(c_summary[c_summary.window.eq("validation_2022_2023")].sort_values(["total_ll", "c"], ascending=[False, True]).iloc[0].c)
    hold = selected[selected.event_date >= HOLDOUT_START].copy()
    hold["selected_ll"] = hold[f"predictive_ll_c_{best_c:g}"]
    vs_population = bootstrap(hold, "selected_ll", "population_ll", args.bootstrap_draws, SEED)
    vs_legacy = bootstrap(hold, "selected_ll", "legacy_ll", args.bootstrap_draws, SEED + 1)
    buckets = hold.groupby("prior_bucket").apply(lambda x: pd.Series({
        "rows": len(x), "fights": x.fight_id.nunique(),
        "delta_ll_vs_population": (x.selected_ll - x.population_ll).sum(),
        "delta_ll_vs_legacy": (x.selected_ll - x.legacy_ll).sum(),
    }), include_groups=False).reset_index()
    sweep.to_csv(args.output_dir / "sigma_sweep.csv", index=False)
    c_summary.to_csv(args.output_dir / "variance_multiplier_sweep.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_rows.csv", index=False)
    buckets.to_csv(args.output_dir / "holdout_prior_buckets.csv", index=False)
    print("=" * 120)
    print("FSR V3 ACTIVE TRAIT AUDIT — SUBMISSION CONVERSION OFFENSE/DEFENSE")
    print("=" * 120)
    print(f"attempt-active fighter-fights={len(obs):,}")
    print("TOP VALIDATION SIGMAS")
    print(sweep.sort_values("total_ll", ascending=False).head(12).to_string(index=False))
    print(f"selected sigma_offense={sigma_offense:g} sigma_defense={sigma_defense:g}")
    print("EPISTEMIC C")
    print(c_summary.to_string(index=False))
    print(f"selected c={best_c:g}")
    print(f"HOLDOUT vs population conversion baseline LL={vs_population[0]:+.3f} CI[{vs_population[1]:+.3f},{vs_population[2]:+.3f}] P>0={vs_population[3]:.3f}")
    print(f"HOLDOUT vs inherited V2 offense/defense LL={vs_legacy[0]:+.3f} CI[{vs_legacy[1]:+.3f},{vs_legacy[2]:+.3f}] P>0={vs_legacy[3]:.3f}")
    print(buckets.to_string(index=False))
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
