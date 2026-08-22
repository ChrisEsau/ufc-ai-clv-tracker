"""Measurement-only prior/variance audit for the live FSR escape/retention family.

No production files are modified. The native target is qualified control seconds
per modeled ground entry, the exact quantity consumed by Event Clock V2's
retention feature:

    mean_seconds_per_entry = population_mean * exp(retention_attacker - escape_defender)

where the current schema calls the attacker retention effect ``escape_defense``
and the defender escape effect ``escape_offense``.

The study does four things chronologically:
1. scores the frozen V2 K=5-entry prior against alternative K values;
2. tests whether attacker and defender fighter heterogeneity are independently useful;
3. selects Gaussian population prior SDs on a validation window;
4. evaluates epistemic variance multipliers c on an untouched holdout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds
from pipeline.fsr_v3.replay.math import nb2_log_likelihood, normalize_log_weights, weighted_mean_sd

SEED = 20260822
VALIDATION_START = pd.Timestamp("2022-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
DEFAULT_OUT = Path("data/diagnostics/fsr_v3/active_trait_audit/escape")

K_CANDIDATES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
SIGMA_CANDIDATES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
C_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25)
GRID = np.linspace(-2.5, 2.5, 501)
GH_X, GH_W = hermgauss(15)
GH_W = GH_W / np.sqrt(np.pi)


def _bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3plus"


def build_fighter_fights() -> pd.DataFrame:
    paired = build_paired_rounds()
    sums = [
        "qualified_control_inflicted_seconds",
        "qualified_control_suffered_seconds",
        "ground_entries",
        "opponent_ground_entries",
        "round_elapsed_seconds",
    ]
    keys = [
        "event_date", "fight_id", "fighter_id", "fighter_name",
        "opponent_id", "opponent_name",
    ]
    fights = paired.groupby(keys, as_index=False)[sums].sum()
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="raise").dt.normalize()
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["fighter_id"] = fights["fighter_id"].astype(str)
    fights["opponent_id"] = fights["opponent_id"].astype(str)

    appearances = fights[["event_date", "fighter_id"]].drop_duplicates()
    by_date = appearances.groupby(["fighter_id", "event_date"], as_index=False).size()
    by_date = by_date.sort_values(["fighter_id", "event_date"])
    by_date["prior_ufc_fights"] = by_date.groupby("fighter_id")["size"].cumsum() - by_date["size"]
    fights = fights.merge(
        by_date[["fighter_id", "event_date", "prior_ufc_fights"]],
        on=["fighter_id", "event_date"], how="left", validate="many_to_one",
    )
    fights["prior_bucket"] = fights["prior_ufc_fights"].astype(int).map(_bucket)
    return fights.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _fit_alpha(train: pd.DataFrame) -> tuple[float, float]:
    active = train[train["ground_entries"] > 0].copy()
    y = active["qualified_control_inflicted_seconds"].to_numpy(float)
    n = active["ground_entries"].to_numpy(float)
    mu0 = float(y.sum() / max(n.sum(), 1.0))
    mu = np.maximum(n * mu0, 1e-12)

    def objective(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        return -float(np.sum(nb2_log_likelihood(y, mu, alpha)))

    fit = minimize_scalar(objective, bounds=(np.log(1e-3), np.log(100.0)), method="bounded")
    if not fit.success:
        raise RuntimeError(f"escape NB2 alpha fit failed: {fit.message}")
    return mu0, float(np.exp(fit.x))


def _nb_ll(y: float, entries: float, mean_per_entry: float, alpha: float) -> float:
    return float(nb2_log_likelihood(y, max(entries * mean_per_entry, 1e-12), alpha))


def _predictive_ll_normal(
    y: float,
    entries: float,
    pop_mean: float,
    eta_mean: float,
    eta_var: float,
    c: float,
    alpha: float,
) -> float:
    if c <= 0.0 or eta_var <= 1e-14:
        return _nb_ll(y, entries, pop_mean * np.exp(eta_mean), alpha)
    sd = float(np.sqrt(max(c * eta_var, 0.0)))
    eta = eta_mean + np.sqrt(2.0) * sd * GH_X
    means = pop_mean * np.exp(np.clip(eta, -6.0, 6.0))
    lls = nb2_log_likelihood(y, entries * means, alpha)
    m = float(np.max(lls))
    return float(m + np.log(np.sum(GH_W * np.exp(lls - m))))


def _prior_weights(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        w = np.zeros_like(GRID)
        w[np.argmin(np.abs(GRID))] = 1.0
        return w
    logw = -0.5 * (GRID / sigma) ** 2 - np.log(sigma)
    return normalize_log_weights(logw)


@dataclass
class ReplayOutput:
    rows: pd.DataFrame


def replay_paired(
    fights: pd.DataFrame,
    *,
    sigma_retention: float,
    sigma_escape: float,
    alpha: float,
    initial_pop_mean: float,
    c_values=C_CANDIDATES,
) -> ReplayOutput:
    ret_state: dict[str, np.ndarray] = {}
    esc_state: dict[str, np.ndarray] = {}
    ret_prior = _prior_weights(sigma_retention)
    esc_prior = _prior_weights(sigma_escape)
    pop_y = 0.0
    pop_n = 0.0
    rows: list[dict[str, object]] = []

    for _, batch in fights.groupby("event_date", sort=True):
        pop_mean = pop_y / pop_n if pop_n > 0 else initial_pop_mean
        pending_ret: dict[str, np.ndarray] = {}
        pending_esc: dict[str, np.ndarray] = {}
        pop_updates: list[tuple[float, float]] = []

        for record in batch.to_dict("records"):
            entries = float(record["ground_entries"])
            y = float(record["qualified_control_inflicted_seconds"])
            if entries <= 0.0:
                continue
            attacker = str(record["fighter_id"])
            defender = str(record["opponent_id"])
            rw = ret_state.get(attacker, ret_prior)
            ew = esc_state.get(defender, esc_prior)
            rmean, rsd = weighted_mean_sd(GRID, rw)
            emean, esd = weighted_mean_sd(GRID, ew)
            eta_mean = rmean - emean
            eta_var = rsd * rsd + esd * esd
            plugin_mean = pop_mean * np.exp(np.clip(eta_mean, -6.0, 6.0))

            row = dict(record)
            row.update({
                "population_mean_seconds_per_entry": pop_mean,
                "retention_pre_mean": rmean,
                "retention_pre_sd": rsd,
                "escape_pre_mean": emean,
                "escape_pre_sd": esd,
                "eta_pre_mean": eta_mean,
                "eta_pre_var": eta_var,
                "plugin_mean_seconds_per_entry": plugin_mean,
                "plugin_ll": _nb_ll(y, entries, plugin_mean, alpha),
                "population_ll": _nb_ll(y, entries, pop_mean, alpha),
                "plugin_abs_error_seconds": abs(y - entries * plugin_mean),
                "population_abs_error_seconds": abs(y - entries * pop_mean),
            })
            for c in c_values:
                row[f"predictive_ll_c_{c:g}"] = _predictive_ll_normal(
                    y, entries, pop_mean, eta_mean, eta_var, float(c), alpha
                )
            rows.append(row)

            ret_ll = nb2_log_likelihood(
                y,
                entries * pop_mean * np.exp(np.clip(GRID - emean, -6.0, 6.0)),
                alpha,
            )
            esc_ll = nb2_log_likelihood(
                y,
                entries * pop_mean * np.exp(np.clip(rmean - GRID, -6.0, 6.0)),
                alpha,
            )
            pending_ret[attacker] = pending_ret.get(attacker, np.zeros_like(GRID)) + ret_ll
            pending_esc[defender] = pending_esc.get(defender, np.zeros_like(GRID)) + esc_ll
            pop_updates.append((y, entries))

        for fighter, ll in pending_ret.items():
            base = ret_state.get(fighter, ret_prior)
            ret_state[fighter] = normalize_log_weights(np.log(np.maximum(base, 1e-300)) + ll)
        for fighter, ll in pending_esc.items():
            base = esc_state.get(fighter, esc_prior)
            esc_state[fighter] = normalize_log_weights(np.log(np.maximum(base, 1e-300)) + ll)
        for y, entries in pop_updates:
            pop_y += y
            pop_n += entries

    return ReplayOutput(pd.DataFrame(rows))


def replay_legacy_k(fights: pd.DataFrame, *, k: float, alpha: float, initial_pop_mean: float) -> pd.DataFrame:
    inflicted = defaultdict(lambda: [0.0, 0.0])
    suffered = defaultdict(lambda: [0.0, 0.0])
    pop_y = 0.0
    pop_n = 0.0
    rows = []
    for _, batch in fights.groupby("event_date", sort=True):
        pop_mean = pop_y / pop_n if pop_n > 0 else initial_pop_mean
        pending = []
        for record in batch.to_dict("records"):
            entries = float(record["ground_entries"])
            y = float(record["qualified_control_inflicted_seconds"])
            if entries <= 0.0:
                continue
            attacker = str(record["fighter_id"])
            defender = str(record["opponent_id"])
            iy, inn = inflicted[attacker]
            sy, snn = suffered[defender]
            att_mean = (iy + pop_mean * k) / (inn + k)
            def_mean = (sy + pop_mean * k) / (snn + k)
            pair_mean = att_mean * def_mean / max(pop_mean, 1e-12)
            rows.append({
                **record,
                "k": float(k),
                "population_mean_seconds_per_entry": pop_mean,
                "legacy_mean_seconds_per_entry": pair_mean,
                "legacy_ll": _nb_ll(y, entries, pair_mean, alpha),
                "legacy_abs_error_seconds": abs(y - entries * pair_mean),
            })
            pending.append((attacker, defender, y, entries))
        for attacker, defender, y, entries in pending:
            inflicted[attacker][0] += y
            inflicted[attacker][1] += entries
            suffered[defender][0] += y
            suffered[defender][1] += entries
            pop_y += y
            pop_n += entries
    return pd.DataFrame(rows)


def summarize_window(rows: pd.DataFrame, start, end=None, *, ll_col: str, mae_col: str) -> dict[str, float]:
    x = rows[rows["event_date"] >= pd.Timestamp(start)]
    if end is not None:
        x = x[x["event_date"] < pd.Timestamp(end)]
    return {
        "rows": int(len(x)),
        "fights": int(x["fight_id"].nunique()),
        "total_ll": float(x[ll_col].sum()),
        "mean_ll": float(x[ll_col].mean()),
        "mae_seconds": float(x[mae_col].mean()),
    }


def bootstrap_delta(rows: pd.DataFrame, a: str, b: str, *, draws=2000, seed=SEED) -> dict[str, float]:
    x = rows[["fight_id", a, b]].dropna().copy()
    clustered = x.groupby("fight_id")[[a, b]].sum()
    diff = (clustered[a] - clustered[b]).to_numpy(float)
    if len(diff) == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_gt_0": np.nan}
    rng = np.random.default_rng(seed)
    sims = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(diff), len(diff))
        sims[i] = diff[idx].sum()
    return {
        "delta": float(diff.sum()),
        "ci_low": float(np.quantile(sims, 0.025)),
        "ci_high": float(np.quantile(sims, 0.975)),
        "p_gt_0": float(np.mean(sims > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fights = build_fighter_fights()
    train = fights[fights["event_date"] < VALIDATION_START]
    initial_pop_mean, alpha = _fit_alpha(train)

    active = fights[fights["ground_entries"] > 0].copy()
    zero_control = active["qualified_control_inflicted_seconds"].eq(0)
    print("=" * 120)
    print("FSR V3 ACTIVE TRAIT AUDIT — ESCAPE / RETENTION")
    print("=" * 120)
    print(f"fighter-fights total: {len(fights):,}")
    print(f"native rows with >=1 modeled ground entry: {len(active):,}")
    print(f"zero-control share conditional on entry: {zero_control.mean():.2%}")
    print(f"pre-2022 population mean control/entry: {initial_pop_mean:.3f} sec")
    print(f"fitted NB2 alpha: {alpha:.6f}")

    legacy_records = []
    legacy_frames = {}
    for k in K_CANDIDATES:
        frame = replay_legacy_k(fights, k=k, alpha=alpha, initial_pop_mean=initial_pop_mean)
        legacy_frames[k] = frame
        val = summarize_window(frame, VALIDATION_START, HOLDOUT_START, ll_col="legacy_ll", mae_col="legacy_abs_error_seconds")
        hold = summarize_window(frame, HOLDOUT_START, None, ll_col="legacy_ll", mae_col="legacy_abs_error_seconds")
        legacy_records.append({"k": k, "window": "validation_2022_2023", **val})
        legacy_records.append({"k": k, "window": "holdout_2024plus", **hold})
    legacy_summary = pd.DataFrame(legacy_records)
    val_legacy = legacy_summary[legacy_summary["window"] == "validation_2022_2023"]
    best_k = float(val_legacy.sort_values(["total_ll", "k"], ascending=[False, True]).iloc[0]["k"])

    model_records = []
    replay_cache = {}
    for sr in SIGMA_CANDIDATES:
        for se in SIGMA_CANDIDATES:
            out = replay_paired(
                fights,
                sigma_retention=sr,
                sigma_escape=se,
                alpha=alpha,
                initial_pop_mean=initial_pop_mean,
            ).rows
            replay_cache[(sr, se)] = out
            val = summarize_window(
                out, VALIDATION_START, HOLDOUT_START,
                ll_col="predictive_ll_c_1", mae_col="plugin_abs_error_seconds",
            )
            model_records.append({
                "sigma_retention": sr,
                "sigma_escape": se,
                "window": "validation_2022_2023",
                **val,
            })
    model_summary = pd.DataFrame(model_records)
    best = model_summary.sort_values(
        ["total_ll", "sigma_retention", "sigma_escape"], ascending=[False, True, True]
    ).iloc[0]
    best_sr = float(best["sigma_retention"])
    best_se = float(best["sigma_escape"])
    selected = replay_cache[(best_sr, best_se)]

    c_records = []
    for c in C_CANDIDATES:
        ll_col = f"predictive_ll_c_{c:g}"
        for label, start, end in (
            ("validation_2022_2023", VALIDATION_START, HOLDOUT_START),
            ("holdout_2024plus", HOLDOUT_START, None),
        ):
            s = summarize_window(selected, start, end, ll_col=ll_col, mae_col="plugin_abs_error_seconds")
            c_records.append({
                "sigma_retention": best_sr,
                "sigma_escape": best_se,
                "c": c,
                "window": label,
                **s,
            })
    c_summary = pd.DataFrame(c_records)
    val_c = c_summary[c_summary["window"] == "validation_2022_2023"]
    best_c = float(val_c.sort_values(["total_ll", "c"], ascending=[False, True]).iloc[0]["c"])

    hold = selected[selected["event_date"] >= HOLDOUT_START].copy()
    legacy_best = legacy_frames[best_k]
    legacy_hold = legacy_best[legacy_best["event_date"] >= HOLDOUT_START][
        ["fight_id", "fighter_id", "legacy_ll", "legacy_abs_error_seconds"]
    ]
    hold = hold.merge(legacy_hold, on=["fight_id", "fighter_id"], how="inner", validate="one_to_one")
    final_ll = f"predictive_ll_c_{best_c:g}"
    hold["selected_ll"] = hold[final_ll]
    hold["selected_vs_population"] = hold["selected_ll"] - hold["population_ll"]
    hold["selected_vs_legacy"] = hold["selected_ll"] - hold["legacy_ll"]

    b_pop = bootstrap_delta(hold, "selected_ll", "population_ll", draws=args.bootstrap_draws)
    b_legacy = bootstrap_delta(hold, "selected_ll", "legacy_ll", draws=args.bootstrap_draws, seed=SEED + 1)

    bucket_rows = []
    for bucket, part in hold.groupby("prior_bucket", sort=False):
        bucket_rows.append({
            "prior_bucket": bucket,
            "rows": int(len(part)),
            "fights": int(part["fight_id"].nunique()),
            "selected_total_ll": float(part["selected_ll"].sum()),
            "population_total_ll": float(part["population_ll"].sum()),
            "legacy_total_ll": float(part["legacy_ll"].sum()),
            "delta_ll_vs_population": float(part["selected_vs_population"].sum()),
            "delta_ll_vs_legacy": float(part["selected_vs_legacy"].sum()),
            "plugin_mae_seconds": float(part["plugin_abs_error_seconds"].mean()),
            "legacy_mae_seconds": float(part["legacy_abs_error_seconds"].mean()),
            "population_mae_seconds": float(part["population_abs_error_seconds"].mean()),
        })
    bucket_summary = pd.DataFrame(bucket_rows)

    legacy_summary.to_csv(args.output_dir / "legacy_k_sweep.csv", index=False)
    model_summary.to_csv(args.output_dir / "sigma_sweep.csv", index=False)
    c_summary.to_csv(args.output_dir / "variance_multiplier_sweep.csv", index=False)
    bucket_summary.to_csv(args.output_dir / "holdout_prior_buckets.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_rows.csv", index=False)

    print("\nLEGACY PRIOR K")
    print(val_legacy.sort_values("total_ll", ascending=False).to_string(index=False))
    print(f"selected legacy K on validation: {best_k:g} entries (current frozen V2 K=5)")

    print("\nPAIRED LATENT PRIOR SD — TOP 12 VALIDATION")
    print(model_summary.sort_values("total_ll", ascending=False).head(12).to_string(index=False))
    print(f"selected sigma_retention={best_sr:g}, sigma_escape={best_se:g}")

    print("\nEPISTEMIC VARIANCE MULTIPLIER")
    print(c_summary.to_string(index=False))
    print(f"selected c on validation: {best_c:g}")

    print("\nUNTOUCHED HOLDOUT 2024+")
    print(f"rows={len(hold):,} fights={hold['fight_id'].nunique():,}")
    print(
        f"selected vs population LL delta={b_pop['delta']:+.3f} "
        f"CI[{b_pop['ci_low']:+.3f},{b_pop['ci_high']:+.3f}] P(>0)={b_pop['p_gt_0']:.3f}"
    )
    print(
        f"selected vs best-legacy LL delta={b_legacy['delta']:+.3f} "
        f"CI[{b_legacy['ci_low']:+.3f},{b_legacy['ci_high']:+.3f}] P(>0)={b_legacy['p_gt_0']:.3f}"
    )
    print(bucket_summary.to_string(index=False))
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
