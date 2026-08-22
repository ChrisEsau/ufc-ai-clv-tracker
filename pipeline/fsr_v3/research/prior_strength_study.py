"""Measurement-only prior-strength audit for validated FSR V3 rate tendencies.

This study does NOT modify locked FSR V3 configuration or Event Clock mechanics.
It asks one narrow question: after 0/1/2/3+ prior UFC fights, does a different
Gamma prior evidence strength K improve next-fight prediction of each trait's
native target?

Families audited independently:
- standing striking tendency: distance significant-strike attempts per standing exposure
- takedown tendency: takedown attempts per eligible takedown exposure

Primary score is posterior-predictive NB2 negative log likelihood.  A plug-in
posterior-mean NB2 score and count error are retained as secondary diagnostics.
Population rate and NB2 dispersion follow the exact leakage-safe V3 replay and
are therefore identical across K candidates; only prior evidence strength moves.
"""

from __future__ import annotations

from dataclasses import replace
from math import lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pipeline.fsr_v3.config import FSRV3Config
from pipeline.fsr_v3.replay.math import nb2_log_likelihood
from pipeline.fsr_v3.replay.rate_families import (
    RateFamilySpec,
    build_rate_fighter_fights,
    replay_tendency,
    standing_spec,
    takedown_spec,
)


OUT_DIR = Path("data/diagnostics/fsr_v3_prior_strength")
MODERN_START = pd.Timestamp("2020-01-01")
FRESH_START = pd.Timestamp("2025-03-29")
BOOTSTRAP_REPS = 5_000
BOOTSTRAP_SEED = 20260821

# Include weaker as well as stronger priors so the direction is empirical.
CANDIDATES = {
    "standing_striking_tendency": [50.0, 87.78, 150.0, 250.0, 400.0, 600.0, 900.0],
    "takedown_tendency": [150.0, 250.0, 350.0, 468.48, 600.0, 900.0, 1350.0, 1800.0],
}


def _log_gamma_prior(grid: np.ndarray, mean: float, shape: float) -> np.ndarray:
    mean = max(float(mean), 1e-9)
    shape = max(float(shape), 1e-9)
    rate = shape / mean
    return (
        shape * log(rate)
        - lgamma(shape)
        + (shape - 1.0) * np.log(grid)
        - rate * grid
    )


def _prior_bucket(prior_fights: int) -> str:
    if prior_fights >= 3:
        return "3plus"
    return str(int(prior_fights))


def _score_family(spec: RateFamilySpec, candidates: list[float]) -> pd.DataFrame:
    """Replay one family once, then score all K values on the exact same states."""
    fights = build_rate_fighter_fights(spec)
    baseline = replay_tendency(fights, spec)

    q_grid = np.linspace(spec.tendency_grid_min, spec.tendency_grid_max, spec.tendency_grid_points)
    states: dict[str, np.ndarray] = {}
    prior_counts: dict[str, int] = {}
    rows: list[dict] = []
    current_k = float(spec.tendency_prior_seconds)
    current_pre_diffs: list[float] = []

    for event_date, batch in baseline.groupby("event_date", sort=True):
        pending_state: list[tuple[str, np.ndarray | None, float]] = []
        pending_fights: list[str] = []

        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            y = float(record["numerator"])
            exposure = float(record["denominator"])
            q_pop = float(record["population_rate_15m"])
            alpha = float(record["observation_alpha"])
            prior_fights = int(prior_counts.get(fighter, 0))
            state = states.get(fighter)

            observation_ll = None
            if exposure > 0.0:
                mu_grid = exposure / 900.0 * q_grid
                observation_ll = nb2_log_likelihood(y, mu_grid, alpha)

            # K only changes today's population prior.  The accumulated fighter
            # observation likelihood is shared across every K candidate.
            for k_seconds in candidates:
                prior_shape = max(q_pop * float(k_seconds) / 900.0, 1e-9)
                lp = _log_gamma_prior(q_grid, q_pop, prior_shape)
                if state is not None:
                    lp = lp + state
                log_z = float(logsumexp(lp))
                weights = np.exp(lp - log_z)
                pre_mean = float(np.sum(q_grid * weights))

                if np.isclose(float(k_seconds), current_k, atol=1e-12, rtol=0.0):
                    current_pre_diffs.append(abs(pre_mean - float(record["pre_rating"])))

                if observation_ll is None:
                    posterior_predictive_ll = np.nan
                    plugin_ll = np.nan
                    predicted_count = 0.0
                else:
                    posterior_predictive_ll = float(logsumexp(lp + observation_ll) - log_z)
                    predicted_count = exposure / 900.0 * pre_mean
                    plugin_ll = float(nb2_log_likelihood(y, predicted_count, alpha))

                rows.append(
                    {
                        "family": spec.tendency_trait,
                        "event_date": pd.Timestamp(event_date),
                        "fight_id": str(record["fight_id"]),
                        "fighter_id": fighter,
                        "fighter_name": record["fighter_name"],
                        "opponent_id": str(record["opponent_id"]),
                        "opponent_name": record["opponent_name"],
                        "prior_fights": prior_fights,
                        "prior_bucket": _prior_bucket(prior_fights),
                        "k_seconds": float(k_seconds),
                        "is_current_k": bool(np.isclose(float(k_seconds), current_k, atol=1e-12, rtol=0.0)),
                        "population_rate_15m": q_pop,
                        "observation_alpha": alpha,
                        "pre_rate_15m": pre_mean,
                        "actual_count": y,
                        "exposure_seconds": exposure,
                        "predicted_count": predicted_count,
                        "posterior_predictive_nll": -posterior_predictive_ll if np.isfinite(posterior_predictive_ll) else np.nan,
                        "plugin_nll": -plugin_ll if np.isfinite(plugin_ll) else np.nan,
                        "count_error": predicted_count - y if exposure > 0.0 else np.nan,
                        "abs_count_error": abs(predicted_count - y) if exposure > 0.0 else np.nan,
                    }
                )

            pending_state.append((fighter, observation_ll, exposure))
            pending_fights.append(fighter)

        # Match production same-event delayed updates.
        for fighter, observation_ll, exposure in pending_state:
            if exposure > 0.0 and observation_ll is not None:
                if fighter in states:
                    states[fighter] = states[fighter] + observation_ll
                    states[fighter] -= np.max(states[fighter])
                else:
                    states[fighter] = observation_ll - np.max(observation_ll)
        for fighter in pending_fights:
            prior_counts[fighter] = int(prior_counts.get(fighter, 0)) + 1

    max_diff = max(current_pre_diffs) if current_pre_diffs else np.nan
    if not np.isfinite(max_diff) or max_diff > 1e-7:
        raise RuntimeError(
            f"{spec.tendency_trait}: research replay failed current-K parity; max pre-rating diff={max_diff}"
        )
    print(f"{spec.tendency_trait}: current-K replay parity max diff={max_diff:.3e}")
    return pd.DataFrame(rows)


def _mask_window(frame: pd.DataFrame, window: str) -> pd.Series:
    if window == "modern_2020plus":
        return frame["event_date"] >= MODERN_START
    if window == "fresh_2025_0329plus":
        return frame["event_date"] >= FRESH_START
    raise ValueError(window)


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    output: list[dict] = []
    windows = ["modern_2020plus", "fresh_2025_0329plus"]
    buckets = ["0", "1", "2", "3plus", "ALL"]

    for family in sorted(scores["family"].unique()):
        family_frame = scores[scores["family"] == family]
        for k_seconds in sorted(family_frame["k_seconds"].unique()):
            candidate = family_frame[family_frame["k_seconds"] == k_seconds]
            for window in windows:
                window_frame = candidate[_mask_window(candidate, window)]
                for bucket in buckets:
                    part = window_frame if bucket == "ALL" else window_frame[window_frame["prior_bucket"] == bucket]
                    part = part[np.isfinite(part["posterior_predictive_nll"])]
                    if part.empty:
                        continue
                    output.append(
                        {
                            "family": family,
                            "k_seconds": float(k_seconds),
                            "is_current_k": bool(part["is_current_k"].iloc[0]),
                            "window": window,
                            "prior_bucket": bucket,
                            "rows": int(len(part)),
                            "fights": int(part["fight_id"].nunique()),
                            "posterior_predictive_nll": float(part["posterior_predictive_nll"].mean()),
                            "plugin_nll": float(part["plugin_nll"].mean()),
                            "mean_count_bias": float(part["count_error"].mean()),
                            "mean_abs_count_error": float(part["abs_count_error"].mean()),
                            "mean_predicted_count": float(part["predicted_count"].mean()),
                            "mean_actual_count": float(part["actual_count"].mean()),
                            "mean_pre_rate_15m": float(part["pre_rate_15m"].mean()),
                        }
                    )

    metrics = pd.DataFrame(output)
    current = metrics[metrics["is_current_k"]][
        ["family", "window", "prior_bucket", "posterior_predictive_nll", "plugin_nll", "mean_abs_count_error"]
    ].rename(
        columns={
            "posterior_predictive_nll": "current_posterior_predictive_nll",
            "plugin_nll": "current_plugin_nll",
            "mean_abs_count_error": "current_mean_abs_count_error",
        }
    )
    metrics = metrics.merge(current, on=["family", "window", "prior_bucket"], how="left", validate="many_to_one")
    metrics["delta_posterior_predictive_nll_vs_current"] = (
        metrics["posterior_predictive_nll"] - metrics["current_posterior_predictive_nll"]
    )
    metrics["delta_plugin_nll_vs_current"] = metrics["plugin_nll"] - metrics["current_plugin_nll"]
    metrics["delta_mae_vs_current"] = metrics["mean_abs_count_error"] - metrics["current_mean_abs_count_error"]
    return metrics


def _cluster_bootstrap_delta(
    scores: pd.DataFrame,
    family: str,
    candidate_k: float,
    window: str,
    bucket: str,
    current_k: float,
) -> dict:
    frame = scores[scores["family"] == family].copy()
    frame = frame[_mask_window(frame, window)]
    if bucket != "ALL":
        frame = frame[frame["prior_bucket"] == bucket]
    frame = frame[np.isfinite(frame["posterior_predictive_nll"])]

    index_cols = ["event_date", "fight_id", "fighter_id"]
    pivot = frame.pivot_table(
        index=index_cols,
        columns="k_seconds",
        values="posterior_predictive_nll",
        aggfunc="first",
    ).reset_index()
    if candidate_k not in pivot.columns or current_k not in pivot.columns:
        raise RuntimeError(f"missing candidate/current score for bootstrap {family} {candidate_k}")
    pivot["delta"] = pivot[candidate_k] - pivot[current_k]
    clusters = pivot.groupby("fight_id")["delta"].agg(["sum", "count"]).reset_index()
    sums = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    n = len(clusters)
    if n == 0:
        return {
            "family": family,
            "candidate_k": candidate_k,
            "current_k": current_k,
            "window": window,
            "prior_bucket": bucket,
            "rows": 0,
            "fights": 0,
            "mean_delta_nll": np.nan,
            "ci_2_5": np.nan,
            "ci_97_5": np.nan,
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED + int(round(candidate_k * 10)) + len(family) + len(window) + len(bucket))
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        draws[i] = sums[idx].sum() / counts[idx].sum()
    return {
        "family": family,
        "candidate_k": float(candidate_k),
        "current_k": float(current_k),
        "window": window,
        "prior_bucket": bucket,
        "rows": int(counts.sum()),
        "fights": int(n),
        "mean_delta_nll": float(sums.sum() / counts.sum()),
        "ci_2_5": float(np.quantile(draws, 0.025)),
        "ci_97_5": float(np.quantile(draws, 0.975)),
    }


def _bootstrap(scores: pd.DataFrame, specs: dict[str, RateFamilySpec]) -> pd.DataFrame:
    rows: list[dict] = []
    for family, spec in specs.items():
        current_k = float(spec.tendency_prior_seconds)
        for candidate_k in CANDIDATES[family]:
            if np.isclose(candidate_k, current_k, atol=1e-12, rtol=0.0):
                continue
            for window in ["modern_2020plus", "fresh_2025_0329plus"]:
                for bucket in ["1", "ALL"]:
                    rows.append(
                        _cluster_bootstrap_delta(
                            scores,
                            family,
                            float(candidate_k),
                            window,
                            bucket,
                            current_k,
                        )
                    )
    return pd.DataFrame(rows)


def _lookup_delta(metrics: pd.DataFrame, family: str, k: float, window: str, bucket: str) -> float:
    row = metrics[
        (metrics["family"] == family)
        & np.isclose(metrics["k_seconds"], k)
        & (metrics["window"] == window)
        & (metrics["prior_bucket"] == bucket)
    ]
    return float(row.iloc[0]["delta_posterior_predictive_nll_vs_current"]) if not row.empty else np.nan


def _headline(metrics: pd.DataFrame, bootstrap: pd.DataFrame, specs: dict[str, RateFamilySpec]) -> pd.DataFrame:
    rows: list[dict] = []
    for family, spec in specs.items():
        current_k = float(spec.tendency_prior_seconds)
        primary = metrics[
            (metrics["family"] == family)
            & (metrics["window"] == "modern_2020plus")
            & (metrics["prior_bucket"] == "1")
        ].sort_values("posterior_predictive_nll")
        best = primary.iloc[0]
        best_k = float(best["k_seconds"])
        modern_one_delta = float(best["delta_posterior_predictive_nll_vs_current"])
        fresh_one_delta = _lookup_delta(metrics, family, best_k, "fresh_2025_0329plus", "1")
        modern_all_delta = _lookup_delta(metrics, family, best_k, "modern_2020plus", "ALL")
        modern_two_delta = _lookup_delta(metrics, family, best_k, "modern_2020plus", "2")
        modern_three_delta = _lookup_delta(metrics, family, best_k, "modern_2020plus", "3plus")

        if np.isclose(best_k, current_k, atol=1e-12, rtol=0.0):
            ci_low = ci_high = 0.0
            verdict = "KEEP_CURRENT"
        else:
            boot = bootstrap[
                (bootstrap["family"] == family)
                & np.isclose(bootstrap["candidate_k"], best_k)
                & (bootstrap["window"] == "modern_2020plus")
                & (bootstrap["prior_bucket"] == "1")
            ].iloc[0]
            ci_low = float(boot["ci_2_5"])
            ci_high = float(boot["ci_97_5"])
            stronger = best_k > current_k
            consistent = (
                modern_one_delta < 0.0
                and fresh_one_delta < 0.0
                and modern_all_delta <= 0.0
                and modern_three_delta <= 0.0
            )
            if stronger and ci_high < 0.0 and consistent:
                verdict = "STRONGER_PRIOR_SUPPORTED"
            elif stronger and modern_one_delta < 0.0:
                verdict = "STRONGER_PRIOR_MIXED_NO_PROMOTION"
            elif best_k < current_k:
                verdict = "WEAKER_PRIOR_BEST_NO_STRONGER_SUPPORT"
            else:
                verdict = "NO_PROMOTION"

        rows.append(
            {
                "family": family,
                "current_k_seconds": current_k,
                "best_modern_1prior_k_seconds": best_k,
                "modern_1prior_delta_nll": modern_one_delta,
                "modern_1prior_boot_ci_2_5": ci_low,
                "modern_1prior_boot_ci_97_5": ci_high,
                "fresh_1prior_delta_nll": fresh_one_delta,
                "modern_2prior_delta_nll": modern_two_delta,
                "modern_3plus_delta_nll": modern_three_delta,
                "modern_all_delta_nll": modern_all_delta,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = FSRV3Config()
    specs = {
        "standing_striking_tendency": standing_spec(config),
        "takedown_tendency": takedown_spec(config),
    }

    all_scores = []
    for family, spec in specs.items():
        print("=" * 110)
        print(f"PRIOR STRENGTH STUDY — {family}")
        print(f"current K: {spec.tendency_prior_seconds:.2f} seconds")
        print(f"candidates: {CANDIDATES[family]}")
        all_scores.append(_score_family(spec, CANDIDATES[family]))

    scores = pd.concat(all_scores, ignore_index=True)
    metrics = _summarize(scores)
    bootstrap = _bootstrap(scores, specs)
    headline = _headline(metrics, bootstrap, specs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "prior_strength_scores.csv", index=False)
    metrics.to_csv(OUT_DIR / "prior_strength_metrics.csv", index=False)
    bootstrap.to_csv(OUT_DIR / "prior_strength_bootstrap.csv", index=False)
    headline.to_csv(OUT_DIR / "prior_strength_headline.csv", index=False)

    print("\n" + "=" * 110)
    print("HEADLINE")
    print(headline.to_string(index=False))
    print("\nPrimary modern exactly-one-prior candidate table (negative delta is better):")
    primary = metrics[
        (metrics["window"] == "modern_2020plus")
        & (metrics["prior_bucket"] == "1")
    ][
        [
            "family",
            "k_seconds",
            "rows",
            "fights",
            "posterior_predictive_nll",
            "delta_posterior_predictive_nll_vs_current",
            "plugin_nll",
            "delta_plugin_nll_vs_current",
            "mean_abs_count_error",
            "delta_mae_vs_current",
        ]
    ]
    print(primary.sort_values(["family", "k_seconds"]).to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
