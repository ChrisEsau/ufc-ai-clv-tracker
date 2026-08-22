"""Sequential FSR V3 tendency replay with an optional external prior seed."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pipeline.fsr_v3.replay.math import nb2_log_likelihood, normalize_log_weights, weighted_mean_sd
from pipeline.fsr_v3.replay.rate_families import RateFamilySpec

from .priors import combine_positive_rate_prior, gamma_logweights

KEYS = ["event_date", "fight_id", "fighter_id"]


def add_prior_ufc_fight_count(frame: pd.DataFrame) -> pd.DataFrame:
    """Count only fights on strictly earlier dates; same-date updates stay delayed."""
    x = frame.copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="raise").dt.normalize()
    counts: dict[str, int] = defaultdict(int)
    out = np.zeros(len(x), dtype=int)
    for _, idx in x.groupby("event_date", sort=True).groups.items():
        positions = list(idx)
        for pos in positions:
            out[x.index.get_loc(pos)] = counts[str(x.loc[pos, "fighter_id"])]
        for pos in positions:
            counts[str(x.loc[pos, "fighter_id"])] += 1
    x["prior_ufc_fights"] = out
    return x


def score_seeded_tendency(
    history: pd.DataFrame,
    spec: RateFamilySpec,
    *,
    external_rate_column: str = "external_predicted_rate_15m",
    external_bucket_column: str = "evidence_bucket",
    extra_seconds_by_bucket: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reconstruct the native posterior predictive with optional external seed.

    ``history`` is the normal V3 tendency replay joined to leakage-safe external
    features. UFC observation likelihoods are accumulated exactly as production
    does. The only change is the prior kernel used before those UFC likelihoods:
    population Gamma prior multiplied by calibrated external pseudo-evidence.
    """
    extra_seconds_by_bucket = dict(extra_seconds_by_bucket or {})
    required = {
        "event_date", "fight_id", "fighter_id", "fighter_name", "numerator",
        "denominator", "population_rate_15m", "observation_alpha",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"seeded tendency history missing columns: {missing}")

    q_grid = np.linspace(
        spec.tendency_grid_min,
        spec.tendency_grid_max,
        spec.tendency_grid_points,
    )
    x = history.copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="raise").dt.normalize()
    x["fight_id"] = x["fight_id"].astype(str)
    x["fighter_id"] = x["fighter_id"].astype(str)
    x = x.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)

    evidence: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for event_date, batch in x.groupby("event_date", sort=True):
        pending: list[tuple[str, np.ndarray]] = []
        for record in batch.to_dict("records"):
            fighter = str(record["fighter_id"])
            q_pop = float(record["population_rate_15m"])
            alpha = float(record["observation_alpha"])
            exposure = float(record["denominator"])
            y = float(record["numerator"])

            bucket = str(record.get(external_bucket_column, "none") or "none")
            k_ext = max(float(extra_seconds_by_bucket.get(bucket, 0.0)), 0.0)
            q_ext_raw = record.get(external_rate_column, np.nan)
            q_ext = float(q_ext_raw) if pd.notna(q_ext_raw) and float(q_ext_raw) > 0.0 else None
            if bucket == "none":
                k_ext = 0.0
                q_ext = None

            prior = combine_positive_rate_prior(
                population_mean_rate_15m=q_pop,
                population_seconds=float(spec.tendency_prior_seconds),
                external_mean_rate_15m=q_ext,
                extra_seconds=k_ext,
            )
            lp = gamma_logweights(q_grid, prior)
            if fighter in evidence:
                lp = lp + evidence[fighter]
            weights = normalize_log_weights(lp)
            pre_mean, pre_sd = weighted_mean_sd(q_grid, weights)

            if exposure > 0.0:
                observation_ll = nb2_log_likelihood(
                    y,
                    exposure / 900.0 * q_grid,
                    alpha,
                )
                predictive_ll = float(logsumexp(lp + observation_ll) - logsumexp(lp))
                post_weights = normalize_log_weights(lp + observation_ll)
                post_mean, post_sd = weighted_mean_sd(q_grid, post_weights)
                pending.append((fighter, observation_ll))
            else:
                observation_ll = None
                predictive_ll = np.nan
                post_mean, post_sd = pre_mean, pre_sd

            rows.append(
                {
                    **record,
                    "cold_pre_rating": pre_mean,
                    "cold_pre_posterior_sd": pre_sd,
                    "cold_post_rating": post_mean,
                    "cold_post_posterior_sd": post_sd,
                    "cold_predictive_ll": predictive_ll,
                    "cold_external_rate_15m": q_ext,
                    "cold_external_seconds": k_ext,
                    "cold_total_prior_seconds": prior.total_seconds,
                }
            )

        # Same-date delayed update, matching production semantics.
        for fighter, observation_ll in pending:
            if fighter in evidence:
                evidence[fighter] = evidence[fighter] + observation_ll
                evidence[fighter] -= np.max(evidence[fighter])
            else:
                evidence[fighter] = observation_ll - np.max(observation_ll)

    return pd.DataFrame(rows)


def paired_fight_bootstrap(
    scored: pd.DataFrame,
    *,
    delta_column: str,
    n_boot: int = 5000,
    seed: int = 20260821,
) -> dict[str, float]:
    """Bootstrap whole fights for paired log-likelihood deltas."""
    by_fight = (
        scored.dropna(subset=[delta_column])
        .groupby("fight_id", as_index=False)[delta_column]
        .sum()
    )
    values = by_fight[delta_column].to_numpy(float)
    if len(values) == 0:
        return {"fights": 0, "rows": 0, "delta_ll": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_boot), len(values)), replace=True).sum(axis=1)
    return {
        "fights": int(len(values)),
        "rows": int(scored[delta_column].notna().sum()),
        "delta_ll": float(values.sum()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }
