"""Generic, simultaneous-by-date chronological FSR V2 replay engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log

import numpy as np
import pandas as pd

from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.traits.registry import TraitGroup


FIGHT_KEYS = ["event_date", "fight_id", "fighter_id", "fighter_name", "opponent_id", "opponent_name"]

# Submission tendency is sparse enough that a fighter with little UFC history
# should retain meaningful population-level submission threat.  The population
# prior fades by UFC fight count while fighter-specific evidence comes from
# effective submission attempts.
SUBMISSION_TENDENCY_INITIAL_RATE = 0.0005947774095442215
SUBMISSION_TENDENCY_PRIOR_SECONDS = 2700.0
SUBMISSION_SUPPRESSION_PRIOR_EXPECTED_ATTEMPTS = 3.0

# Validated Stage-1 Endpoint-2 rule: retain the standard 900-second
# population prior through one prior UFC fight, then use the fighter's
# raw observed takedown tendency from two prior UFC fights onward.
TAKEDOWN_TENDENCY_RAW_AFTER_PRIOR_FIGHTS = 2


def aggregate_fights(rounds: pd.DataFrame) -> pd.DataFrame:
    """Create an in-memory replay frame; this frame is never persisted."""
    sum_columns = [
        "distance_landed", "distance_attempted", "head_attempted", "body_attempted",
        "leg_attempted", "td_landed", "td_attempted", "ctrl_sec", "ground_entries",
        "ground_landed", "ground_attempted", "effective_submission_attempts", "sub_att", "rev",
        "opponent_distance_attempted", "opponent_td_landed", "opponent_td_attempted", "opponent_ctrl_sec",
        "opponent_ground_entries", "opponent_ground_attempted",
        "opponent_effective_submission_attempts", "opponent_sub_att",
        "standing_exposure_seconds", "td_tendency_exposure_seconds",
        "td_suppression_exposure_seconds", "modeled_ground_exposure_seconds",
        "qualified_control_inflicted_seconds", "qualified_control_suffered_seconds",
        "round_elapsed_seconds",
    ]
    aggregations = {column: "sum" for column in sum_columns}
    aggregations.update({
        "submission_finish": "max", "method": "first", "winner_id": "first",
        "ground_exposure_fallback_used": "sum", "inferred_ground_entry": "sum",
        "match_time_interpretation": "first",
    })
    fights = rounds.groupby(FIGHT_KEYS, as_index=False).agg(aggregations)
    fights["fight_elapsed_seconds"] = fights["round_elapsed_seconds"]
    fights["head_body_attempted"] = fights["head_attempted"] + fights["body_attempted"]
    fights["target_attempted"] = (
        fights["head_attempted"] + fights["body_attempted"] + fights["leg_attempted"]
    )
    return fights.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(-30.0, min(30.0, value))))


def _logit(value: float) -> float:
    value = min(1 - 1e-6, max(1e-6, value))
    return log(value / (1 - value))


@dataclass
class ReplayResult:
    history: pd.DataFrame
    state: dict
    population: dict


class ReplayEngine:
    def __init__(self, config: FSRV2Config | None = None):
        self.config = config or FSRV2Config()

    def replay(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        if group.name == "submission_tendency":
            return self._submission_tendency(group, fights)
        if group.name == "submission_suppression":
            return self._submission_suppression(group, fights)
        if group.kind in {"behavior", "composition"}:
            return self._behavior(group, fights)
        if group.kind == "suppression":
            return self._suppression(group, fights)
        if group.kind == "paired":
            return self._paired(group, fights)
        if group.kind == "takedown":
            return self._takedown(group, fights)
        if group.kind == "escape":
            return self._escape(group, fights)
        raise ValueError(f"Unsupported replay kind: {group.kind}")

    def _submission_tendency(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        """Replay submission attempts per fight-second with a population exposure prior."""
        states: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        prior_seconds = SUBMISSION_TENDENCY_PRIOR_SECONDS

        for _, batch in fights.groupby("event_date", sort=True):
            pending: list[tuple[str, float, float]] = []

            global_rate = (
                population[0] / population[1]
                if population[1] > 0
                else SUBMISSION_TENDENCY_INITIAL_RATE
            )

            for record in batch.to_dict("records"):
                fighter = record["fighter_id"]
                numerator = float(record[group.numerator])
                denominator = float(record[group.denominator])

                prior_n, prior_d = states[fighter]

                pre = (
                    prior_n + global_rate * prior_seconds
                ) / (
                    prior_d + prior_seconds
                )

                observation = (
                    numerator / denominator
                    if denominator > 0
                    else np.nan
                )

                post_n = prior_n + numerator
                post_d = prior_d + denominator

                post = (
                    post_n + global_rate * prior_seconds
                ) / (
                    post_d + prior_seconds
                )

                row = self._row(
                    record,
                    group.traits[0],
                    pre,
                    post,
                    observation,
                    numerator,
                    denominator,
                )
                row.update({
                    "population_prior_rate": global_rate,
                    "population_prior_seconds": prior_seconds,
                    "fighter_prior_attempts": prior_n,
                    "fighter_prior_exposure_seconds": prior_d,
                })
                rows.append(row)

                pending.append((fighter, numerator, denominator))

            # Same-event delayed updates prevent leakage.
            for fighter, numerator, denominator in pending:
                if denominator > 0:
                    states[fighter][0] += numerator
                    states[fighter][1] += denominator
                    population[0] += numerator
                    population[1] += denominator

        history = pd.DataFrame(rows)

        final_rate = (
            population[0] / population[1]
            if population[1] > 0
            else SUBMISSION_TENDENCY_INITIAL_RATE
        )

        history["latest_rating"] = np.nan
        history["latest_population_baseline"] = final_rate

        for fighter, (numerator, denominator) in states.items():
            latest = (
                numerator + final_rate * prior_seconds
            ) / (
                denominator + prior_seconds
            )

            mask = history["fighter_id"].eq(fighter)
            if mask.any():
                history.loc[
                    history.index[mask][-1],
                    "latest_rating",
                ] = latest

        return ReplayResult(
            history,
            dict(states),
            {
                "numerator": population[0],
                "denominator": population[1],
            },
        )

    def _submission_suppression(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        """Replay multiplicative submission-attempt suppression.

        Rating semantics:
            1.0 = neutral
            <1.0 = suppresses opponent submission attempts
            >1.0 = permits more opponent submission attempts than expected
        """
        prior_expected = SUBMISSION_SUPPRESSION_PRIOR_EXPECTED_ATTEMPTS

        # Build leakage-safe prefight opponent submission tendencies using
        # the exact submission-tendency replay semantics.
        tendency_group = TraitGroup(
            name="submission_tendency",
            kind="behavior",
            traits=("submission_tendency",),
            numerator="effective_submission_attempts",
            denominator="fight_elapsed_seconds",
        )
        tendency_history = self._submission_tendency(
            tendency_group, fights
        ).history

        tendency_lookup = {
            (row.fight_id, row.fighter_id): float(row.pre_rating)
            for row in tendency_history.itertuples()
            if row.trait == "submission_tendency"
        }

        # defender -> [actual opponent attempts allowed,
        #              expected opponent attempts]
        states: dict[str, list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        rows: list[dict] = []

        for _, batch in fights.groupby("event_date", sort=True):
            pending: list[tuple[str, float, float]] = []

            for record in batch.to_dict("records"):
                defender = record["fighter_id"]
                opponent = record["opponent_id"]

                actual_hist, expected_hist = states[defender]

                pre = (
                    actual_hist + prior_expected
                ) / (
                    expected_hist + prior_expected
                )

                seconds = float(record["fight_elapsed_seconds"])
                opponent_rate = tendency_lookup[
                    (record["fight_id"], opponent)
                ]

                expected_attempts = opponent_rate * seconds
                actual_attempts = float(record[group.numerator])

                post = (
                    actual_hist + actual_attempts + prior_expected
                ) / (
                    expected_hist + expected_attempts + prior_expected
                )

                observed = (
                    actual_attempts / expected_attempts
                    if expected_attempts > 0
                    else np.nan
                )

                row = self._row(
                    record,
                    group.traits[0],
                    pre,
                    post,
                    observed,
                    actual_attempts,
                    expected_attempts,
                )
                row.update({
                    "opponent_expected_rate": opponent_rate,
                    "opponent_expected_attempts": expected_attempts,
                    "opponent_actual_attempts": actual_attempts,
                    "suppression_prior_expected_attempts": prior_expected,
                })
                rows.append(row)

                pending.append(
                    (defender, actual_attempts, expected_attempts)
                )

            # Same-event delayed updates prevent leakage.
            for defender, actual_attempts, expected_attempts in pending:
                states[defender][0] += actual_attempts
                states[defender][1] += expected_attempts

        history = pd.DataFrame(rows)
        history["latest_rating"] = np.nan

        for fighter, (actual_hist, expected_hist) in states.items():
            latest = (
                actual_hist + prior_expected
            ) / (
                expected_hist + prior_expected
            )

            mask = history["fighter_id"].eq(fighter)
            if mask.any():
                history.loc[
                    history.index[mask][-1],
                    "latest_rating",
                ] = latest

        return ReplayResult(
            history,
            dict(states),
            {
                "actual_attempts": sum(v[0] for v in states.values()),
                "expected_attempts": sum(v[1] for v in states.values()),
            },
        )

    def _behavior(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        states: dict[str, list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        fight_counts: dict[str, int] = defaultdict(int)
        population = [0.0, 0.0]
        rows: list[dict] = []

        base_prior_weight = (
            self.config.target_composition_prior_attempts
            if group.kind == "composition"
            else self.config.behavior_prior_seconds
        )

        def effective_prior_weight(
            prior_ufc_fights: int,
        ) -> float:
            if (
                group.name == "takedown_tendency"
                and prior_ufc_fights
                >= TAKEDOWN_TENDENCY_RAW_AFTER_PRIOR_FIGHTS
            ):
                return 0.0

            return float(base_prior_weight)

        def blended_rate(
            numerator: float,
            denominator: float,
            population_rate: float,
            prior_weight: float,
        ) -> float:
            total_denominator = (
                denominator + prior_weight
            )

            if total_denominator <= 0:
                return population_rate

            return (
                numerator
                + population_rate * prior_weight
            ) / total_denominator

        for _, batch in fights.groupby(
            "event_date",
            sort=True,
        ):
            pending_updates: list[
                tuple[str, float, float]
            ] = []

            pending_fight_counts: list[str] = []

            global_rate = (
                population[0] / population[1]
                if population[1] > 0
                else 0.0
            )

            for record in batch.to_dict("records"):
                fighter = record["fighter_id"]

                numerator = float(
                    record[group.numerator]
                )

                denominator = float(
                    record[group.denominator]
                )

                prior_n, prior_d = states[fighter]
                prior_fights = fight_counts[fighter]

                pre_prior_weight = (
                    effective_prior_weight(
                        prior_fights
                    )
                )

                pre = blended_rate(
                    prior_n,
                    prior_d,
                    global_rate,
                    pre_prior_weight,
                )

                observation = (
                    numerator / denominator
                    if denominator > 0
                    else np.nan
                )

                if denominator > 0:
                    post_n = prior_n + numerator
                    post_d = prior_d + denominator
                else:
                    post_n = prior_n
                    post_d = prior_d

                post_prior_weight = (
                    effective_prior_weight(
                        prior_fights + 1
                    )
                )

                post = blended_rate(
                    post_n,
                    post_d,
                    global_rate,
                    post_prior_weight,
                )

                row = self._row(
                    record,
                    group.traits[0],
                    pre,
                    post,
                    observation,
                    numerator,
                    denominator,
                )

                if group.name == "takedown_tendency":
                    row.update(
                        {
                            "prior_ufc_fights":
                                prior_fights,
                            "population_prior_seconds":
                                pre_prior_weight,
                        }
                    )

                rows.append(row)

                if denominator > 0:
                    pending_updates.append(
                        (
                            fighter,
                            numerator,
                            denominator,
                        )
                    )

                pending_fight_counts.append(
                    fighter
                )

            # Same-event delayed updates preserve
            # leakage-safe pre-fight state.
            for (
                fighter,
                numerator,
                denominator,
            ) in pending_updates:
                states[fighter][0] += numerator
                states[fighter][1] += denominator

                population[0] += numerator
                population[1] += denominator

            for fighter in pending_fight_counts:
                fight_counts[fighter] += 1

        history = pd.DataFrame(rows)

        final_rate = (
            population[0] / population[1]
            if population[1] > 0
            else 0.0
        )

        history["latest_rating"] = np.nan

        for fighter, (
            numerator,
            denominator,
        ) in states.items():
            mask = history[
                "fighter_id"
            ].eq(fighter)

            if not mask.any():
                continue

            final_prior_weight = (
                effective_prior_weight(
                    fight_counts[fighter]
                )
            )

            latest = blended_rate(
                numerator,
                denominator,
                final_rate,
                final_prior_weight,
            )

            history.loc[
                history.index[mask][-1],
                "latest_rating",
            ] = latest

        history[
            "latest_population_baseline"
        ] = final_rate

        if len(group.traits) == 2:
            complement = history.copy()

            complement["trait"] = (
                group.traits[1]
            )

            for column in [
                "pre_rating",
                "post_rating",
                "observed",
            ]:
                complement[column] = (
                    1.0
                    - complement[column]
                )

            complement[
                "latest_rating"
            ] = (
                1.0
                - complement[
                    "latest_rating"
                ]
            )

            complement[
                "latest_population_baseline"
            ] = (
                1.0
                - complement[
                    "latest_population_baseline"
                ]
            )

            history = pd.concat(
                [
                    history,
                    complement,
                ],
                ignore_index=True,
            )

        return ReplayResult(
            history,
            dict(states),
            {
                "numerator": population[0],
                "denominator": population[1],
            },
        )

    def _suppression(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        states: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        opponent_states: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        for date, batch in fights.groupby("event_date", sort=True):
            pending_suppression = []
            pending_tendency = []
            global_rate = population[0] / population[1] if population[1] else 0.0
            for record in batch.to_dict("records"):
                opponent_n, opponent_d = opponent_states[record["opponent_id"]]
                expected = (opponent_n + global_rate * self.config.behavior_prior_seconds) / (
                    opponent_d + self.config.behavior_prior_seconds
                )
                denominator = float(record[group.denominator])
                actual = float(record[group.numerator]) / denominator if denominator > 0 else np.nan
                residual = expected - actual if denominator > 0 else np.nan
                sum_residual, exposure = states[record["fighter_id"]]
                pre = sum_residual / (exposure + self.config.suppression_prior_seconds)
                post = (sum_residual + (residual * denominator if denominator > 0 else 0.0)) / (
                    exposure + max(0.0, denominator) + self.config.suppression_prior_seconds
                )
                row = self._row(record, group.traits[0], pre, post, residual, float(record[group.numerator]), denominator)
                row.update({"opponent_expected_rate": expected, "opponent_actual_rate": actual})
                rows.append(row)
                if denominator > 0:
                    pending_suppression.append((record["fighter_id"], residual * denominator, denominator))
                    pending_tendency.append((record["opponent_id"], float(record[group.numerator]), denominator))
            for fighter, residual_sum, denominator in pending_suppression:
                states[fighter][0] += residual_sum; states[fighter][1] += denominator
            for fighter, numerator, denominator in pending_tendency:
                opponent_states[fighter][0] += numerator; opponent_states[fighter][1] += denominator
                population[0] += numerator; population[1] += denominator
        return ReplayResult(pd.DataFrame(rows), dict(states), {"numerator": population[0], "denominator": population[1]})

    def _paired(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        offense: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        defense: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        for date, batch in fights.groupby("event_date", sort=True):
            pending = []
            baseline = population[0] / population[1] if population[1] else 0.5
            for record in batch.to_dict("records"):
                numerator = float(record[group.numerator])
                denominator = float(record[group.denominator])
                if group.name == "submission_effectiveness":
                    denominator = max(denominator, numerator)
                off_pre = offense[record["fighter_id"]][0]
                def_pre = defense[record["opponent_id"]][0]
                expected = _logistic(_logit(baseline) + (off_pre - def_pre) / self.config.rating_scale)
                observed = numerator / denominator if denominator > 0 else np.nan
                saturation_attempts = (
                    self.config.submission_effectiveness_saturation_attempts
                    if group.name == "submission_effectiveness"
                    else self.config.evidence_saturation_attempts
                )
                evidence = (
                    1.0 - exp(-denominator / saturation_attempts)
                    if denominator > 0
                    else 0.0
                )
                update = self.config.elo_k * evidence * (observed - expected) if denominator > 0 else 0.0
                common = self._row(record, group.traits[0], off_pre, off_pre + update, observed, numerator, denominator)
                common.update({"opponent_pre_rating": def_pre, "expected": expected, "evidence_strength": evidence, "update": update})
                common["population_baseline"] = baseline
                rows.append(common)
                defense_row = common.copy()
                defense_row.update({"fighter_id": record["opponent_id"], "fighter_name": record["opponent_name"], "opponent_id": record["fighter_id"], "opponent_name": record["fighter_name"], "trait": group.traits[1], "pre_rating": def_pre, "opponent_pre_rating": off_pre, "post_rating": def_pre - update, "update": -update})
                rows.append(defense_row)
                if denominator > 0:
                    pending.append((record["fighter_id"], record["opponent_id"], update, denominator, numerator))
            for attacker, defender, update, denominator, numerator in pending:
                offense[attacker][0] += update; offense[attacker][1] += denominator
                defense[defender][0] -= update; defense[defender][1] += denominator
                population[0] += numerator; population[1] += denominator
        history = pd.DataFrame(rows)
        final_baseline = population[0] / population[1] if population[1] else 0.5
        history["latest_population_baseline"] = final_baseline
        return ReplayResult(history, {"offense": dict(offense), "defense": dict(defense)}, {"numerator": population[0], "denominator": population[1]})

    def _takedown(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        """Final cumulative, population-centered TD offense/defense model."""
        offense: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        defense: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        prior = self.config.takedown_effectiveness_prior_attempts
        for _, batch in fights.groupby("event_date", sort=True):
            baseline = population[0] / population[1] if population[1] else 0.5
            stop_baseline = 1.0 - baseline
            pending = []
            for record in batch.to_dict("records"):
                landed = float(record["td_landed"])
                attempted = float(record["td_attempted"])
                opp_attempted = float(record["opponent_td_attempted"])
                opp_landed = float(record.get("opponent_td_landed", 0.0))
                off_landed, off_attempted = offense[record["fighter_id"]]
                stopped, faced = defense[record["fighter_id"]]
                off_rate = (off_landed + baseline * prior) / (off_attempted + prior)
                stop_rate = (stopped + stop_baseline * prior) / (faced + prior)
                off_pre = _logit(off_rate) - _logit(baseline)
                def_pre = _logit(stop_rate) - _logit(stop_baseline)
                off_post = _logit((off_landed + landed + baseline * prior) /
                                  (off_attempted + attempted + prior)) - _logit(baseline)
                def_post = _logit((stopped + opp_attempted - opp_landed + stop_baseline * prior) /
                                  (faced + opp_attempted + prior)) - _logit(stop_baseline)
                off_row = self._row(record, group.traits[0], off_pre, off_post,
                                    landed / attempted if attempted else np.nan,
                                    landed, attempted)
                off_row["population_baseline"] = baseline
                rows.append(off_row)
                def_row = self._row(record, group.traits[1], def_pre, def_post,
                                    (opp_attempted - opp_landed) / opp_attempted if opp_attempted else np.nan,
                                    opp_attempted - opp_landed, opp_attempted)
                def_row["population_baseline"] = stop_baseline
                rows.append(def_row)
                pending.append((record["fighter_id"], landed, attempted,
                                opp_attempted - opp_landed, opp_attempted))
            for fighter, landed, attempted, stopped, faced in pending:
                offense[fighter][0] += landed; offense[fighter][1] += attempted
                defense[fighter][0] += stopped; defense[fighter][1] += faced
                population[0] += landed; population[1] += attempted
        final_baseline = population[0] / population[1] if population[1] else 0.5
        history = pd.DataFrame(rows)
        history["latest_rating"] = np.nan
        history["latest_population_baseline"] = np.where(
            history["trait"].eq(group.traits[0]), final_baseline, 1.0 - final_baseline
        )
        for fighter in set(offense) | set(defense):
            values = {
                group.traits[0]: _logit((offense[fighter][0] + final_baseline * prior) /
                                        (offense[fighter][1] + prior)) - _logit(final_baseline),
                group.traits[1]: _logit((defense[fighter][0] + (1-final_baseline) * prior) /
                                        (defense[fighter][1] + prior)) - _logit(1-final_baseline),
            }
            for trait, value in values.items():
                mask = history["fighter_id"].eq(fighter) & history["trait"].eq(trait)
                if mask.any(): history.loc[history.index[mask][-1], "latest_rating"] = value
        return ReplayResult(history, {"offense": dict(offense), "defense": dict(defense)},
                            {"numerator": population[0], "denominator": population[1]})

    def _escape(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        suffered: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        inflicted: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        prior = self.config.escape_prior_entries
        for _, batch in fights.groupby("event_date", sort=True):
            mu_pop = population[0] / population[1] if population[1] else 60.0
            pending = []
            for record in batch.to_dict("records"):
                suffered_duration = float(record["qualified_control_suffered_seconds"])
                suffered_entries = float(record["opponent_ground_entries"])
                inflicted_duration = float(record["qualified_control_inflicted_seconds"])
                inflicted_entries = float(record["ground_entries"])
                s_duration, s_entries = suffered[record["fighter_id"]]
                i_duration, i_entries = inflicted[record["fighter_id"]]
                mu_suffered = (s_duration + mu_pop * prior) / (s_entries + prior)
                mu_inflicted = (i_duration + mu_pop * prior) / (i_entries + prior)
                off = log(mu_pop / mu_suffered)
                defense = log(mu_inflicted / mu_pop)
                off_post = log(mu_pop / ((s_duration + suffered_duration + mu_pop * prior) /
                                         (s_entries + suffered_entries + prior)))
                def_post = log(((i_duration + inflicted_duration + mu_pop * prior) /
                                (i_entries + inflicted_entries + prior)) / mu_pop)
                off_row = self._row(record, group.traits[0], off, off_post,
                                    suffered_duration / suffered_entries if suffered_entries else np.nan,
                                    suffered_duration, suffered_entries)
                def_row = self._row(record, group.traits[1], defense, def_post,
                                    inflicted_duration / inflicted_entries if inflicted_entries else np.nan,
                                    inflicted_duration, inflicted_entries)
                for row in (off_row, def_row):
                    row["population_duration_baseline_seconds"] = mu_pop
                    rows.append(row)
                pending.append((record["fighter_id"], suffered_duration, suffered_entries,
                                inflicted_duration, inflicted_entries))
            for fighter, sd, se, iduration, ie in pending:
                suffered[fighter][0] += sd; suffered[fighter][1] += se
                inflicted[fighter][0] += iduration; inflicted[fighter][1] += ie
                population[0] += iduration; population[1] += ie
        final_mean = population[0] / population[1] if population[1] else 60.0
        history = pd.DataFrame(rows)
        history["latest_rating"] = np.nan
        history["latest_population_duration_baseline_seconds"] = final_mean
        for fighter in set(suffered) | set(inflicted):
            latest = {
                group.traits[0]: log(final_mean / ((suffered[fighter][0] + final_mean * prior) /
                                                    (suffered[fighter][1] + prior))),
                group.traits[1]: log(((inflicted[fighter][0] + final_mean * prior) /
                                      (inflicted[fighter][1] + prior)) / final_mean),
            }
            for trait, value in latest.items():
                mask = history["fighter_id"].eq(fighter) & history["trait"].eq(trait)
                if mask.any(): history.loc[history.index[mask][-1], "latest_rating"] = value
        return ReplayResult(history, {"suffered": dict(suffered), "inflicted": dict(inflicted)},
                            {"duration": population[0], "entries": population[1]})

    @staticmethod
    def _row(record: dict, trait: str, pre: float, post: float, observed: float, numerator: float, denominator: float) -> dict:
        return {
            "event_date": record["event_date"], "fight_id": record["fight_id"],
            "fighter_id": record["fighter_id"], "fighter_name": record["fighter_name"],
            "opponent_id": record["opponent_id"], "opponent_name": record["opponent_name"],
            "trait": trait, "pre_rating": pre, "post_rating": post,
            "observed": observed, "raw_numerator": numerator, "raw_denominator": denominator,
        }
