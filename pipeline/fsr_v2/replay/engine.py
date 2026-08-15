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
    ]
    aggregations = {column: "sum" for column in sum_columns}
    aggregations.update({
        "submission_finish": "max", "method": "first", "winner_id": "first",
        "ground_exposure_fallback_used": "sum", "inferred_ground_entry": "sum",
        "match_time_interpretation": "first",
    })
    fights = rounds.groupby(FIGHT_KEYS, as_index=False).agg(aggregations)
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

    def _behavior(self, group: TraitGroup, fights: pd.DataFrame) -> ReplayResult:
        states: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        population = [0.0, 0.0]
        rows: list[dict] = []
        prior_weight = (self.config.target_composition_prior_attempts
                        if group.kind == "composition"
                        else self.config.behavior_prior_seconds)
        for date, batch in fights.groupby("event_date", sort=True):
            pending: list[tuple[str, float, float]] = []
            global_rate = population[0] / population[1] if population[1] else 0.0
            for record in batch.to_dict("records"):
                numerator = float(record[group.numerator])
                denominator = float(record[group.denominator])
                prior_n, prior_d = states[record["fighter_id"]]
                pre = (prior_n + global_rate * prior_weight) / (
                    prior_d + prior_weight
                )
                observation = numerator / denominator if denominator > 0 else np.nan
                if denominator > 0:
                    post = (prior_n + numerator + global_rate * prior_weight) / (
                        prior_d + denominator + prior_weight
                    )
                else:
                    post = pre
                rows.append(self._row(record, group.traits[0], pre, post, observation, numerator, denominator))
                if denominator > 0:
                    pending.append((record["fighter_id"], numerator, denominator))
            for fighter, numerator, denominator in pending:
                states[fighter][0] += numerator
                states[fighter][1] += denominator
                population[0] += numerator
                population[1] += denominator
        history = pd.DataFrame(rows)
        final_rate = population[0] / population[1] if population[1] else 0.0
        history["latest_rating"] = np.nan
        for fighter, (numerator, denominator) in states.items():
            mask = history["fighter_id"].eq(fighter)
            if mask.any():
                index = history.index[mask][-1]
                history.loc[index, "latest_rating"] = (
                    numerator + final_rate * prior_weight
                ) / (denominator + prior_weight)
        history["latest_population_baseline"] = final_rate
        if len(group.traits) == 2:  # complementary head/body histories
            complement = history.copy()
            complement["trait"] = group.traits[1]
            for column in ["pre_rating", "post_rating", "observed"]:
                complement[column] = 1.0 - complement[column]
            complement["latest_rating"] = 1.0 - complement["latest_rating"]
            complement["latest_population_baseline"] = 1.0 - complement["latest_population_baseline"]
            history = pd.concat([history, complement], ignore_index=True)
        return ReplayResult(history, dict(states), {"numerator": population[0], "denominator": population[1]})

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
                evidence = 1.0 - exp(-denominator / self.config.evidence_saturation_attempts) if denominator > 0 else 0.0
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
