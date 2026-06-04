from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


STATUS_ORDER = [
    "OFFICIAL BET",
    "WATCHLIST",
    "LOW ODDS MATCH",
    "SPARSE FEATURES",
    "INVALID MODEL DATA",
    "NO BET",
]

SCENARIO_FILTER_COLUMNS = [
    "scenario_passes_model_quality_filter",
    "scenario_passes_feature_validation_filter",
    "scenario_passes_odds_match_filter",
    "scenario_passes_edge_filter",
    "scenario_passes_confidence_filter",
    "scenario_passes_odds_range_filter",
    "scenario_passes_positive_ev_filter",
]

SCENARIO_BETTING_THRESHOLD_COLUMNS = [
    "scenario_passes_edge_filter",
    "scenario_passes_confidence_filter",
    "scenario_passes_odds_range_filter",
    "scenario_passes_positive_ev_filter",
]


@dataclass(frozen=True)
class BettingRules:
    min_edge: float = 0.05
    min_confidence: float = 70.0
    min_odds: int = -250
    max_odds: int = 400
    require_positive_ev: bool = True
    watchlist_max_failed_thresholds: int = 2
    watchlist_high_ev_override: float = 0.25
    bankroll: float = 10000.0
    kelly_fraction: float = 0.50
    max_stake_pct: float = 0.03
    min_stake: float = 0.0
    stake_rounding: float = 1.0


def default_betting_rules() -> BettingRules:
    return BettingRules()


def rules_to_dict(rules: BettingRules) -> dict:
    return asdict(rules)


def rules_changed_from_default(rules: BettingRules) -> bool:
    return rules_to_dict(rules) != rules_to_dict(default_betting_rules())


def american_to_decimal(odds):
    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds > 0:
        return 1 + odds / 100

    return 1 + 100 / abs(odds)


def kelly_fraction(model_prob, american_odds):
    if pd.isna(model_prob) or pd.isna(american_odds):
        return 0.0

    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    p = float(model_prob)
    q = 1 - p

    if b <= 0:
        return 0.0

    return max(0.0, ((b * p) - q) / b)


def scaled_kelly_stake(
    bankroll,
    model_prob,
    american_odds,
    kelly_multiplier,
    max_stake_pct,
    min_stake=0.0,
    stake_rounding=1.0,
):
    full_kelly = kelly_fraction(model_prob, american_odds)
    raw_stake = float(bankroll) * full_kelly * float(kelly_multiplier)
    capped_stake = min(raw_stake, float(bankroll) * float(max_stake_pct))

    if capped_stake < float(min_stake):
        return 0.0

    if stake_rounding and float(stake_rounding) > 0:
        capped_stake = round(capped_stake / float(stake_rounding)) * float(stake_rounding)

    return round(float(capped_stake), 2)


def _bool_series(df, column, default=False):
    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return df[column].fillna(default).astype(bool)


def _numeric_series(df, column, default=np.nan):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")

    return pd.to_numeric(df[column], errors="coerce")


def build_failed_filter_reason(row):
    failed = []

    for col in SCENARIO_FILTER_COLUMNS:
        if not row[col]:
            failed.append(
                col.replace("scenario_passes_", "")
                .replace("_filter", "")
            )

    return ", ".join(failed)


def apply_betting_rules(board_df, rules: BettingRules):
    scenario = board_df.copy()

    scenario["scenario_passes_model_quality_filter"] = _bool_series(
        scenario,
        "passes_model_quality_filter",
    )
    scenario["scenario_passes_feature_validation_filter"] = _bool_series(
        scenario,
        "passes_feature_validation_filter",
    )
    scenario["scenario_passes_odds_match_filter"] = _bool_series(
        scenario,
        "passes_odds_match_filter",
    )

    best_edge = _numeric_series(scenario, "best_edge")
    best_confidence = _numeric_series(scenario, "best_confidence")
    best_american_odds = _numeric_series(scenario, "best_american_odds")
    best_ev = _numeric_series(scenario, "best_ev")

    scenario["scenario_passes_edge_filter"] = best_edge >= rules.min_edge
    scenario["scenario_passes_confidence_filter"] = best_confidence >= rules.min_confidence
    scenario["scenario_passes_odds_range_filter"] = (
        (best_american_odds >= rules.min_odds)
        &
        (best_american_odds <= rules.max_odds)
    )
    scenario["scenario_passes_positive_ev_filter"] = (
        best_ev > 0
        if rules.require_positive_ev
        else pd.Series(True, index=scenario.index)
    )

    scenario["scenario_passes_all_bet_filters"] = scenario[SCENARIO_FILTER_COLUMNS].all(axis=1)
    scenario["scenario_failed_filter_count"] = (
        len(SCENARIO_FILTER_COLUMNS)
        - scenario[SCENARIO_FILTER_COLUMNS].sum(axis=1)
    )
    scenario["scenario_failed_filters"] = scenario.apply(
        build_failed_filter_reason,
        axis=1,
    )

    scenario["scenario_passes_core_data_filters"] = (
        scenario["scenario_passes_model_quality_filter"]
        &
        scenario["scenario_passes_feature_validation_filter"]
        &
        scenario["scenario_passes_odds_match_filter"]
    )

    scenario["scenario_failed_betting_threshold_count"] = (
        len(SCENARIO_BETTING_THRESHOLD_COLUMNS)
        - scenario[SCENARIO_BETTING_THRESHOLD_COLUMNS].sum(axis=1)
    )

    scenario["scenario_is_official_bet"] = scenario["scenario_passes_all_bet_filters"]
    scenario["scenario_is_watchlist_bet"] = (
        (~scenario["scenario_is_official_bet"])
        &
        scenario["scenario_passes_core_data_filters"]
        &
        (
            (scenario["scenario_failed_betting_threshold_count"] <= rules.watchlist_max_failed_thresholds)
            |
            (best_ev > rules.watchlist_high_ev_override)
        )
    )

    scenario["scenario_recommended_stake"] = scenario.apply(
        lambda row: scaled_kelly_stake(
            bankroll=rules.bankroll,
            model_prob=row.get("best_prob"),
            american_odds=row.get("best_american_odds"),
            kelly_multiplier=rules.kelly_fraction,
            max_stake_pct=rules.max_stake_pct,
            min_stake=rules.min_stake,
            stake_rounding=rules.stake_rounding,
        )
        if row["scenario_passes_all_bet_filters"]
        else 0.0,
        axis=1,
    )

    scenario["scenario_bet_status"] = "NO BET"
    scenario.loc[
        ~scenario["scenario_passes_model_quality_filter"],
        "scenario_bet_status",
    ] = "INVALID MODEL DATA"
    scenario.loc[
        scenario["scenario_passes_model_quality_filter"]
        &
        ~scenario["scenario_passes_feature_validation_filter"],
        "scenario_bet_status",
    ] = "SPARSE FEATURES"
    scenario.loc[
        scenario["scenario_passes_model_quality_filter"]
        &
        scenario["scenario_passes_feature_validation_filter"]
        &
        ~scenario["scenario_passes_odds_match_filter"],
        "scenario_bet_status",
    ] = "LOW ODDS MATCH"
    scenario.loc[
        scenario["scenario_is_watchlist_bet"],
        "scenario_bet_status",
    ] = "WATCHLIST"
    scenario.loc[
        scenario["scenario_is_official_bet"],
        "scenario_bet_status",
    ] = "OFFICIAL BET"

    scenario["scenario_bet_reason"] = np.where(
        scenario["scenario_is_official_bet"],
        "All scenario betting filters passed",
        scenario["scenario_failed_filters"],
    )

    return scenario


def scenario_comparison(board_df, scenario_df):
    production_status = board_df.get("bet_status", pd.Series(index=board_df.index, dtype="object"))
    scenario_status = scenario_df.get("scenario_bet_status", pd.Series(index=scenario_df.index, dtype="object"))
    production_stake = _numeric_series(board_df, "recommended_stake", default=0).fillna(0)
    scenario_stake = _numeric_series(scenario_df, "scenario_recommended_stake", default=0).fillna(0)

    return {
        "production_official_bets": int((production_status == "OFFICIAL BET").sum()),
        "scenario_official_bets": int((scenario_status == "OFFICIAL BET").sum()),
        "production_watchlist": int((production_status == "WATCHLIST").sum()),
        "scenario_watchlist": int((scenario_status == "WATCHLIST").sum()),
        "production_total_stake": float(production_stake.sum()),
        "scenario_total_stake": float(scenario_stake.sum()),
        "added_official_bets": int(((production_status != "OFFICIAL BET") & (scenario_status == "OFFICIAL BET")).sum()),
        "removed_official_bets": int(((production_status == "OFFICIAL BET") & (scenario_status != "OFFICIAL BET")).sum()),
        "stake_delta": float(scenario_stake.sum() - production_stake.sum()),
    }
