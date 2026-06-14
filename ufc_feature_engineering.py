# ============================================================
# ufc_feature_engineering.py
# Shared feature engineering helpers for UFC pipeline
# Champion Clean Set truth-source version
# ============================================================

import numpy as np
import pandas as pd


def safe_col(df, col, default=0.0):
    """
    Safely return a numeric dataframe column.
    If missing, return a default-filled Series.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)

    return pd.Series(default, index=df.index)


def to_decimal_rate(series_or_value):
    """
    Convert percentage-style values to decimal rates.
    Works for Series or scalar values.
    """
    if isinstance(series_or_value, pd.Series):
        s = pd.to_numeric(series_or_value, errors="coerce").fillna(0)

        return pd.Series(
            np.where(s > 1, s / 100, s),
            index=series_or_value.index,
        )

    try:
        if pd.isna(series_or_value):
            return 0.0

        value = float(series_or_value)

        if value > 1:
            return value / 100.0

        return value

    except Exception:
        return 0.0


def add_feature_block(df, feature_dict):
    """
    Add many engineered features at once to avoid dataframe fragmentation.
    """
    feature_df = pd.DataFrame(feature_dict, index=df.index)

    return pd.concat([df, feature_df], axis=1).copy()


def add_v5_engineered_features(df):
    """
    Add Champion Clean Set engineered features.

    This intentionally matches the old V4/Champion feature formulas, with
    additional competition-adjusted interaction features for Model Lab testing.
    """
    out = df.copy()

    r_str_def_decimal = to_decimal_rate(
        safe_col(out, "r_pre_str_def")
    )

    b_str_def_decimal = to_decimal_rate(
        safe_col(out, "b_pre_str_def")
    )

    r_vs_b_str_def = to_decimal_rate(
        safe_col(out, "b_pre_str_def")
    )

    b_vs_r_str_def = to_decimal_rate(
        safe_col(out, "r_pre_str_def")
    )

    r_vs_b_td_def = to_decimal_rate(
        safe_col(out, "b_pre_td_def")
    )

    b_vs_r_td_def = to_decimal_rate(
        safe_col(out, "r_pre_td_def")
    )

    features = {
        # ----------------------------------------------------
        # Core physical matchup
        # ----------------------------------------------------
        "age_diff": (
            safe_col(out, "r_age")
            - safe_col(out, "b_age")
        ),

        "height_diff": (
            safe_col(out, "r_height")
            - safe_col(out, "b_height")
        ),

        "reach_diff": (
            safe_col(out, "r_reach")
            - safe_col(out, "b_reach")
        ),

        "weight_diff": (
            safe_col(out, "r_weight")
            - safe_col(out, "b_weight")
        ),

        # ----------------------------------------------------
        # Striking edge
        # ----------------------------------------------------
        "striking_edge": (
            safe_col(out, "splm_diff")
            - safe_col(out, "sapm_diff")
            + safe_col(out, "str_acc_diff")
            + safe_col(out, "str_def_diff")
        ),

        # ----------------------------------------------------
        # Grappling edge
        # ----------------------------------------------------
        "grappling_edge": (
            safe_col(out, "td_avg_diff")
            + safe_col(out, "sub_avg_diff")
            - safe_col(out, "td_def_diff")
        ),

        # ----------------------------------------------------
        # Finish volatility
        # ----------------------------------------------------
        "finish_volatility": (
            safe_col(out, "finish_rate_diff")
            - safe_col(out, "finish_loss_rate_diff")
        ),

        # ----------------------------------------------------
        # Wrestling pressure vs defense
        # ----------------------------------------------------
        "wrestling_pressure_vs_defense": (
            safe_col(out, "td_avg_diff")
            - safe_col(out, "td_def_diff")
        ),
    }

    features["reach_striking_combo"] = (
        features["reach_diff"]
        * safe_col(out, "splm_diff")
    )

    # --------------------------------------------------------
    # Chin / durability risk
    # --------------------------------------------------------
    features["r_chin_risk"] = (
        safe_col(out, "r_pre_sapm")
        * (1 - r_str_def_decimal)
    )

    features["b_chin_risk"] = (
        safe_col(out, "b_pre_sapm")
        * (1 - b_str_def_decimal)
    )

    features["chin_risk_diff"] = (
        features["r_chin_risk"]
        - features["b_chin_risk"]
    )

    # --------------------------------------------------------
    # Experience efficiency
    # --------------------------------------------------------
    features["r_experience_ratio"] = (
        safe_col(out, "r_pre_fights")
        / (safe_col(out, "r_pre_losses") + 1)
    )

    features["b_experience_ratio"] = (
        safe_col(out, "b_pre_fights")
        / (safe_col(out, "b_pre_losses") + 1)
    )

    features["experience_ratio_diff"] = (
        features["r_experience_ratio"]
        - features["b_experience_ratio"]
    )

    # --------------------------------------------------------
    # Aggression index
    # --------------------------------------------------------
    features["r_aggression_index"] = (
        safe_col(out, "r_pre_splm")
        + safe_col(out, "r_pre_td_avg")
    )

    features["b_aggression_index"] = (
        safe_col(out, "b_pre_splm")
        + safe_col(out, "b_pre_td_avg")
    )

    features["aggression_index_diff"] = (
        features["r_aggression_index"]
        - features["b_aggression_index"]
    )

    # --------------------------------------------------------
    # Age curve
    # --------------------------------------------------------
    features["r_age_squared"] = (
        safe_col(out, "r_age") ** 2
    )

    features["b_age_squared"] = (
        safe_col(out, "b_age") ** 2
    )

    features["age_squared_diff"] = (
        features["r_age_squared"]
        - features["b_age_squared"]
    )

    # --------------------------------------------------------
    # Pressure striking advantage
    # --------------------------------------------------------
    features["r_pressure_striking_adv"] = (
        safe_col(out, "r_pre_splm")
        * (1 - r_vs_b_str_def)
    )

    features["b_pressure_striking_adv"] = (
        safe_col(out, "b_pre_splm")
        * (1 - b_vs_r_str_def)
    )

    features["pressure_striking_adv_diff"] = (
        features["r_pressure_striking_adv"]
        - features["b_pressure_striking_adv"]
    )

    # --------------------------------------------------------
    # Wrestling mismatch
    # --------------------------------------------------------
    features["r_wrestling_mismatch"] = (
        safe_col(out, "r_pre_td_avg")
        * (1 - r_vs_b_td_def)
    )

    features["b_wrestling_mismatch"] = (
        safe_col(out, "b_pre_td_avg")
        * (1 - b_vs_r_td_def)
    )

    features["wrestling_mismatch_diff"] = (
        features["r_wrestling_mismatch"]
        - features["b_wrestling_mismatch"]
    )

    # --------------------------------------------------------
    # Submission mismatch
    # --------------------------------------------------------
    features["r_submission_mismatch"] = (
        safe_col(out, "r_pre_sub_avg")
        + safe_col(out, "b_pre_ctrl_against_per_min")
    )

    features["b_submission_mismatch"] = (
        safe_col(out, "b_pre_sub_avg")
        + safe_col(out, "r_pre_ctrl_against_per_min")
    )

    features["submission_mismatch_diff"] = (
        features["r_submission_mismatch"]
        - features["b_submission_mismatch"]
    )

    # --------------------------------------------------------
    # Competition-adjusted matchup credibility
    # --------------------------------------------------------
    # Normalize ELO diffs only inside interaction terms so magnitudes stay
    # interpretable while preserving raw ELO features elsewhere.
    avg_opp_elo_norm = safe_col(out, "avg_opponent_elo_diff") / 100.0
    ewm_avg_opp_elo_norm = safe_col(out, "ewm_avg_opponent_elo_diff") / 100.0

    features["striking_edge_x_avg_opp_elo"] = (
        features["striking_edge"] * avg_opp_elo_norm
    )
    features["striking_edge_x_ewm_avg_opp_elo"] = (
        features["striking_edge"] * ewm_avg_opp_elo_norm
    )
    features["grappling_edge_x_avg_opp_elo"] = (
        features["grappling_edge"] * avg_opp_elo_norm
    )
    features["grappling_edge_x_ewm_avg_opp_elo"] = (
        features["grappling_edge"] * ewm_avg_opp_elo_norm
    )
    features["wrestling_mismatch_x_avg_opp_elo"] = (
        features["wrestling_mismatch_diff"] * avg_opp_elo_norm
    )
    features["submission_mismatch_x_avg_opp_elo"] = (
        features["submission_mismatch_diff"] * avg_opp_elo_norm
    )

    out = add_feature_block(out, features)

    return out.copy()


def get_engineered_feature_list():
    """
    Champion Clean Set registry features.
    """
    return [
        "age_diff",
        "height_diff",
        "reach_diff",
        "weight_diff",
        "striking_edge",
        "grappling_edge",
        "finish_volatility",
        "wrestling_pressure_vs_defense",
        "reach_striking_combo",
        "chin_risk_diff",
        "experience_ratio_diff",
        "aggression_index_diff",
        "age_squared_diff",
        "pressure_striking_adv_diff",
        "wrestling_mismatch_diff",
        "submission_mismatch_diff",
        "striking_edge_x_avg_opp_elo",
        "striking_edge_x_ewm_avg_opp_elo",
        "grappling_edge_x_avg_opp_elo",
        "grappling_edge_x_ewm_avg_opp_elo",
        "wrestling_mismatch_x_avg_opp_elo",
        "submission_mismatch_x_avg_opp_elo",
    ]


def save_engineered_feature_registry(features, output_path):
    """
    Save engineered feature registry as CSV.
    """
    registry_df = pd.DataFrame({"feature": list(features)})
    registry_df.to_csv(output_path, index=False)

    return registry_df
