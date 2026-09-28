"""Leakage-safe fighter-round training data for simulator parameter models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.common.fight_time import ROUND_SECONDS, repair_elapsed_match_time


class SimulationTrainingDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimulationTrainingBuildResult:
    dataset: pd.DataFrame
    audit: pd.DataFrame


ROUND_KEYS = ["fight_id", "fighter_id", "round"]
ROUND_REQUIRED = [
    "fight_id", "fighter_id", "opponent_id", "corner", "round",
    "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
    "control_seconds", "kd",
]
MASTER_REQUIRED = [
    "fight_id", "method", "finish_round", "match_time_sec", "total_rounds", "winner_id"
]
ROUND_NUMERIC = [
    "round", "sig_str_landed", "sig_str_attempted", "total_str_landed",
    "total_str_attempted", "td_landed", "td_attempted", "control_seconds",
    "kd", "sub_att", "rev", "ground_landed", "ground_attempted",
]
OPTIONAL_ZERO = [
    "total_str_landed", "total_str_attempted", "sub_att", "rev",
    "ground_landed", "ground_attempted",
]
PRIOR_STATS = [column for column in ROUND_NUMERIC if column not in {"round", "rev"}]
ALIASES = {
    "round": ("round", "round_number"),
    "sig_str_landed": ("sig_str_landed", "sig_landed"),
    "sig_str_attempted": ("sig_str_attempted", "sig_attempted"),
    "total_str_landed": ("total_str_landed", "total_landed"),
    "total_str_attempted": ("total_str_attempted", "total_attempted"),
    "td_landed": ("td_landed", "takedowns_landed"),
    "td_attempted": ("td_attempted", "takedowns_attempted"),
    "control_seconds": (
        "control_seconds", "ctrl_seconds", "ctrl_sec", "control_time_seconds",
        "control_time_sec", "ctrl", "control", "control_time",
    ),
    "kd": ("kd", "knockdowns"),
    "sub_att": ("sub_att", "submission_attempts"),
    "rev": ("rev", "reversals"),
    "ground_landed": ("ground_landed", "sig_ground_landed"),
    "ground_attempted": ("ground_attempted", "sig_ground_attempted"),
}
TARGET_COLUMNS = [
    "target_sig_landed", "target_sig_attempted", "target_sig_accuracy",
    "target_total_landed", "target_total_attempted", "target_td_landed",
    "target_td_attempted", "target_td_accuracy", "target_control_seconds",
    "target_knockdowns", "target_submission_attempts", "target_ground_landed",
    "target_ground_attempted", "target_fighter_ko_tko_finish",
    "target_opponent_ko_tko_finish", "target_fighter_submission_finish",
    "target_opponent_submission_finish", "target_any_stoppage",
    "target_no_stoppage", "target_fight_reaches_next_round",
    "target_round_completed", "target_finish_time_in_round_seconds",
    "target_elapsed_fight_seconds",
]


def _require(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SimulationTrainingDataError(f"{label} is missing required columns: {missing}")


def _seconds(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        return float(pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0])
    try:
        minutes, seconds = (int(part) for part in text.split(":"))
    except (ValueError, TypeError):
        return float("nan")
    return float(minutes * 60 + seconds) if minutes >= 0 and 0 <= seconds < 60 else float("nan")


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    ).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def standardize_round_stats(round_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = round_stats_df.copy()
    for canonical, aliases in ALIASES.items():
        if canonical not in df.columns:
            source = next((name for name in aliases if name in df.columns), None)
            if source:
                df[canonical] = df[source]
    for column in OPTIONAL_ZERO:
        if column not in df.columns:
            df[column] = 0.0
    _require(df, ROUND_REQUIRED, "round stats")

    if "date" not in df.columns and "event_date" in df.columns:
        df["date"] = df["event_date"]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["corner"] = df["corner"].astype("string").str.strip().str.lower()
    if (~df["corner"].isin(["red", "blue"])).any():
        raise SimulationTrainingDataError("round stats has invalid corner values")
    for column in ROUND_NUMERIC:
        df[column] = (
            df[column].map(_seconds)
            if column == "control_seconds"
            else pd.to_numeric(df[column], errors="coerce")
        )
    if df["round"].isna().any() or df["round"].lt(1).any():
        raise SimulationTrainingDataError("round stats has invalid round values")
    df["round"] = df["round"].astype(int)
    if any(df[column].lt(0).fillna(False).any() for column in ROUND_NUMERIC if column != "round"):
        raise SimulationTrainingDataError("round stats contains negative values")
    if df.duplicated(ROUND_KEYS).any():
        raise SimulationTrainingDataError("round stats has duplicate fighter-round keys")
    return df


def _method_family(method: object) -> str:
    text = "" if pd.isna(method) else str(method).strip().lower()
    if "ko" in text or "tko" in text:
        return "ko_tko"
    if "sub" in text:
        return "submission"
    if "dec" in text:
        return "decision"
    if "draw" in text or "no contest" in text or text in {"nc", "overturned"}:
        return "draw_no_contest"
    return "other"


def standardize_master_fights(master_df: pd.DataFrame) -> pd.DataFrame:
    df = repair_elapsed_match_time(master_df.copy())
    _require(df, MASTER_REQUIRED, "master fights")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("finish_round", "match_time_sec", "total_rounds"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df["finish_round"].isna().any() or df["finish_round"].lt(1).any():
        raise SimulationTrainingDataError("master fights has invalid finish_round")
    if df["total_rounds"].isna().any() or not df["total_rounds"].isin([3, 5]).all():
        raise SimulationTrainingDataError("master fights total_rounds must be 3 or 5")
    if df["match_time_sec"].isna().any() or df["match_time_sec"].lt(0).any():
        raise SimulationTrainingDataError("master fights has invalid match_time_sec")
    if df.duplicated(["fight_id"]).any():
        raise SimulationTrainingDataError("master fights has duplicate fight_id values")
    df["finish_round"] = df["finish_round"].astype(int)
    df["total_rounds"] = df["total_rounds"].astype(int)
    df["method_family"] = df["method"].map(_method_family)
    if df["match_time_sec"].gt(df["total_rounds"] * ROUND_SECONDS).any():
        raise SimulationTrainingDataError("master fights has time beyond scheduled duration")
    keep = [
        "fight_id", "event_id", "event_name", "date", "division", "title_fight",
        "method", "method_family", "finish_round", "match_time_sec", "total_rounds",
        "winner_id",
    ]
    return df[[column for column in keep if column in df.columns]].copy()


def _coalesce(df: pd.DataFrame, column: str) -> pd.DataFrame:
    left, right = f"{column}_round", f"{column}_master"
    if left not in df.columns and right not in df.columns:
        return df
    a = df[left] if left in df.columns else pd.Series(pd.NA, index=df.index)
    b = df[right] if right in df.columns else pd.Series(pd.NA, index=df.index)
    if column == "date":
        a, b = pd.to_datetime(a, errors="coerce"), pd.to_datetime(b, errors="coerce")
        mismatch = a.notna() & b.notna() & a.ne(b)
    else:
        mismatch = a.notna() & b.notna() & a.astype("string").ne(b.astype("string"))
    if mismatch.any():
        raise SimulationTrainingDataError(f"round stats and master disagree on {column}")
    df[column] = a.combine_first(b)
    return df.drop(columns=[name for name in (left, right) if name in df.columns])


def merge_rounds_with_master(rounds: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    df = rounds.merge(
        master, on="fight_id", how="left", validate="many_to_one",
        suffixes=("_round", "_master"), indicator=True,
    )
    if df["_merge"].ne("both").any():
        raise SimulationTrainingDataError("round stats contains fights missing from master")
    df = df.drop(columns="_merge")
    for column in ("event_id", "event_name", "date"):
        df = _coalesce(df, column)
    if "date" not in df.columns or df["date"].isna().any():
        raise SimulationTrainingDataError("training rows require a fight date")
    if df["round"].gt(df["finish_round"]).any():
        raise SimulationTrainingDataError("round stats contains rounds after the finish")
    if df["round"].gt(df["total_rounds"]).any():
        raise SimulationTrainingDataError("round stats contains rounds after scheduled duration")
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    target_map = {
        "target_sig_landed": "sig_str_landed",
        "target_sig_attempted": "sig_str_attempted",
        "target_total_landed": "total_str_landed",
        "target_total_attempted": "total_str_attempted",
        "target_td_landed": "td_landed",
        "target_td_attempted": "td_attempted",
        "target_control_seconds": "control_seconds",
        "target_knockdowns": "kd",
        "target_submission_attempts": "sub_att",
        "target_ground_landed": "ground_landed",
        "target_ground_attempted": "ground_attempted",
    }
    for target, source in target_map.items():
        out[target] = out[source]
    out["target_sig_accuracy"] = _ratio(out["sig_str_landed"], out["sig_str_attempted"])
    out["target_td_accuracy"] = _ratio(out["td_landed"], out["td_attempted"])

    fighter_won = out["winner_id"].astype("string").eq(out["fighter_id"].astype("string"))
    opponent_won = out["winner_id"].astype("string").eq(out["opponent_id"].astype("string"))
    finish_round = out["round"].eq(out["finish_round"])
    ko = finish_round & out["method_family"].eq("ko_tko")
    sub = finish_round & out["method_family"].eq("submission")
    stoppage = ko | sub
    out["target_fighter_ko_tko_finish"] = (ko & fighter_won).astype(int)
    out["target_opponent_ko_tko_finish"] = (ko & opponent_won).astype(int)
    out["target_fighter_submission_finish"] = (sub & fighter_won).astype(int)
    out["target_opponent_submission_finish"] = (sub & opponent_won).astype(int)
    out["target_any_stoppage"] = stoppage.astype(int)
    out["target_no_stoppage"] = (~stoppage).astype(int)
    out["target_fight_reaches_next_round"] = out["round"].lt(out["finish_round"]).astype(int)
    out["target_round_completed"] = (
        out["round"].lt(out["finish_round"]) | (finish_round & ~stoppage)
    ).astype(int)
    clock = (out["match_time_sec"] - (out["round"] - 1) * ROUND_SECONDS).clip(
        lower=0, upper=ROUND_SECONDS
    )
    out["target_finish_time_in_round_seconds"] = np.where(
        stoppage, clock, float(ROUND_SECONDS)
    )
    out["target_elapsed_fight_seconds"] = out["match_time_sec"].astype(float)
    return out


def _add_prior_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(ROUND_KEYS).copy()
    keys = [out["fight_id"], out["fighter_id"]]
    grouped = out.groupby(["fight_id", "fighter_id"], sort=False)
    out["prior_rounds_completed"] = grouped.cumcount()
    out["rounds_remaining_including_current"] = out["total_rounds"] - out["round"] + 1
    out["elapsed_seconds_before_round"] = out["prior_rounds_completed"] * ROUND_SECONDS
    out["scheduled_fight_seconds"] = out["total_rounds"] * ROUND_SECONDS
    for column in PRIOR_STATS:
        values = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        out[column] = values
        cumulative = values.groupby(keys).cumsum() - values
        out[f"prior_{column}_cumulative"] = cumulative
        out[f"prior_{column}_last_round"] = grouped[column].shift(1).fillna(0.0)
        out[f"prior_{column}_per_completed_round"] = _ratio(
            cumulative, out["prior_rounds_completed"]
        )
    context = [column for column in out.columns if column.startswith("prior_")]
    context += ["elapsed_seconds_before_round", "rounds_remaining_including_current"]
    opponent = out[["fight_id", "fighter_id", "round", *context]].rename(
        columns={"fighter_id": "opponent_id", **{c: f"opponent_{c}" for c in context}}
    )
    out = out.merge(
        opponent, on=["fight_id", "opponent_id", "round"], how="left", validate="one_to_one"
    )
    if out[[f"opponent_{column}" for column in context]].isna().all(axis=1).any():
        raise SimulationTrainingDataError("round stats is missing paired opponent rows")
    return out


def _state_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        column for column in df.columns
        if column.startswith("rfs_") and "_fight_" not in column
    )


def join_prefight_state_sources(
    training_df: pd.DataFrame, state_sources: Mapping[str, pd.DataFrame] | None
) -> pd.DataFrame:
    out = training_df.copy()
    for source_name, raw in (state_sources or {}).items():
        state = raw.copy()
        _require(state, ["fight_id", "fighter_id"], f"state source {source_name!r}")
        if state.duplicated(["fight_id", "fighter_id"]).any():
            raise SimulationTrainingDataError(f"state source {source_name!r} has duplicate keys")
        columns = _state_columns(state)
        if not columns:
            raise SimulationTrainingDataError(
                f"state source {source_name!r} has no leakage-safe rfs_* columns"
            )
        date_column = next(
            (candidate for candidate in ("date", "state_date", "latest_date") if candidate in state),
            None,
        )
        selected = ["fight_id", "fighter_id", *columns] + ([date_column] if date_column else [])
        state = state[selected]
        for side, id_column in (("fighter", "fighter_id"), ("opponent", "opponent_id")):
            renames = {column: f"{side}_{column}" for column in columns}
            if side == "opponent":
                renames["fighter_id"] = "opponent_id"
            temp_date = f"_{side}_{source_name}_state_date"
            if date_column:
                renames[date_column] = temp_date
            joined = state.rename(columns=renames)
            collisions = set(out).intersection(
                [value for key, value in renames.items() if key != "fighter_id"]
            )
            if collisions:
                raise SimulationTrainingDataError(
                    f"state source {source_name!r} collides with columns: {sorted(collisions)}"
                )
            out = out.merge(
                joined, on=["fight_id", id_column], how="left", validate="many_to_one"
            )
            side_columns = [f"{side}_{column}" for column in columns]
            out[f"{side}_{source_name}_state_available"] = out[side_columns].notna().any(axis=1).astype(int)
            if date_column:
                source_date = pd.to_datetime(out[temp_date], errors="coerce")
                target_date = pd.to_datetime(out["date"], errors="coerce")
                if (source_date.notna() & target_date.notna() & source_date.gt(target_date)).any():
                    raise SimulationTrainingDataError(
                        f"state source {source_name!r} contains future-dated rows"
                    )
                out = out.drop(columns=temp_date)
    return out


def _drop_current_round(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ROUND_NUMERIC if column != "round"]
    columns += ["method", "method_family", "finish_round", "match_time_sec", "winner_id"]
    return df.drop(columns=[column for column in columns if column in df])


def validate_training_dataset(df: pd.DataFrame) -> None:
    _require(df, [*ROUND_KEYS, *TARGET_COLUMNS], "training dataset")
    if df.duplicated(ROUND_KEYS).any():
        raise SimulationTrainingDataError("training dataset has duplicate keys")
    leaky = [
        column for column in df
        if "_fight_" in column and column.startswith(("fighter_rfs_", "opponent_rfs_"))
    ]
    if leaky:
        raise SimulationTrainingDataError(f"training dataset contains realized RFS columns: {leaky}")
    raw = [column for column in ROUND_NUMERIC if column != "round" and column in df]
    if raw:
        raise SimulationTrainingDataError(f"current-round observations remain as predictors: {raw}")
    binary = [
        "target_fighter_ko_tko_finish", "target_opponent_ko_tko_finish",
        "target_fighter_submission_finish", "target_opponent_submission_finish",
        "target_any_stoppage", "target_no_stoppage",
        "target_fight_reaches_next_round", "target_round_completed",
    ]
    if any((~df[column].isin([0, 1])).any() for column in binary):
        raise SimulationTrainingDataError("binary targets contain invalid values")
    if not df.groupby(["fight_id", "round"])["fighter_id"].nunique().eq(2).all():
        raise SimulationTrainingDataError("training rows are missing a paired opponent")
    components = df[[
        "target_fighter_ko_tko_finish", "target_opponent_ko_tko_finish",
        "target_fighter_submission_finish", "target_opponent_submission_finish",
    ]].sum(axis=1)
    if components.gt(1).any() or not components.eq(df["target_any_stoppage"]).all():
        raise SimulationTrainingDataError("competing finish targets are inconsistent")


def build_training_audit(
    df: pd.DataFrame, state_sources: Mapping[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    rows = [
        ("row_count", len(df), len(df) > 0),
        ("fight_count", df["fight_id"].nunique(), df["fight_id"].nunique() > 0),
        ("fighter_count", df["fighter_id"].nunique(), df["fighter_id"].nunique() > 1),
        ("duplicate_fighter_round_keys", int(df.duplicated(ROUND_KEYS).sum()), not df.duplicated(ROUND_KEYS).any()),
        ("fighter_ko_tko_positive_rows", int(df["target_fighter_ko_tko_finish"].sum()), True),
        ("fighter_submission_positive_rows", int(df["target_fighter_submission_finish"].sum()), True),
        ("stoppage_rows", int(df["target_any_stoppage"].sum()), True),
    ]
    for source in (state_sources or {}):
        for side in ("fighter", "opponent"):
            column = f"{side}_{source}_state_available"
            if column in df:
                rows.append((f"{column}_coverage", float(df[column].mean()), True))
    return pd.DataFrame(rows, columns=["check", "value", "passed"]).assign(detail="")


def build_simulation_training_dataset(
    round_stats_df: pd.DataFrame,
    master_df: pd.DataFrame,
    state_sources: Mapping[str, pd.DataFrame] | None = None,
) -> SimulationTrainingBuildResult:
    rounds = standardize_round_stats(round_stats_df)
    master = standardize_master_fights(master_df)
    dataset = merge_rounds_with_master(rounds, master)
    dataset = _add_targets(dataset)
    dataset = _add_prior_context(dataset)
    dataset = join_prefight_state_sources(dataset, state_sources)
    dataset = _drop_current_round(dataset)
    dataset = dataset.sort_values(["date", "fight_id", "round", "corner"]).reset_index(drop=True)
    validate_training_dataset(dataset)
    return SimulationTrainingBuildResult(dataset, build_training_audit(dataset, state_sources))
