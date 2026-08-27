#!/usr/bin/env python3
"""
Standalone UFC historical prediction audit.

Designed to run from the repository root in GitHub Codespaces.

What it does
------------
1. Loads current prediction parquet/CSV artifacts.
2. Optionally recovers older committed prediction artifacts from Git history.
3. Auto-discovers a completed-fight results dataset, or accepts --results PATH.
4. Grades moneyline model picks against actual winners.
5. Writes both all-snapshot and latest pre-fight performance summaries.

The script is read-only with respect to the production pipeline. It writes only
inside --output-dir (default: data/audits/prediction_performance).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/audits/prediction_performance")
DEFAULT_PREDICTION_GLOBS = (
    "data/predictions/model_outcomes.parquet",
    "data/predictions/model_outcomes.csv",
    "data/predictions/by_model/*/model_outcomes.parquet",
    "data/predictions/by_model/*/model_outcomes.csv",
    "data/predictions/history/*.parquet",
    "data/predictions/history/*.csv",
)

PREDICTION_REQUIRED_ANY = (
    {"fight_id", "model_probability"},
    {"red_fighter", "blue_fighter", "model_probability"},
)

ALIASES: dict[str, tuple[str, ...]] = {
    "fight_id": (
        "fight_id", "bout_id", "match_id", "ufcstats_fight_id",
        "fight_url", "bout_url",
    ),
    "event_id": (
        "event_id", "ufcstats_event_id", "event_url",
    ),
    "event_name": (
        "event_name", "event", "event_title",
    ),
    "event_date": (
        "event_date", "date", "commence_time", "event_datetime",
    ),
    "red_fighter": (
        "red_fighter", "r_fighter", "r_name", "fighter_red", "red_name",
        "fighter_1", "fighter1", "fighter_a",
    ),
    "blue_fighter": (
        "blue_fighter", "b_fighter", "b_name", "fighter_blue", "blue_name",
        "fighter_2", "fighter2", "fighter_b",
    ),
    "red_fighter_id": (
        "red_fighter_id", "r_fighter_id", "r_id", "fighter_red_id",
        "fighter_1_id", "fighter1_id", "fighter_a_id",
    ),
    "blue_fighter_id": (
        "blue_fighter_id", "b_fighter_id", "b_id", "fighter_blue_id",
        "fighter_2_id", "fighter2_id", "fighter_b_id",
    ),
    "winner": (
        "winner", "winner_name", "winning_fighter", "result_winner",
        "winner_fighter",
    ),
    "winner_id": (
        "winner_id", "winner_fighter_id", "winning_fighter_id",
    ),
    "winner_side": (
        "winner_side", "winning_side", "result", "winner_corner",
        "result_side",
    ),
    "red_result": (
        "red_result", "r_result", "result_red",
    ),
    "blue_result": (
        "blue_result", "b_result", "result_blue",
    ),
    "method": (
        "method", "result_method", "finish_method",
    ),
    "round": (
        "round", "finish_round", "result_round",
    ),
    "time": (
        "time", "finish_time", "result_time",
    ),
}


@dataclass(frozen=True)
class LoadedFrame:
    frame: pd.DataFrame
    source: str
    source_kind: str
    git_commit: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit historical UFC predictions against completed fight results."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results", default=None)
    parser.add_argument("--prediction-path", action="append", default=[])
    parser.add_argument("--prediction-git-path", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-git-history", action="store_true")
    parser.add_argument("--max-git-commits", type=int, default=500)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--market-key", action="append", default=[])
    parser.add_argument("--min-pick-probability", type=float, default=0.0)
    parser.add_argument("--list-result-candidates", action="store_true")
    return parser.parse_args()


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        capture_output=True,
        text=text,
    )


def is_git_repo(repo_root: Path) -> bool:
    try:
        result = run_command(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
        )
        return result.stdout.strip() == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table type: {path}")


def read_table_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    suffix = suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(data))
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    raise ValueError(f"Unsupported historical table type: {suffix}")


def normalize_column_name(name: object) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def first_existing_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    column_set = set(columns)
    for alias in aliases:
        if alias in column_set:
            return alias
    return None


def normalize_person(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def standardize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalized_columns(raw_df)
    rename: dict[str, str] = {}

    for canonical, aliases in ALIASES.items():
        source = first_existing_column(df.columns, aliases)
        if source and source != canonical and canonical not in df.columns:
            rename[source] = canonical

    df = df.rename(columns=rename)

    if "winner" not in df.columns:
        winner = pd.Series(pd.NA, index=df.index, dtype="object")

        if {"red_result", "red_fighter"}.issubset(df.columns):
            red_won = df["red_result"].astype(str).str.upper().str.strip().isin(
                {"W", "WIN", "WINNER"}
            )
            winner.loc[red_won] = df.loc[red_won, "red_fighter"]

        if {"blue_result", "blue_fighter"}.issubset(df.columns):
            blue_won = df["blue_result"].astype(str).str.upper().str.strip().isin(
                {"W", "WIN", "WINNER"}
            )
            winner.loc[blue_won] = df.loc[blue_won, "blue_fighter"]

        if "winner_side" in df.columns:
            side = df["winner_side"].astype(str).str.lower().str.strip()
            if "red_fighter" in df.columns:
                mask = side.isin({"red", "r", "fighter_1", "fighter1", "a"})
                winner.loc[mask] = df.loc[mask, "red_fighter"]
            if "blue_fighter" in df.columns:
                mask = side.isin({"blue", "b", "fighter_2", "fighter2"})
                winner.loc[mask] = df.loc[mask, "blue_fighter"]

        if winner.notna().any():
            df["winner"] = winner

    if (
        "winner_id" not in df.columns
        and {"winner", "red_fighter", "blue_fighter"}.issubset(df.columns)
    ):
        winner_norm = df["winner"].map(normalize_person)
        red_norm = df["red_fighter"].map(normalize_person)
        blue_norm = df["blue_fighter"].map(normalize_person)

        winner_id = pd.Series(pd.NA, index=df.index, dtype="object")
        if "red_fighter_id" in df.columns:
            winner_id.loc[winner_norm.eq(red_norm)] = df.loc[
                winner_norm.eq(red_norm), "red_fighter_id"
            ]
        if "blue_fighter_id" in df.columns:
            winner_id.loc[winner_norm.eq(blue_norm)] = df.loc[
                winner_norm.eq(blue_norm), "blue_fighter_id"
            ]
        if winner_id.notna().any():
            df["winner_id"] = winner_id

    return df


def looks_like_prediction_frame(df: pd.DataFrame) -> bool:
    columns = set(normalize_column_name(c) for c in df.columns)
    return any(required.issubset(columns) for required in PREDICTION_REQUIRED_ANY)


def prediction_candidate_paths(repo_root: Path, explicit: list[str]) -> list[Path]:
    candidates: set[Path] = set()

    for item in explicit:
        path = Path(item)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists() and path.is_file():
            candidates.add(path.resolve())

    if not explicit:
        for pattern in DEFAULT_PREDICTION_GLOBS:
            for path in repo_root.glob(pattern):
                if path.is_file():
                    candidates.add(path.resolve())

    return sorted(candidates)


def load_current_predictions(
    repo_root: Path,
    explicit_paths: list[str],
) -> list[LoadedFrame]:
    loaded: list[LoadedFrame] = []

    for path in prediction_candidate_paths(repo_root, explicit_paths):
        try:
            frame = read_table(path)
        except Exception as exc:
            print(f"WARNING: Could not read prediction file {path}: {exc}", file=sys.stderr)
            continue

        if not looks_like_prediction_frame(frame):
            print(
                f"WARNING: Skipping non-prediction-looking file: {path}",
                file=sys.stderr,
            )
            continue

        loaded.append(
            LoadedFrame(
                frame=frame,
                source=str(path.relative_to(repo_root.resolve())),
                source_kind="working_tree",
            )
        )

    return loaded


def default_git_prediction_paths(repo_root: Path) -> list[str]:
    paths: set[str] = set()

    for path in prediction_candidate_paths(repo_root, []):
        try:
            paths.add(path.relative_to(repo_root.resolve()).as_posix())
        except ValueError:
            continue

    paths.update(
        {
            "data/predictions/model_outcomes.parquet",
            "data/predictions/model_outcomes.csv",
        }
    )
    return sorted(paths)


def recover_git_predictions(
    repo_root: Path,
    git_paths: list[str],
    max_commits: int,
) -> list[LoadedFrame]:
    if not is_git_repo(repo_root):
        print("WARNING: Git history recovery skipped; not inside a Git repository.")
        return []

    loaded: list[LoadedFrame] = []

    for git_path in git_paths:
        try:
            log_result = run_command(
                ["git", "log", "--all", "--format=%H", "--", git_path],
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"WARNING: Could not inspect Git history for {git_path}: {exc.stderr}",
                file=sys.stderr,
            )
            continue

        commits = []
        seen_commits: set[str] = set()
        for commit in log_result.stdout.splitlines():
            commit = commit.strip()
            if commit and commit not in seen_commits:
                commits.append(commit)
                seen_commits.add(commit)
            if len(commits) >= max_commits:
                break

        for commit in commits:
            try:
                show_result = run_command(
                    ["git", "show", f"{commit}:{git_path}"],
                    cwd=repo_root,
                    text=False,
                )
                frame = read_table_bytes(show_result.stdout, Path(git_path).suffix)
            except Exception as exc:
                print(
                    f"WARNING: Could not read {git_path} at {commit[:10]}: {exc}",
                    file=sys.stderr,
                )
                continue

            if not looks_like_prediction_frame(frame):
                continue

            loaded.append(
                LoadedFrame(
                    frame=frame,
                    source=git_path,
                    source_kind="git_history",
                    git_commit=commit,
                )
            )

    return loaded


def normalize_predictions(frames: list[LoadedFrame]) -> pd.DataFrame:
    normalized: list[pd.DataFrame] = []

    for loaded in frames:
        df = normalized_columns(loaded.frame)

        if "model_probability" not in df.columns:
            continue

        df["_prediction_source"] = loaded.source
        df["_prediction_source_kind"] = loaded.source_kind
        df["_prediction_git_commit"] = loaded.git_commit

        if "prediction_timestamp" in df.columns:
            df["prediction_timestamp"] = pd.to_datetime(
                df["prediction_timestamp"], errors="coerce", utc=True
            )

        df["model_probability"] = pd.to_numeric(
            df["model_probability"], errors="coerce"
        )

        if "model_pick_probability" not in df.columns:
            if "model_confidence" in df.columns:
                df["model_pick_probability"] = pd.to_numeric(
                    df["model_confidence"], errors="coerce"
                )
            elif "is_model_pick" in df.columns:
                df["model_pick_probability"] = np.where(
                    df["is_model_pick"].fillna(False).astype(bool),
                    df["model_probability"],
                    np.nan,
                )

        normalized.append(df)

    if not normalized:
        return pd.DataFrame()

    combined = pd.concat(normalized, ignore_index=True, sort=False)

    dedupe_columns = [
        column
        for column in (
            "prediction_run_id",
            "prediction_timestamp",
            "model_id",
            "market_key",
            "fight_id",
            "outcome_join_key",
            "outcome_label",
            "model_probability",
        )
        if column in combined.columns
    ]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")

    return combined.reset_index(drop=True)


def score_result_candidate(path: Path) -> tuple[float, dict[str, object]]:
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            columns = [normalize_column_name(c) for c in pq.ParquetFile(path).schema.names]
        elif path.suffix.lower() == ".csv":
            columns = [
                normalize_column_name(c)
                for c in pd.read_csv(path, nrows=0).columns
            ]
        else:
            return -999.0, {}
    except Exception:
        return -999.0, {}

    column_set = set(columns)
    resolved = {
        canonical: first_existing_column(column_set, aliases)
        for canonical, aliases in ALIASES.items()
    }

    score = 0.0
    if resolved["fight_id"]:
        score += 8
    if resolved["winner"] or resolved["winner_id"]:
        score += 10
    if resolved["winner_side"]:
        score += 5
    if resolved["red_result"] or resolved["blue_result"]:
        score += 5
    if resolved["red_fighter"] and resolved["blue_fighter"]:
        score += 6
    if resolved["red_fighter_id"] and resolved["blue_fighter_id"]:
        score += 4
    if resolved["event_id"]:
        score += 2
    if resolved["event_name"]:
        score += 1
    if resolved["event_date"]:
        score += 1

    path_text = path.as_posix().lower()
    for token in ("master", "fight", "result", "historical", "completed", "bout"):
        if token in path_text:
            score += 0.5
    for token in (
        "prediction", "live_card", "feature", "audit",
        "upcoming", "queue", "status", "staging",
    ):
        if token in path_text:
            score -= 10

    return score, {"columns": columns, "resolved": resolved}


def find_result_candidates(repo_root: Path) -> list[tuple[Path, float, dict[str, object]]]:
    candidates: list[tuple[Path, float, dict[str, object]]] = []
    data_root = repo_root / "data"
    if not data_root.exists():
        return candidates

    for pattern in ("**/*.parquet", "**/*.pq", "**/*.csv"):
        for path in data_root.glob(pattern):
            if not path.is_file():
                continue
            score, details = score_result_candidate(path)
            if score > 0:
                candidates.append((path, score, details))

    candidates.sort(key=lambda item: (-item[1], item[0].as_posix()))
    return candidates


def print_result_candidates(
    candidates: list[tuple[Path, float, dict[str, object]]],
    repo_root: Path,
) -> None:
    print("\nRanked completed-result candidates")
    print("=" * 80)
    if not candidates:
        print("No plausible result files found beneath data/.")
        return

    for index, (path, score, details) in enumerate(candidates[:25], start=1):
        resolved = {
            key: value
            for key, value in details.get("resolved", {}).items()
            if value
        }
        try:
            display_path = path.relative_to(repo_root)
        except ValueError:
            display_path = path
        print(f"{index:>2}. score={score:>5.1f}  {display_path}")
        print(f"    resolved={resolved}")


def choose_results_path(
    repo_root: Path,
    explicit: str | None,
) -> tuple[Path, list[tuple[Path, float, dict[str, object]]]]:
    candidates = find_result_candidates(repo_root)

    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            raise FileNotFoundError(f"Results file does not exist: {path}")
        return path.resolve(), candidates

    if not candidates:
        raise FileNotFoundError(
            "No completed-results dataset could be auto-discovered. "
            "Run with --list-result-candidates or provide --results PATH."
        )

    best_path, best_score, _ = candidates[0]
    if best_score < 12:
        raise RuntimeError(
            "The best auto-discovered results file is not reliable enough to select "
            f"automatically (score={best_score:.1f}, path={best_path}). "
            "Run with --list-result-candidates, then provide --results PATH."
        )

    return best_path.resolve(), candidates


def build_result_lookup(results: pd.DataFrame) -> dict[str, dict[object, list[int]]]:
    lookup: dict[str, dict[object, list[int]]] = {
        "fight_id": {},
        "fighter_ids": {},
        "fighter_names": {},
    }

    for index, row in results.iterrows():
        fight_id = normalize_identifier(row.get("fight_id"))
        if fight_id:
            lookup["fight_id"].setdefault(fight_id, []).append(index)

        red_id = normalize_identifier(row.get("red_fighter_id"))
        blue_id = normalize_identifier(row.get("blue_fighter_id"))
        if red_id and blue_id:
            key = tuple(sorted((red_id, blue_id)))
            lookup["fighter_ids"].setdefault(key, []).append(index)

        red_name = normalize_person(row.get("red_fighter"))
        blue_name = normalize_person(row.get("blue_fighter"))
        if red_name and blue_name:
            key = tuple(sorted((red_name, blue_name)))
            lookup["fighter_names"].setdefault(key, []).append(index)

    return lookup


def narrow_matches(
    prediction: pd.Series,
    results: pd.DataFrame,
    candidate_indices: list[int],
) -> list[int]:
    if len(candidate_indices) <= 1:
        return candidate_indices

    subset = results.loc[candidate_indices]

    prediction_event_id = normalize_identifier(prediction.get("event_id"))
    if prediction_event_id and "event_id" in subset.columns:
        matching = subset[
            subset["event_id"].map(normalize_identifier).eq(prediction_event_id)
        ]
        if len(matching) == 1:
            return matching.index.tolist()
        if len(matching) > 1:
            subset = matching

    prediction_event_name = normalize_person(prediction.get("event_name"))
    if prediction_event_name and "event_name" in subset.columns:
        matching = subset[
            subset["event_name"].map(normalize_person).eq(prediction_event_name)
        ]
        if len(matching) == 1:
            return matching.index.tolist()
        if len(matching) > 1:
            subset = matching

    prediction_time = pd.to_datetime(
        prediction.get("commence_time"), errors="coerce", utc=True
    )
    if pd.notna(prediction_time) and "event_date" in subset.columns:
        result_dates = pd.to_datetime(subset["event_date"], errors="coerce", utc=True)
        date_delta = (result_dates.dt.date - prediction_time.date()).map(
            lambda value: abs(value.days) if pd.notna(value) else 999999
        )
        minimum = date_delta.min()
        if minimum <= 2:
            matching = subset.loc[date_delta.eq(minimum)]
            if len(matching) == 1:
                return matching.index.tolist()
            subset = matching

    return subset.index.tolist()


def find_result_row(
    prediction: pd.Series,
    results: pd.DataFrame,
    lookup: dict[str, dict[object, list[int]]],
) -> tuple[pd.Series | None, str, str]:
    fight_id = normalize_identifier(prediction.get("fight_id"))
    if fight_id and fight_id in lookup["fight_id"]:
        indices = narrow_matches(
            prediction, results, lookup["fight_id"][fight_id]
        )
        if len(indices) == 1:
            return results.loc[indices[0]], "fight_id", ""
        return None, "", f"ambiguous fight_id match ({len(indices)} rows)"

    red_id = normalize_identifier(prediction.get("red_fighter_id"))
    blue_id = normalize_identifier(prediction.get("blue_fighter_id"))
    if red_id and blue_id:
        key = tuple(sorted((red_id, blue_id)))
        if key in lookup["fighter_ids"]:
            indices = narrow_matches(
                prediction, results, lookup["fighter_ids"][key]
            )
            if len(indices) == 1:
                return results.loc[indices[0]], "fighter_ids", ""
            return None, "", f"ambiguous fighter-ID match ({len(indices)} rows)"

    red_name = normalize_person(prediction.get("red_fighter"))
    blue_name = normalize_person(prediction.get("blue_fighter"))
    if red_name and blue_name:
        key = tuple(sorted((red_name, blue_name)))
        if key in lookup["fighter_names"]:
            indices = narrow_matches(
                prediction, results, lookup["fighter_names"][key]
            )
            if len(indices) == 1:
                return results.loc[indices[0]], "fighter_names", ""
            return None, "", f"ambiguous fighter-name match ({len(indices)} rows)"

    return None, "", "no completed-result match"


def determine_actual_winner(result: pd.Series) -> tuple[str, str]:
    winner_id = normalize_identifier(result.get("winner_id"))
    winner_name = str(result.get("winner", "")).strip()

    if not winner_name or winner_name.lower() in {"nan", "none", "<na>"}:
        winner_name = ""

    if not winner_name:
        winner_side = str(result.get("winner_side", "")).lower().strip()
        if winner_side in {"red", "r", "fighter_1", "fighter1", "a"}:
            winner_name = str(result.get("red_fighter", "")).strip()
            winner_id = winner_id or normalize_identifier(
                result.get("red_fighter_id")
            )
        elif winner_side in {"blue", "b", "fighter_2", "fighter2"}:
            winner_name = str(result.get("blue_fighter", "")).strip()
            winner_id = winner_id or normalize_identifier(
                result.get("blue_fighter_id")
            )

    return winner_name, winner_id


def grade_moneyline_pick(
    prediction: pd.Series,
    result: pd.Series,
) -> tuple[bool | None, str, str]:
    actual_winner, actual_winner_id = determine_actual_winner(result)
    if not actual_winner and not actual_winner_id:
        return None, "", "result row has no resolvable winner"

    predicted_fighter_id = normalize_identifier(
        prediction.get("outcome_fighter_id")
    )
    predicted_label = str(
        prediction.get("outcome_label", prediction.get("model_pick", ""))
    ).strip()

    if predicted_fighter_id and actual_winner_id:
        return (
            predicted_fighter_id == actual_winner_id,
            actual_winner,
            "",
        )

    predicted_norm = normalize_person(predicted_label)
    actual_norm = normalize_person(actual_winner)
    if predicted_norm and actual_norm:
        return predicted_norm == actual_norm, actual_winner, ""

    return None, actual_winner, "could not compare predicted fighter to winner"


def select_pick_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()

    if "is_model_pick" in df.columns:
        values = df["is_model_pick"]
        if values.dtype == bool:
            mask = values
        else:
            mask = values.astype(str).str.lower().isin({"true", "1", "yes"})
        picks = df.loc[mask].copy()
    elif {"model_pick", "outcome_label"}.issubset(df.columns):
        picks = df.loc[
            df["model_pick"].astype(str).eq(df["outcome_label"].astype(str))
        ].copy()
    else:
        group_columns = [
            column
            for column in (
                "prediction_run_id",
                "model_id",
                "market_key",
                "fight_id",
            )
            if column in df.columns
        ]
        if not group_columns:
            raise RuntimeError(
                "Predictions contain neither is_model_pick nor enough columns "
                "to infer the selected outcome."
            )
        index = df.groupby(group_columns, dropna=False)["model_probability"].idxmax()
        picks = df.loc[index].copy()

    if "model_pick_probability" not in picks.columns:
        picks["model_pick_probability"] = picks["model_probability"]

    picks["model_pick_probability"] = pd.to_numeric(
        picks["model_pick_probability"], errors="coerce"
    )

    dedupe_columns = [
        column
        for column in (
            "prediction_run_id",
            "model_id",
            "market_key",
            "fight_id",
            "model_pick",
            "outcome_label",
        )
        if column in picks.columns
    ]
    if dedupe_columns:
        picks = picks.drop_duplicates(subset=dedupe_columns, keep="first")

    return picks.reset_index(drop=True)


def grade_predictions(
    pick_rows: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    lookup = build_result_lookup(results)
    audit_rows: list[dict[str, object]] = []

    for _, prediction in pick_rows.iterrows():
        result, match_method, match_error = find_result_row(
            prediction, results, lookup
        )

        record = prediction.to_dict()
        record["result_match_method"] = match_method
        record["audit_status"] = "unmatched"
        record["audit_reason"] = match_error
        record["actual_winner"] = pd.NA
        record["is_correct"] = pd.NA

        market_key = str(prediction.get("market_key", "")).lower().strip()
        model_family = str(prediction.get("model_family", "")).lower().strip()

        if result is None:
            audit_rows.append(record)
            continue

        is_moneyline = (
            market_key in {"", "h2h", "moneyline", "winner", "fight_winner"}
            or model_family == "moneyline"
            or pd.notna(prediction.get("outcome_fighter_id"))
        )
        if not is_moneyline:
            record["audit_status"] = "unsupported_market"
            record["audit_reason"] = (
                f"market '{market_key or model_family}' requires a market-specific "
                "result grader"
            )
            audit_rows.append(record)
            continue

        correct, actual_winner, grade_error = grade_moneyline_pick(
            prediction, result
        )
        record["actual_winner"] = actual_winner

        for column in (
            "winner_id",
            "method",
            "round",
            "time",
            "event_date",
        ):
            if column in result.index:
                record[f"actual_{column}"] = result.get(column)

        if correct is None:
            record["audit_status"] = "ungradable"
            record["audit_reason"] = grade_error
        else:
            record["audit_status"] = "graded"
            record["audit_reason"] = ""
            record["is_correct"] = bool(correct)

        audit_rows.append(record)

    return pd.DataFrame(audit_rows)


def safe_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities.astype(float), 1e-15, 1 - 1e-15)
    y_true = y_true.astype(float)
    return float(
        -np.mean(
            y_true * np.log(probabilities)
            + (1 - y_true) * np.log(1 - probabilities)
        )
    )


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return math.nan

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, probabilities))
    except Exception:
        ranks = pd.Series(probabilities).rank(method="average").to_numpy()
        positive_rank_sum = ranks[y_true == 1].sum()
        auc = (
            positive_rank_sum - positives * (positives + 1) / 2
        ) / (positives * negatives)
        return float(auc)


def summarize_group(group: pd.DataFrame) -> pd.Series:
    y = group["is_correct"].astype(bool).astype(int).to_numpy()
    probabilities = pd.to_numeric(
        group["model_pick_probability"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(probabilities)
    y = y[valid]
    probabilities = probabilities[valid]

    if len(y) == 0:
        return pd.Series(
            {
                "graded_picks": 0,
                "correct_picks": 0,
                "accuracy": math.nan,
                "avg_pick_probability": math.nan,
                "brier_score": math.nan,
                "log_loss": math.nan,
                "roc_auc_pick_correctness": math.nan,
            }
        )

    return pd.Series(
        {
            "graded_picks": int(len(y)),
            "correct_picks": int(y.sum()),
            "accuracy": float(y.mean()),
            "avg_pick_probability": float(probabilities.mean()),
            "brier_score": float(np.mean((probabilities - y) ** 2)),
            "log_loss": safe_log_loss(y, probabilities),
            "roc_auc_pick_correctness": safe_roc_auc(y, probabilities),
        }
    )


def build_summary(graded: pd.DataFrame) -> pd.DataFrame:
    if graded.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in ("model_id", "model_family", "market_key", "algorithm")
        if column in graded.columns
    ]
    if not group_columns:
        graded = graded.copy()
        graded["_all_models"] = "all"
        group_columns = ["_all_models"]

    summary = (
        graded.groupby(group_columns, dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    if "prediction_timestamp" in graded.columns:
        timestamps = graded.groupby(group_columns, dropna=False)[
            "prediction_timestamp"
        ].agg(first_prediction="min", last_prediction="max").reset_index()
        summary = summary.merge(timestamps, on=group_columns, how="left")

    return summary.sort_values(
        ["graded_picks", "accuracy"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_calibration(graded: pd.DataFrame) -> pd.DataFrame:
    if graded.empty:
        return pd.DataFrame()

    df = graded.copy()
    df["model_pick_probability"] = pd.to_numeric(
        df["model_pick_probability"], errors="coerce"
    )
    df = df[df["model_pick_probability"].between(0, 1, inclusive="both")]
    if df.empty:
        return pd.DataFrame()

    edges = np.linspace(0.0, 1.0, 11)
    df["probability_bucket"] = pd.cut(
        df["model_pick_probability"],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    group_columns = [
        column
        for column in ("model_id", "market_key")
        if column in df.columns
    ] + ["probability_bucket"]

    calibration = (
        df.groupby(group_columns, observed=True, dropna=False)
        .agg(
            picks=("is_correct", "size"),
            correct=("is_correct", "sum"),
            mean_predicted_probability=("model_pick_probability", "mean"),
            actual_accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    calibration["calibration_error"] = (
        calibration["actual_accuracy"]
        - calibration["mean_predicted_probability"]
    )
    calibration["probability_bucket"] = calibration[
        "probability_bucket"
    ].astype(str)
    return calibration


def select_latest_prefight_predictions(graded: pd.DataFrame) -> pd.DataFrame:
    """Keep one latest available prediction per model, fight, and market.

    The model-market snapshot is append-only and may contain sportsbook copies,
    repeated captures, and multiple prediction runs before one fight.
    """

    if graded.empty:
        return graded.copy()

    df = graded.copy()

    red_id = df.get("red_fighter_id", pd.Series(pd.NA, index=df.index)).map(
        normalize_identifier
    )
    blue_id = df.get("blue_fighter_id", pd.Series(pd.NA, index=df.index)).map(
        normalize_identifier
    )
    fighter_id_key = pd.Series("", index=df.index, dtype="object")
    has_ids = red_id.ne("") & blue_id.ne("")
    fighter_id_key.loc[has_ids] = [
        "__".join(sorted(pair))
        for pair in zip(red_id.loc[has_ids], blue_id.loc[has_ids])
    ]

    red_name = df.get("red_fighter", pd.Series(pd.NA, index=df.index)).map(
        normalize_person
    )
    blue_name = df.get("blue_fighter", pd.Series(pd.NA, index=df.index)).map(
        normalize_person
    )
    fighter_name_key = pd.Series("", index=df.index, dtype="object")
    has_names = red_name.ne("") & blue_name.ne("")
    fighter_name_key.loc[has_names] = [
        "__".join(sorted(pair))
        for pair in zip(red_name.loc[has_names], blue_name.loc[has_names])
    ]

    fight_id_key = df.get(
        "fight_id", pd.Series(pd.NA, index=df.index)
    ).map(normalize_identifier)

    df["_audit_fight_key"] = np.where(
        fighter_id_key.ne(""),
        "ids:" + fighter_id_key,
        np.where(
            fighter_name_key.ne(""),
            "names:" + fighter_name_key,
            "fight_id:" + fight_id_key,
        ),
    )

    if "prediction_timestamp" in df.columns:
        df["prediction_timestamp"] = pd.to_datetime(
            df["prediction_timestamp"], errors="coerce", utc=True
        )
    else:
        df["prediction_timestamp"] = pd.NaT

    if "actual_event_date" in df.columns:
        event_date = pd.to_datetime(
            df["actual_event_date"], errors="coerce", utc=True
        )
        prefight_cutoff = event_date.dt.normalize() + pd.Timedelta(days=1)
        valid_prefight = (
            df["prediction_timestamp"].isna()
            | prefight_cutoff.isna()
            | df["prediction_timestamp"].lt(prefight_cutoff)
        )
        df = df.loc[valid_prefight].copy()

    group_columns = [
        column
        for column in ("model_id", "market_key", "_audit_fight_key")
        if column in df.columns
    ]
    if not group_columns:
        return df.drop(columns=["_audit_fight_key"], errors="ignore")

    df["_timestamp_sort"] = df["prediction_timestamp"].fillna(
        pd.Timestamp("1900-01-01", tz="UTC")
    )
    df = df.sort_values(
        group_columns + ["_timestamp_sort"],
        ascending=[True] * len(group_columns) + [True],
    )
    latest = df.groupby(group_columns, dropna=False, as_index=False).tail(1)

    return latest.drop(
        columns=["_audit_fight_key", "_timestamp_sort"],
        errors="ignore",
    ).reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    audit: pd.DataFrame,
    prediction_sources: pd.DataFrame,
    result_path: Path,
    candidates: list[tuple[Path, float, dict[str, object]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graded_all = audit.loc[audit["audit_status"].eq("graded")].copy()
    if not graded_all.empty:
        graded_all["is_correct"] = graded_all["is_correct"].astype(bool)

    graded_latest = select_latest_prefight_predictions(graded_all)

    all_snapshot_summary = build_summary(graded_all)
    latest_prefight_summary = build_summary(graded_latest)
    all_snapshot_calibration = build_calibration(graded_all)
    latest_prefight_calibration = build_calibration(graded_latest)
    unmatched = audit.loc[~audit["audit_status"].eq("graded")].copy()

    audit.to_parquet(output_dir / "prediction_audit_rows.parquet", index=False)
    audit.to_csv(output_dir / "prediction_audit_rows.csv", index=False)
    graded_latest.to_parquet(
        output_dir / "latest_prefight_prediction_rows.parquet", index=False
    )
    graded_latest.to_csv(
        output_dir / "latest_prefight_prediction_rows.csv", index=False
    )

    latest_prefight_summary.to_csv(
        output_dir / "model_performance_summary.csv", index=False
    )
    latest_prefight_summary.to_csv(
        output_dir / "latest_prefight_model_performance_summary.csv", index=False
    )
    all_snapshot_summary.to_csv(
        output_dir / "all_snapshot_model_performance_summary.csv", index=False
    )

    latest_prefight_calibration.to_csv(
        output_dir / "model_calibration.csv", index=False
    )
    latest_prefight_calibration.to_csv(
        output_dir / "latest_prefight_model_calibration.csv", index=False
    )
    all_snapshot_calibration.to_csv(
        output_dir / "all_snapshot_model_calibration.csv", index=False
    )

    unmatched.to_csv(
        output_dir / "unmatched_or_ungraded_predictions.csv", index=False
    )
    prediction_sources.to_csv(
        output_dir / "prediction_source_inventory.csv", index=False
    )

    candidate_payload = []
    for path, score, details in candidates[:50]:
        candidate_payload.append(
            {
                "path": str(path),
                "score": score,
                "resolved_columns": {
                    key: value
                    for key, value in details.get("resolved", {}).items()
                    if value
                },
            }
        )

    diagnostics = {
        "selected_results_path": str(result_path),
        "prediction_rows_total": int(len(audit)),
        "graded_snapshot_rows": int(len(graded_all)),
        "graded_latest_prefight_rows": int(len(graded_latest)),
        "duplicate_or_earlier_snapshot_rows_removed": int(
            len(graded_all) - len(graded_latest)
        ),
        "unmatched_or_ungraded_rows": int(len(unmatched)),
        "official_summary": str(output_dir / "model_performance_summary.csv"),
        "result_candidates": candidate_payload,
    }
    (output_dir / "audit_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nAudit outputs")
    print("=" * 80)
    for filename in (
        "prediction_audit_rows.parquet",
        "prediction_audit_rows.csv",
        "latest_prefight_prediction_rows.parquet",
        "latest_prefight_prediction_rows.csv",
        "model_performance_summary.csv",
        "latest_prefight_model_performance_summary.csv",
        "all_snapshot_model_performance_summary.csv",
        "model_calibration.csv",
        "latest_prefight_model_calibration.csv",
        "all_snapshot_model_calibration.csv",
        "unmatched_or_ungraded_predictions.csv",
        "prediction_source_inventory.csv",
        "audit_diagnostics.json",
    ):
        print(output_dir / filename)

    print("\nEvaluation row counts")
    print("=" * 80)
    print(f"All graded snapshots: {len(graded_all):,}")
    print(f"Latest pre-fight model/fight rows: {len(graded_latest):,}")
    print(
        "Earlier or duplicate snapshots removed: "
        f"{len(graded_all) - len(graded_latest):,}"
    )

    if not latest_prefight_summary.empty:
        print("\nOfficial model performance: latest pre-fight prediction per fight")
        print("=" * 80)
        display = latest_prefight_summary.copy()
        for column in (
            "accuracy",
            "avg_pick_probability",
            "brier_score",
            "log_loss",
            "roc_auc_pick_correctness",
        ):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: f"{value:.4f}" if pd.notna(value) else ""
                )
        print(display.to_string(index=False))
    else:
        print("\nNo predictions were successfully graded.")
        if not unmatched.empty and "audit_reason" in unmatched.columns:
            print("\nTop audit failure reasons:")
            print(unmatched["audit_reason"].value_counts().head(10).to_string())

    if not all_snapshot_summary.empty:
        print("\nAll-snapshot comparison")
        print("=" * 80)
        display = all_snapshot_summary.copy()
        for column in (
            "accuracy",
            "avg_pick_probability",
            "brier_score",
            "log_loss",
            "roc_auc_pick_correctness",
        ):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: f"{value:.4f}" if pd.notna(value) else ""
                )
        print(display.to_string(index=False))


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    if not repo_root.exists():
        print(f"ERROR: Repository root does not exist: {repo_root}", file=sys.stderr)
        return 2

    candidates = find_result_candidates(repo_root)
    if args.list_result_candidates:
        print_result_candidates(candidates, repo_root)
        return 0

    print("=" * 80)
    print("UFC HISTORICAL PREDICTION AUDIT")
    print("=" * 80)
    print(f"Repository root: {repo_root}")

    current_frames = load_current_predictions(
        repo_root, args.prediction_path
    )
    historical_frames: list[LoadedFrame] = []

    if not args.no_git_history:
        git_paths = (
            args.prediction_git_path
            if args.prediction_git_path
            else default_git_prediction_paths(repo_root)
        )
        print(f"Git prediction paths: {len(git_paths)}")
        historical_frames = recover_git_predictions(
            repo_root,
            git_paths,
            max_commits=max(1, args.max_git_commits),
        )

    all_frames = current_frames + historical_frames
    if not all_frames:
        print(
            "ERROR: No prediction artifacts were found. Supply --prediction-path "
            "or commit prediction artifacts before running the audit.",
            file=sys.stderr,
        )
        return 2

    predictions = normalize_predictions(all_frames)
    if predictions.empty:
        print("ERROR: Prediction artifacts contained no usable rows.", file=sys.stderr)
        return 2

    if args.model_id and "model_id" in predictions.columns:
        predictions = predictions[
            predictions["model_id"].astype(str).isin(args.model_id)
        ].copy()

    if args.market_key and "market_key" in predictions.columns:
        predictions = predictions[
            predictions["market_key"].astype(str).isin(args.market_key)
        ].copy()

    pick_rows = select_pick_rows(predictions)
    pick_rows = pick_rows[
        pick_rows["model_pick_probability"].fillna(-np.inf)
        >= args.min_pick_probability
    ].copy()

    print(f"Prediction snapshots loaded: {len(all_frames)}")
    print(f"Unique outcome rows: {len(predictions):,}")
    print(f"Selected model-pick rows: {len(pick_rows):,}")

    try:
        results_path, candidates = choose_results_path(
            repo_root, args.results
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_result_candidates(candidates, repo_root)
        return 2

    print(f"Completed-results file: {results_path}")
    raw_results = read_table(results_path)
    results = standardize_results(raw_results)

    if "winner" not in results.columns and "winner_id" not in results.columns:
        print(
            "ERROR: Selected results file has no resolvable winner column. "
            "Run --list-result-candidates and pass --results explicitly.",
            file=sys.stderr,
        )
        print(f"Columns: {list(results.columns)}", file=sys.stderr)
        return 2

    audit = grade_predictions(pick_rows, results)

    source_rows = []
    for loaded in all_frames:
        source_rows.append(
            {
                "source": loaded.source,
                "source_kind": loaded.source_kind,
                "git_commit": loaded.git_commit,
                "rows": len(loaded.frame),
                "columns": ",".join(map(str, loaded.frame.columns)),
            }
        )
    source_inventory = pd.DataFrame(source_rows)

    write_outputs(
        output_dir,
        audit,
        source_inventory,
        results_path,
        candidates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
