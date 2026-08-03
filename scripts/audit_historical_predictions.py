#!/usr/bin/env python3
"""Audit historical UFC moneyline predictions against completed results.

Run from the repository root in Codespaces:

    python scripts/audit_historical_predictions.py

Useful options:

    python scripts/audit_historical_predictions.py --list-result-candidates
    python scripts/audit_historical_predictions.py --results PATH
    python scripts/audit_historical_predictions.py --no-git-history

The script is read-only outside its output directory:
    data/audits/prediction_performance/
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/audits/prediction_performance")
PREDICTION_PATHS = (
    "data/predictions/model_outcomes.parquet",
    "data/predictions/model_outcomes.csv",
)
PREDICTION_GLOBS = (
    "data/predictions/by_model/*/model_outcomes.parquet",
    "data/predictions/by_model/*/model_outcomes.csv",
    "data/predictions/history/*.parquet",
    "data/predictions/history/*.csv",
)

ALIASES = {
    "fight_id": ("fight_id", "bout_id", "match_id", "ufcstats_fight_id"),
    "event_id": ("event_id", "ufcstats_event_id"),
    "event_name": ("event_name", "event", "event_title"),
    "event_date": ("event_date", "date", "commence_time"),
    "red_fighter": (
        "red_fighter", "r_fighter", "fighter_red", "fighter_1", "fighter1"
    ),
    "blue_fighter": (
        "blue_fighter", "b_fighter", "fighter_blue", "fighter_2", "fighter2"
    ),
    "red_fighter_id": (
        "red_fighter_id", "r_fighter_id", "fighter_red_id", "fighter_1_id"
    ),
    "blue_fighter_id": (
        "blue_fighter_id", "b_fighter_id", "fighter_blue_id", "fighter_2_id"
    ),
    "winner": (
        "winner", "winner_name", "winning_fighter", "result_winner"
    ),
    "winner_id": ("winner_id", "winner_fighter_id", "winning_fighter_id"),
    "winner_side": (
        "winner_side", "winning_side", "winner_corner", "result_side"
    ),
    "red_result": ("red_result", "r_result", "result_red"),
    "blue_result": ("blue_result", "b_result", "result_blue"),
    "method": ("method", "result_method", "finish_method"),
    "round": ("round", "finish_round", "result_round"),
    "time": ("time", "finish_time", "result_time"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results", default=None)
    parser.add_argument("--prediction-path", action="append", default=[])
    parser.add_argument("--prediction-git-path", action="append", default=[])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--no-git-history", action="store_true")
    parser.add_argument("--max-git-commits", type=int, default=500)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--market-key", action="append", default=[])
    parser.add_argument("--min-pick-probability", type=float, default=0.0)
    parser.add_argument("--list-result-candidates", action="store_true")
    return parser.parse_args()


def run_git(repo: Path, args: list[str], *, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=not binary,
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def read_table_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    if suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(data))
    if suffix.lower() == ".csv":
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    raise ValueError(f"Unsupported historical file type: {suffix}")


def norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def norm_person(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("’", "'")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def norm_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [norm_col(column) for column in result.columns]
    return result


def first_alias(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    return next((alias for alias in aliases if alias in available), None)


def standardize_results(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)
    rename = {}
    for canonical, aliases in ALIASES.items():
        source = first_alias(df.columns, aliases)
        if source and canonical not in df.columns:
            rename[source] = canonical
    df = df.rename(columns=rename)

    if "winner" not in df.columns:
        winner = pd.Series(pd.NA, index=df.index, dtype="object")
        if {"red_result", "red_fighter"}.issubset(df.columns):
            mask = df["red_result"].astype(str).str.upper().isin({"W", "WIN", "WINNER"})
            winner.loc[mask] = df.loc[mask, "red_fighter"]
        if {"blue_result", "blue_fighter"}.issubset(df.columns):
            mask = df["blue_result"].astype(str).str.upper().isin({"W", "WIN", "WINNER"})
            winner.loc[mask] = df.loc[mask, "blue_fighter"]
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
        winner_norm = df["winner"].map(norm_person)
        winner_id = pd.Series(pd.NA, index=df.index, dtype="object")
        if "red_fighter_id" in df.columns:
            mask = winner_norm.eq(df["red_fighter"].map(norm_person))
            winner_id.loc[mask] = df.loc[mask, "red_fighter_id"]
        if "blue_fighter_id" in df.columns:
            mask = winner_norm.eq(df["blue_fighter"].map(norm_person))
            winner_id.loc[mask] = df.loc[mask, "blue_fighter_id"]
        if winner_id.notna().any():
            df["winner_id"] = winner_id

    return df


def prediction_paths(repo: Path, explicit: list[str]) -> list[Path]:
    paths: set[Path] = set()
    if explicit:
        for item in explicit:
            path = Path(item)
            path = path if path.is_absolute() else repo / path
            if path.is_file():
                paths.add(path.resolve())
    else:
        for item in PREDICTION_PATHS:
            path = repo / item
            if path.is_file():
                paths.add(path.resolve())
        for pattern in PREDICTION_GLOBS:
            paths.update(path.resolve() for path in repo.glob(pattern) if path.is_file())
    return sorted(paths)


def load_working_predictions(repo: Path, explicit: list[str]) -> list[pd.DataFrame]:
    frames = []
    for path in prediction_paths(repo, explicit):
        try:
            df = normalize_columns(read_table(path))
        except Exception as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        if "model_probability" not in df.columns:
            continue
        df["_prediction_source"] = str(path.relative_to(repo))
        df["_prediction_source_kind"] = "working_tree"
        df["_prediction_git_commit"] = pd.NA
        frames.append(df)
    return frames


def recover_git_predictions(
    repo: Path,
    paths: list[str],
    max_commits: int,
) -> list[pd.DataFrame]:
    frames = []
    try:
        run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    except Exception:
        print("WARNING: Git history recovery skipped; not a Git repository.")
        return frames

    for path in paths:
        try:
            commits = run_git(repo, ["log", "--all", "--format=%H", "--", path]).stdout.splitlines()
        except subprocess.CalledProcessError:
            continue
        seen = set()
        for commit in commits:
            if commit in seen:
                continue
            seen.add(commit)
            if len(seen) > max_commits:
                break
            try:
                raw = run_git(repo, ["show", f"{commit}:{path}"], binary=True).stdout
                df = normalize_columns(read_table_bytes(raw, Path(path).suffix))
            except Exception:
                continue
            if "model_probability" not in df.columns:
                continue
            df["_prediction_source"] = path
            df["_prediction_source_kind"] = "git_history"
            df["_prediction_git_commit"] = commit
            frames.append(df)
    return frames


def combine_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "prediction_timestamp" in df.columns:
        df["prediction_timestamp"] = pd.to_datetime(
            df["prediction_timestamp"], errors="coerce", utc=True
        )
    df["model_probability"] = pd.to_numeric(df["model_probability"], errors="coerce")
    if "model_pick_probability" not in df.columns:
        if "model_confidence" in df.columns:
            df["model_pick_probability"] = pd.to_numeric(
                df["model_confidence"], errors="coerce"
            )
        else:
            df["model_pick_probability"] = df["model_probability"]
    dedupe = [
        column for column in (
            "prediction_run_id", "prediction_timestamp", "model_id", "market_key",
            "fight_id", "outcome_label", "model_probability"
        ) if column in df.columns
    ]
    return df.drop_duplicates(dedupe).reset_index(drop=True) if dedupe else df


def select_pick_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "is_model_pick" in df.columns:
        mask = df["is_model_pick"].astype(str).str.lower().isin({"true", "1", "yes"})
        picks = df.loc[mask].copy()
    elif {"model_pick", "outcome_label"}.issubset(df.columns):
        picks = df.loc[df["model_pick"].astype(str).eq(df["outcome_label"].astype(str))].copy()
    else:
        groups = [
            column for column in ("prediction_run_id", "model_id", "market_key", "fight_id")
            if column in df.columns
        ]
        if not groups:
            raise RuntimeError("Cannot determine model-pick rows.")
        picks = df.loc[df.groupby(groups, dropna=False)["model_probability"].idxmax()].copy()
    picks["model_pick_probability"] = pd.to_numeric(
        picks["model_pick_probability"], errors="coerce"
    )
    return picks.reset_index(drop=True)


def result_candidate_score(path: Path) -> tuple[float, dict[str, str]]:
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            columns = [norm_col(column) for column in pq.ParquetFile(path).schema.names]
        else:
            columns = [norm_col(column) for column in pd.read_csv(path, nrows=0).columns]
    except Exception:
        return -999.0, {}

    resolved = {
        canonical: first_alias(columns, aliases)
        for canonical, aliases in ALIASES.items()
    }
    score = 0.0
    score += 8 if resolved["fight_id"] else 0
    score += 10 if resolved["winner"] or resolved["winner_id"] else 0
    score += 5 if resolved["winner_side"] else 0
    score += 5 if resolved["red_result"] or resolved["blue_result"] else 0
    score += 6 if resolved["red_fighter"] and resolved["blue_fighter"] else 0
    score += 4 if resolved["red_fighter_id"] and resolved["blue_fighter_id"] else 0
    text = path.as_posix().lower()
    score += sum(0.5 for token in ("master", "fight", "result", "historical") if token in text)
    score -= sum(3 for token in ("prediction", "live_card", "feature", "audit") if token in text)
    return score, {key: value for key, value in resolved.items() if value}


def result_candidates(repo: Path):
    candidates = []
    data = repo / "data"
    if not data.exists():
        return candidates
    for pattern in ("**/*.parquet", "**/*.pq", "**/*.csv"):
        for path in data.glob(pattern):
            score, resolved = result_candidate_score(path)
            if score > 0:
                candidates.append((path.resolve(), score, resolved))
    return sorted(candidates, key=lambda item: (-item[1], item[0].as_posix()))


def print_candidates(candidates, repo: Path) -> None:
    print("\nRanked completed-result candidates")
    print("=" * 80)
    for index, (path, score, resolved) in enumerate(candidates[:25], start=1):
        try:
            path = path.relative_to(repo)
        except ValueError:
            pass
        print(f"{index:>2}. score={score:>5.1f}  {path}")
        print(f"    resolved={resolved}")


def select_results(repo: Path, explicit: str | None, candidates) -> Path:
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else repo / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    if not candidates or candidates[0][1] < 12:
        raise RuntimeError("No reliable results file was auto-detected; use --results PATH.")
    return candidates[0][0]


def build_lookup(results: pd.DataFrame):
    lookup = {"fight_id": {}, "fighter_ids": {}, "fighter_names": {}}
    for index, row in results.iterrows():
        fight_id = norm_id(row.get("fight_id"))
        if fight_id:
            lookup["fight_id"].setdefault(fight_id, []).append(index)
        red_id, blue_id = norm_id(row.get("red_fighter_id")), norm_id(row.get("blue_fighter_id"))
        if red_id and blue_id:
            lookup["fighter_ids"].setdefault(tuple(sorted((red_id, blue_id))), []).append(index)
        red, blue = norm_person(row.get("red_fighter")), norm_person(row.get("blue_fighter"))
        if red and blue:
            lookup["fighter_names"].setdefault(tuple(sorted((red, blue))), []).append(index)
    return lookup


def find_result(prediction: pd.Series, results: pd.DataFrame, lookup):
    keys = [
        ("fight_id", norm_id(prediction.get("fight_id"))),
        ("fighter_ids", tuple(sorted((norm_id(prediction.get("red_fighter_id")), norm_id(prediction.get("blue_fighter_id")))))),
        ("fighter_names", tuple(sorted((norm_person(prediction.get("red_fighter")), norm_person(prediction.get("blue_fighter")))))),
    ]
    for method, key in keys:
        if not key or (isinstance(key, tuple) and not all(key)):
            continue
        matches = lookup[method].get(key, [])
        if len(matches) == 1:
            return results.loc[matches[0]], method, ""
        if len(matches) > 1:
            return None, "", f"ambiguous {method} match ({len(matches)} rows)"
    return None, "", "no completed-result match"


def actual_winner(result: pd.Series) -> tuple[str, str]:
    winner = str(result.get("winner", "")).strip()
    if winner.lower() in {"", "nan", "none", "<na>"}:
        winner = ""
    winner_id = norm_id(result.get("winner_id"))
    return winner, winner_id


def grade(picks: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    lookup = build_lookup(results)
    rows = []
    for _, prediction in picks.iterrows():
        row = prediction.to_dict()
        result, method, reason = find_result(prediction, results, lookup)
        row.update({
            "result_match_method": method,
            "audit_status": "unmatched",
            "audit_reason": reason,
            "actual_winner": pd.NA,
            "is_correct": pd.NA,
        })
        if result is None:
            rows.append(row)
            continue

        market = str(prediction.get("market_key", "")).lower()
        family = str(prediction.get("model_family", "")).lower()
        moneyline = market in {"", "h2h", "moneyline", "winner", "fight_winner"} or family == "moneyline"
        if not moneyline:
            row["audit_status"] = "unsupported_market"
            row["audit_reason"] = f"market '{market or family}' needs a market-specific grader"
            rows.append(row)
            continue

        winner, winner_id = actual_winner(result)
        predicted_id = norm_id(prediction.get("outcome_fighter_id"))
        predicted_name = str(prediction.get("outcome_label", prediction.get("model_pick", "")))
        if winner_id and predicted_id:
            correct = winner_id == predicted_id
        elif winner and predicted_name:
            correct = norm_person(winner) == norm_person(predicted_name)
        else:
            row["audit_status"] = "ungradable"
            row["audit_reason"] = "winner or predicted fighter could not be resolved"
            rows.append(row)
            continue

        row["audit_status"] = "graded"
        row["audit_reason"] = ""
        row["actual_winner"] = winner
        row["is_correct"] = bool(correct)
        for column in ("winner_id", "method", "round", "time", "event_date"):
            if column in result.index:
                row[f"actual_{column}"] = result.get(column)
        rows.append(row)
    return pd.DataFrame(rows)


def roc_auc(y: np.ndarray, probability: np.ndarray) -> float:
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def summarize(group: pd.DataFrame) -> pd.Series:
    y = group["is_correct"].astype(bool).astype(int).to_numpy()
    probability = pd.to_numeric(group["model_pick_probability"], errors="coerce").to_numpy(float)
    valid = np.isfinite(probability)
    y, probability = y[valid], probability[valid]
    if not len(y):
        return pd.Series({"graded_picks": 0})
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    return pd.Series({
        "graded_picks": int(len(y)),
        "correct_picks": int(y.sum()),
        "accuracy": float(y.mean()),
        "avg_pick_probability": float(probability.mean()),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "roc_auc_pick_correctness": roc_auc(y, probability),
    })


def build_summary(graded: pd.DataFrame) -> pd.DataFrame:
    if graded.empty:
        return pd.DataFrame()
    groups = [column for column in ("model_id", "model_family", "market_key", "algorithm") if column in graded.columns]
    if not groups:
        graded = graded.assign(all_models="all")
        groups = ["all_models"]
    return (
        graded.groupby(groups, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
        .sort_values(["graded_picks", "accuracy"], ascending=[False, False])
    )


def build_calibration(graded: pd.DataFrame) -> pd.DataFrame:
    if graded.empty:
        return pd.DataFrame()
    df = graded.copy()
    df["model_pick_probability"] = pd.to_numeric(df["model_pick_probability"], errors="coerce")
    df = df[df["model_pick_probability"].between(0, 1)]
    if df.empty:
        return pd.DataFrame()
    df["probability_bucket"] = pd.cut(
        df["model_pick_probability"], np.linspace(0, 1, 11), include_lowest=True
    )
    groups = [column for column in ("model_id", "market_key") if column in df.columns] + ["probability_bucket"]
    result = df.groupby(groups, observed=True, dropna=False).agg(
        picks=("is_correct", "size"),
        correct=("is_correct", "sum"),
        mean_predicted_probability=("model_pick_probability", "mean"),
        actual_accuracy=("is_correct", "mean"),
    ).reset_index()
    result["calibration_error"] = result["actual_accuracy"] - result["mean_predicted_probability"]
    result["probability_bucket"] = result["probability_bucket"].astype(str)
    return result


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir)
    output = output if output.is_absolute() else repo / output

    candidates = result_candidates(repo)
    if args.list_result_candidates:
        print_candidates(candidates, repo)
        return 0

    print("=" * 80)
    print("UFC HISTORICAL PREDICTION AUDIT")
    print("=" * 80)

    frames = load_working_predictions(repo, args.prediction_path)
    if not args.no_git_history:
        git_paths = args.prediction_git_path or list(PREDICTION_PATHS)
        git_paths.extend(
            str(path.relative_to(repo))
            for path in prediction_paths(repo, [])
            if str(path.relative_to(repo)) not in git_paths
        )
        frames.extend(recover_git_predictions(repo, git_paths, args.max_git_commits))

    predictions = combine_predictions(frames)
    if predictions.empty:
        print("ERROR: no usable prediction artifacts found.", file=sys.stderr)
        return 2

    if args.model_id and "model_id" in predictions.columns:
        predictions = predictions[predictions["model_id"].astype(str).isin(args.model_id)]
    if args.market_key and "market_key" in predictions.columns:
        predictions = predictions[predictions["market_key"].astype(str).isin(args.market_key)]

    picks = select_pick_rows(predictions)
    picks = picks[picks["model_pick_probability"] >= args.min_pick_probability]

    try:
        results_path = select_results(repo, args.results, candidates)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_candidates(candidates, repo)
        return 2

    results = standardize_results(read_table(results_path))
    if "winner" not in results.columns and "winner_id" not in results.columns:
        print("ERROR: selected results file has no resolvable winner.", file=sys.stderr)
        return 2

    audit = grade(picks, results)
    graded = audit[audit["audit_status"].eq("graded")].copy()
    if not graded.empty:
        graded["is_correct"] = graded["is_correct"].astype(bool)

    summary = build_summary(graded)
    calibration = build_calibration(graded)
    unmatched = audit[~audit["audit_status"].eq("graded")]

    output.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output / "prediction_audit_rows.parquet", index=False)
    audit.to_csv(output / "prediction_audit_rows.csv", index=False)
    summary.to_csv(output / "model_performance_summary.csv", index=False)
    calibration.to_csv(output / "model_calibration.csv", index=False)
    unmatched.to_csv(output / "unmatched_or_ungraded_predictions.csv", index=False)
    (output / "audit_diagnostics.json").write_text(json.dumps({
        "selected_results_path": str(results_path),
        "prediction_outcome_rows": int(len(predictions)),
        "model_pick_rows": int(len(picks)),
        "graded_rows": int(len(graded)),
        "unmatched_or_ungraded_rows": int(len(unmatched)),
    }, indent=2), encoding="utf-8")

    print(f"Results file: {results_path}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Model picks: {len(picks):,}")
    print(f"Graded picks: {len(graded):,}")
    print(f"Outputs: {output}")
    if not summary.empty:
        print("\n" + summary.to_string(index=False))
    elif not unmatched.empty:
        print("\nNo predictions graded. Top reasons:")
        print(unmatched["audit_reason"].value_counts().head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
