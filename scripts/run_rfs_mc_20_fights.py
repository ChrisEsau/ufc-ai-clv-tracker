from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


MASTER_PATH = Path("data/master/ufc_master.parquet")
PATH_COUNT = 2000
SEED = 42
TARGET_YEAR = 2026
FIGHT_COUNT = 20


def find_column(
    columns: pd.Index,
    candidates: list[str],
) -> str:
    """Return the first available column from a candidate list."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        f"Could not find any column from {candidates}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the batch runner."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile-source",
        choices=("ewm", "last3"),
        default="ewm",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=PATH_COUNT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--fight-count",
        type=int,
        default=FIGHT_COUNT,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(
        "data/simulation/"
        f"rfs_mc_v1_20_fight_batch_{args.profile_source}"
    )
    summary_path = output_dir / "summary.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(MASTER_PATH)

    fight_id_col = find_column(
        master.columns,
        ["fight_id"],
    )
    red_id_col = find_column(
        master.columns,
        [
            "r_fighter_id",
            "red_fighter_id",
            "r_id",
            "red_id",
        ],
    )
    blue_id_col = find_column(
        master.columns,
        [
            "b_fighter_id",
            "blue_fighter_id",
            "b_id",
            "blue_id",
        ],
    )
    date_col = find_column(
        master.columns,
        ["date", "fight_date", "event_date"],
    )
    weight_col = find_column(
        master.columns,
        ["weight_class", "division"],
    )
    rounds_col = find_column(
        master.columns,
        ["total_rounds", "scheduled_rounds"],
    )

    master = master.copy()
    master[date_col] = pd.to_datetime(
        master[date_col],
        errors="coerce",
    )

    fights = (
        master.loc[
            master[date_col].dt.year.eq(TARGET_YEAR)
            & master[red_id_col].notna()
            & master[blue_id_col].notna()
            & master[weight_col].notna()
            & master[rounds_col].isin([3, 5])
        ]
        .sort_values(date_col)
        .drop_duplicates(subset=[fight_id_col])
    )

    if fights.empty:
        raise RuntimeError(
            f"No candidate fights found from {TARGET_YEAR}"
        )

    summaries: list[dict[str, object]] = []
    successful_count = 0
    skipped_count = 0

    for _, row in fights.iterrows():
        if successful_count >= args.fight_count:
            break

        batch_number = successful_count + 1
        fight_id = str(row[fight_id_col])
        target_date = row[date_col].strftime("%Y-%m-%d")
        scheduled_rounds = int(row[rounds_col])

        output_path = output_dir / f"{fight_id}.json"

        print(
            f"\n[{batch_number:02d}/{args.fight_count}] "
            f"Running fight {fight_id} — {target_date}",
            flush=True,
        )

        command = [
            sys.executable,
            "-m",
            "pipeline.simulation.rfs_mc_v1.run_simulation",
            "--red-fighter-id",
            str(row[red_id_col]),
            "--blue-fighter-id",
            str(row[blue_id_col]),
            "--target-date",
            target_date,
            "--weight-class",
            str(row[weight_col]),
            "--gender",
            "male",
            "--scheduled-rounds",
            str(scheduled_rounds),
            "--paths",
            str(args.paths),
            "--seed",
            str(args.seed),
            "--profile-source",
            args.profile_source,
            "--output",
            str(output_path),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            skipped_count += 1

            print(
                f"SKIPPED: {fight_id} — profile unavailable "
                "or simulator error",
                flush=True,
            )

            error_lines = [
                line
                for line in completed.stderr.splitlines()
                if line.strip()
            ]
            if error_lines:
                print(error_lines[-1], flush=True)

            continue

        successful_count += 1

        result = json.loads(
            output_path.read_text(encoding="utf-8")
        )
        request = result["request"]
        simulation = result["simulation"]
        methods = simulation["method_probabilities"]
        finish = simulation["finish_distribution"]

        red_name = request["red_fighter_name"]
        blue_name = request["blue_fighter_name"]

        red_win = simulation["red_win_probability"]
        blue_win = simulation["blue_win_probability"]

        predicted_winner = (
            red_name
            if red_win >= blue_win
            else blue_name
        )
        predicted_probability = max(red_win, blue_win)

        total_ko = (
            methods["red_ko_tko"]
            + methods["blue_ko_tko"]
        )
        total_submission = (
            methods["red_submission"]
            + methods["blue_submission"]
        )
        total_decision = (
            methods["red_decision"]
            + methods["blue_decision"]
        )

        print(
            f"COMPLETE: {red_name} vs {blue_name}",
            flush=True,
        )
        print(
            f"  Pick: {predicted_winner} "
            f"({predicted_probability:.1%})",
            flush=True,
        )
        print(
            f"  Red/Blue: {red_win:.1%} / {blue_win:.1%}",
            flush=True,
        )
        print(
            f"  Methods: KO {total_ko:.1%} | "
            f"SUB {total_submission:.1%} | "
            f"DEC {total_decision:.1%}",
            flush=True,
        )
        print(
            "  Finish/Distance: "
            f"{simulation['finish_probability']:.1%} / "
            f"{simulation['distance_probability']:.1%}",
            flush=True,
        )

        summaries.append(
            {
                "fight_id": fight_id,
                "date": target_date,
                "profile_source": args.profile_source,
                "red_fighter": red_name,
                "blue_fighter": blue_name,
                "weight_class": request["weight_class"],
                "scheduled_rounds": scheduled_rounds,
                "predicted_winner": predicted_winner,
                "predicted_probability": predicted_probability,
                "red_win_probability": red_win,
                "blue_win_probability": blue_win,
                "draw_probability": simulation[
                    "draw_probability"
                ],
                "red_ko_tko": methods["red_ko_tko"],
                "red_submission": methods["red_submission"],
                "red_decision": methods["red_decision"],
                "blue_ko_tko": methods["blue_ko_tko"],
                "blue_submission": methods["blue_submission"],
                "blue_decision": methods["blue_decision"],
                "total_ko_probability": total_ko,
                "total_submission_probability": total_submission,
                "total_decision_probability": total_decision,
                "finish_probability": simulation[
                    "finish_probability"
                ],
                "distance_probability": simulation[
                    "distance_probability"
                ],
                "mean_finish_round": (
                    finish["mean_round"]
                    if finish is not None
                    else None
                ),
                "mean_elapsed_seconds": (
                    finish["mean_elapsed_seconds"]
                    if finish is not None
                    else None
                ),
            }
        )

    if not summaries:
        raise RuntimeError(
            "No fights completed successfully."
        )

    pd.DataFrame(summaries).to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print(
        f"Completed: {successful_count} | "
        f"Skipped: {skipped_count}"
    )
    print(f"Profile source: {args.profile_source}")
    print(f"Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
