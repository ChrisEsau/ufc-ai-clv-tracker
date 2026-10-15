"""Summarize an RFS Monte Carlo batch-run text log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, median


COMPLETE_RE = re.compile(r"^COMPLETE:\s+(.+?)\s+vs\s+(.+?)$")
PICK_RE = re.compile(r"^\s*Pick:\s+(.+?)\s+\(([\d.]+)%\)")
RED_BLUE_RE = re.compile(
    r"^\s*Red/Blue:\s+([\d.]+)%\s*/\s*([\d.]+)%"
)
METHOD_RE = re.compile(
    r"^\s*Methods:\s+KO\s+([\d.]+)%"
    r"\s*\|\s*SUB\s+([\d.]+)%"
    r"\s*\|\s*DEC\s+([\d.]+)%"
)
FINISH_RE = re.compile(
    r"^\s*Finish/Distance:\s+([\d.]+)%\s*/\s*([\d.]+)%"
)
SKIPPED_RE = re.compile(r"^SKIPPED:\s+(.+?)\s+—")


def parse_log(path: Path) -> tuple[list[dict], list[str]]:
    """Parse completed and skipped fights from a runner log."""

    fights: list[dict] = []
    skipped: list[str] = []
    current: dict | None = None

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        line = raw_line.rstrip()

        complete_match = COMPLETE_RE.match(line)
        if complete_match:
            current = {
                "red": complete_match.group(1),
                "blue": complete_match.group(2),
            }
            fights.append(current)
            continue

        skipped_match = SKIPPED_RE.match(line)
        if skipped_match:
            skipped.append(skipped_match.group(1))
            current = None
            continue

        if current is None:
            continue

        pick_match = PICK_RE.match(line)
        if pick_match:
            current["pick"] = pick_match.group(1)
            current["pick_probability"] = float(
                pick_match.group(2)
            )
            continue

        red_blue_match = RED_BLUE_RE.match(line)
        if red_blue_match:
            current["red_probability"] = float(
                red_blue_match.group(1)
            )
            current["blue_probability"] = float(
                red_blue_match.group(2)
            )
            continue

        method_match = METHOD_RE.match(line)
        if method_match:
            current["ko"] = float(method_match.group(1))
            current["submission"] = float(
                method_match.group(2)
            )
            current["decision"] = float(
                method_match.group(3)
            )
            continue

        finish_match = FINISH_RE.match(line)
        if finish_match:
            current["finish"] = float(
                finish_match.group(1)
            )
            current["distance"] = float(
                finish_match.group(2)
            )

    complete_fights = [
        fight
        for fight in fights
        if all(
            key in fight
            for key in (
                "pick",
                "pick_probability",
                "ko",
                "submission",
                "decision",
                "finish",
                "distance",
            )
        )
    ]

    return complete_fights, skipped


def print_ranked(
    title: str,
    fights: list[dict],
    key: str,
    *,
    limit: int = 10,
) -> None:
    """Print fights ranked by one probability field."""

    print(f"\n{title}")
    print("-" * len(title))

    ranked = sorted(
        fights,
        key=lambda fight: fight[key],
        reverse=True,
    )[:limit]

    for index, fight in enumerate(ranked, start=1):
        matchup = f"{fight['red']} vs {fight['blue']}"
        print(
            f"{index:>2}. {fight[key]:>5.1f}%  "
            f"{matchup}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an RFS Monte Carlo batch log."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the batch-run text file.",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        raise SystemExit(
            f"Log file not found: {args.log_path}"
        )

    fights, skipped = parse_log(args.log_path)

    if not fights:
        raise SystemExit(
            "No completed fights were parsed from the log."
        )

    ko_values = [fight["ko"] for fight in fights]
    sub_values = [
        fight["submission"] for fight in fights
    ]
    decision_values = [
        fight["decision"] for fight in fights
    ]
    finish_values = [
        fight["finish"] for fight in fights
    ]
    pick_values = [
        fight["pick_probability"] for fight in fights
    ]

    print("=" * 72)
    print("RFS MONTE CARLO BATCH SUMMARY")
    print("=" * 72)
    print(f"Log:               {args.log_path}")
    print(f"Completed fights:  {len(fights)}")
    print(f"Skipped fights:    {len(skipped)}")
    print(f"Total processed:   {len(fights) + len(skipped)}")

    print("\nAVERAGE METHOD PROBABILITIES")
    print("----------------------------")
    print(f"KO/TKO:     {mean(ko_values):6.2f}%")
    print(f"Submission: {mean(sub_values):6.2f}%")
    print(f"Decision:   {mean(decision_values):6.2f}%")
    print(f"Finish:     {mean(finish_values):6.2f}%")
    print(f"Distance:   {100.0 - mean(finish_values):6.2f}%")

    print("\nMEDIAN METHOD PROBABILITIES")
    print("---------------------------")
    print(f"KO/TKO:     {median(ko_values):6.2f}%")
    print(f"Submission: {median(sub_values):6.2f}%")
    print(f"Decision:   {median(decision_values):6.2f}%")
    print(f"Finish:     {median(finish_values):6.2f}%")

    print("\nPICK CONFIDENCE")
    print("---------------")
    print(f"Average:    {mean(pick_values):6.2f}%")
    print(f"Median:     {median(pick_values):6.2f}%")
    print(
        "70%+ picks: "
        f"{sum(value >= 70 for value in pick_values)}"
    )
    print(
        "80%+ picks: "
        f"{sum(value >= 80 for value in pick_values)}"
    )
    print(
        "90%+ picks: "
        f"{sum(value >= 90 for value in pick_values)}"
    )

    print_ranked(
        "HIGHEST SUBMISSION PROBABILITIES",
        fights,
        "submission",
    )
    print_ranked(
        "HIGHEST KO/TKO PROBABILITIES",
        fights,
        "ko",
    )
    print_ranked(
        "HIGHEST TOTAL FINISH PROBABILITIES",
        fights,
        "finish",
    )
    print_ranked(
        "HIGHEST DECISION PROBABILITIES",
        fights,
        "decision",
    )
    print_ranked(
        "STRONGEST WINNER PICKS",
        fights,
        "pick_probability",
    )

    if skipped:
        print("\nSKIPPED FIGHTS")
        print("--------------")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
