"""JSON input/output helpers for the simulation shadow pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.simulation.contracts import MatchupSimulationInput, SimulationSummary


class SimulationIOError(RuntimeError):
    """Raised when simulator input or output cannot be read or written."""


def load_matchup(path: str | Path) -> MatchupSimulationInput:
    input_path = Path(path)
    if not input_path.exists():
        raise SimulationIOError(f"Simulation matchup input not found: {input_path}")

    try:
        payload: dict[str, Any] = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimulationIOError(f"Invalid simulation matchup JSON: {input_path}: {exc}") from exc

    return MatchupSimulationInput.from_mapping(payload)


def write_summary(summary: SimulationSummary, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
