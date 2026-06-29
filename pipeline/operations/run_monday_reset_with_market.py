from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from pipeline.operations import run_monday_reset as base

_ORIGINAL = base._run_function_step


def _patched_run_function_step(
    results: list[base.StepResult],
    step_id: str,
    name: str,
    fn: Callable[[], object],
    outputs: Sequence[Path | str],
) -> object:
    value = _ORIGINAL(results, step_id, name, fn, outputs)
    if step_id == "set_target_event":
        base._run_command_step(
            results,
            "run_market_refresh",
            "Run Market Refresh",
            base._python_module(
                "pipeline.operations.run_market_refresh",
                "--mode",
                "production",
            ),
            [
                "data/market/canonical_market_catalog.parquet",
                "data/market/market_outcomes.parquet",
                "data/audits/ufc_market_match_audit_v2.parquet",
            ],
        )
    return value


def main() -> None:
    base._run_function_step = _patched_run_function_step
    base.main()


if __name__ == "__main__":
    main()
