"""Single approved fight-agnostic entry point for locked Brain MC research.

ALL Brain MC research runs MUST execute through this module unless the user
explicitly approves an exception. Do not create or execute one-off runners,
ad-hoc fight scripts, alternate harnesses, or workflow monkeypatch runners.

This module owns run-time targeting and path-count selection while delegating the
frozen mechanics implementation to the locked implementation module. It sets the
fight id consistently across every historical research seam before execution.

CLI examples:

    python -m pipeline.research.locked_brain_mc --fight-id 419fff06f338f5c6 --paths 500
    python -m pipeline.research.locked_brain_mc --fight-id 419fff06f338f5c6 --paths 1

When --paths 1 is used, the locked implementation automatically emits the full
one-path event report with Brain probabilities and mechanic probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.research import locked_brain_mc_allen_shahbazyan as locked_impl

DEFAULT_PATHS = 500
OUTROOT = Path("data/research/locked_brain_mc")
APPROVED_ENTRY_POINT = "pipeline.research.locked_brain_mc"
RUN_POLICY = (
    "ALL Brain MC research runs must execute through this harness; no one-off "
    "or alternate runners unless explicitly approved by the user."
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the single approved fight-agnostic locked Brain MC harness"
    )
    parser.add_argument(
        "--fight-id",
        required=True,
        help="target fight_id from the repository master/prefight data",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=DEFAULT_PATHS,
        help=(
            f"number of matched-seed paths (default: {DEFAULT_PATHS}); "
            "--paths 1 automatically emits event_report.json/csv"
        ),
    )
    args = parser.parse_args(argv)
    args.fight_id = str(args.fight_id).strip()
    if not args.fight_id:
        parser.error("--fight-id must be non-empty")
    if args.paths < 1:
        parser.error("--paths must be >= 1")
    return args


def _set_fight_id(fight_id: str) -> None:
    """Set the target through every shared seam used by the locked stack."""
    locked_impl.validated_kd.base_trace.FIGHT_ID = fight_id
    locked_impl.time_ko.base_trace.FIGHT_ID = fight_id
    locked_impl.scored.base_trace.FIGHT_ID = fight_id
    locked_impl.timing.base_trace.FIGHT_ID = fight_id
    locked_impl.timing.target.FIGHT_ID = fight_id
    locked_impl.scored.pressure_mod.FIGHT_ID = fight_id


def _finalize_manifest(fight_id: str, paths: int, outdir: Path) -> None:
    manifest_path = outdir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"locked run did not write manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "entry_point": APPROVED_ENTRY_POINT,
            "fight_id": fight_id,
            "paths": int(paths),
            "fight_id_source": "CLI --fight-id",
            "paths_source": "CLI --paths or generic locked default",
            "run_policy": RUN_POLICY,
            "single_approved_harness": True,
            "legacy_implementation_module": (
                "pipeline.research.locked_brain_mc_allen_shahbazyan"
            ),
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(*, fight_id: str, paths: int = DEFAULT_PATHS) -> None:
    fight_id = str(fight_id).strip()
    if not fight_id:
        raise ValueError("fight_id must be non-empty")
    if isinstance(paths, bool) or not isinstance(paths, int) or paths < 1:
        raise ValueError("paths must be an integer >= 1")

    _set_fight_id(fight_id)
    outdir = OUTROOT / fight_id
    outdir.mkdir(parents=True, exist_ok=True)

    original_outdir = locked_impl.OUTDIR
    try:
        locked_impl.OUTDIR = outdir
        print(
            "LOCKED_BRAIN_MC_GENERIC_TARGET",
            {
                "entry_point": APPROVED_ENTRY_POINT,
                "fight_id": fight_id,
                "paths": paths,
                "single_approved_harness": True,
                "production_changed": False,
            },
        )
        locked_impl.main(paths=paths)
        _finalize_manifest(fight_id, paths, outdir)
    finally:
        locked_impl.OUTDIR = original_outdir


if __name__ == "__main__":
    args = _parse_args()
    main(fight_id=args.fight_id, paths=args.paths)
