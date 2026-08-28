"""Single approved fight-agnostic locked Brain MC entry point.

The validated simulation core is preserved byte-for-byte in
pipeline.research.locked_brain_mc_legacy. This facade adds one reusable historical
bundle. Normal runs REQUIRE and consume the bundle; --build-bundle materializes
historical databases once and does not run a fight simulation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.research import locked_brain_mc_legacy as _core
from pipeline.research.locked_brain_bundle import DEFAULT_BUNDLE_DIR, build_bundle, install_bundle_runtime

APPROVED_ENTRY_POINT = _core.APPROVED_ENTRY_POINT
RUN_POLICY = _core.RUN_POLICY
LOCKED_PATHS = _core.LOCKED_PATHS


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the single approved bundle-backed locked Brain MC harness")
    parser.add_argument("--fight-id", required=False)
    parser.add_argument("--paths", type=int, default=LOCKED_PATHS)
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--build-bundle", action="store_true", help="Build reusable historical databases once, then exit")
    args = parser.parse_args(argv)
    if args.build_bundle:
        return args
    args.fight_id = str(args.fight_id or "").strip()
    if not args.fight_id:
        parser.error("--fight-id is required unless --build-bundle is set")
    if args.paths < 1:
        parser.error("--paths must be >= 1")
    return args


def main(*, fight_id: str, paths: int = LOCKED_PATHS, bundle_dir: Path | str = DEFAULT_BUNDLE_DIR) -> None:
    fight_id = str(fight_id).strip()
    if not fight_id:
        raise ValueError("fight_id must be non-empty")
    if isinstance(paths, bool) or not isinstance(paths, int) or paths < 1:
        raise ValueError("paths must be an integer >= 1")
    manifest = install_bundle_runtime(_core, fight_id, bundle_dir)
    print("LOCKED_BRAIN_BUNDLE_LOADED")
    print(f"schema_version={manifest['schema_version']} bundle_dir={Path(bundle_dir)} fight_id={fight_id}")
    _core.main(fight_id=fight_id, paths=paths)


if __name__ == "__main__":
    args = _parse_args()
    if args.build_bundle:
        manifest = build_bundle(args.bundle_dir)
        print("LOCKED_BRAIN_BUNDLE_BUILT")
        print(f"schema_version={manifest['schema_version']} bundle_dir={Path(args.bundle_dir)}")
    else:
        main(fight_id=args.fight_id, paths=args.paths, bundle_dir=args.bundle_dir)
