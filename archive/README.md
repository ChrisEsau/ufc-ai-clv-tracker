# Archive

This folder stores root-level legacy files and duplicate generated artifacts that were moved out of the repository root during cleanup.

The active platform uses canonical runtime artifact paths under `data/` and shared path constants from `pipeline.common.paths`.

Archived root-level duplicates should not be used as active pipeline inputs unless a future cleanup intentionally restores or migrates them.

Archived workflow files under `archive/.github/workflows/` are retained for historical reference and should not be treated as active GitHub Actions entry points.
