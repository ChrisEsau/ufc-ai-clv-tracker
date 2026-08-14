"""Atomic JSON checkpoint manifests for independently replayable groups."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(path)
