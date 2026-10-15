"""Source and code fingerprint helpers."""

import hashlib
from pathlib import Path

import pandas as pd


def source_fingerprint(path: Path, frame: pd.DataFrame) -> str:
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{tuple(frame.columns)}:{tuple(map(str, frame.dtypes))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
