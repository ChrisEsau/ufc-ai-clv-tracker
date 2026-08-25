"""Order-independent matched path seeds."""

from __future__ import annotations
import hashlib


def derive_path_seed(seed_set_version: str, bout_id: str, path_id: int) -> int:
    if not seed_set_version or not str(bout_id) or path_id < 0:
        raise ValueError("seed set, bout ID, and non-negative path ID are required")
    material = f"{seed_set_version}|{bout_id}|{path_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
