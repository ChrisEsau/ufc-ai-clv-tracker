"""Shared UFC fight-context normalization helpers.

These helpers keep historical ingestion and live prediction aligned for fight-level
context fields used by feature views and prop models.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_string(value: Any) -> str | None:
    """Return a normalized string or None for blank/null placeholder values."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "nat", "<na>"}:
        return None

    return value


def clean_division(value: Any) -> str | None:
    """Clean UFC division/weight-class text for master/live consistency."""

    value = clean_string(value)

    if value is None:
        return None

    value = re.sub(r"\btitle\s+bout\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\btitle\b", "", value, flags=re.IGNORECASE)
    value = " ".join(value.replace("\n", " ").split())

    return value.lower() if value else None


def title_fight_flag(value: Any) -> int:
    """Infer title-fight flag from UFCStats division/weight-class text."""

    value = clean_string(value)

    if value is None:
        return 0

    return int("title" in value.lower())


def total_rounds_from_time_format(value: Any, title_fight: Any = 0) -> int:
    """Infer scheduled rounds from UFCStats time-format text.

    Historical completed-fight pages may expose strings such as ``5 Rnd`` or a
    parenthesized round format. Upcoming fight-card rows often do not expose a
    time-format value, so this intentionally falls back the same way the master
    mapper historically did: title fights are 5 rounds, non-title fights are 3.
    """

    value = clean_string(value)

    if value is not None:
        match = re.search(r"(\d+)\s*rnd", value, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))

        rounds = re.search(r"\(([^)]*)\)", value)

        if rounds:
            round_count = len([part for part in rounds.group(1).split("-") if part])

            if round_count > 0:
                return round_count

    return 5 if int(title_fight or 0) == 1 else 3
