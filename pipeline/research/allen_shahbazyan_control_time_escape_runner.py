"""Compatibility runner for the control-time escape diagnostic."""
from __future__ import annotations

from pipeline.research import allen_shahbazyan_one_path_brain_trace_v1 as mod

_orig_col = mod._col


def _col_with_round_stats_aliases(df, *names):
    lower = {str(c).lower(): c for c in df.columns}
    if any(name.lower() in {"control_seconds", "ctrl_seconds", "ctrl", "control"} for name in names):
        for alias in ("ctrl_sec", "control_seconds", "ctrl_seconds", "ctrl", "control"):
            if alias in lower:
                return lower[alias]
    return _orig_col(df, *names)


mod._col = _col_with_round_stats_aliases

if __name__ == "__main__":
    mod.main()
