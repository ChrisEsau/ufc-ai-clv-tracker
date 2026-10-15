"""Validated FSR V3 escape/retention replay.

The active-trait audit found that the existing escape semantics remain useful,
but the V2 five-entry prior is too weak.  V3 therefore keeps the same
leakage-safe cumulative duration/entry estimator with an eight-entry prior.
Epistemic sampling was rejected (c=0), so this history is mean-only.
"""

from __future__ import annotations

import pandas as pd

from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.traits.registry import GROUPS
from pipeline.fsr_v3.active_config import ActiveTraitConfig


def replay_escape(
    paired_rounds: pd.DataFrame,
    config: ActiveTraitConfig | None = None,
) -> pd.DataFrame:
    config = config or ActiveTraitConfig()
    fights = aggregate_fights(paired_rounds)
    v2_config = FSRV2Config(escape_prior_entries=float(config.escape_prior_entries))
    history = ReplayEngine(v2_config).replay(GROUPS["escape_effectiveness"], fights).history.copy()

    history["pre_posterior_sd"] = 0.0
    history["post_posterior_sd"] = 0.0
    history["variance_multiplier"] = float(config.escape_variance_multiplier)
    history["sampling_enabled"] = False
    history["posterior_family"] = "deterministic_mean"
    history["validated_prior_entries"] = float(config.escape_prior_entries)
    return history
