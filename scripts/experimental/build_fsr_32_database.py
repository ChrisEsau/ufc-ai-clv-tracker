"""Build the shadow FSR-32 simulator-facing contract.

FSR-32 starts from FSR-28, appends three explicit fighter stamina parameters,
and replaces the inherited ``striking_power`` trait with the approved fresh
striking-power model.

Fresh striking power
--------------------
Positive evidence uses the frozen V8 hierarchy:

- Round-1 KO/TKO: primary evidence
- later-round KO/TKO: discounted by finish round
- Round-1 KD: small secondary evidence

Before a fighter has any positive power event, repeated landed Round-1
significant-strike opportunity can move the rating below the neutral 50 prior.
The selected V9 low-end curve is:

    opportunity_i = 1 - exp(-R1_SIG_LANDED_i / 20)
    O = sum(opportunity_i)
    power = 50 - 15 * (1 - exp(-O / 6))

Once positive power has been demonstrated, the V8 positive curve takes over.
The stored fresh-power state is non-degrading: later quiet fights cannot erase a
previous demonstration. Age effects remain external to this stored fresh trait.

All power values written to a prefight snapshot are leakage-safe: evidence from
the current fight is applied only after that fight, so it can affect the next
snapshot but never the current one.

Stamina contract
----------------
FSR-32 appends:

- stamina_capacity
- stamina_depletion_resistance
- stamina_performance_resilience

Recovery is intentionally NOT fighter-specific. The historical recovery proxy
was functionally non-informative, so recovery remains simulator-global physics.

Shadow/research only. No production schema is modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_striking_power_evidence_v2_sweep as power_v2
from scripts.experimental import fsr_striking_power_evidence_v4_aggregation_sweep as power_v4
from scripts.experimental import fsr_striking_power_evidence_v8_hierarchical_ko_kd_sweep as power_v8


FSR_28_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_28_shadow/"
    "fsr_28_prefight_snapshots.parquet"
)
OUTPUT_DIR = Path("data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow")
OUTPUT_PATH = OUTPUT_DIR / "fsr_32_prefight_snapshots.parquet"

STAMINA_CAPACITY = "stamina_capacity"
STAMINA_DEPLETION_RESISTANCE = "stamina_depletion_resistance"
STAMINA_PERFORMANCE_RESILIENCE = "stamina_performance_resilience"

# Legacy symbol only so frozen V3/V3.2 code and tests remain import-compatible.
# It is intentionally excluded from STAMINA_COLUMNS and is not emitted by the
# current FSR-32 builder.
STAMINA_RECOVERY_ABILITY = "stamina_recovery_ability"

DEFAULT_STAMINA_CAPACITY = 100.0

STAMINA_SOURCE_COLUMNS = {
    STAMINA_DEPLETION_RESISTANCE: "fatigue_accumulation_resistance",
    STAMINA_PERFORMANCE_RESILIENCE: "fatigue_performance_resilience",
}

STAMINA_COLUMNS = (
    STAMINA_CAPACITY,
    STAMINA_DEPLETION_RESISTANCE,
    STAMINA_PERFORMANCE_RESILIENCE,
)

REMOVED_RECOVERY_COLUMNS = (
    "recovery_ability",
    STAMINA_RECOVERY_ABILITY,
)

STRIKING_POWER = "striking_power"
POWER_NEUTRAL = 50.0
POWER_MIN = 35.0
POWER_MAX = 90.0
LOW_END_SIG_TAU = 20.0
LOW_END_MAX_PENALTY = 15.0
LOW_END_OPPORTUNITY_TAU = 6.0


def _positive_power_rating(
    *,
    prior_fights: int,
    prior_power_events: int,
    peak_single_fight_evidence: float,
) -> float:
    """Map cumulative positive V8 state onto the established display scale."""
    fights = max(float(prior_fights), 1.0)
    events = max(float(prior_power_events), 0.0)
    peak = max(float(peak_single_fight_evidence), 0.0)

    event_rate = np.clip(events / fights, 0.0, 1.0)
    repeatability = 1.0 - np.exp(-events / power_v4.REPEATABILITY_TAU)
    confidence = 1.0 - np.exp(-fights / power_v4.CONFIDENCE_TAU)
    adjusted_frequency = np.sqrt(event_rate) * confidence
    compressed_peak = 1.0 - np.exp(-peak / power_v4.PEAK_TAU)

    score = (
        power_v4.REPEATABILITY_WEIGHT * repeatability
        + power_v4.FREQUENCY_WEIGHT * adjusted_frequency
        + power_v4.PEAK_WEIGHT * compressed_peak
    )
    rating = POWER_NEUTRAL + 40.0 * (
        1.0 - np.exp(-score / power_v4.DISPLAY_RATING_SATURATION)
    )
    return float(np.clip(rating, POWER_NEUTRAL, POWER_MAX))


def _low_end_power_rating(opportunity_score: float) -> float:
    """Selected V9 L=15, tau=6 curve for fighters with no positive event."""
    opportunity = max(float(opportunity_score), 0.0)
    rating = POWER_NEUTRAL - LOW_END_MAX_PENALTY * (
        1.0 - np.exp(-opportunity / LOW_END_OPPORTUNITY_TAU)
    )
    return float(np.clip(rating, POWER_MIN, POWER_NEUTRAL))


def _fight_order_table(master: pd.DataFrame) -> pd.DataFrame:
    required = {"fight_id", "date"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise RuntimeError(f"Master missing power chronology columns: {missing}")

    order = master[["fight_id", "date"]].copy().reset_index(drop=False)
    order = order.rename(columns={"index": "_source_order"})
    order["fight_id"] = order["fight_id"].astype(str)
    order["date"] = pd.to_datetime(order["date"], errors="coerce")
    if order["date"].isna().any():
        raise RuntimeError("Master contains invalid dates required for power chronology")

    # One master row per fight is expected; tolerate duplicate copies by keeping
    # the first source occurrence before chronological ordering.
    order = order.drop_duplicates("fight_id", keep="first")
    order = order.sort_values(["date", "_source_order", "fight_id"]).reset_index(drop=True)
    order["fight_order"] = np.arange(len(order), dtype=np.int64)
    return order[["fight_id", "fight_order"]]


def _power_evidence_by_fighter_fight(
    master: pd.DataFrame,
    rounds: pd.DataFrame,
    fight_order: pd.DataFrame,
) -> pd.DataFrame:
    """Return one leakage-safe evidence row per fighter-fight with R1 stats."""
    _, scored = power_v8.build_v8_rankings(master, rounds)

    required = {
        "fighter_id",
        "fight_id",
        "sig_str_landed",
        "fight_power_evidence_v8",
        "power_event",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise RuntimeError(f"V8 scored evidence missing required columns: {missing}")

    detail = scored[list(required)].copy()
    detail["fighter_id"] = detail["fighter_id"].astype(str)
    detail["fight_id"] = detail["fight_id"].astype(str)
    detail["sig_str_landed"] = pd.to_numeric(
        detail["sig_str_landed"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    detail["fight_power_evidence_v8"] = pd.to_numeric(
        detail["fight_power_evidence_v8"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    detail["power_event"] = detail["power_event"].fillna(False).astype(bool)

    # Historical fighter-name variants can create duplicate fighter_id groups in
    # research summaries. The final power contract is keyed only by fighter_id.
    detail = detail.groupby(["fighter_id", "fight_id"], as_index=False).agg(
        sig_str_landed=("sig_str_landed", "max"),
        fight_power_evidence_v8=("fight_power_evidence_v8", "max"),
        power_event=("power_event", "max"),
    )
    detail["opportunity"] = 1.0 - np.exp(
        -detail["sig_str_landed"] / LOW_END_SIG_TAU
    )

    detail = detail.merge(
        fight_order,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )
    if detail["fight_order"].isna().any():
        raise RuntimeError("Power evidence contains fights missing from master chronology")
    detail["fight_order"] = detail["fight_order"].astype(np.int64)
    return detail.sort_values(["fighter_id", "fight_order", "fight_id"]).reset_index(drop=True)


def build_prefight_striking_power(
    fsr_rows: pd.DataFrame,
    evidence: pd.DataFrame,
    fight_order: pd.DataFrame,
) -> pd.DataFrame:
    """Build one leakage-safe fresh-power value for every FSR snapshot.

    Evidence from fight N is applied after fight N. Therefore the prefight row
    for fight N can only see evidence with ``fight_order < current fight_order``.
    """
    keys = ["fight_id", "fighter_id"]
    missing = [c for c in keys if c not in fsr_rows.columns]
    if missing:
        raise RuntimeError(f"FSR rows missing power keys: {missing}")
    if fsr_rows.duplicated(keys).any():
        raise RuntimeError("FSR rows violate fighter-fight grain")

    snapshots = fsr_rows[keys].copy()
    snapshots["fight_id"] = snapshots["fight_id"].astype(str)
    snapshots["fighter_id"] = snapshots["fighter_id"].astype(str)
    snapshots = snapshots.merge(
        fight_order,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )
    if snapshots["fight_order"].isna().any():
        raise RuntimeError("FSR snapshots contain fights missing from master chronology")
    snapshots["fight_order"] = snapshots["fight_order"].astype(np.int64)

    result = np.full(len(snapshots), POWER_NEUTRAL, dtype=float)
    snapshot_positions = snapshots.groupby("fighter_id", sort=False).indices

    evidence_groups = {
        fighter_id: group.sort_values("fight_order").reset_index(drop=True)
        for fighter_id, group in evidence.groupby("fighter_id", sort=False)
    }

    for fighter_id, positions in snapshot_positions.items():
        hist = evidence_groups.get(str(fighter_id))
        if hist is None or hist.empty:
            continue

        orders = hist["fight_order"].to_numpy(dtype=np.int64)
        opportunities = hist["opportunity"].to_numpy(dtype=float)
        event_flags = hist["power_event"].to_numpy(dtype=bool)
        event_evidence = hist["fight_power_evidence_v8"].to_numpy(dtype=float)

        # State after each historical fight. Low-end evidence can reduce an
        # undemonstrated fighter below 50. After the first positive event, fresh
        # power is non-degrading and can only hold or rise.
        state_after = np.empty(len(hist), dtype=float)
        cumulative_opportunity = 0.0
        cumulative_events = 0
        peak_evidence = 0.0
        running_positive_rating = POWER_NEUTRAL

        for i in range(len(hist)):
            cumulative_opportunity += float(opportunities[i])
            if bool(event_flags[i]):
                cumulative_events += 1
                peak_evidence = max(peak_evidence, float(event_evidence[i]))

            if cumulative_events == 0:
                state = _low_end_power_rating(cumulative_opportunity)
            else:
                candidate = _positive_power_rating(
                    prior_fights=i + 1,
                    prior_power_events=cumulative_events,
                    peak_single_fight_evidence=peak_evidence,
                )
                running_positive_rating = max(running_positive_rating, candidate)
                state = running_positive_rating
            state_after[i] = state

        current_orders = snapshots.loc[positions, "fight_order"].to_numpy(dtype=np.int64)
        prior_counts = np.searchsorted(orders, current_orders, side="left")
        fighter_values = np.full(len(positions), POWER_NEUTRAL, dtype=float)
        has_prior = prior_counts > 0
        fighter_values[has_prior] = state_after[prior_counts[has_prior] - 1]
        result[np.asarray(positions, dtype=int)] = fighter_values

    out = snapshots[keys].copy()
    out[STRIKING_POWER] = np.clip(result, POWER_MIN, POWER_MAX)
    return out


def build_fsr_32_database(
    fsr_28: pd.DataFrame,
    *,
    master: pd.DataFrame | None = None,
    rounds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    keys = ["fight_id", "fighter_id"]
    base = fsr_28.copy()

    missing = [
        column
        for column in [*keys, *STAMINA_SOURCE_COLUMNS.values()]
        if column not in base.columns
    ]
    if missing:
        raise RuntimeError(f"FSR-28 missing required stamina-source columns: {missing}")
    if base.duplicated(keys).any():
        raise RuntimeError("FSR-28 violates fighter-fight grain")

    base = base.drop(columns=list(REMOVED_RECOVERY_COLUMNS), errors="ignore")

    base[STAMINA_CAPACITY] = float(DEFAULT_STAMINA_CAPACITY)
    for target, source in STAMINA_SOURCE_COLUMNS.items():
        base[target] = pd.to_numeric(base[source], errors="coerce")

    # Preserve the one-argument function contract for frozen tests/importers.
    # The actual artifact build in main() always provides source data and thus
    # always emits the new fresh striking-power trait.
    if (master is None) != (rounds is None):
        raise RuntimeError("master and rounds must be provided together")
    if master is not None and rounds is not None:
        order = _fight_order_table(master)
        evidence = _power_evidence_by_fighter_fight(master, rounds, order)
        power = build_prefight_striking_power(base, evidence, order)
        base = base.drop(columns=[STRIKING_POWER], errors="ignore").merge(
            power,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if base[STRIKING_POWER].isna().any():
            raise RuntimeError("FSR-32 fresh striking_power failed to populate all rows")

    numeric = base[list(STAMINA_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("FSR-32 contains missing or non-finite stamina parameters")
    if (numeric[STAMINA_CAPACITY] <= 0.0).any():
        raise RuntimeError("FSR-32 stamina_capacity must be positive")

    for column in (
        STAMINA_DEPLETION_RESISTANCE,
        STAMINA_PERFORMANCE_RESILIENCE,
    ):
        if ((numeric[column] < 10.0) | (numeric[column] > 90.0)).any():
            raise RuntimeError(f"FSR-32 {column} is outside the established 10-90 FSR range")

    for target, source in STAMINA_SOURCE_COLUMNS.items():
        if not np.allclose(
            pd.to_numeric(base[target], errors="coerce"),
            pd.to_numeric(base[source], errors="coerce"),
            equal_nan=False,
        ):
            raise RuntimeError(f"FSR-32 alias mismatch: {target} != {source}")

    if STRIKING_POWER in base.columns:
        power_numeric = pd.to_numeric(base[STRIKING_POWER], errors="coerce")
        if power_numeric.isna().any() or not np.isfinite(power_numeric.to_numpy()).all():
            raise RuntimeError("FSR-32 striking_power contains missing/non-finite values")
        if ((power_numeric < POWER_MIN) | (power_numeric > POWER_MAX)).any():
            raise RuntimeError("FSR-32 striking_power is outside the 35-90 fresh-power range")

    if any(column in base.columns for column in REMOVED_RECOVERY_COLUMNS):
        raise RuntimeError("FSR-32 recovery columns were not removed")

    sort_cols = [c for c in ("date", "fight_id", "fighter_id") if c in base.columns]
    return base.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    if not FSR_28_PATH.exists():
        raise RuntimeError(f"FSR-28 artifact not found: {FSR_28_PATH}")
    if not power_v2.MASTER_PATH.exists():
        raise RuntimeError(f"Master artifact not found: {power_v2.MASTER_PATH}")
    if not power_v2.ROUND_STATS_PATH.exists():
        raise RuntimeError(f"Round-stats artifact not found: {power_v2.ROUND_STATS_PATH}")

    print(f"[FSR-32] loading FSR-28: {FSR_28_PATH}", flush=True)
    fsr_28 = pd.read_parquet(FSR_28_PATH)
    master = pd.read_parquet(power_v2.MASTER_PATH)
    rounds = pd.read_parquet(power_v2.ROUND_STATS_PATH)
    print(f"[FSR-32] FSR-28 rows: {len(fsr_28):,}", flush=True)

    database = build_fsr_32_database(fsr_28, master=master, rounds=rounds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database.to_parquet(OUTPUT_PATH, index=False)

    power = pd.to_numeric(database[STRIKING_POWER], errors="coerce")
    print(f"Wrote {len(database):,} FSR-32 pre-fight rows to {OUTPUT_PATH}", flush=True)
    print("FSR-32 fresh striking-power contract:", flush=True)
    print(f"  range: {power.min():.3f} .. {power.max():.3f}", flush=True)
    print(f"  mean / median: {power.mean():.3f} / {power.median():.3f}", flush=True)
    print(f"  below 50: {(power < 50.0).sum():,}", flush=True)
    print(f"  exactly 50: {np.isclose(power, 50.0).sum():,}", flush=True)
    print(f"  above 50: {(power > 50.0).sum():,}", flush=True)
    print("FSR-32 stamina contract:", flush=True)
    for column in STAMINA_COLUMNS:
        print(f"  - {column}", flush=True)
    print("Removed recovery columns:", flush=True)
    for column in REMOVED_RECOVERY_COLUMNS:
        print(f"  - {column}", flush=True)
    print("Between-round recovery is simulator-global physics.", flush=True)


if __name__ == "__main__":
    main()
