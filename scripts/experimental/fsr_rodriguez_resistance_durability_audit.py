"""Audit Daniel Rodriguez's exact pre-Medic KD-resistance and durability evidence.

Diagnostic only. Replays the leakage-safe reservoir-trait builder through the
2026-08-01 target date and prints the historical observations and component
scores that produced Rodriguez's pre-fight ratings.

No simulator constants or FSR values are changed.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_finish_reservoir_traits_v1 as traits
from scripts.experimental import fsr_static_mc_ko_tko_v2_2020plus_mature_r1_severity_decomposition as severity
from scripts.experimental import fsr_static_mc_v0 as base

BOUT_ID = "68ae50dbf98dc15f"
TARGET_DATE = pd.Timestamp("2026-08-01")


def _target_profile() -> tuple[pd.Series, str]:
    master = severity.modern._load_master(severity.modern.MASTER_PATH)
    candidate = severity.modern._build_outcome_cohort(master)
    cohort, pairs = severity.modern._load_fsr_pairs_for_cohort(severity.modern.FSR_PATH, candidate)
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    if BOUT_ID not in pairs:
        raise KeyError(f"Bout {BOUT_ID} missing from pre-fight FSR pair set")
    red, blue = pairs[BOUT_ID]
    for profile in (red, blue):
        if "rodriguez" in base._display_name(profile).lower():
            return profile, str(profile["fighter_id"])
    raise KeyError("Could not identify Daniel Rodriguez in target bout")


def _prepare_rfs(path: Path) -> pd.DataFrame:
    rfs = pd.read_parquet(path).copy()
    missing = sorted(traits.REQUIRED_COLUMNS - set(rfs.columns))
    if missing:
        raise ValueError(f"RFS missing required columns: {missing}")
    rfs["fight_id"] = rfs["fight_id"].astype(str)
    rfs["fighter_id"] = rfs["fighter_id"].astype(str)
    rfs["date"] = pd.to_datetime(rfs["date"], errors="coerce")
    if rfs["date"].isna().any():
        raise ValueError("Invalid dates in RFS")
    for col in (
        traits.KD_COL,
        traits.SIG_ABS_COL,
        traits.HEAD_ABS_COL,
        traits.GROUND_ABS_COL,
        traits.OPP_CTRL_COL,
        traits.ROUNDS_COL,
        traits.KO_LOSS_COL,
    ):
        rfs[col] = pd.to_numeric(rfs[col], errors="coerce")
    rfs["_damage_exposure"] = rfs.apply(traits._damage_exposure, axis=1)
    return rfs.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _state_and_peer_context(rfs: pd.DataFrame, fighter_id: str):
    states: dict[str, dict[str, float]] = defaultdict(traits._new_state)
    prior_sig_exposures: list[float] = []
    prior_damage_exposures: list[float] = []
    fighter_rows: list[dict[str, object]] = []

    dates = sorted(d for d in rfs["date"].dropna().unique() if pd.Timestamp(d) < TARGET_DATE)
    for fight_date in dates:
        date_rows = rfs[rfs["date"].eq(fight_date)].copy()
        kd_threshold = traits._quantile(prior_sig_exposures, traits.KD_HIGH_EXPOSURE_QUANTILE)
        dur_threshold = traits._quantile(prior_damage_exposures, traits.DURABILITY_HIGH_EXPOSURE_QUANTILE)

        for _, row in date_rows.iterrows():
            fid = str(row["fighter_id"])
            if fid != fighter_id:
                continue
            fighter_rows.append(
                {
                    "date": pd.Timestamp(fight_date).date(),
                    "fight_id": str(row["fight_id"]),
                    "kd_abs": max(0.0, traits._finite(row.get(traits.KD_COL))),
                    "sig_abs": max(0.0, traits._finite(row.get(traits.SIG_ABS_COL))),
                    "head_abs": max(0.0, traits._finite(row.get(traits.HEAD_ABS_COL))),
                    "ground_abs": max(0.0, traits._finite(row.get(traits.GROUND_ABS_COL))),
                    "opp_ctrl_sec": max(0.0, traits._finite(row.get(traits.OPP_CTRL_COL))),
                    "rounds": max(1.0, traits._finite(row.get(traits.ROUNDS_COL), 1.0)),
                    "ko_loss": int(traits._finite(row.get(traits.KO_LOSS_COL)) >= 0.5),
                    "damage_exposure": float(row["_damage_exposure"]),
                    "kd_high_threshold": kd_threshold,
                    "dur_high_threshold": dur_threshold,
                }
            )

        # Same update logic as production shadow builder.
        for _, row in date_rows.iterrows():
            fid = str(row["fighter_id"])
            state = states[fid]
            kd_abs = max(0.0, traits._finite(row.get(traits.KD_COL)))
            sig_abs = max(0.0, traits._finite(row.get(traits.SIG_ABS_COL)))
            exposure = max(0.0, traits._finite(row.get("_damage_exposure")))
            ko_loss = 1.0 if traits._finite(row.get(traits.KO_LOSS_COL)) >= 0.5 else 0.0

            state["fights"] += 1.0
            state["kd_absorbed"] += kd_abs
            state["sig_absorbed"] += sig_abs
            if kd_abs <= 0:
                state["kd_free_fights"] += 1.0

            if kd_threshold is not None and sig_abs >= kd_threshold:
                state["kd_high_exposure_fights"] += 1.0
                if kd_abs <= 0:
                    state["kd_free_high_exposure"] += 1.0

            if dur_threshold is not None and exposure >= dur_threshold:
                state["dur_high_exposure_fights"] += 1.0
                state["dur_high_exposure_sum"] += exposure
                if ko_loss < 0.5:
                    state["dur_high_survivals"] += 1.0
                    state["dur_high_survived_exposure_sum"] += exposure

            if ko_loss < 0.5:
                state["survived_fights"] += 1.0
                state["survived_exposure_sum"] += exposure
            else:
                state["ko_losses"] += 1.0

            prior_sig_exposures.append(sig_abs)
            prior_damage_exposures.append(exposure)

    # Peer populations strictly before target date.
    peer_avoidance: list[float] = []
    peer_free: list[float] = []
    peer_high: list[float] = []
    for state in states.values():
        if state["fights"] <= 0:
            continue
        avoidance, free_rate, high_rate = traits._kd_raw_components(state)
        peer_avoidance.append(avoidance)
        peer_free.append(free_rate)
        if high_rate is not None:
            peer_high.append(high_rate)

    return (
        states[fighter_id],
        pd.DataFrame(fighter_rows),
        peer_avoidance,
        peer_free,
        peer_high,
        traits._quantile(prior_damage_exposures, traits.DURABILITY_HIGH_EXPOSURE_QUANTILE),
    )


def _pct(value: float, population: list[float]) -> float:
    p = traits._percentile(value, population)
    return float("nan") if p is None else float(p)


def main() -> None:
    profile, fighter_id = _target_profile()
    name = base._display_name(profile)
    rfs = _prepare_rfs(traits.RFS_PATH)
    state, history, peer_avoidance, peer_free, peer_high, dur_threshold = _state_and_peer_context(rfs, fighter_id)

    avoidance, free_rate, high_rate = traits._kd_raw_components(state)
    avoid_pct = _pct(avoidance, peer_avoidance)
    free_pct = _pct(free_rate, peer_free)
    high_pct = _pct(high_rate, peer_high) if high_rate is not None else float("nan")
    kd_parts = [x for x in (avoid_pct, free_pct, high_pct) if np.isfinite(x)]
    kd_score = float(np.mean(kd_parts)) if kd_parts else 0.5

    dur_parts: list[tuple[str, float, float]] = []
    if state["dur_high_exposure_fights"] > 0:
        high_survival = state["dur_high_survivals"] / state["dur_high_exposure_fights"]
        dur_parts.append(("high-exposure survival", 0.35, high_survival))
        if state["dur_high_exposure_sum"] > 0:
            sev_survival = state["dur_high_survived_exposure_sum"] / state["dur_high_exposure_sum"]
            dur_parts.append(("severity-weighted survival", 0.30, sev_survival))
    if dur_threshold is not None and dur_threshold > 0 and state["survived_fights"] > 0:
        avg_survived = state["survived_exposure_sum"] / state["survived_fights"]
        punishment = float(np.clip(avg_survived / dur_threshold, 0.0, 2.0) / 2.0)
        dur_parts.append(("punishment tolerance", 0.20, punishment))
    if state["fights"] > 0:
        overall_survival = 1.0 - state["ko_losses"] / state["fights"]
        dur_parts.append(("overall KO survival", 0.15, overall_survival))

    dur_weight = sum(w for _, w, _ in dur_parts)
    dur_score = sum(w * v for _, w, v in dur_parts) / dur_weight if dur_weight else 0.5

    print("=" * 110)
    print(f"{name.upper()} — PRE-MEDIC RESISTANCE / DURABILITY AUDIT")
    print("=" * 110)
    print(f"fighter_id: {fighter_id}")
    print(f"target bout: {BOUT_ID} | target date: {TARGET_DATE.date()}")
    print(f"prior UFC fights in RFS state: {int(state['fights'])}")
    print(f"stored pre-fight knockdown_resistance: {float(profile['knockdown_resistance']):.3f}")
    print(f"stored pre-fight damage_durability: {float(profile['damage_durability']):.3f}")

    print("\nPRIOR-FIGHT EVIDENCE")
    if history.empty:
        print("No prior history rows found.")
    else:
        view = history.copy()
        view["kd_high"] = view.apply(
            lambda r: bool(pd.notna(r["kd_high_threshold"]) and r["sig_abs"] >= r["kd_high_threshold"]), axis=1
        )
        view["dur_high"] = view.apply(
            lambda r: bool(pd.notna(r["dur_high_threshold"]) and r["damage_exposure"] >= r["dur_high_threshold"]), axis=1
        )
        cols = [
            "date", "fight_id", "kd_abs", "sig_abs", "head_abs", "ground_abs", "opp_ctrl_sec",
            "rounds", "ko_loss", "damage_exposure", "kd_high", "dur_high"
        ]
        print(view[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nKNOCKDOWN-RESISTANCE COMPONENTS")
    print(f"career KD absorbed: {state['kd_absorbed']:.0f}")
    print(f"career sig strikes absorbed: {state['sig_absorbed']:.0f}")
    print(f"KD-free fights: {state['kd_free_fights']:.0f}/{state['fights']:.0f} = {free_rate:.3f}")
    print(
        f"KD-free high-exposure fights: {state['kd_free_high_exposure']:.0f}/"
        f"{state['kd_high_exposure_fights']:.0f} = "
        f"{(high_rate if high_rate is not None else float('nan')):.3f}"
    )
    print(f"smoothed KD avoidance raw: {avoidance:.6f}; peer percentile={avoid_pct:.3f}")
    print(f"KD-free-rate peer percentile: {free_pct:.3f}")
    print(f"high-exposure KD-free peer percentile: {high_pct:.3f}")
    print(f"combined KD evidence score: {kd_score * 100:.3f}")
    print(
        f"recomputed rating: {traits._rating_from_score(kd_score, int(state['fights'])):.3f}"
    )

    print("\nDAMAGE-DURABILITY COMPONENTS")
    print(f"KO losses: {state['ko_losses']:.0f}/{state['fights']:.0f}")
    print(
        f"high-exposure survivals: {state['dur_high_survivals']:.0f}/"
        f"{state['dur_high_exposure_fights']:.0f}"
    )
    print(f"target-date durability high-exposure threshold: {dur_threshold}")
    for label, weight, value in dur_parts:
        print(f"{label:28s} weight={weight:.2f} value={value:.4f} contribution={weight * value:.4f}")
    print(f"combined durability evidence score: {dur_score * 100:.3f}")
    print(
        f"recomputed rating: {traits._rating_from_score(dur_score, int(state['fights'])):.3f}"
    )

    print("\nNOTE")
    print("This is a reconstruction of the committed leakage-safe builder using strictly pre-2026-08-01 evidence.")
    print("No simulator constants or FSR values were changed.")


if __name__ == "__main__":
    main()
