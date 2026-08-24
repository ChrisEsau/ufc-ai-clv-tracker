from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    FighterPathTraits,
    derive_runtime_inputs,
    load_latest_profiles,
    load_prefight_snapshots,
)
from .policy import Action, Capability, FightState, action_probabilities


def pct(series: pd.Series, value: float) -> float:
    arr = np.asarray(series, dtype=float)
    return float(np.mean(arr <= float(value)))


def traits_from_row(row: pd.Series) -> FighterPathTraits:
    vals = {}
    for k, v in row.items():
        if k == "fighter_id":
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            vals[k] = x
    return FighterPathTraits(str(row["fighter_id"]), vals, False)


def median_row(frame: pd.DataFrame) -> pd.Series:
    row = {"fighter_id": "POP_MEDIAN"}
    for c in frame.columns:
        if c == "fighter_id":
            continue
        if pd.api.types.is_numeric_dtype(frame[c]):
            row[c] = float(frame[c].median())
    return pd.Series(row)


def runtime_population(latest: pd.DataFrame) -> pd.DataFrame:
    med = traits_from_row(median_row(latest))
    rows = []
    for _, row in latest.iterrows():
        try:
            rt = derive_runtime_inputs(traits_from_row(row), med)
        except Exception:
            continue
        rows.append({
            "fighter_id": str(row["fighter_id"]),
            "standing_rate": rt.standing_rate_15m,
            "standing_acc": rt.standing_accuracy,
            "td_rate": rt.takedown_rate_15m,
            "td_comp": rt.takedown_completion,
            "ground_rate": rt.ground_slope_rate_15m_own_control,
            "ground_acc": rt.ground_accuracy,
        })
    out = pd.DataFrame(rows)
    if len(out) < 50:
        raise RuntimeError(f"too few valid FSR V3 profiles for percentile audit: {len(out)}")
    return out


def capability(attacker: pd.Series, defender: pd.Series, pop: pd.DataFrame) -> tuple[Capability, dict[str, float]]:
    rt = derive_runtime_inputs(traits_from_row(attacker), traits_from_row(defender))
    sr = pct(pop["standing_rate"], rt.standing_rate_15m)
    sa = pct(pop["standing_acc"], rt.standing_accuracy)
    tr = pct(pop["td_rate"], rt.takedown_rate_15m)
    tc = pct(pop["td_comp"], rt.takedown_completion)
    gr = pct(pop["ground_rate"], rt.ground_slope_rate_15m_own_control)
    ga = pct(pop["ground_acc"], rt.ground_accuracy)
    cap = Capability(
        standing=(sr + sa) / 2.0,
        counter=sa,
        pressure=sr,
        clinch=.35,  # no validated V3 clinch capability; hold neutral in this audit
        takedown=(tr + tc) / 2.0,
        ground_top=(gr + ga) / 2.0,
        submission=.30,
        escape=.40,
        reversal=.30,
    )
    raw = {
        "stand_rate15": rt.standing_rate_15m,
        "stand_acc": rt.standing_accuracy,
        "td_rate15": rt.takedown_rate_15m,
        "td_comp": rt.takedown_completion,
        "stand_cap": cap.standing,
        "td_cap": cap.takedown,
        "ground_cap": cap.ground_top,
    }
    return cap, raw


def label(row: pd.Series) -> str:
    for c in ("fighter_name", "name", "fighter"):
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return str(row["fighter_id"])


def show_probs(state: FightState, cap: Capability) -> str:
    probs = action_probabilities(state, cap)
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{a.value}={p*100:.1f}%" for a, p in ordered)


def main() -> None:
    latest = load_latest_profiles()
    snapshots = load_prefight_snapshots()
    pop = runtime_population(latest)

    print("STANDARD FIGHTER V1 — REAL FSR V3 CAPABILITY AUDIT")
    print(f"latest profiles={len(latest):,}; valid percentile population={len(pop):,}")
    print("mapping: standing=(rate percentile + accuracy percentile)/2; counter=accuracy percentile; pressure=rate percentile")
    print("mapping: takedown=(opponent-specific TD rate percentile + completion percentile)/2; clinch held neutral=.35")

    groups = []
    ordered = snapshots.sort_values(["event_date", "fight_id"], ascending=[False, False])
    for (date, fight_id), g in ordered.groupby(["event_date", "fight_id"], sort=False):
        if len(g) != 2:
            continue
        if not set(g["fighter_id"]).issubset(set(latest["fighter_id"])):
            continue
        groups.append((date, fight_id, g.copy()))
        if len(groups) >= 6:
            break

    if not groups:
        raise RuntimeError("no recent two-fighter FSR V3 matchup groups found")

    states = {
        "neutral": FightState(),
        "losing_striking": FightState(striking_edge=-.85),
        "losing_plus_td_failing": FightState(striking_edge=-.85, td_failure_recent=.85),
        "hurt": FightState(own_hurt=.90),
        "opponent_hurt": FightState(opponent_hurt=.90),
    }

    for date, fight_id, g in groups:
        a, b = g.iloc[0], g.iloc[1]
        print(f"\n=== {pd.Timestamp(date).date()} fight={fight_id}: {label(a)} vs {label(b)} ===")
        for fighter, opp in ((a, b), (b, a)):
            cap, raw = capability(fighter, opp, pop)
            print(f"\n{label(fighter)} vs {label(opp)}")
            print(
                f"  FSR matchup: stand_rate15={raw['stand_rate15']:.3f} stand_acc={raw['stand_acc']:.3f} "
                f"td_rate15={raw['td_rate15']:.3f} td_comp={raw['td_comp']:.3f}"
            )
            print(
                f"  brain capabilities: standing={raw['stand_cap']:.3f} takedown={raw['td_cap']:.3f} "
                f"ground_top={raw['ground_cap']:.3f} clinch=0.350(neutral)"
            )
            for sname, state in states.items():
                print(f"  {sname:24s} {show_probs(state, cap)}")

    print("\nSTANDARD FIGHTER V1 REAL FSR AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
