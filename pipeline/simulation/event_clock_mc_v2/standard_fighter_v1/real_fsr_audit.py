from __future__ import annotations

import pandas as pd

from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    load_latest_profiles,
    load_prefight_snapshots,
)
from .capability_translation import (
    CapabilityReference,
    prior_snapshot_count,
    translate_capability,
)
from .policy import Capability, FightState, action_probabilities


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
    reference = CapabilityReference.from_latest(latest)

    print("STANDARD FIGHTER V1 — FORMAL REAL FSR V3 CAPABILITY AUDIT")
    print(f"latest profiles={len(latest):,}; valid reference population={len(reference.runtime):,}")
    print("translator: matchup-aware empirical ranks; no fighter archetypes")
    print("standing=(rate rank + accuracy rank)/2; counter=accuracy rank; pressure=rate rank")
    print("takedown=(opponent-specific TD rate rank + completion rank)/2")
    print("ground_top=(ground rate rank + ground accuracy rank)/2")
    print("unsupported mappings held explicit neutral: clinch=.35 submission=.30 escape=.40 reversal=.30")
    print("cold_start flag: zero strictly prior-date UFC FSR snapshots; flag only, no capability adjustment")

    groups = []
    ordered = snapshots.sort_values(["event_date", "fight_id"], ascending=[False, False])
    latest_ids = set(latest["fighter_id"].astype(str))
    for (date, fight_id), g in ordered.groupby(["event_date", "fight_id"], sort=False):
        if len(g) != 2:
            continue
        if not set(g["fighter_id"].astype(str)).issubset(latest_ids):
            continue
        groups.append((date, fight_id, g.copy()))
        if len(groups) >= 8:
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

    audited = 0
    cold = 0
    cold_fights: set[str] = set()

    for date, fight_id, g in groups:
        a, b = g.iloc[0], g.iloc[1]
        print(f"\n=== {pd.Timestamp(date).date()} fight={fight_id}: {label(a)} vs {label(b)} ===")
        fight_has_cold = False
        for fighter, opp in ((a, b), (b, a)):
            prior = prior_snapshot_count(
                snapshots,
                fighter_id=str(fighter["fighter_id"]),
                event_date=date,
            )
            translated = translate_capability(
                fighter,
                opp,
                reference,
                prior_ufc_fights=prior,
            )
            cap = translated.capability
            audited += 1
            cold += int(translated.cold_start)
            fight_has_cold = fight_has_cold or translated.cold_start

            status = "COLD_START" if translated.cold_start else "established"
            print(f"\n{label(fighter)} vs {label(opp)}")
            print(f"  experience: prior_ufc_fights={translated.prior_ufc_fights} status={status}")
            print(
                f"  FSR matchup: stand_rate15={translated.standing_rate_15m:.3f} "
                f"stand_acc={translated.standing_accuracy:.3f} "
                f"td_rate15={translated.takedown_rate_15m:.3f} "
                f"td_comp={translated.takedown_completion:.3f}"
            )
            print(
                f"  brain capabilities: standing={cap.standing:.3f} "
                f"takedown={cap.takedown:.3f} ground_top={cap.ground_top:.3f} "
                f"clinch={cap.clinch:.3f}(neutral)"
            )
            print(
                f"  component ranks: stand_rate={translated.standing_rate_percentile:.3f} "
                f"stand_acc={translated.standing_accuracy_percentile:.3f} "
                f"td_rate={translated.takedown_rate_percentile:.3f} "
                f"td_comp={translated.takedown_completion_percentile:.3f}"
            )
            for sname, state in states.items():
                print(f"  {sname:24s} {show_probs(state, cap)}")

        if fight_has_cold:
            cold_fights.add(str(fight_id))

    print("\n=== COLD-START FLAGS ===")
    print(f"audited fighters={audited}; cold-start fighters={cold}; cold-start fights={len(cold_fights)}")
    if cold_fights:
        print("flagged fight_ids=" + ",".join(sorted(cold_fights)))
    print("cold-start flags are reporting-only; no policy or capability values were changed")
    print("\nSTANDARD FIGHTER V1 FORMAL REAL FSR AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
