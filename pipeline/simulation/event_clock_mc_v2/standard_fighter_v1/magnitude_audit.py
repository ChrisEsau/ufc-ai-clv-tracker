from __future__ import annotations

from dataclasses import replace

from .policy import Action, Capability, FightState, action_probabilities


def pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def prob(state: FightState, cap: Capability, action: Action) -> float:
    return action_probabilities(state, cap)[action]


def sweep(label: str, values: list[float], make_state, cap: Capability, actions: tuple[Action, ...]) -> None:
    print(f"\n=== {label} ===")
    print("value  " + "  ".join(f"{a.value:>16}" for a in actions))
    for value in values:
        state = make_state(value)
        vals = [prob(state, cap, action) for action in actions]
        print(f"{value:4.2f}  " + "  ".join(f"{pct(v):>16}" for v in vals))


def compare(label: str, actual: float, estimated: float) -> None:
    err_pp = 100.0 * (actual - estimated)
    print(f"{label:58} actual={pct(actual)}  estimated={100*estimated:6.2f}%  error={err_pp:+6.2f} pp")


def main() -> None:
    neutral = FightState()
    balanced = Capability(
        standing=.45, counter=.35, pressure=.35, clinch=.35, takedown=.35,
        ground_top=.35, submission=.30, escape=.40, reversal=.30,
    )
    striker = replace(balanced, standing=.85, counter=.70, pressure=.65, takedown=.10, clinch=.20)
    wrestler = replace(balanced, standing=.30, counter=.20, pressure=.55, takedown=.90, clinch=.75, ground_top=.85)

    print("STANDARD FIGHTER V1 MAGNITUDE AUDIT")
    print("temperature=0.55; coefficients unchanged")

    print("\n=== NEUTRAL BASELINES ===")
    for name, cap in (("balanced", balanced), ("striker", striker), ("wrestler", wrestler)):
        probs = action_probabilities(neutral, cap)
        print(name)
        for action, value in probs.items():
            print(f"  {action.value:16} {pct(value)}")

    grid = [0.00, 0.25, 0.50, 0.75, 1.00]

    sweep(
        "WRESTLER: INCREASING STRIKING DISADVANTAGE",
        grid,
        lambda x: replace(neutral, striking_edge=-x),
        wrestler,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK),
    )
    sweep(
        "STRIKER: INCREASING STRIKING DISADVANTAGE",
        grid,
        lambda x: replace(neutral, striking_edge=-x),
        striker,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK, Action.STAND_COUNTER),
    )
    sweep(
        "WRESTLER: INCREASING OWN HURT",
        grid,
        lambda x: replace(neutral, own_hurt=x),
        wrestler,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK),
    )
    sweep(
        "STRIKER: INCREASING OWN HURT",
        grid,
        lambda x: replace(neutral, own_hurt=x),
        striker,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK),
    )
    sweep(
        "WRESTLER: RECENT TD SUCCESS",
        grid,
        lambda x: replace(neutral, td_success_recent=x),
        wrestler,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK),
    )
    sweep(
        "WRESTLER: RECENT TD FAILURE",
        grid,
        lambda x: replace(neutral, td_failure_recent=x),
        wrestler,
        (Action.TAKEDOWN_ENTRY, Action.CLINCH_ENTRY, Action.RESET_RANGE, Action.STAND_ATTACK),
    )
    sweep(
        "WRESTLER: FATIGUE",
        grid,
        lambda x: replace(neutral, fatigue=x),
        wrestler,
        (Action.TAKEDOWN_ENTRY, Action.PRESSURE, Action.RESET_RANGE, Action.STAND_ATTACK),
    )

    combo_outstruck_td_success = replace(neutral, striking_edge=-1.0, td_success_recent=1.0)
    combo_outstruck_hurt = replace(neutral, striking_edge=-1.0, own_hurt=1.0)
    opp_hurt = replace(neutral, opponent_hurt=1.0)

    print("\n=== STACKED / MIXED SIGNALS ===")
    for label, state, cap in (
        ("wrestler: badly outstruck + TDs succeeding", combo_outstruck_td_success, wrestler),
        ("wrestler: badly outstruck + badly hurt", combo_outstruck_hurt, wrestler),
        ("wrestler: opponent badly hurt", opp_hurt, wrestler),
    ):
        print(label)
        for action, value in sorted(action_probabilities(state, cap).items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {action.value:16} {pct(value)}")

    # Earlier assistant estimates, recorded verbatim/approximately as decimals.
    # This section exists specifically to audit those estimates against real execution.
    estimates = [
        ("neutral balanced stand attack", prob(neutral, balanced, Action.STAND_ATTACK), .33),
        ("neutral balanced stand counter", prob(neutral, balanced, Action.STAND_COUNTER), .20),
        ("neutral balanced pressure", prob(neutral, balanced, Action.PRESSURE), .16),
        ("neutral balanced TD", prob(neutral, balanced, Action.TAKEDOWN_ENTRY), .12),
        ("neutral wrestler TD", prob(neutral, wrestler, Action.TAKEDOWN_ENTRY), .24),
        ("neutral striker TD", prob(neutral, striker, Action.TAKEDOWN_ENTRY), .06),
        ("wrestler max striking disadvantage TD", prob(replace(neutral, striking_edge=-1.0), wrestler, Action.TAKEDOWN_ENTRY), .56),
        ("striker max striking disadvantage TD", prob(replace(neutral, striking_edge=-1.0), striker, Action.TAKEDOWN_ENTRY), .10),
        ("striker max striking disadvantage reset", prob(replace(neutral, striking_edge=-1.0), striker, Action.RESET_RANGE), .22),
        ("striker max hurt reset", prob(replace(neutral, own_hurt=1.0), striker, Action.RESET_RANGE), .48),
        ("wrestler max hurt reset", prob(replace(neutral, own_hurt=1.0), wrestler, Action.RESET_RANGE), .24),
        ("wrestler max hurt clinch", prob(replace(neutral, own_hurt=1.0), wrestler, Action.CLINCH_ENTRY), .23),
        ("wrestler max hurt TD", prob(replace(neutral, own_hurt=1.0), wrestler, Action.TAKEDOWN_ENTRY), .45),
        ("wrestler max TD success TD", prob(replace(neutral, td_success_recent=1.0), wrestler, Action.TAKEDOWN_ENTRY), .56),
        ("wrestler max TD failure TD", prob(replace(neutral, td_failure_recent=1.0), wrestler, Action.TAKEDOWN_ENTRY), .08),
        ("wrestler max fatigue TD", prob(replace(neutral, fatigue=1.0), wrestler, Action.TAKEDOWN_ENTRY), .13),
        ("wrestler max fatigue pressure", prob(replace(neutral, fatigue=1.0), wrestler, Action.PRESSURE), .08),
        ("outstruck + TD success wrestler TD", prob(combo_outstruck_td_success, wrestler, Action.TAKEDOWN_ENTRY), .83),
        ("outstruck + hurt wrestler TD", prob(combo_outstruck_hurt, wrestler, Action.TAKEDOWN_ENTRY), .62),
        ("outstruck + hurt wrestler clinch", prob(combo_outstruck_hurt, wrestler, Action.CLINCH_ENTRY), .18),
        ("outstruck + hurt wrestler reset", prob(combo_outstruck_hurt, wrestler, Action.RESET_RANGE), .18),
        ("opponent hurt wrestler stand attack", prob(opp_hurt, wrestler, Action.STAND_ATTACK), .52),
        ("opponent hurt wrestler pressure", prob(opp_hurt, wrestler, Action.PRESSURE), .30),
        ("opponent hurt wrestler TD", prob(opp_hurt, wrestler, Action.TAKEDOWN_ENTRY), .08),
    ]

    print("\n=== EARLIER ESTIMATE VS ACTUAL EXECUTION ===")
    abs_errors = []
    for label, actual, estimate in estimates:
        compare(label, actual, estimate)
        abs_errors.append(abs(100.0 * (actual - estimate)))

    print(f"\nmean absolute estimate error: {sum(abs_errors) / len(abs_errors):.3f} pp")
    print(f"max absolute estimate error:  {max(abs_errors):.3f} pp")
    print("\nSTANDARD FIGHTER V1 MAGNITUDE AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
