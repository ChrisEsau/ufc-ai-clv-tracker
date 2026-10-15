from __future__ import annotations

from dataclasses import replace

from .policy import Action, Capability, FightState, Phase, action_probabilities


def p(state: FightState, cap: Capability, action: Action) -> float:
    return action_probabilities(state, cap)[action]


def assert_gt(label: str, a: float, b: float) -> None:
    if not a > b:
        raise AssertionError(f"{label}: expected {a:.4f} > {b:.4f}")
    print(f"PASS {label}: {a:.4f} > {b:.4f}")


def assert_lt(label: str, a: float, b: float) -> None:
    if not a < b:
        raise AssertionError(f"{label}: expected {a:.4f} < {b:.4f}")
    print(f"PASS {label}: {a:.4f} < {b:.4f}")


def main() -> None:
    neutral = FightState()
    balanced = Capability(standing=.45, counter=.35, pressure=.35, clinch=.35, takedown=.35, ground_top=.35, submission=.30, escape=.40, reversal=.30)
    striker = replace(balanced, standing=.85, counter=.70, pressure=.65, takedown=.10, clinch=.20)
    wrestler = replace(balanced, standing=.30, counter=.20, pressure=.55, takedown=.90, clinch=.75, ground_top=.85)
    weak_wrestler = replace(striker, takedown=-.25, clinch=.05)

    # 1-5 standing phase value and alternatives.
    win = replace(neutral, striking_edge=.85)
    lose = replace(neutral, striking_edge=-.85)
    assert_gt("elite striker winning -> more stand attack", p(win, striker, Action.STAND_ATTACK), p(neutral, striker, Action.STAND_ATTACK))
    assert_lt("elite striker winning -> fewer TD entries", p(win, striker, Action.TAKEDOWN_ENTRY), p(neutral, striker, Action.TAKEDOWN_ENTRY))
    assert_gt("strong wrestler losing striking -> more TD", p(lose, wrestler, Action.TAKEDOWN_ENTRY), p(neutral, wrestler, Action.TAKEDOWN_ENTRY))
    assert_gt("weak wrestler losing striking -> more reset", p(lose, weak_wrestler, Action.RESET_RANGE), p(neutral, weak_wrestler, Action.RESET_RANGE))
    assert_gt("strong wrestler chooses TD more than weak wrestler when losing striking", p(lose, wrestler, Action.TAKEDOWN_ENTRY), p(lose, weak_wrestler, Action.TAKEDOWN_ENTRY))

    # 6-7 hurt logic.
    hurt = replace(neutral, own_hurt=.9)
    opp_hurt = replace(neutral, opponent_hurt=.9)
    assert_gt("own hurt -> more reset", p(hurt, balanced, Action.RESET_RANGE), p(neutral, balanced, Action.RESET_RANGE))
    assert_gt("opponent hurt -> more stand attack", p(opp_hurt, balanced, Action.STAND_ATTACK), p(neutral, balanced, Action.STAND_ATTACK))

    # 8-10 wrestling reinforcement and abandonment.
    td_working = replace(neutral, td_success_recent=.9)
    td_failing = replace(neutral, td_failure_recent=.9)
    assert_gt("successful TDs -> more TD intent", p(td_working, wrestler, Action.TAKEDOWN_ENTRY), p(neutral, wrestler, Action.TAKEDOWN_ENTRY))
    assert_lt("stuffed TDs -> less TD intent", p(td_failing, wrestler, Action.TAKEDOWN_ENTRY), p(neutral, wrestler, Action.TAKEDOWN_ENTRY))
    assert_gt("failed TDs still more viable for wrestler than weak wrestler", p(td_failing, wrestler, Action.TAKEDOWN_ENTRY), p(td_failing, weak_wrestler, Action.TAKEDOWN_ENTRY))

    # 11-12 cage logic.
    opp_cage = replace(neutral, opponent_back_to_cage=1.0)
    own_cage = replace(neutral, own_back_to_cage=1.0)
    assert_gt("opponent on fence + wrestler -> more TD", p(opp_cage, wrestler, Action.TAKEDOWN_ENTRY), p(neutral, wrestler, Action.TAKEDOWN_ENTRY))
    assert_gt("own back on fence -> more reset", p(own_cage, balanced, Action.RESET_RANGE), p(neutral, balanced, Action.RESET_RANGE))

    # 13-15 ground common sense.
    top_neutral = FightState(phase=Phase.GROUND_TOP)
    top_dom = FightState(phase=Phase.GROUND_TOP, dominant_top_position=1.0)
    assert_lt("dominant top -> less disengage", p(top_dom, wrestler, Action.DISENGAGE), p(top_neutral, wrestler, Action.DISENGAGE))
    bottom_neutral = FightState(phase=Phase.GROUND_BOTTOM)
    bottom_bad = FightState(phase=Phase.GROUND_BOTTOM, bad_bottom_position=1.0)
    assert_gt("bad bottom -> improve position", p(bottom_bad, balanced, Action.IMPROVE_POSITION), p(bottom_neutral, balanced, Action.IMPROVE_POSITION))
    assert_gt("standup-oriented fighter bottom -> escape prioritized", p(bottom_neutral, striker, Action.ESCAPE_STAND), p(bottom_neutral, striker, Action.BOTTOM_STRIKE))

    # 16-17 score/time urgency.
    behind_late = replace(neutral, score_state=-1.0, late_fight=1.0)
    ahead_late = replace(neutral, score_state=1.0, late_fight=1.0)
    assert_gt("behind late -> more attack", p(behind_late, balanced, Action.STAND_ATTACK), p(neutral, balanced, Action.STAND_ATTACK))
    assert_gt("ahead late -> more reset", p(ahead_late, balanced, Action.RESET_RANGE), p(neutral, balanced, Action.RESET_RANGE))

    print("\nSTANDARD FIGHTER V1 SYNTHETIC SCENARIOS: PASS")


if __name__ == "__main__":
    main()
