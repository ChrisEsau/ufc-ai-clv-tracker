from types import SimpleNamespace

from pipeline.simulation.event_mc_v1.events import ConsequenceEvent, FightFinished
from pipeline.simulation.event_mc_v1.flow_stats import FlowStatsSink
from pipeline.simulation.event_mc_v1.judging import RoundScore
from pipeline.simulation.event_mc_v1.sinks import FullTraceEventSink
from pipeline.simulation.event_mc_v1.single_fight import (
    _aggregate_results,
    build_engine,
    resolve_fight,
    run_summary,
    run_trace,
)

FIGHT_ID = "d14fea43712707f0"


def test_historical_fight_resolves_expected_prefight_profiles():
    fight = resolve_fight(FIGHT_ID)
    assert fight.fight_id == FIGHT_ID
    assert (fight.red_name, fight.blue_name, fight.date) == ("Benoit Saint Denis", "Mauricio Ruffy", "2025-09-06")
    assert fight.profiles.red.fighter_name == fight.red_name
    assert fight.profiles.blue.fighter_name == fight.blue_name
    assert fight.profiles.red.fighter_id != fight.profiles.blue.fighter_id


def test_trace_and_summary_modes_execute_and_are_reproducible(capsys):
    fight = resolve_fight(FIGHT_ID)
    run_trace(fight, 42)
    trace = capsys.readouterr().out
    assert "CHRONOLOGICAL TRACE" in trace and "RoundStarted" in trace and "PATH SUMMARY" in trace
    first = run_summary(fight, 3, 77); capsys.readouterr()
    second = run_summary(fight, 3, 77); summary = capsys.readouterr().out
    assert first == second
    assert "Completed: 3/3" in summary and "CHRONOLOGICAL TRACE" not in summary
    assert "Finish rounds:" in summary
    assert "TD attempts/completions=" in summary
    assert "SUB attempts=" in summary
    assert "KO_TKO=" in summary and "SUB=" in summary and "Scheduled horizon=" in summary


def test_trace_is_ordered_terminal_and_exactly_seed_reproducible():
    fight = resolve_fight(FIGHT_ID)
    first_engine, _, _ = build_engine(fight, 314, FullTraceEventSink())
    second_engine, _, _ = build_engine(fight, 314, FullTraceEventSink())
    first = first_engine.run().sink_result
    second = second_engine.run().sink_result

    assert first == second
    timestamps = [entry.timestamp_seconds for entry in first]
    assert timestamps == sorted(timestamps)
    finished_indexes = [
        index
        for index, entry in enumerate(first)
        if entry.kind == "event" and isinstance(entry.payload, FightFinished)
    ]
    assert len(finished_indexes) == 1
    assert finished_indexes[0] == len(first) - 1


def _controlled_result(winner, finish_time, attempts, outcomes, knockdowns=()):
    return SimpleNamespace(
        state=SimpleNamespace(
            winner=winner,
            finish_method="KO_TKO" if winner else None,
            fight_time_seconds=finish_time,
        ),
        sink_result={
            "attempts": attempts,
            "outcomes": outcomes,
            "physiology": tuple(SimpleNamespace(knockdown=value) for value in knockdowns),
        },
    )


def test_controlled_summary_arithmetic():
    empty = {"red": {}, "blue": {}}
    rows = [
        _controlled_result(
            "red",
            250.0,
            {"red": {"takedown": 2, "submission_attempt": 1}, "blue": {}},
            {"red": {"takedown_landed": 1}, "blue": {}},
            (True, False),
        ),
        _controlled_result(
            "blue",
            601.0,
            {"red": {}, "blue": {"clinch_takedown": 3, "submission_attempt": 2}},
            {"red": {}, "blue": {"clinch_takedown_landed": 2}},
            (True, True),
        ),
        _controlled_result(None, 900.0, empty, empty),
    ]

    summary = _aggregate_results(rows)
    assert summary["finish_rounds"] == {1: 1, 3: 1}
    assert summary["scheduled_horizon"] == 1
    assert summary["knockdowns"] == (1, 2, 0)
    assert summary["sides"]["red"] == {
        "attempts": {"takedown": 2, "submission_attempt": 1},
        "td_attempts": 2,
        "td_completions": 1,
        "submission_attempts": 1,
        "ko_wins": 1,
        "submission_finishes": 0,
        "decision_wins": 0,
    }
    assert summary["sides"]["blue"]["td_attempts"] == 3
    assert summary["sides"]["blue"]["td_completions"] == 2
    assert summary["sides"]["blue"]["submission_attempts"] == 2


def test_lewis_daukaus_finishing_strike_accounting_preserves_physics():
    fight = resolve_fight("4b7ec02b39fc6f70")
    engine, _, _ = build_engine(
        fight,
        20260813,
        FlowStatsSink(),
    )
    result = engine.run()

    assert result.state.finished
    assert result.state.finish_method == "KO_TKO"

    landed_strikes = sum(
        count
        for side in ("red", "blue")
        for key, count in result.sink_result[
            "outcomes"
        ][side].items()
        if "strike" in key
        and key.endswith("_landed")
    )

    # Every modeled landed strike enters physiology exactly
    # once and receives exactly one KO/TKO finish check.
    assert landed_strikes > 0
    assert len(
        result.sink_result["physiology"]
    ) == landed_strikes
    assert len(
        result.sink_result["finishes"]
    ) == landed_strikes

    # Because this path terminates by KO/TKO, the final
    # landed-strike finish check must be the terminal one.
    assert result.sink_result[
        "finishes"
    ][-1].finished


def test_submission_trace_and_finishing_attempt_accounting(capsys):
    fight = resolve_fight("b22eab3aa1522f40")
    engine, _, _ = build_engine(fight, 22, FlowStatsSink())
    result = engine.run()
    assert result.state.finish_method == "SUB"
    assert sum(result.sink_result["attempts"][side].get("submission_attempt", 0) for side in ("red", "blue")) == 1
    assert sum(result.sink_result["outcomes"][side].get("submission_attempt_attempted", 0) for side in ("red", "blue")) == 1
    assert len(result.sink_result["submission_checks"]) == 1
    assert result.sink_result["submission_checks"][0].finished

    run_trace(fight, 22)
    trace = capsys.readouterr().out
    assert "SUBMISSION CHECK red->blue" in trace
    for field in ("threat=", "resistance=", "position=", "stamina/context=", "pSUB=", "finished=True"):
        assert field in trace


def test_decision_trace_has_five_ten_nine_cards_and_one_terminal_event(capsys):
    fight = resolve_fight("4b7ec02b39fc6f70")
    engine, _, _ = build_engine(fight, 70, FullTraceEventSink())
    result = engine.run()
    events = [entry.payload for entry in result.sink_result if entry.kind == "event"]
    cards = [event.payload for event in events if isinstance(event, ConsequenceEvent) and isinstance(event.payload, RoundScore)]
    assert result.state.finish_method == "DEC" and result.state.winner in {"red", "blue"}
    assert len(cards) == 5 and all({card.red_score, card.blue_score} == {9, 10} for card in cards)
    assert sum(isinstance(event, FightFinished) for event in events) == 1
    assert isinstance(events[-1], FightFinished)

    run_trace(fight, 70)
    trace = capsys.readouterr().out
    assert trace.count(" JUDGING ") == 5
    assert "FINAL DECISION" in trace and "method=DEC" in trace
    run_summary(fight, 1, 70)
    summary = capsys.readouterr().out
    assert "DEC=1 (100.0%)" in summary
