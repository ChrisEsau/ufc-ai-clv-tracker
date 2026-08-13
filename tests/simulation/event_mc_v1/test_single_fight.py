from pipeline.simulation.event_mc_v1.single_fight import resolve_fight, run_summary, run_trace

FIGHT_ID = "d14fea43712707f0"


def test_historical_fight_resolves_expected_prefight_profiles():
    fight = resolve_fight(FIGHT_ID)
    assert (fight.red_name, fight.blue_name, fight.date) == ("Benoit Saint Denis", "Mauricio Ruffy", "2025-09-06")
    assert fight.profiles.red.fighter_name == fight.red_name


def test_trace_and_summary_modes_execute_and_are_reproducible(capsys):
    fight = resolve_fight(FIGHT_ID)
    run_trace(fight, 42)
    trace = capsys.readouterr().out
    assert "CHRONOLOGICAL TRACE" in trace and "RoundStarted" in trace and "PATH SUMMARY" in trace
    first = run_summary(fight, 3, 77); capsys.readouterr()
    second = run_summary(fight, 3, 77); summary = capsys.readouterr().out
    assert first == second
    assert "Completed: 3/3" in summary and "CHRONOLOGICAL TRACE" not in summary
