from pipeline.simulation.event_mc_v1.rng import RNGManager, RNGStream


def _draw(root: int, stream: RNGStream, count: int = 8):
    return RNGManager(root).stream(stream).random(count).tolist()


def test_same_seed_and_each_named_stream_reproduce() -> None:
    for stream in RNGStream:
        assert _draw(17, stream) == _draw(17, stream)


def test_different_seed_diverges() -> None:
    assert _draw(17, RNGStream.SCHEDULER) != _draw(18, RNGStream.SCHEDULER)


def test_other_stream_draws_do_not_perturb_scheduler() -> None:
    manager = RNGManager(99)
    manager.stream(RNGStream.DAMAGE).random(100_000)
    observed = manager.stream(RNGStream.SCHEDULER).random(8).tolist()
    assert observed == _draw(99, RNGStream.SCHEDULER)


def test_request_order_does_not_change_stream_sequences() -> None:
    first = RNGManager(123)
    damage_first = first.stream(RNGStream.DAMAGE).random(8).tolist()
    scheduler_second = first.stream(RNGStream.SCHEDULER).random(8).tolist()
    second = RNGManager(123)
    scheduler_first = second.stream(RNGStream.SCHEDULER).random(8).tolist()
    damage_second = second.stream(RNGStream.DAMAGE).random(8).tolist()
    assert damage_first == damage_second
    assert scheduler_second == scheduler_first


def test_locked_stream_ids() -> None:
    assert {stream.name: stream.value for stream in RNGStream} == {
        "SCHEDULER": 10,
        "STRIKE_RESOLUTION": 20,
        "TAKEDOWN": 30,
        "SUBMISSION": 40,
        "DAMAGE": 50,
        "KNOCKDOWN_FINISH": 60,
        "JUDGING": 70,
    }
