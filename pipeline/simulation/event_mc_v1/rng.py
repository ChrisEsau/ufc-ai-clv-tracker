"""Central, call-order-independent named random streams."""

from enum import IntEnum

import numpy as np


class RNGStream(IntEnum):
    SCHEDULER = 10
    STRIKE_RESOLUTION = 20
    TAKEDOWN = 30
    SUBMISSION = 40
    DAMAGE = 50
    KNOCKDOWN_FINISH = 60
    JUDGING = 70


class RNGManager:
    def __init__(self, root_seed: int) -> None:
        if isinstance(root_seed, bool) or not isinstance(root_seed, int):
            raise TypeError("root_seed must be an integer")
        self.root_seed = root_seed
        self._streams: dict[RNGStream, np.random.Generator] = {}

    def stream(self, name: RNGStream) -> np.random.Generator:
        stream_name = RNGStream(name)
        if stream_name not in self._streams:
            seed = np.random.SeedSequence([self.root_seed, int(stream_name)])
            self._streams[stream_name] = np.random.default_rng(seed)
        return self._streams[stream_name]
