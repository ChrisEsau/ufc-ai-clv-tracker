"""Bruno Silva vs Edgar Chairez pure-EWM 0.50 six-way shadow.

Research-only wrapper around the frozen Schnell-Costa EWM05 harness. It changes
only the target matchup; EWM decay, canonical blend, matched seeds, standing
cadence, Brain shadow behavior, mechanics, and judging remain identical.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io

from pipeline.simulation.event_clock_mc_v2.diagnostics import schnell_costa_ewm05_sixway as base


def main() -> None:
    base.FIGHTER_A = "Bruno Silva"
    base.FIGHTER_B = "Edgar Chairez"

    buf = io.StringIO()
    with redirect_stdout(buf):
        base.main()

    text = buf.getvalue()
    text = text.replace("SCHNELL_COSTA_EWM05_SIXWAY", "BRUNO_SILVA_CHAIREZ_EWM05_SIXWAY")
    text = text.replace("Schnell-Costa pure EWM 0.50 six-way shadow", "Bruno Silva-Chairez pure EWM 0.50 six-way shadow")
    print(text, end="")


if __name__ == "__main__":
    main()
