"""Replay one actual R1-KO bout the MC strongly identified and one it missed.

Selection contract
------------------
- Start from the existing 2020+ mature-fighter Round-1 severity decomposition.
- Restrict to the actual Round-1 KO/TKO cohort.
- "Hit" = actual R1 KO/TKO bout with the highest observed MC P(R1 KO).
- "Miss" = actual R1 KO/TKO bout with the lowest observed MC P(R1 KO).
- Resolve the leakage-safe pre-fight FSR pair for each bout.
- Freshly re-simulate each selected matchup under the current shadow simulator.

This is a diagnostic/replay tool only. It changes no FSR values or simulator constants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import fsr_static_mc_ko_tko_v2_2020plus_mature_r1_severity_decomposition as severity


DEFAULT_INPUT = severity.OUTPUT_PATH
DEFAULT_FRESH_PATHS = 100
DEFAULT_SEED = 20260810

TRAITS = (
    "striking_power",
    "knockdown_resistance",
    "damage_durability",
    "distance_striking_pressure",
    "distance_precision",
    "distance_striking_precision",
    "distance_defense",
    "distance_striking_defense",
    "clinch_striking_pressure",
    "clinch_striking_precision",
    "clinch_striking_defense",
    "ground_striking_pressure",
    "ground_striking_precision",
    "ground_striking_defense",
)


def _load_path_diagnostic(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"R1 severity decomposition not found: {path}. "
            "Run fsr_static_mc_ko_tko_v2_2020plus_mature_r1_severity_decomposition.py first."
        )
    frame = pd.read_parquet(path).copy()
    required = {
        "bout_id",
        "cohort_group",
        "actual_r1_ko",
        "r1_ko",
        "r1_any_kd",
        "r1_kd",
        "r1_sig_landed",
        "r1_max_shock",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"R1 severity artifact missing required columns: {missing}")
    frame["bout_id"] = frame["bout_id"].astype(str)
    return frame


def _rank_actual_r1_ko_bouts(paths: pd.DataFrame) -> pd.DataFrame:
    actual = paths[
        paths["actual_r1_ko"].eq(1)
        & paths["cohort_group"].eq("actual_r1_ko")
    ].copy()
    if actual.empty:
        raise ValueError("No actual R1 KO/TKO paths found in diagnostic artifact.")

    summary = (
        actual.groupby("bout_id", as_index=False)
        .agg(
            diagnostic_paths=("r1_ko", "size"),
            mc_p_r1_ko=("r1_ko", "mean"),
            mc_p_r1_kd=("r1_any_kd", "mean"),
            mean_r1_kd=("r1_kd", "mean"),
            mean_r1_sig_landed=("r1_sig_landed", "mean"),
            mean_r1_max_shock=("r1_max_shock", "mean"),
        )
    )
    return summary


def _select_hit_miss(summary: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    hit = summary.sort_values(
        ["mc_p_r1_ko", "mc_p_r1_kd", "mean_r1_max_shock"],
        ascending=[False, False, False],
    ).iloc[0]
    miss = summary.sort_values(
        ["mc_p_r1_ko", "mc_p_r1_kd", "mean_r1_max_shock"],
        ascending=[True, True, True],
    ).iloc[0]
    return hit, miss


def _resolve_current_cohort_and_pairs(
    fsr_path: Path,
    master_path: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    modern = severity.modern
    master = modern._load_master(master_path)
    candidate = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(fsr_path, candidate)
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    return cohort, pairs


def _fighter_name(profile: pd.Series) -> str:
    return base._display_name(profile)


def _maybe_winner_name(bout: pd.Series, red: pd.Series, blue: pd.Series) -> str:
    if "actual_winner_id" not in bout.index or pd.isna(bout.get("actual_winner_id")):
        return "unknown"
    winner_id = str(bout["actual_winner_id"])
    if winner_id == str(red["fighter_id"]):
        return _fighter_name(red)
    if winner_id == str(blue["fighter_id"]):
        return _fighter_name(blue)
    return winner_id


def _print_trait_table(red: pd.Series, blue: pd.Series) -> None:
    red_name = _fighter_name(red)
    blue_name = _fighter_name(blue)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for trait in TRAITS:
        # Avoid printing aliases twice when only one naming convention is populated.
        if trait in seen:
            continue
        if trait not in red.index and trait not in blue.index:
            continue
        rv = pd.to_numeric(pd.Series([red.get(trait)]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([blue.get(trait)]), errors="coerce").iloc[0]
        rows.append(
            {
                "trait": trait,
                red_name: float(rv) if pd.notna(rv) else np.nan,
                blue_name: float(bv) if pd.notna(bv) else np.nan,
            }
        )
        seen.add(trait)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _run_fresh_batch(
    red: pd.Series,
    blue: pd.Series,
    *,
    paths: int,
    seed: int,
) -> tuple[pd.DataFrame, list[tuple[int, severity.R1SeverityTraceSimulator, object]]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    sims: list[tuple[int, severity.R1SeverityTraceSimulator, object]] = []

    for path_index in range(paths):
        path_seed = int(rng.integers(0, 2**31 - 1))
        sim = severity.R1SeverityTraceSimulator(
            red,
            blue,
            collapse=severity.STRONG,
            seed=path_seed,
        )
        path = sim.run()
        trace = pd.DataFrame(sim.strike_trace)
        rows.append(
            {
                "path_index": path_index,
                "seed": path_seed,
                "r1_ko": int(getattr(path, "finish", None) is not None),
                "r1_any_kd": int(
                    sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored > 0
                ),
                "r1_kd": int(sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored),
                "r1_sig_att": int(sim.stats[0].sig_att + sim.stats[1].sig_att),
                "r1_sig_landed": int(sim.stats[0].sig_landed + sim.stats[1].sig_landed),
                "r1_max_shock": float(trace["shock_fraction"].max()) if len(trace) else 0.0,
                "r1_mean_shock": float(trace["shock_fraction"].mean()) if len(trace) else 0.0,
            }
        )
        sims.append((path_seed, sim, path))

    return pd.DataFrame(rows), sims


def _representative_sim(
    batch: pd.DataFrame,
    sims: list[tuple[int, severity.R1SeverityTraceSimulator, object]],
    *,
    prefer_finish: bool,
) -> tuple[int, severity.R1SeverityTraceSimulator, object]:
    candidates = batch[batch["r1_ko"].eq(int(prefer_finish))]
    if candidates.empty:
        candidates = batch

    # Prefer a path near the candidate group's median max shock rather than the
    # most extreme realization, so the printed replay is reasonably representative.
    target = float(candidates["r1_max_shock"].median())
    row = candidates.assign(
        distance=(candidates["r1_max_shock"] - target).abs()
    ).sort_values(["distance", "path_index"]).iloc[0]
    idx = int(row["path_index"])
    return sims[idx]


def _print_path(sim: severity.R1SeverityTraceSimulator, path: object, seed: int) -> None:
    print(f"\nREPRESENTATIVE PATH | seed={seed}")
    print("-" * 120)
    for event in path.events:
        owner = ""
        if event.get("top_start"):
            owner = f" | top={event['top_start']}"
        elif event.get("clinch_controller_start"):
            owner = f" | clinch_ctrl={event['clinch_controller_start']}"
        print(
            f"R{event['round']} {event['clock_start']} S{event['segment']:02d} "
            f"[{event['phase_start']:8s}->{event['phase_end']:8s}] "
            f"{event['striking']} | {event['transition']}{owner}"
        )

    finish = getattr(path, "finish", None)
    print("\nPATH RESULT")
    if finish is None:
        print("No R1 KO/TKO in this path.")
    else:
        winner = sim.names[int(finish.winner)]
        loser = sim.names[int(finish.loser)]
        print(f"KO/TKO: {winner} over {loser}")
        print(
            f"finish strike raw={finish.raw_strike_damage:.3f}; "
            f"effective={finish.effective_strike_damage:.3f}; "
            f"reservoir {finish.reservoir_before:.3f}->{finish.reservoir_after:.3f}; "
            f"KD on strike={finish.knockdown_on_strike}; "
            f"recent KD before={finish.recent_kd_before}"
        )

    for i, name in enumerate(sim.names):
        stats = sim.stats[i]
        state = sim.damage_state[i]
        print(
            f"{name}: sig={stats.sig_landed}/{stats.sig_att}; "
            f"KD scored={stats.knockdowns_scored}; "
            f"damage dealt={stats.damage_dealt:.2f}; "
            f"reservoir remaining={state.reservoir_current:.2f}/{state.reservoir_capacity:.2f}"
        )


def _print_case(
    label: str,
    selected: pd.Series,
    cohort: pd.DataFrame,
    pairs: dict[str, tuple[pd.Series, pd.Series]],
    *,
    fresh_paths: int,
    seed: int,
) -> None:
    bout_id = str(selected["bout_id"])
    if bout_id not in pairs:
        raise KeyError(f"Selected bout {bout_id} missing from current FSR pairs.")
    red, blue = pairs[bout_id]
    bout_rows = cohort[cohort["bout_id"].eq(bout_id)]
    bout = bout_rows.iloc[0] if len(bout_rows) else pd.Series(dtype=object)

    print("\n" + "=" * 120)
    print(f"{label}: {_fighter_name(red)} vs {_fighter_name(blue)}")
    print("=" * 120)
    print(f"bout_id: {bout_id}")
    if "event_date" in bout.index and pd.notna(bout.get("event_date")):
        print(f"event_date: {pd.Timestamp(bout['event_date']).date()}")
    print("actual outcome: Round-1 KO/TKO")
    print(f"actual winner: {_maybe_winner_name(bout, red, blue)}")
    print(
        "diagnostic MC: "
        f"P(R1 KO)={float(selected['mc_p_r1_ko']):.2%}; "
        f"P(R1 KD)={float(selected['mc_p_r1_kd']):.2%}; "
        f"mean sig landed={float(selected['mean_r1_sig_landed']):.2f}; "
        f"mean max shock={float(selected['mean_r1_max_shock']):.4f}; "
        f"paths={int(selected['diagnostic_paths'])}"
    )

    print("\nPRE-FIGHT FSR TRAITS")
    _print_trait_table(red, blue)

    batch, sims = _run_fresh_batch(
        red,
        blue,
        paths=fresh_paths,
        seed=seed,
    )
    print("\nFRESH CURRENT-MC REPLAY")
    print(
        f"paths={len(batch)}; "
        f"P(R1 KO)={batch['r1_ko'].mean():.2%}; "
        f"P(R1 KD)={batch['r1_any_kd'].mean():.2%}; "
        f"mean KD={batch['r1_kd'].mean():.3f}; "
        f"mean sig attempts={batch['r1_sig_att'].mean():.2f}; "
        f"mean sig landed={batch['r1_sig_landed'].mean():.2f}; "
        f"mean max shock={batch['r1_max_shock'].mean():.4f}"
    )

    representative = _representative_sim(
        batch,
        sims,
        prefer_finish=(label == "HIT"),
    )
    _print_path(representative[1], representative[2], representative[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one actual R1 KO the MC hit and one actual R1 KO it missed"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--master", type=Path, default=severity.modern.MASTER_PATH)
    parser.add_argument("--fsr-path", type=Path, default=severity.modern.FSR_PATH)
    parser.add_argument("--fresh-paths", type=int, default=DEFAULT_FRESH_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.fresh_paths <= 0:
        raise ValueError("--fresh-paths must be positive")

    diagnostic = _load_path_diagnostic(args.input)
    summary = _rank_actual_r1_ko_bouts(diagnostic)
    hit, miss = _select_hit_miss(summary)
    cohort, pairs = _resolve_current_cohort_and_pairs(args.fsr_path, args.master)

    print("=" * 120)
    print("ACTUAL ROUND-1 KO/TKO — MC HIT VS MISS REPLAY")
    print("=" * 120)
    print(
        f"actual R1 KO bouts ranked: {len(summary):,}; "
        f"selection based on existing {int(summary['diagnostic_paths'].median())}-path diagnostic"
    )
    print(
        f"HIT selected at P(R1 KO)={float(hit['mc_p_r1_ko']):.2%}; "
        f"MISS selected at P(R1 KO)={float(miss['mc_p_r1_ko']):.2%}"
    )

    _print_case(
        "HIT",
        hit,
        cohort,
        pairs,
        fresh_paths=args.fresh_paths,
        seed=args.seed + 1,
    )
    _print_case(
        "MISS",
        miss,
        cohort,
        pairs,
        fresh_paths=args.fresh_paths,
        seed=args.seed + 2,
    )

    print("\nNo simulator constants or FSR values were changed.")


if __name__ == "__main__":
    main()
