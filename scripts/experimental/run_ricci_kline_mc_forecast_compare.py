"""Compare Ricci vs Kline MC under current, raw-trend, and shrunk-trend FSR-32.

Uses the exact same Monte Carlo engine, path seeds, ages, and 3-round horizon for
all three variants. Only the fighter FSR rows change.

Shadow/research only. No canonical artifact or simulator constant is modified.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental.fsr_32_trajectory_forecast_audit import FSR_PATH, build_forecasts

TARGET_FIGHT_ID = "52ddf20a10890b41"
RICCI_ID = "d25240135aee03e5"
KLINE_ID = "745fa7b605f8e2da"
FIGHTER_IDS = {RICCI_ID, KLINE_ID}
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
HEARTBEAT_EVERY = 100


def _age(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _load_target_context() -> tuple[pd.Series, tuple[pd.Series, pd.Series]]:
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    selected = cohort.loc[cohort["bout_id"].eq(TARGET_FIGHT_ID)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one target cohort row, found {len(selected)}")
    return selected.iloc[0], pairs[TARGET_FIGHT_ID]


def _forecast_target_rows() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    print("[forecast] loading Ricci/Kline FSR histories...", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    subset = fsr.loc[fsr["fighter_id"].isin(FIGHTER_IDS)].copy()
    print(f"[forecast] {len(subset)} historical prefight rows; fitting forecasts...", flush=True)

    raw, shrunk, _audit, _traits = build_forecasts(subset)
    raw_target = raw.loc[raw["fight_id"].eq(TARGET_FIGHT_ID)].copy()
    shrunk_target = shrunk.loc[shrunk["fight_id"].eq(TARGET_FIGHT_ID)].copy()
    if len(raw_target) != 2 or len(shrunk_target) != 2:
        raise RuntimeError("Could not resolve both target fighter forecast rows")

    raw_map = {str(r["fighter_id"]): r for _, r in raw_target.iterrows()}
    shrunk_map = {str(r["fighter_id"]): r for _, r in shrunk_target.iterrows()}
    return raw_map, shrunk_map


def _variant_pair(
    current_pair: tuple[pd.Series, pd.Series],
    replacement: dict[str, pd.Series] | None,
) -> tuple[pd.Series, pd.Series]:
    if replacement is None:
        return current_pair
    out: list[pd.Series] = []
    for row in current_pair:
        fighter_id = str(row.get("fighter_id"))
        if fighter_id not in replacement:
            raise RuntimeError(f"Forecast replacement missing fighter {fighter_id}")
        new_row = row.copy()
        source = replacement[fighter_id]
        for col in source.index:
            if col in new_row.index:
                new_row[col] = source[col]
        out.append(new_row)
    return out[0], out[1]


def _run_variant(
    label: str,
    pair: tuple[pd.Series, pd.Series],
    seeds: np.ndarray,
    red_age: float | None,
    blue_age: float | None,
) -> dict[str, object]:
    red, blue = pair
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)
    winners = {0: 0, 1: 0}
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    winner_methods = {
        0: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
        1: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
    }

    started = time.perf_counter()
    total = len(seeds)
    print(f"\n[{label}] starting {total:,} paths: {red_name} vs {blue_name}", flush=True)

    for i, seed in enumerate(seeds, start=1):
        sim = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winner = int(path.winner)
        method = str(path.method)
        winners[winner] += 1
        methods[method] = methods.get(method, 0) + 1
        winner_methods[winner][method] = winner_methods[winner].get(method, 0) + 1

        if i % HEARTBEAT_EVERY == 0 or i == total:
            elapsed = time.perf_counter() - started
            pct = 100.0 * i / total
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / rate if rate > 0 else 0.0
            print(
                f"[{label}] {i:,}/{total:,} ({pct:5.1f}%) | "
                f"elapsed={elapsed:6.1f}s | eta={eta:6.1f}s",
                flush=True,
            )

    n = float(total)
    return {
        "variant": label,
        "red_name": red_name,
        "blue_name": blue_name,
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
        "p_red_ko_tko": winner_methods[0]["KO/TKO"] / n,
        "p_blue_ko_tko": winner_methods[1]["KO/TKO"] / n,
        "p_red_sub": winner_methods[0]["SUB"] / n,
        "p_blue_sub": winner_methods[1]["SUB"] / n,
        "p_red_dec": winner_methods[0]["DEC"] / n,
        "p_blue_dec": winner_methods[1]["DEC"] / n,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    overall_started = time.perf_counter()
    print("[setup] loading aligned historical target bout...", flush=True)
    bout, current_pair = _load_target_context()
    red_age = _age(bout, "r_age")
    blue_age = _age(bout, "b_age")

    raw_map, shrunk_map = _forecast_target_rows()
    variants = {
        "CURRENT": current_pair,
        "RAW": _variant_pair(current_pair, raw_map),
        "SHRUNK": _variant_pair(current_pair, shrunk_map),
    }

    red_name = base._display_name(current_pair[0])
    blue_name = base._display_name(current_pair[1])
    print(
        f"[setup] target={TARGET_FIGHT_ID} | RED={red_name} | BLUE={blue_name} | "
        f"ages={red_age}/{blue_age}",
        flush=True,
    )
    print("[setup] generating shared seeds...", flush=True)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    rows = []
    for label in ("CURRENT", "RAW", "SHRUNK"):
        rows.append(_run_variant(label, variants[label], seeds, red_age, blue_age))

    result = pd.DataFrame(rows)
    print("\n" + "=" * 118)
    print("RICCI vs KLINE — MC FORECAST COMPARISON")
    print("=" * 118)
    display = result.copy()
    probability_cols = [c for c in display.columns if c.startswith("p_")]
    for col in probability_cols:
        display[col] = display[col].map(lambda x: f"{x:.2%}")
    print(
        display[
            [
                "variant",
                "p_red_win",
                "p_blue_win",
                "p_ko_tko",
                "p_sub",
                "p_dec",
                "p_red_ko_tko",
                "p_blue_ko_tko",
                "p_red_sub",
                "p_blue_sub",
                "p_red_dec",
                "p_blue_dec",
                "elapsed_seconds",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.1f}")
    )

    print("\nWIN-PROBABILITY MOVEMENT VS CURRENT")
    current = result.iloc[0]
    for _, row in result.iloc[1:].iterrows():
        print(
            f"{row['variant']:6s}: RED {row['p_red_win'] - current['p_red_win']:+.2%} | "
            f"BLUE {row['p_blue_win'] - current['p_blue_win']:+.2%} | "
            f"KO {row['p_ko_tko'] - current['p_ko_tko']:+.2%} | "
            f"SUB {row['p_sub'] - current['p_sub']:+.2%} | "
            f"DEC {row['p_dec'] - current['p_dec']:+.2%}"
        )

    print(f"\n[done] total elapsed={time.perf_counter() - overall_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
