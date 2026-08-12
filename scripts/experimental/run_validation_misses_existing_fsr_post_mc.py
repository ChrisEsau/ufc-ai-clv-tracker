"""Zero-replay PRE vs observed-POST FSR MC audit for frozen validation misses.

This script intentionally DOES NOT rebuild any FSR database.

For each missed fight in the frozen 34-fight validation baseline:
1. use the original aligned FSR-32 pre-fight profiles;
2. recover each fighter's stored post-fight state from that fighter's next
   pre-fight FSR-32 snapshot (stored FSR does not change between fights);
3. if no later snapshot exists, optionally use an already-generated explicit
   canonical target pre/post CSV (currently Ricci/Kline);
4. run PRE and POST Monte Carlo arms with identical seeds.

If either fighter has no recoverable post-fight state, the bout is reported as
unavailable. No fallback historical replay is allowed.

Diagnostic only: POST arms are intentionally leaky/counterfactual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_single_historical_full_fight_bout as historical


BASELINE_PATH = Path(
    "data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv"
)
FSR32_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
RICCI_KLINE_EXPLICIT = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_canonical_shadow/"
    "fsr_canonical_52ddf20a10890b41_pre_post_traits.csv"
)
OUTPUT_PATH = Path(
    "data/experimental/validation_misses_existing_fsr_post_mc.csv"
)
UNAVAILABLE_PATH = Path(
    "data/experimental/validation_misses_existing_fsr_post_unavailable.csv"
)
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811


def _date_col(df: pd.DataFrame) -> str:
    for col in ("date", "event_date", "fight_date"):
        if col in df.columns:
            return col
    raise RuntimeError("FSR frame has no date/event_date/fight_date column")


def _overlay_canonical(profile: pd.Series, values: dict[str, float]) -> pd.Series:
    out = profile.copy()
    for trait, value in values.items():
        out[trait] = float(value)
    out["distance_precision"] = float(out["distance_striking_precision"])
    out["distance_defense"] = float(out["distance_striking_defense"])
    out["stamina_depletion_resistance"] = float(out["fatigue_accumulation_resistance"])
    out["stamina_performance_resilience"] = float(out["fatigue_performance_resilience"])
    out["stamina_capacity"] = float(canonical.STAMINA_CAPACITY)
    return out


def _load_explicit_post() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not RICCI_KLINE_EXPLICIT.exists():
        return out
    df = pd.read_csv(RICCI_KLINE_EXPLICIT)
    required = {"fight_id", "fighter_id", "trait", "post_fsr"}
    if not required.issubset(df.columns):
        return out
    for (fight_id, fighter_id), group in df.groupby(["fight_id", "fighter_id"], sort=False):
        vals: dict[str, float] = {}
        for row in group.itertuples(index=False):
            trait = str(row.trait)
            if trait in canonical.CANONICAL_RATINGS:
                vals[trait] = float(row.post_fsr)
        if set(vals) == set(canonical.CANONICAL_RATINGS):
            out[(str(fight_id), str(fighter_id))] = vals
    return out


def _next_snapshot(
    fsr: pd.DataFrame,
    fighter_id: str,
    target_date: pd.Timestamp,
    target_fight_id: str,
) -> pd.Series | None:
    rows = fsr.loc[fsr["fighter_id"].eq(str(fighter_id))].copy()
    rows = rows.loc[~rows["fight_id"].eq(str(target_fight_id))]
    rows = rows.loc[rows["_date"] > target_date]
    if rows.empty:
        return None
    rows = rows.sort_values(["_date", "fight_id"])
    return rows.iloc[0].copy()


def _run_arm(
    red: pd.Series,
    blue: pd.Series,
    *,
    seeds: np.ndarray,
    red_age: float | None,
    blue_age: float | None,
) -> dict[str, float]:
    winners = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winners[int(path.winner)] += 1
        methods[str(path.method)] += 1
    n = float(len(seeds))
    return {
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    for path in (BASELINE_PATH, FSR32_PATH):
        if not path.exists():
            raise RuntimeError(f"required input not found: {path}")

    print("[zero-replay] loading frozen validation baseline...", flush=True)
    baseline = pd.read_csv(BASELINE_PATH)
    misses = baseline.loc[pd.to_numeric(baseline["mc_correct"], errors="coerce").eq(0)].copy()
    print(f"[zero-replay] missed bouts: {len(misses):,}", flush=True)

    print("[zero-replay] loading existing FSR-32 snapshots...", flush=True)
    fsr = pd.read_parquet(FSR32_PATH)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    dcol = _date_col(fsr)
    fsr["_date"] = pd.to_datetime(fsr[dcol], errors="raise")
    print(f"[zero-replay] FSR-32 rows: {len(fsr):,}", flush=True)

    print("[zero-replay] loading aligned historical cohort once...", flush=True)
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    bout_meta = cohort.set_index("bout_id", drop=False)

    explicit = _load_explicit_post()
    print(f"[zero-replay] explicit post-fight fighter states available: {len(explicit):,}", flush=True)

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    results: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []

    total = len(misses)
    for idx, miss in enumerate(misses.itertuples(index=False), start=1):
        bout_id = str(miss.bout_id)
        print(f"\n[zero-replay] bout {idx}/{total} | {bout_id} | {miss.red} vs {miss.blue}", flush=True)
        if bout_id not in pairs or bout_id not in bout_meta.index:
            unavailable.append({"bout_id": bout_id, "reason": "not in aligned cohort"})
            print("  unavailable: not in aligned cohort", flush=True)
            continue

        bout = bout_meta.loc[bout_id]
        pre_red, pre_blue = pairs[bout_id]
        target_date = pd.Timestamp(pd.to_datetime(miss.event_date))
        red_id = str(pre_red["fighter_id"])
        blue_id = str(pre_blue["fighter_id"])

        sources: dict[str, str] = {}
        post_profiles: list[pd.Series] = []
        missing_side = None
        for side, fighter_id, pre_profile in (
            ("red", red_id, pre_red),
            ("blue", blue_id, pre_blue),
        ):
            next_row = _next_snapshot(fsr, fighter_id, target_date, bout_id)
            if next_row is not None:
                post_profiles.append(next_row)
                sources[side] = f"next_prefight:{next_row['fight_id']}@{next_row['_date'].date()}"
                continue
            vals = explicit.get((bout_id, fighter_id))
            if vals is not None:
                post_profiles.append(_overlay_canonical(pre_profile, vals))
                sources[side] = "explicit_postfight_csv"
                continue
            missing_side = side
            break

        if missing_side is not None or len(post_profiles) != 2:
            reason = f"no later snapshot/explicit post state for {missing_side or 'fighter'}"
            unavailable.append({
                "bout_id": bout_id,
                "red": miss.red,
                "blue": miss.blue,
                "actual_winner": miss.actual_winner,
                "reason": reason,
            })
            print(f"  unavailable: {reason}", flush=True)
            continue

        post_red, post_blue = post_profiles
        red_age = historical._age(bout, "r_age")
        blue_age = historical._age(bout, "b_age")

        print(f"  red post source : {sources['red']}", flush=True)
        print(f"  blue post source: {sources['blue']}", flush=True)
        print(f"  running PRE {args.paths:,} paths...", flush=True)
        pre = _run_arm(pre_red, pre_blue, seeds=seeds, red_age=red_age, blue_age=blue_age)
        print(f"  running POST {args.paths:,} paths...", flush=True)
        post = _run_arm(post_red, post_blue, seeds=seeds, red_age=red_age, blue_age=blue_age)

        actual = str(miss.actual_winner)
        red_name = str(miss.red)
        blue_name = str(miss.blue)
        if actual == red_name:
            pre_actual = pre["p_red_win"]
            post_actual = post["p_red_win"]
            post_favorite = red_name if post["p_red_win"] >= 0.5 else blue_name
        elif actual == blue_name:
            pre_actual = pre["p_blue_win"]
            post_actual = post["p_blue_win"]
            post_favorite = blue_name if post["p_blue_win"] >= 0.5 else red_name
        else:
            raise RuntimeError(f"actual winner {actual!r} does not match red/blue for {bout_id}")

        delta = post_actual - pre_actual
        flipped = post_favorite == actual
        print(
            f"  ACTUAL {actual}: PRE {pre_actual:.1%} -> POST {post_actual:.1%} "
            f"({100*delta:+.1f} pp) | flipped={flipped}",
            flush=True,
        )

        results.append({
            "bout_id": bout_id,
            "event_date": miss.event_date,
            "red": red_name,
            "blue": blue_name,
            "actual_winner": actual,
            "actual_method": miss.actual_method,
            "red_age": red_age,
            "blue_age": blue_age,
            "red_prior_ufc_fights": pre_red.get("prior_ufc_fights", np.nan),
            "blue_prior_ufc_fights": pre_blue.get("prior_ufc_fights", np.nan),
            "red_post_source": sources["red"],
            "blue_post_source": sources["blue"],
            "pre_p_red_win": pre["p_red_win"],
            "pre_p_blue_win": pre["p_blue_win"],
            "post_p_red_win": post["p_red_win"],
            "post_p_blue_win": post["p_blue_win"],
            "pre_p_actual_winner": pre_actual,
            "post_p_actual_winner": post_actual,
            "actual_winner_delta": delta,
            "moved_toward_actual": delta > 0.0,
            "post_flipped_to_actual": flipped,
            "pre_p_ko_tko": pre["p_ko_tko"],
            "post_p_ko_tko": post["p_ko_tko"],
            "pre_p_sub": pre["p_sub"],
            "post_p_sub": post["p_sub"],
            "pre_p_dec": pre["p_dec"],
            "post_p_dec": post["p_dec"],
        })

    out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.DataFrame(unavailable).to_csv(UNAVAILABLE_PATH, index=False)

    print("\n" + "=" * 100)
    print("ZERO-REPLAY VALIDATION MISS SUMMARY")
    print("=" * 100)
    print(f"available bouts:   {len(out)}/{len(misses)}")
    print(f"unavailable bouts: {len(unavailable)}/{len(misses)}")
    if not out.empty:
        moved = int(out["moved_toward_actual"].sum())
        flipped = int(out["post_flipped_to_actual"].sum())
        print(f"moved toward actual winner: {moved}/{len(out)} ({moved/len(out):.1%})")
        print(f"flipped to actual winner:   {flipped}/{len(out)} ({flipped/len(out):.1%})")
        print(f"mean actual-winner change:   {100*out['actual_winner_delta'].mean():+.2f} pp")
        print(f"median actual-winner change: {100*out['actual_winner_delta'].median():+.2f} pp")
        compact = out[[
            "red", "blue", "actual_winner", "pre_p_actual_winner",
            "post_p_actual_winner", "actual_winner_delta",
            "post_flipped_to_actual",
        ]].copy()
        for col in ("pre_p_actual_winner", "post_p_actual_winner"):
            compact[col] = compact[col].map(lambda x: f"{x:.1%}")
        compact["actual_winner_delta"] = compact["actual_winner_delta"].map(lambda x: f"{100*x:+.1f} pp")
        print("\n" + compact.to_string(index=False))
    print(f"\nwrote: {args.output}")
    print(f"unavailable: {UNAVAILABLE_PATH}")
    print("No FSR replay was performed.")


if __name__ == "__main__":
    main()
