"""Batch diagnostic: rerun every saved MC winner miss with observed post-fight FSR.

This is intentionally leaky/counterfactual.  The goal is not to claim a better
historical prediction; it is to measure whether the actual fight's FSR update
moves the unchanged simulator toward the fighter who actually won.

The expensive canonical replay happens ONCE for the whole miss cohort.  The
resulting pre/post trait artifact is persisted and reused on later invocations,
so repeated MC audits do not replay FSR history.

Post-fight extraction strategy
------------------------------
* If a fighter has a later real UFC snapshot, that next pre-fight snapshot is
  the stored state after the target fight (ratings do not change between UFC
  fights under the current stored-FSR contract).
* If the target is that fighter's latest UFC fight, one synthetic sentinel bout
  is appended after the end of the real dataset.  The sentinel pre-fight state
  therefore snapshots the fighter after all real evidence, including the
  target fight.  Sentinels are used only for terminal target fighters and are
  all placed after real history, so their copied evidence cannot contaminate
  any real historical snapshot.

Research/shadow only.  No production artifacts are modified.
"""
from __future__ import annotations

import argparse
import time
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
OUTPUT_DIR = Path("data/experimental/validation_misses_pre_post_fsr")
PRE_POST_TRAITS_PATH = OUTPUT_DIR / "validation_misses_pre_post_traits.csv"
PRE_POST_WIDE_PATH = OUTPUT_DIR / "validation_misses_pre_post_wide.parquet"
MC_RESULTS_PATH = OUTPUT_DIR / "validation_misses_pre_post_mc.csv"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
SENTINEL_PREFIX = "__validation_postfight__"


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def _load_misses() -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise RuntimeError(f"saved validation baseline not found: {BASELINE_PATH}")
    baseline = pd.read_csv(BASELINE_PATH)
    required = {
        "bout_id", "red", "blue", "actual_winner", "actual_method",
        "p_red_win", "p_blue_win", "mc_correct", "card_no", "event_date",
    }
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise RuntimeError(f"validation baseline missing columns: {missing}")
    misses = baseline.loc[pd.to_numeric(baseline["mc_correct"], errors="coerce").eq(0)].copy()
    misses["bout_id"] = misses["bout_id"].astype(str)
    misses["event_date"] = pd.to_datetime(misses["event_date"], errors="raise")
    if len(misses) != 16:
        raise RuntimeError(f"expected 16 saved MC winner misses, found {len(misses)}")
    return misses.reset_index(drop=True)


def _sentinel_id(fight_id: str) -> str:
    return f"{SENTINEL_PREFIX}{fight_id}"


def _augment_with_terminal_sentinels(
    rfs: pd.DataFrame,
    rounds: pd.DataFrame,
    master: pd.DataFrame,
    misses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    target_ids = set(misses["bout_id"].astype(str))
    target_rfs = rfs.loc[rfs["fight_id"].astype(str).isin(target_ids)].copy()
    counts = target_rfs.groupby("fight_id").size()
    bad = counts[counts.ne(2)]
    if not bad.empty:
        raise RuntimeError(f"target fights must have exactly two RFS rows: {bad.to_dict()}")

    latest_date = rfs.groupby(rfs["fighter_id"].astype(str))["date"].max()
    terminal_target_fights: set[str] = set()
    for row in target_rfs.itertuples(index=False):
        fighter_id = str(row.fighter_id)
        if pd.Timestamp(row.date) == pd.Timestamp(latest_date.loc[fighter_id]):
            terminal_target_fights.add(str(row.fight_id))

    if not terminal_target_fights:
        return rfs, rounds, master, set()

    # All sentinels share one date strictly after all real history.  Same-date
    # family contracts snapshot before sentinel updates.  Assert a fighter is
    # not present in two sentinel bouts to keep fresh-power ordering harmless.
    sentinel_date = pd.Timestamp(rfs["date"].max()) + pd.Timedelta(nanoseconds=1)
    sentinel_rfs_parts: list[pd.DataFrame] = []
    sentinel_round_parts: list[pd.DataFrame] = []
    sentinel_master_parts: list[pd.DataFrame] = []
    sentinel_fighters: list[str] = []
    round_date_col = canonical._date_column(rounds)
    master_date_col = canonical._date_column(master)

    for fight_id in sorted(terminal_target_fights):
        sid = _sentinel_id(fight_id)

        rf = rfs.loc[rfs["fight_id"].eq(fight_id)].copy()
        sentinel_fighters.extend(rf["fighter_id"].astype(str).tolist())
        rf["fight_id"] = sid
        rf["date"] = sentinel_date
        if "event_date" in rf.columns:
            rf["event_date"] = sentinel_date
        sentinel_rfs_parts.append(rf)

        rr = rounds.loc[rounds["fight_id"].eq(fight_id)].copy()
        if rr.empty:
            raise RuntimeError(f"target {fight_id} has no round rows for sentinel")
        rr["fight_id"] = sid
        rr[round_date_col] = sentinel_date
        if "date" in rr.columns:
            rr["date"] = sentinel_date
        if "event_date" in rr.columns:
            rr["event_date"] = sentinel_date
        sentinel_round_parts.append(rr)

        mm = master.loc[master["fight_id"].eq(fight_id)].copy()
        if len(mm) != 1:
            raise RuntimeError(f"target {fight_id} expected one master row, found {len(mm)}")
        mm["fight_id"] = sid
        mm[master_date_col] = sentinel_date
        if "date" in mm.columns:
            mm["date"] = sentinel_date
        sentinel_master_parts.append(mm)

    dupes = pd.Series(sentinel_fighters).value_counts()
    dupes = dupes[dupes.gt(1)]
    if not dupes.empty:
        raise RuntimeError(
            "a fighter appears in multiple terminal sentinels; batch sentinel ordering would "
            f"be ambiguous: {dupes.to_dict()}"
        )

    rfs_work = pd.concat([rfs, *sentinel_rfs_parts], ignore_index=True)
    rounds_work = pd.concat([rounds, *sentinel_round_parts], ignore_index=True)
    master_work = pd.concat([master, *sentinel_master_parts], ignore_index=True)
    return rfs_work, rounds_work, master_work, terminal_target_fights


def _extract_pre_post_from_replay(
    replay: pd.DataFrame,
    misses: pd.DataFrame,
    terminal_target_fights: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = replay.loc[~replay["fight_id"].astype(str).str.startswith(SENTINEL_PREFIX)].copy()
    real["date"] = pd.to_datetime(real["date"], errors="raise")

    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    for miss in misses.itertuples(index=False):
        fight_id = str(miss.bout_id)
        target = real.loc[real["fight_id"].eq(fight_id)].copy()
        if len(target) != 2:
            raise RuntimeError(f"fight {fight_id}: expected two canonical pre rows, found {len(target)}")
        target_date = pd.Timestamp(target["date"].iloc[0])

        for pre_row in target.itertuples(index=False):
            fighter_id = str(pre_row.fighter_id)
            later = real.loc[
                real["fighter_id"].astype(str).eq(fighter_id)
                & real["date"].gt(target_date)
            ].sort_values(["date", "fight_id"])

            if not later.empty:
                post_series = later.iloc[0]
                post_source = "next_real_prefight"
            else:
                sid = _sentinel_id(fight_id)
                post = replay.loc[
                    replay["fight_id"].eq(sid)
                    & replay["fighter_id"].astype(str).eq(fighter_id)
                ]
                if len(post) != 1:
                    raise RuntimeError(
                        f"fight {fight_id} fighter {fighter_id}: terminal state needs one sentinel; "
                        f"found {len(post)}"
                    )
                post_series = post.iloc[0]
                post_source = "terminal_sentinel"

            wide: dict[str, object] = {
                "fight_id": fight_id,
                "event_date": target_date,
                "card_no": int(miss.card_no),
                "fighter_id": fighter_id,
                "fighter_name": str(pre_row.fighter_name),
                "actual_winner": str(miss.actual_winner),
                "actual_method": str(miss.actual_method),
                "prior_ufc_fights_pre": int(pre_row.prior_ufc_fights),
                "post_source": post_source,
            }
            for trait in canonical.CANONICAL_RATINGS:
                pre_value = float(getattr(pre_row, trait))
                post_value = float(post_series[trait])
                wide[f"{trait}_pre"] = pre_value
                wide[f"{trait}_post"] = post_value
                wide[f"{trait}_delta"] = post_value - pre_value
                long_rows.append({
                    "fight_id": fight_id,
                    "event_date": target_date,
                    "fighter_id": fighter_id,
                    "fighter_name": str(pre_row.fighter_name),
                    "trait": trait,
                    "pre_fsr": pre_value,
                    "post_fsr": post_value,
                    "delta": post_value - pre_value,
                    "post_source": post_source,
                })
            wide_rows.append(wide)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    if len(wide_df) != 32:
        raise RuntimeError(f"expected 32 fighter-fight pre/post rows, found {len(wide_df)}")
    return wide_df, long_df


def _build_or_load_pre_post(misses: pd.DataFrame, rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PRE_POST_WIDE_PATH.exists() and PRE_POST_TRAITS_PATH.exists() and not rebuild:
        print(f"[batch] reusing persisted pre/post states: {PRE_POST_WIDE_PATH}", flush=True)
        return pd.read_parquet(PRE_POST_WIDE_PATH), pd.read_csv(PRE_POST_TRAITS_PATH)

    print("[batch] no reusable pre/post artifact; starting ONE canonical historical replay", flush=True)
    rfs, rounds, master = canonical._load_inputs()
    rfs_work, rounds_work, master_work, terminal_fights = _augment_with_terminal_sentinels(
        rfs, rounds, master, misses
    )
    print(
        f"[batch] misses={len(misses)} | terminal-sentinel fights={len(terminal_fights)} | "
        f"augmented rfs={len(rfs_work):,}",
        flush=True,
    )
    replay = canonical.build_canonical_prefight(rfs_work, rounds_work, master_work, progress=True)
    wide, long = _extract_pre_post_from_replay(replay, misses, terminal_fights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(PRE_POST_WIDE_PATH, index=False)
    long.to_csv(PRE_POST_TRAITS_PATH, index=False)
    print(f"[batch] wrote reusable pre/post wide: {PRE_POST_WIDE_PATH}", flush=True)
    print(f"[batch] wrote reusable pre/post traits: {PRE_POST_TRAITS_PATH}", flush=True)
    return wide, long


def _overlay(profile: pd.Series, row: pd.Series, suffix: str) -> pd.Series:
    out = profile.copy()
    for trait in canonical.CANONICAL_RATINGS:
        out[trait] = float(row[f"{trait}_{suffix}"])
    out["distance_precision"] = float(out["distance_striking_precision"])
    out["distance_defense"] = float(out["distance_striking_defense"])
    out["stamina_depletion_resistance"] = float(out["fatigue_accumulation_resistance"])
    out["stamina_performance_resilience"] = float(out["fatigue_performance_resilience"])
    out["stamina_capacity"] = float(canonical.STAMINA_CAPACITY)
    return out


def _run_arm(
    red: pd.Series,
    blue: pd.Series,
    seeds: np.ndarray,
    red_age: float | None,
    blue_age: float | None,
    label: str,
) -> dict[str, float]:
    winners = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    heartbeat = max(1, len(seeds) // 5)
    for i, seed in enumerate(seeds, start=1):
        path = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        ).run()
        winners[int(path.winner)] += 1
        methods[str(path.method)] += 1
        if i % heartbeat == 0 or i == len(seeds):
            print(f"    [{label}] {i:,}/{len(seeds):,} ({100*i/len(seeds):.0f}%)", flush=True)
    n = float(len(seeds))
    return {
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
    }


def _days_since_previous_fight(replay_wide: pd.DataFrame, fighter_id: str, fight_id: str) -> float:
    # This helper receives only target rows, so layoff is added later from raw RFS if desired.
    return float("nan")


def _run_mc(misses: pd.DataFrame, wide: pd.DataFrame, paths: int, seed: int) -> pd.DataFrame:
    start = time.perf_counter()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort_by_id = cohort.set_index("bout_id", drop=False)

    rows: list[dict[str, object]] = []
    for idx, miss in enumerate(misses.itertuples(index=False), start=1):
        fight_id = str(miss.bout_id)
        if fight_id not in pairs or fight_id not in cohort_by_id.index:
            raise RuntimeError(f"missed fight {fight_id} absent from aligned historical cohort")
        bout = cohort_by_id.loc[fight_id]
        original_red, original_blue = pairs[fight_id]
        red_id = str(original_red["fighter_id"])
        blue_id = str(original_blue["fighter_id"])
        fight_states = wide.loc[wide["fight_id"].astype(str).eq(fight_id)].set_index("fighter_id")
        if red_id not in fight_states.index or blue_id not in fight_states.index:
            raise RuntimeError(f"fight {fight_id}: canonical fighter IDs do not match MC pair")

        pre_red = _overlay(original_red, fight_states.loc[red_id], "pre")
        pre_blue = _overlay(original_blue, fight_states.loc[blue_id], "pre")
        post_red = _overlay(original_red, fight_states.loc[red_id], "post")
        post_blue = _overlay(original_blue, fight_states.loc[blue_id], "post")
        red_name = base._display_name(original_red)
        blue_name = base._display_name(original_blue)
        red_age = historical._age(bout, "r_age")
        blue_age = historical._age(bout, "b_age")

        # Same exact seed vector in PRE and POST arms, and same base seed across fights,
        # matching the historical single-bout runner convention.
        rng = np.random.default_rng(seed)
        seeds = rng.integers(1, np.iinfo(np.int32).max, size=paths, dtype=np.int64)

        print(
            f"\n[MC] fight {idx}/{len(misses)} | {red_name} vs {blue_name} | "
            f"actual={miss.actual_winner} | elapsed={_elapsed(start)}",
            flush=True,
        )
        pre = _run_arm(pre_red, pre_blue, seeds, red_age, blue_age, "PRE")
        post = _run_arm(post_red, post_blue, seeds, red_age, blue_age, "POST")

        if str(miss.actual_winner) == red_name:
            actual_side = "red"
            pre_actual = pre["p_red_win"]
            post_actual = post["p_red_win"]
        elif str(miss.actual_winner) == blue_name:
            actual_side = "blue"
            pre_actual = pre["p_blue_win"]
            post_actual = post["p_blue_win"]
        else:
            raise RuntimeError(
                f"fight {fight_id}: actual winner {miss.actual_winner!r} matches neither "
                f"{red_name!r} nor {blue_name!r}"
            )

        delta_actual = post_actual - pre_actual
        flipped = bool(pre_actual < 0.5 and post_actual > 0.5)
        moved_toward = bool(delta_actual > 0.0)
        rows.append({
            "fight_id": fight_id,
            "card_no": int(miss.card_no),
            "event_date": pd.Timestamp(miss.event_date),
            "red": red_name,
            "blue": blue_name,
            "actual_winner": str(miss.actual_winner),
            "actual_side": actual_side,
            "actual_method": str(miss.actual_method),
            "red_age": red_age,
            "blue_age": blue_age,
            "red_prior_ufc_fights": int(fight_states.loc[red_id, "prior_ufc_fights_pre"]),
            "blue_prior_ufc_fights": int(fight_states.loc[blue_id, "prior_ufc_fights_pre"]),
            "baseline_p_red_win": float(miss.p_red_win),
            "baseline_p_blue_win": float(miss.p_blue_win),
            "pre_p_red_win": pre["p_red_win"],
            "pre_p_blue_win": pre["p_blue_win"],
            "post_p_red_win": post["p_red_win"],
            "post_p_blue_win": post["p_blue_win"],
            "pre_p_actual_winner": pre_actual,
            "post_p_actual_winner": post_actual,
            "delta_p_actual_winner": delta_actual,
            "moved_toward_actual": moved_toward,
            "flipped_to_actual_winner": flipped,
            "pre_p_ko_tko": pre["p_ko_tko"],
            "post_p_ko_tko": post["p_ko_tko"],
            "pre_p_sub": pre["p_sub"],
            "post_p_sub": post["p_sub"],
            "pre_p_dec": pre["p_dec"],
            "post_p_dec": post["p_dec"],
        })

        print(
            f"    RESULT actual-winner P: {pre_actual:.1%} -> {post_actual:.1%} "
            f"({100*delta_actual:+.1f} pp) | toward={moved_toward} | flip={flipped}",
            flush=True,
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rebuild-pre-post",
        action="store_true",
        help="Force the expensive one-time canonical replay even if reusable pre/post artifacts exist.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build/reuse the 16-fight pre/post artifact but do not run Monte Carlo.",
    )
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    misses = _load_misses()
    print("=" * 108)
    print("VALIDATION MISSES — PRE-FIGHT VS OBSERVED POST-FIGHT FSR MONTE CARLO")
    print("=" * 108)
    print(f"saved winner misses: {len(misses)}")
    print("POST arm is intentionally leaky/counterfactual; diagnostic use only.")
    print("A reusable pre/post artifact prevents repeat historical FSR replays.\n")

    wide, _ = _build_or_load_pre_post(misses, rebuild=args.rebuild_pre_post)
    if args.build_only:
        return

    results = _run_mc(misses, wide, args.paths, args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(MC_RESULTS_PATH, index=False)

    moved = int(results["moved_toward_actual"].sum())
    flipped = int(results["flipped_to_actual_winner"].sum())
    mean_delta = float(results["delta_p_actual_winner"].mean())
    median_delta = float(results["delta_p_actual_winner"].median())

    print("\n" + "=" * 108)
    print("SUMMARY")
    print("=" * 108)
    print(f"moved toward actual winner : {moved}/{len(results)} ({moved/len(results):.1%})")
    print(f"flipped to actual winner   : {flipped}/{len(results)} ({flipped/len(results):.1%})")
    print(f"mean actual-winner change  : {100*mean_delta:+.1f} pp")
    print(f"median actual-winner change: {100*median_delta:+.1f} pp")

    display = results[[
        "red", "blue", "actual_winner", "pre_p_actual_winner", "post_p_actual_winner",
        "delta_p_actual_winner", "moved_toward_actual", "flipped_to_actual_winner",
    ]].copy()
    for col in ("pre_p_actual_winner", "post_p_actual_winner"):
        display[col] = display[col].map(lambda x: f"{x:.1%}")
    display["delta_p_actual_winner"] = display["delta_p_actual_winner"].map(
        lambda x: f"{100*x:+.1f} pp"
    )
    print("\n" + display.to_string(index=False))
    print(f"\nwrote: {MC_RESULTS_PATH}")
    print(f"reusable pre/post states: {PRE_POST_WIDE_PATH}")


if __name__ == "__main__":
    main()
