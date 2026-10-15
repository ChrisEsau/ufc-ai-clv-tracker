"""Run locked Damage Reservoir V1 / KD=80 on real historical UFC matchups.

Purpose
-------
Before adding any KO/TKO stoppage mechanics, measure the behavior of the current
reservoir + knockdown layer on real historical bout pairings rather than random
fighter matchups.

This audit samples 300 historical bouts from the leakage-safe FSR-28 pre-fight
snapshot artifact. Each bout is simulated repeatedly with the current
``StaticFSRMCDamageV1`` engine. KO/TKO stoppages are intentionally disabled.

For this checkpoint, a simulated path is classified only as:

- ``KD path``: at least one knockdown occurred;
- ``no-KD path``: no knockdown occurred through the scheduled simulation.

Because KO/TKO is disabled, ``no-KD`` is the useful current proxy for the
user-requested "goes the distance" comparison. The script explicitly labels it
that way so it is not confused with a final production distance prediction.

The audit refuses to create random pairings. A real bout/fight identifier with
exactly two fighter rows is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_damage_v1 as damage


FSR_PATH = damage.FSR_PATH
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_kd_audit.parquet"
)
DEFAULT_BOUTS = 300
DEFAULT_PATHS_PER_BOUT = 20
DEFAULT_SEED = 20260809
DEFAULT_ROUNDS = 3

BOUT_KEY_CANDIDATES = (
    "bout_id",
    "fight_id",
    "ufcstats_fight_id",
    "fight_key",
    "bout_key",
)
ROUND_COLUMN_CANDIDATES = (
    "scheduled_rounds",
    "bout_rounds",
    "num_rounds",
    "rounds_scheduled",
)
DATE_COLUMN_CANDIDATES = (
    "event_date",
    "fight_date",
    "date",
)
NAME_COLUMN_CANDIDATES = (
    "fighter_name",
    "name",
)


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return None


def _resolve_bout_key(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(
                f"Requested --bout-key {explicit!r} not found. "
                f"Available columns include: {sorted(frame.columns)[:80]}"
            )
        return explicit

    key = _first_present(frame.columns, BOUT_KEY_CANDIDATES)
    if key is None:
        raise ValueError(
            "Could not identify a historical bout key in the FSR snapshot. "
            f"Tried {BOUT_KEY_CANDIDATES}. Pass --bout-key explicitly after "
            "inspecting the artifact schema. Random fighter pairing is disabled."
        )
    return key


def _prepare_historical_bouts(
    frame: pd.DataFrame,
    *,
    bout_key: str,
) -> tuple[list[tuple[str, pd.Series, pd.Series]], pd.DataFrame]:
    """Return only real two-fighter historical bouts with complete Damage V1 inputs."""
    required = set(damage.base.REQUIRED_COLUMNS) | damage.REQUIRED_DAMAGE_COLUMNS
    required.add("fighter_id")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FSR snapshot missing Damage V1 required columns: {missing}")

    work = frame.copy()
    work["fighter_id"] = work["fighter_id"].astype(str)
    work[bout_key] = work[bout_key].astype(str)

    # Require one row per fighter within the pre-fight bout snapshot and exactly
    # two distinct fighters. This prevents accidental malformed or random pairs.
    counts = work.groupby(bout_key, sort=False).agg(
        rows=("fighter_id", "size"),
        fighters=("fighter_id", "nunique"),
    )
    valid_keys = counts.index[(counts["rows"] == 2) & (counts["fighters"] == 2)]
    valid = work[work[bout_key].isin(valid_keys)].copy()

    bouts: list[tuple[str, pd.Series, pd.Series]] = []
    for key, group in valid.groupby(bout_key, sort=False):
        group = group.reset_index(drop=True)
        bouts.append((str(key), group.iloc[0], group.iloc[1]))

    return bouts, counts


def _rounds_for_bout(red: pd.Series, blue: pd.Series, round_col: str | None) -> int:
    if round_col is None:
        return DEFAULT_ROUNDS

    values = pd.to_numeric(
        pd.Series([red.get(round_col), blue.get(round_col)]), errors="coerce"
    ).dropna()
    if values.empty:
        return DEFAULT_ROUNDS

    rounds = int(round(values.iloc[0]))
    return rounds if rounds in (3, 5) else DEFAULT_ROUNDS


def _display_name(row: pd.Series, name_col: str | None) -> str:
    if name_col and pd.notna(row.get(name_col)):
        return str(row[name_col])
    return str(row["fighter_id"])


def _run_audit(
    selected: list[tuple[str, pd.Series, pd.Series]],
    *,
    paths_per_bout: int,
    seed: int,
    round_col: str | None,
    date_col: str | None,
    name_col: str | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    total_paths = len(selected) * paths_per_bout
    path_counter = 0

    for bout_index, (bout_id, red, blue) in enumerate(selected, start=1):
        rounds = _rounds_for_bout(red, blue, round_col)
        event_date = red.get(date_col) if date_col else None

        for path_index in range(paths_per_bout):
            path_seed = int(rng.integers(0, 2**31 - 1))
            sim = damage.StaticFSRMCDamageV1(red, blue, rounds=rounds, seed=path_seed)
            sim.run()

            red_kd = int(sim.stats[0].knockdowns_scored)
            blue_kd = int(sim.stats[1].knockdowns_scored)
            total_kd = red_kd + blue_kd
            any_kd = total_kd > 0

            rows.append(
                {
                    "bout_index": bout_index,
                    "bout_id": bout_id,
                    "event_date": event_date,
                    "rounds": rounds,
                    "path_index": path_index,
                    "path_seed": path_seed,
                    "red_fighter_id": str(red["fighter_id"]),
                    "blue_fighter_id": str(blue["fighter_id"]),
                    "red_name": _display_name(red, name_col),
                    "blue_name": _display_name(blue, name_col),
                    "red_knockdowns": red_kd,
                    "blue_knockdowns": blue_kd,
                    "total_knockdowns": total_kd,
                    "any_knockdown": int(any_kd),
                    "path_class": "KD path" if any_kd else "no-KD path",
                    "red_reservoir_fraction_end": sim.damage_state[0].reservoir_fraction,
                    "blue_reservoir_fraction_end": sim.damage_state[1].reservoir_fraction,
                }
            )

            path_counter += 1
            if path_counter % 500 == 0 or path_counter == total_paths:
                print(
                    f"[historical KD audit] paths {path_counter:,}/{total_paths:,}; "
                    f"bouts_started={bout_index:,}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def _print_summary(paths: pd.DataFrame, selected_bouts: int) -> None:
    total_paths = len(paths)
    kd_paths = int(paths["any_knockdown"].sum())
    no_kd_paths = total_paths - kd_paths
    kd_rate = kd_paths / total_paths if total_paths else float("nan")

    bout = (
        paths.groupby("bout_id", as_index=False)
        .agg(
            rounds=("rounds", "first"),
            red_name=("red_name", "first"),
            blue_name=("blue_name", "first"),
            paths=("path_index", "size"),
            p_any_kd=("any_knockdown", "mean"),
            mean_total_kd=("total_knockdowns", "mean"),
            mean_red_res_end=("red_reservoir_fraction_end", "mean"),
            mean_blue_res_end=("blue_reservoir_fraction_end", "mean"),
        )
    )

    expected_kd_bouts = float(bout["p_any_kd"].sum())
    expected_no_kd_bouts = float(selected_bouts - expected_kd_bouts)

    print("\n" + "=" * 120)
    print("HISTORICAL 300-BOUT DAMAGE RESERVOIR / KD=80 AUDIT")
    print("=" * 120)
    print(f"historical bouts sampled: {selected_bouts:,}")
    print(f"MC paths: {total_paths:,}")
    print(f"paths per historical bout: {total_paths // selected_bouts if selected_bouts else 0:,}")
    print(f"KD shock coefficient: {damage.KD_SHOCK_COEFFICIENT:g}")

    print("\nPATH-LEVEL RESULT")
    print(f"KD paths: {kd_paths:,} ({kd_rate:.2%})")
    print(f"no-KD paths: {no_kd_paths:,} ({1.0 - kd_rate:.2%})")
    print(
        "NOTE: KO/TKO is disabled, so 'no-KD path' is the current distance proxy; "
        "this is not yet a final distance prediction."
    )

    print("\nEXPECTED COUNT ACROSS THE 300 HISTORICAL MATCHUPS")
    print(f"expected bouts with >=1 KD: {expected_kd_bouts:.1f} / {selected_bouts}")
    print(f"expected no-KD bouts:       {expected_no_kd_bouts:.1f} / {selected_bouts}")

    print("\nKD COUNT DISTRIBUTION — PATH LEVEL")
    dist = (
        paths["total_knockdowns"]
        .value_counts(normalize=False)
        .sort_index()
        .rename_axis("total_KD")
        .reset_index(name="paths")
    )
    dist["share"] = dist["paths"] / total_paths
    print(dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBY SCHEDULED ROUNDS")
    by_rounds = (
        paths.groupby("rounds", as_index=False)
        .agg(paths=("path_index", "size"), kd_path_rate=("any_knockdown", "mean"))
        .sort_values("rounds")
    )
    print(by_rounds.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nBOUT-LEVEL KD PROBABILITY DISTRIBUTION")
    qs = bout["p_any_kd"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).rename("p_any_kd")
    print(qs.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nRESEARCH BOUNDARY")
    print("- Real historical pairings only; no random fighter pairing.")
    print("- Leakage-safe FSR pre-fight profiles feed the MC.")
    print("- Damage reservoir and locked KD=80 are active.")
    print("- KO/TKO stoppage is disabled for this checkpoint.")
    print("- Do not interpret no-KD as final production goes-distance probability yet.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 300 real historical UFC matchups through Damage V1 / KD=80"
    )
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    parser.add_argument("--paths-per-bout", type=int, default=DEFAULT_PATHS_PER_BOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bout-key", default=None)
    args = parser.parse_args()

    print(f"[historical KD audit] loading {args.fsr_path}", flush=True)
    frame = pd.read_parquet(args.fsr_path)
    print(
        f"[historical KD audit] snapshot rows={len(frame):,}; columns={len(frame.columns):,}",
        flush=True,
    )

    bout_key = _resolve_bout_key(frame, args.bout_key)
    round_col = _first_present(frame.columns, ROUND_COLUMN_CANDIDATES)
    date_col = _first_present(frame.columns, DATE_COLUMN_CANDIDATES)
    name_col = _first_present(frame.columns, NAME_COLUMN_CANDIDATES)

    print(f"[historical KD audit] bout key: {bout_key}", flush=True)
    print(f"[historical KD audit] rounds column: {round_col or 'none -> default 3'}", flush=True)
    print(f"[historical KD audit] date column: {date_col or 'none'}", flush=True)

    bouts, counts = _prepare_historical_bouts(frame, bout_key=bout_key)
    malformed = int(((counts["rows"] != 2) | (counts["fighters"] != 2)).sum())
    print(
        f"[historical KD audit] valid two-fighter historical bouts={len(bouts):,}; "
        f"excluded malformed groups={malformed:,}",
        flush=True,
    )

    if len(bouts) < args.bouts:
        raise ValueError(
            f"Requested {args.bouts} historical bouts but only {len(bouts)} valid "
            "two-fighter pre-fight pairings are available."
        )

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(len(bouts), size=args.bouts, replace=False)
    selected = [bouts[int(i)] for i in selected_indices]

    print(
        f"[historical KD audit] selected={len(selected):,}; "
        f"paths_per_bout={args.paths_per_bout}; "
        f"total_paths={len(selected) * args.paths_per_bout:,}",
        flush=True,
    )

    paths = _run_audit(
        selected,
        paths_per_bout=args.paths_per_bout,
        seed=args.seed + 1,
        round_col=round_col,
        date_col=date_col,
        name_col=name_col,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths.to_parquet(args.output, index=False)
    _print_summary(paths, len(selected))
    print(f"\n[historical KD audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
