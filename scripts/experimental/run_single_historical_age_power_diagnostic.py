"""Run one historical FSR-32 bout with the uniform physical age experiment.

Configuration:
- stored prefight FSR; no trajectory
- no YAML age configuration
- age <= 30: no adjustment
- age > 30: -2 FSR points/year after 30
- identical rule applied to striking_power, knockdown_resistance, and
  damage_durability

The diagnostic aggregates path-level offense and phase statistics so a simulated
matchup can be compared with the historical fight structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_2026_baseline_age_power_same_decay as age_power

DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
OUT_DIR = Path("data/experimental/single_historical_age_power_diagnostic")


def _norm_name(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _age(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _select_bout(red_query: str, blue_query: str, date_query: str | None):
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)

    date_col = next((c for c in ("event_date", "fight_date", "date") if c in cohort.columns), None)
    if date_col is not None:
        cohort["_date"] = pd.to_datetime(cohort[date_col], errors="coerce").dt.normalize()

    rq = _norm_name(red_query)
    bq = _norm_name(blue_query)
    matches: list[tuple[pd.Series, pd.Series, pd.Series]] = []
    for _, row in cohort.iterrows():
        bid = str(row["bout_id"])
        pair = pairs.get(bid)
        if pair is None:
            continue
        red, blue = pair
        rn = _norm_name(base._display_name(red))
        bn = _norm_name(base._display_name(blue))
        names_match = (rn == rq and bn == bq) or (rn == bq and bn == rq)
        if not names_match:
            continue
        if date_query is not None:
            if "_date" not in row.index or pd.isna(row["_date"]):
                continue
            if pd.Timestamp(row["_date"]).normalize() != pd.Timestamp(date_query).normalize():
                continue
        matches.append((row, red, blue))

    if not matches:
        raise ValueError(f"No aligned mature bout found for {red_query!r} vs {blue_query!r} date={date_query!r}")
    if len(matches) > 1:
        detail = []
        for row, red, blue in matches:
            d = row.get("_date", "unknown")
            detail.append(f"{row['bout_id']} {d} {base._display_name(red)} vs {base._display_name(blue)}")
        raise ValueError("Multiple matching bouts; supply --date:\n  " + "\n  ".join(detail))
    return matches[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--red", required=True)
    p.add_argument("--blue", required=True)
    p.add_argument("--date")
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    bout, red, blue = _select_bout(args.red, args.blue, args.date)
    bid = str(bout["bout_id"])
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)
    red_age = _age(bout, "r_age")
    blue_age = _age(bout, "b_age")

    age_power._install_uniform_physical_age_layer()
    red_power, red_mod = age_power._apply_same_physical_age_decay(red, red_age, enabled=True)
    blue_power, blue_mod = age_power._apply_same_physical_age_decay(blue, blue_age, enabled=True)
    red_effective, red_other = age_power.runner.age_modifiers.apply_age_modifiers(red_power, red_age)
    blue_effective, blue_other = age_power.runner.age_modifiers.apply_age_modifiers(blue_power, blue_age)

    print("=" * 142)
    print("SINGLE HISTORICAL UNIFORM PHYSICAL AGE DIAGNOSTIC")
    print("=" * 142)
    print(f"bout_id: {bid}")
    print(f"fight: {red_name} vs {blue_name}")
    print(f"date: {bout.get('_date', args.date)}")
    print(f"ages: {red_age} / {blue_age}")
    print(f"paths: {args.paths:,}")
    print("contract: stored prefight FSR | no trajectory | NO YAML | all 3 physical traits 0<=30 then -2/year")
    print(f"uniform age modifiers: {red_name} {red_mod:+.2f} | {blue_name} {blue_mod:+.2f}")

    physical = ["striking_power", "knockdown_resistance", "damage_durability"]
    print("\nPHYSICAL FSR: STORED -> FIGHT-NIGHT EFFECTIVE")
    for trait in physical:
        print(
            f"{trait:24s} | {red_name}: {float(red[trait]):6.2f}->{float(red_effective[trait]):6.2f} | "
            f"{blue_name}: {float(blue[trait]):6.2f}->{float(blue_effective[trait]):6.2f}"
        )

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    winner_counts = [0, 0]
    method_counts = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    values: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        values.setdefault(key, []).append(float(value))

    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(
            red_power,
            blue_power,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winner_counts[int(path.winner)] += 1
        method_counts[path.method] += 1

        for i, side in ((0, "red"), (1, "blue")):
            s = sim.stats[i]
            add(f"{side}_sig_att", getattr(s, "sig_att", 0))
            add(f"{side}_sig_landed", getattr(s, "sig_landed", 0))
            add(f"{side}_td_att", getattr(s, "td_att", 0))
            add(f"{side}_td_landed", getattr(s, "td_landed", 0))
            add(f"{side}_control_seconds", getattr(s, "control_seconds", 0))
            add(f"{side}_ground_control_seconds", getattr(s, "ground_control_seconds", 0))
            add(f"{side}_clinch_control_seconds", getattr(s, "clinch_control_seconds", 0))
            add(f"{side}_sub_att", getattr(s, "sub_att", 0))
            add(f"{side}_reversals", getattr(s, "reversals", 0))
            add(f"{side}_knockdowns", getattr(s, "knockdowns_scored", 0))

        ps = getattr(sim.stats[0], "phase_segments", {})
        add("distance_segments", ps.get("DISTANCE", 0))
        add("clinch_segments", ps.get("CLINCH", 0))
        add("ground_segments", ps.get("GROUND", 0))

    n = float(args.paths)
    print("\nOUTCOME")
    print(f"{red_name}: {winner_counts[0] / n:.1%}")
    print(f"{blue_name}: {winner_counts[1] / n:.1%}")
    print(
        f"KO/SUB/DEC: {method_counts['KO/TKO'] / n:.1%} / "
        f"{method_counts['SUB'] / n:.1%} / {method_counts['DEC'] / n:.1%}"
    )

    print("\nSIMULATED PER-FIGHT AVERAGES")
    print(f"{'metric':28s} {red_name:>22s} {blue_name:>22s}")
    print("-" * 78)
    for label, key in (
        ("significant attempts", "sig_att"),
        ("significant landed", "sig_landed"),
        ("takedown attempts", "td_att"),
        ("takedowns landed", "td_landed"),
        ("control seconds", "control_seconds"),
        ("ground control seconds", "ground_control_seconds"),
        ("clinch control seconds", "clinch_control_seconds"),
        ("submission attempts", "sub_att"),
        ("reversals", "reversals"),
        ("knockdowns", "knockdowns"),
    ):
        print(f"{label:28s} {_mean(values['red_' + key]):22.2f} {_mean(values['blue_' + key]):22.2f}")

    print("\nAVERAGE PHASE TIME")
    for phase, key in (("DISTANCE", "distance_segments"), ("CLINCH", "clinch_segments"), ("GROUND", "ground_segments")):
        segs = _mean(values[key])
        print(f"{phase:8s}: {segs:6.2f} segments = {segs * 10.0:7.1f} sec")

    row = {
        "bout_id": bid,
        "red": red_name,
        "blue": blue_name,
        "red_age": red_age,
        "blue_age": blue_age,
        "paths": args.paths,
        "p_red_win": winner_counts[0] / n,
        "p_blue_win": winner_counts[1] / n,
        "p_ko": method_counts["KO/TKO"] / n,
        "p_sub": method_counts["SUB"] / n,
        "p_dec": method_counts["DEC"] / n,
        "red_uniform_age_modifier": red_mod,
        "blue_uniform_age_modifier": blue_mod,
    }
    for key, vals in values.items():
        row[f"mean_{key}"] = _mean(vals)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{bid}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
