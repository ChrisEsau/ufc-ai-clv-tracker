"""Decision-only historical stat audit for FSR Static MC V0.

V0 does not yet model finishes, so its cleanest historical validation cohort is
fights that completed all scheduled rounds. This audit uses leakage-safe pre-fight
FSR-26 snapshots, simulates each historical matchup repeatedly for its full 3- or
5-round duration, and compares mean simulated fighter totals with realized RFS
fighter-fight totals.

Control accounting
------------------
V0 records clinch control and ground control separately. UFCStats provides one
combined CTRL total, so only simulated total control (clinch + ground) is scored
against the historical target. The two simulated components are printed as
unscored diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental.fsr_static_mc_v0 import StaticFSRMCV0

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_v0_decision_stats_audit.parquet"
)

RFS = {
    "rounds": "rfs_finish_state_fight_rounds_observed",
    "sig_att": "rfs_finish_state_fight_sig_strike_attempts",
    "sig_landed": "rfs_finish_state_fight_sig_strikes_landed",
    "td_att": "rfs_phase_interact_fight_td_attempts",
    "td_landed": "rfs_finish_state_fight_takedowns_landed",
    "control_seconds": "rfs_finish_state_fight_control_seconds",
    "sub_att": "rfs_finish_state_fight_submission_attempts",
    "reversals": "rfs_phase_interact_fight_reversals",
    "ko_loss": "rfs_finish_state_fight_ko_tko_loss_indicator",
    "sub_loss": "rfs_finish_state_fight_submission_loss_indicator",
}


def _safe_spearman(frame: pd.DataFrame, a: str, b: str) -> float:
    x = pd.to_numeric(frame[a], errors="coerce")
    y = pd.to_numeric(frame[b], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"a": x[mask], "b": y[mask]}).corr(method="spearman").iloc[0, 1])


def _load_frames(fsr_path: Path, rfs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[decision audit] loading FSR-26 from {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path).copy()
    print(f"[decision audit] loaded {len(fsr):,} FSR rows", flush=True)
    print(f"[decision audit] loading RFS history from {rfs_path}", flush=True)
    rfs = pd.read_parquet(rfs_path).copy()
    print(f"[decision audit] loaded {len(rfs):,} RFS rows", flush=True)
    for frame in (fsr, rfs):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)
    missing = [column for column in RFS.values() if column not in rfs.columns]
    if missing:
        raise ValueError(f"RFS history missing required decision-audit columns: {missing}")
    return fsr, rfs


def _decision_fight_table(rfs: pd.DataFrame) -> pd.DataFrame:
    cols = ["fight_id", "fighter_id", *RFS.values()]
    work = rfs[cols].copy()
    for column in RFS.values():
        work[column] = pd.to_numeric(work[column], errors="coerce")

    valid_ids: list[str] = []
    rounds_by_fight: dict[str, int] = {}
    for fight_id, fight in work.groupby("fight_id", sort=False):
        if fight["fighter_id"].nunique() != 2 or len(fight) != 2:
            continue
        rounds = fight[RFS["rounds"]].dropna().astype(int).unique()
        if len(rounds) != 1 or int(rounds[0]) not in (3, 5):
            continue
        ko_loss = fight[RFS["ko_loss"]].fillna(0.0)
        sub_loss = fight[RFS["sub_loss"]].fillna(0.0)
        if (ko_loss > 0.5).any() or (sub_loss > 0.5).any():
            continue
        valid_ids.append(str(fight_id))
        rounds_by_fight[str(fight_id)] = int(rounds[0])

    out = work[work["fight_id"].isin(valid_ids)].copy()
    out["scheduled_rounds"] = out["fight_id"].map(rounds_by_fight).astype(int)
    return out


def _simulate_fight(
    pair: pd.DataFrame,
    rounds: int,
    sims_per_fight: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    pair = pair.reset_index(drop=True)
    red = pair.iloc[0]
    blue = pair.iloc[1]
    totals = [
        {
            "sig_att": 0.0,
            "sig_landed": 0.0,
            "td_att": 0.0,
            "td_landed": 0.0,
            "control_seconds": 0.0,
            "clinch_control_seconds": 0.0,
            "ground_control_seconds": 0.0,
            "sub_att": 0.0,
            "reversals": 0.0,
        }
        for _ in range(2)
    ]

    for sim_i in range(sims_per_fight):
        path = StaticFSRMCV0(red, blue, rounds=rounds, seed=seed + sim_i).run()
        for i, stats in enumerate(path.stats):
            totals[i]["sig_att"] += stats.sig_att
            totals[i]["sig_landed"] += stats.sig_landed
            totals[i]["td_att"] += stats.td_att
            totals[i]["td_landed"] += stats.td_landed
            totals[i]["control_seconds"] += stats.control_seconds
            totals[i]["clinch_control_seconds"] += stats.clinch_control_seconds
            totals[i]["ground_control_seconds"] += stats.ground_control_seconds
            totals[i]["sub_att"] += stats.sub_att
            totals[i]["reversals"] += stats.reversals

    rows: list[dict[str, float | str | int]] = []
    for i in range(2):
        row: dict[str, float | str | int] = {
            "fight_id": str(pair.iloc[i]["fight_id"]),
            "fighter_id": str(pair.iloc[i]["fighter_id"]),
            "scheduled_rounds": rounds,
        }
        for metric, value in totals[i].items():
            row[f"sim_{metric}"] = value / sims_per_fight
        row["sim_sig_accuracy"] = (
            row["sim_sig_landed"] / row["sim_sig_att"] if row["sim_sig_att"] else 0.0
        )
        row["sim_td_completion"] = (
            row["sim_td_landed"] / row["sim_td_att"] if row["sim_td_att"] else np.nan
        )
        rows.append(row)
    return rows


def _attach_realized(sim: pd.DataFrame, decision_rows: pd.DataFrame) -> pd.DataFrame:
    realized = decision_rows[["fight_id", "fighter_id", "scheduled_rounds"]].copy()
    for metric in (
        "sig_att", "sig_landed", "td_att", "td_landed",
        "control_seconds", "sub_att", "reversals",
    ):
        realized[f"real_{metric}"] = pd.to_numeric(decision_rows[RFS[metric]], errors="coerce").to_numpy()

    realized["real_sig_accuracy"] = np.divide(
        realized["real_sig_landed"], realized["real_sig_att"],
        out=np.full(len(realized), np.nan),
        where=realized["real_sig_att"].to_numpy(dtype=float) > 0,
    )
    realized["real_td_completion"] = np.divide(
        realized["real_td_landed"], realized["real_td_att"],
        out=np.full(len(realized), np.nan),
        where=realized["real_td_att"].to_numpy(dtype=float) > 0,
    )
    return sim.merge(
        realized,
        on=["fight_id", "fighter_id", "scheduled_rounds"],
        how="inner",
        validate="one_to_one",
    )


def _metric_table(audit: pd.DataFrame, title: str) -> None:
    metrics = [
        ("sig strike attempts", "sig_att"),
        ("sig strikes landed", "sig_landed"),
        ("sig strike accuracy", "sig_accuracy"),
        ("TD attempts", "td_att"),
        ("TD landed", "td_landed"),
        ("TD completion", "td_completion"),
        ("control seconds", "control_seconds"),
        ("submission attempts", "sub_att"),
        ("reversals", "reversals"),
    ]
    rows = []
    for label, metric in metrics:
        sim_col = f"sim_{metric}"
        real_col = f"real_{metric}"
        sim = pd.to_numeric(audit[sim_col], errors="coerce")
        real = pd.to_numeric(audit[real_col], errors="coerce")
        mask = sim.notna() & real.notna()
        if not mask.any():
            continue
        diff = sim[mask] - real[mask]
        rows.append({
            "metric": label,
            "rows": int(mask.sum()),
            "sim_mean": float(sim[mask].mean()),
            "real_mean": float(real[mask].mean()),
            "bias": float(diff.mean()),
            "mae": float(np.abs(diff).mean()),
            "median_ae": float(np.abs(diff).median()),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "spearman": _safe_spearman(audit.loc[mask], sim_col, real_col),
        })
    print(f"\n{title}")
    print("=" * 132)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_control_diagnostics(audit: pd.DataFrame, title: str) -> None:
    print(f"\n{title} — SIMULATED CONTROL COMPONENTS")
    print("-" * 82)
    clinch = pd.to_numeric(audit["sim_clinch_control_seconds"], errors="coerce")
    ground = pd.to_numeric(audit["sim_ground_control_seconds"], errors="coerce")
    total = pd.to_numeric(audit["sim_control_seconds"], errors="coerce")
    real = pd.to_numeric(audit["real_control_seconds"], errors="coerce")
    print(f"sim clinch control mean : {clinch.mean():8.2f}s")
    print(f"sim ground control mean : {ground.mean():8.2f}s")
    print(f"sim total control mean  : {total.mean():8.2f}s")
    print(f"real UFCStats CTRL mean : {real.mean():8.2f}s")
    check = np.nanmax(np.abs((clinch + ground - total).to_numpy(dtype=float)))
    print(f"component sum check     : max |clinch + ground - total| = {check:.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision-only historical stat replay for FSR Static MC V0")
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--rfs-path", type=Path, default=RFS_PATH)
    parser.add_argument("--sims-per-fight", type=int, default=50)
    parser.add_argument("--max-fights", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2901)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.sims_per_fight <= 0 or args.max_fights <= 0:
        raise SystemExit("--sims-per-fight and --max-fights must be positive")

    fsr, rfs = _load_frames(args.fsr_path, args.rfs_path)
    decisions = _decision_fight_table(rfs)
    decision_ids = decisions["fight_id"].drop_duplicates().tolist()
    fsr_counts = fsr.groupby("fight_id")["fighter_id"].nunique()
    valid_fsr = set(fsr_counts[fsr_counts == 2].index.astype(str))
    decision_ids = [fight_id for fight_id in decision_ids if fight_id in valid_fsr]

    if len(decision_ids) > args.max_fights:
        idx = np.linspace(0, len(decision_ids) - 1, args.max_fights, dtype=int)
        decision_ids = [decision_ids[i] for i in idx]

    rounds_map = (
        decisions[["fight_id", "scheduled_rounds"]]
        .drop_duplicates("fight_id")
        .set_index("fight_id")["scheduled_rounds"]
        .to_dict()
    )

    print(
        f"[decision audit] replaying {len(decision_ids):,} full-distance fights x "
        f"{args.sims_per_fight:,} paths = {len(decision_ids) * args.sims_per_fight:,} paths",
        flush=True,
    )

    grouped = fsr.set_index("fight_id", drop=False)
    rows: list[dict[str, float | str | int]] = []
    for fight_i, fight_id in enumerate(decision_ids, 1):
        pair = grouped.loc[[fight_id]].copy()
        if len(pair) != 2:
            continue
        rounds = int(rounds_map[fight_id])
        rows.extend(_simulate_fight(
            pair,
            rounds=rounds,
            sims_per_fight=args.sims_per_fight,
            seed=args.seed + fight_i * 100_003,
        ))
        if fight_i == 1 or fight_i % 25 == 0 or fight_i == len(decision_ids):
            print(f"[decision audit] fight {fight_i:,}/{len(decision_ids):,}", flush=True)

    sim = pd.DataFrame(rows)
    selected_decisions = decisions[decisions["fight_id"].isin(decision_ids)].copy()
    audit = _attach_realized(sim, selected_decisions)

    n3 = audit.loc[audit["scheduled_rounds"] == 3, "fight_id"].nunique()
    n5 = audit.loc[audit["scheduled_rounds"] == 5, "fight_id"].nunique()
    print(f"[decision audit] matched fighter-fight rows: {len(audit):,}", flush=True)
    print(f"[decision audit] fight cohort: {n3:,} three-round decisions; {n5:,} five-round decisions", flush=True)

    _metric_table(audit, "ALL FULL-DISTANCE FIGHTS — FIGHTER TOTALS")
    _print_control_diagnostics(audit, "ALL FULL-DISTANCE FIGHTS")

    if (audit["scheduled_rounds"] == 3).any():
        cohort3 = audit[audit["scheduled_rounds"] == 3]
        _metric_table(cohort3, "THREE-ROUND FULL-DISTANCE FIGHTS")
        _print_control_diagnostics(cohort3, "THREE-ROUND FULL-DISTANCE FIGHTS")
    if (audit["scheduled_rounds"] == 5).any():
        cohort5 = audit[audit["scheduled_rounds"] == 5]
        _metric_table(cohort5, "FIVE-ROUND FULL-DISTANCE FIGHTS")
        _print_control_diagnostics(cohort5, "FIVE-ROUND FULL-DISTANCE FIGHTS")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.output, index=False)
    print(f"\n[decision audit] wrote {args.output}", flush=True)
    print(
        "\nINTERPRETATION: total simulated control = clinch control + ground control and is "
        "the quantity compared with historical UFCStats CTRL. Means/bias test absolute "
        "calibration; MAE/RMSE test fighter-fight error; Spearman tests matchup ranking. "
        "V0 finishes and dynamic state remain disabled."
    )


if __name__ == "__main__":
    main()
