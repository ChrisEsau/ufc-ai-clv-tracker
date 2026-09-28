"""Study 9: does post-KD survival need a distinct fighter trait?

Shadow-only research. No simulator or production FSR logic is modified.

Purpose
-------
Determine whether the candidate ``damage_durability`` signal from Study 8
already explains survival after a knockdown, or whether the existing
``chin_resistance`` rating adds meaningful information once durability,
knockdown count, and realized damage exposure are held roughly constant.

Interpretation
--------------
If durability separates KO/TKO survival strongly inside KD-exposed cohorts and
chin_resistance adds little within durability-matched strata, prefer the simpler
ontology and do not create a separate post-KD-survival trait. If chin_resistance
adds stable incremental separation, retain/rebuild a distinct post-KD trait.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DURABILITY_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_damage_durability_study_v0.parquet"
)
FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_26_shadow/"
    "fsr_26_prefight_snapshots.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_ko_reservoir_post_kd_survival_redundancy_study_v0.parquet"
)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return float("nan")
    return float(pd.DataFrame({"a": a[mask], "b": b[mask]}).corr(method="spearman").iloc[0, 1])


def _rank_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    ranks = pd.to_numeric(series, errors="coerce").rank(method="first", pct=True)
    edges = np.linspace(0.0, 1.0, len(labels) + 1)
    return pd.cut(ranks, bins=edges, labels=labels, include_lowest=True)


def _load(durability_path: Path, fsr_path: Path) -> pd.DataFrame:
    print(f"[post-KD survival] loading Study 8 rows: {durability_path}", flush=True)
    durability = pd.read_parquet(durability_path)
    print(f"[post-KD survival] Study 8 rows: {len(durability):,}", flush=True)

    print(f"[post-KD survival] loading FSR snapshots: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path)
    print(f"[post-KD survival] FSR rows: {len(fsr):,}", flush=True)

    required_durability = {
        "fight_id", "fighter_id", "candidate_damage_durability",
        "current_damage_exposure", "current_kd_absorbed", "current_ko_tko_loss",
    }
    missing = sorted(required_durability - set(durability.columns))
    if missing:
        raise ValueError(f"Study 8 artifact missing required columns: {missing}")

    required_fsr = {"fight_id", "fighter_id", "chin_resistance"}
    missing = sorted(required_fsr - set(fsr.columns))
    if missing:
        raise ValueError(f"FSR artifact missing required columns: {missing}")

    for frame in (durability, fsr):
        frame["fight_id"] = frame["fight_id"].astype(str)
        frame["fighter_id"] = frame["fighter_id"].astype(str)

    if fsr.duplicated(["fight_id", "fighter_id"]).any():
        raise ValueError("FSR snapshots violate fighter-fight grain")

    keep = ["fight_id", "fighter_id", "chin_resistance"]
    if "chin_resistance_updates" in fsr.columns:
        keep.append("chin_resistance_updates")

    work = durability.merge(
        fsr[keep], on=["fight_id", "fighter_id"], how="inner", validate="one_to_one"
    )

    numeric = [
        "candidate_damage_durability", "current_damage_exposure",
        "current_kd_absorbed", "current_ko_tko_loss", "chin_resistance",
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=numeric).copy()
    work = work[work["current_kd_absorbed"] >= 1.0].copy()
    work["kd_group"] = np.where(work["current_kd_absorbed"] >= 2.0, "2+ KD", "1 KD")

    print(f"[post-KD survival] KD-exposed eligible rows: {len(work):,}", flush=True)
    return work.reset_index(drop=True)


def _quintile_table(frame: pd.DataFrame, signal: str, label: str) -> pd.DataFrame:
    work = frame.copy()
    work["bucket"] = _rank_bucket(
        work[signal], ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    )
    rows: list[dict[str, object]] = []
    for bucket in ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]:
        g = work[work["bucket"].astype(str) == bucket]
        if g.empty:
            continue
        rows.append({
            "signal": label,
            "bucket": bucket,
            "fighter_fights": int(len(g)),
            "mean_signal": float(g[signal].mean()),
            "mean_kd_absorbed": float(g["current_kd_absorbed"].mean()),
            "mean_damage_exposure": float(g["current_damage_exposure"].mean()),
            "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
            "survival_probability": float(1.0 - g["current_ko_tko_loss"].mean()),
        })
    return pd.DataFrame(rows)


def _kd_count_tables(frame: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for kd_group in ["1 KD", "2+ KD"]:
        g = frame[frame["kd_group"] == kd_group].copy()
        if len(g) < 30:
            continue
        d = _quintile_table(g, "candidate_damage_durability", f"durability | {kd_group}")
        c = _quintile_table(g, "chin_resistance", f"chin | {kd_group}")
        tables.extend([d, c])
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _matched_exposure_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["exposure_bucket"] = _rank_bucket(
        work["current_damage_exposure"], ["E1", "E2", "E3", "E4", "E5"]
    )
    rows: list[dict[str, object]] = []
    for kd_group in ["1 KD", "2+ KD"]:
        kd = work[work["kd_group"] == kd_group].copy()
        for exposure in ["E1", "E2", "E3", "E4", "E5"]:
            group = kd[kd["exposure_bucket"].astype(str) == exposure].copy()
            if len(group) < 45:
                continue
            for signal, label in [
                ("candidate_damage_durability", "durability"),
                ("chin_resistance", "chin"),
            ]:
                group2 = group.copy()
                group2["band"] = _rank_bucket(group2[signal], ["low", "middle", "high"])
                for band in ["low", "middle", "high"]:
                    g = group2[group2["band"].astype(str) == band]
                    if g.empty:
                        continue
                    rows.append({
                        "kd_group": kd_group,
                        "exposure_bucket": exposure,
                        "signal": label,
                        "band": band,
                        "fighter_fights": int(len(g)),
                        "mean_signal": float(g[signal].mean()),
                        "mean_damage_exposure": float(g["current_damage_exposure"].mean()),
                        "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
                    })
    return pd.DataFrame(rows)


def _incremental_chin_within_durability(frame: pd.DataFrame) -> pd.DataFrame:
    """Ask whether chin adds separation after durability + adversity matching."""
    work = frame.copy()
    work["exposure_bucket"] = _rank_bucket(
        work["current_damage_exposure"], ["E1", "E2", "E3", "E4", "E5"]
    )
    rows: list[dict[str, object]] = []
    for kd_group in ["1 KD", "2+ KD"]:
        for exposure in ["E1", "E2", "E3", "E4", "E5"]:
            base = work[
                (work["kd_group"] == kd_group)
                & (work["exposure_bucket"].astype(str) == exposure)
            ].copy()
            if len(base) < 60:
                continue
            base["durability_band"] = _rank_bucket(
                base["candidate_damage_durability"], ["low", "middle", "high"]
            )
            for durability_band in ["low", "middle", "high"]:
                d = base[base["durability_band"].astype(str) == durability_band].copy()
                if len(d) < 20:
                    continue
                d["chin_band"] = _rank_bucket(d["chin_resistance"], ["low", "high"])
                for chin_band in ["low", "high"]:
                    g = d[d["chin_band"].astype(str) == chin_band]
                    if g.empty:
                        continue
                    rows.append({
                        "kd_group": kd_group,
                        "exposure_bucket": exposure,
                        "durability_band": durability_band,
                        "chin_band_within_durability": chin_band,
                        "fighter_fights": int(len(g)),
                        "mean_durability": float(g["candidate_damage_durability"].mean()),
                        "mean_chin": float(g["chin_resistance"].mean()),
                        "mean_damage_exposure": float(g["current_damage_exposure"].mean()),
                        "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
                    })
    return pd.DataFrame(rows)


def _incremental_durability_within_chin(frame: pd.DataFrame) -> pd.DataFrame:
    """Reverse check: does durability add separation after chin matching?"""
    work = frame.copy()
    work["exposure_bucket"] = _rank_bucket(
        work["current_damage_exposure"], ["E1", "E2", "E3", "E4", "E5"]
    )
    rows: list[dict[str, object]] = []
    for kd_group in ["1 KD", "2+ KD"]:
        for exposure in ["E1", "E2", "E3", "E4", "E5"]:
            base = work[
                (work["kd_group"] == kd_group)
                & (work["exposure_bucket"].astype(str) == exposure)
            ].copy()
            if len(base) < 60:
                continue
            base["chin_band"] = _rank_bucket(base["chin_resistance"], ["low", "middle", "high"])
            for chin_band in ["low", "middle", "high"]:
                c = base[base["chin_band"].astype(str) == chin_band].copy()
                if len(c) < 20:
                    continue
                c["durability_band"] = _rank_bucket(
                    c["candidate_damage_durability"], ["low", "high"]
                )
                for durability_band in ["low", "high"]:
                    g = c[c["durability_band"].astype(str) == durability_band]
                    if g.empty:
                        continue
                    rows.append({
                        "kd_group": kd_group,
                        "exposure_bucket": exposure,
                        "chin_band": chin_band,
                        "durability_band_within_chin": durability_band,
                        "fighter_fights": int(len(g)),
                        "mean_durability": float(g["candidate_damage_durability"].mean()),
                        "mean_chin": float(g["chin_resistance"].mean()),
                        "mean_damage_exposure": float(g["current_damage_exposure"].mean()),
                        "ko_tko_loss_probability": float(g["current_ko_tko_loss"].mean()),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 9: post-KD survival trait redundancy")
    parser.add_argument("--durability-path", type=Path, default=DURABILITY_PATH)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    work = _load(args.durability_path, args.fsr_path)

    print("\n" + "=" * 120)
    print("STUDY 9 — POST-KD SURVIVAL TRAIT REDUNDANCY")
    print("=" * 120)
    print(f"KD-exposed fighter-fights: {len(work):,}")
    print("\nCORE SPEARMAN RELATIONSHIPS")
    print(
        "candidate_damage_durability vs KO/TKO loss  "
        f"{_safe_spearman(work['candidate_damage_durability'], work['current_ko_tko_loss']):.5f}"
    )
    print(
        "chin_resistance              vs KO/TKO loss  "
        f"{_safe_spearman(work['chin_resistance'], work['current_ko_tko_loss']):.5f}"
    )
    print(
        "durability vs chin correlation                "
        f"{_safe_spearman(work['candidate_damage_durability'], work['chin_resistance']):.5f}"
    )

    print("\nDURABILITY QUINTILES — KD-EXPOSED")
    print(_quintile_table(work, "candidate_damage_durability", "durability").to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nCHIN QUINTILES — KD-EXPOSED")
    print(_quintile_table(work, "chin_resistance", "chin").to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nWITHIN 1 KD / 2+ KD STRATA")
    print(_kd_count_tables(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nMATCHED KD COUNT + CURRENT DAMAGE EXPOSURE")
    print(_matched_exposure_table(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nINCREMENTAL CHIN WITHIN DURABILITY + ADVERSITY STRATA")
    print(_incremental_chin_within_durability(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nINCREMENTAL DURABILITY WITHIN CHIN + ADVERSITY STRATA")
    print(_incremental_durability_within_chin(work).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(
        "\nINTERPRETATION RULE: if chin_resistance produces little or inconsistent "
        "separation after matching on durability, KD count, and current damage "
        "exposure, a separate post-KD-survival trait is probably redundant. "
        "If chin remains directionally stable and material, retain/rebuild it."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(args.output, index=False)
    print(f"\n[post-KD survival] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
