from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.config import FSRV2Config
from pipeline.fsr_v2.replay.engine import ReplayEngine, aggregate_fights
from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.traits.registry import GROUPS


PRIORS = [2, 5, 10, 20, 40, 80]


def logistic(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p))


def auc(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)

    pos = y == 1
    neg = y == 0

    ranks = pd.Series(p).rank(method="average").to_numpy()

    return (
        ranks[pos].sum()
        - pos.sum() * (pos.sum() + 1) / 2
    ) / (pos.sum() * neg.sum())


def history_for_prior(fights, prior):
    cfg = FSRV2Config(
        takedown_effectiveness_prior_attempts=float(prior)
    )

    h = ReplayEngine(cfg).replay(
        GROUPS["takedown_effectiveness"],
        fights,
    ).history

    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    holdout = pd.read_csv(args.csv)
    holdout["bout_id"] = holdout["bout_id"].astype(str)

    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    )

    fights["fight_id"] = fights["fight_id"].astype(str)

    print("=" * 94)
    print("FSR V2 — TAKEDOWN COMPLETION PRIOR SWEEP")
    print("=" * 94)

    print(
        f"{'prior':>7s}"
        f"{'offense AUC':>15s}"
        f"{'defense AUC':>15s}"
        f"{'full AUC':>12s}"
    )

    for prior in PRIORS:
        h = history_for_prior(fights, prior)

        h["fight_id"] = h["fight_id"].astype(str)

        lookup = {
            (
                str(r.fight_id),
                str(r.fighter_name),
                str(r.trait),
            ): (
                float(r.pre_rating),
                float(r.population_baseline),
            )
            for r in h.itertuples()
        }

        ys = []
        p_off = []
        p_def = []
        p_full = []

        for r in holdout.itertuples():
            fid = str(r.bout_id)

            for side in ("red", "blue"):
                opp = "blue" if side == "red" else "red"

                attacker = str(
                    getattr(r, f"{side}_fighter")
                )
                defender = str(
                    getattr(r, f"{opp}_fighter")
                )

                attempts = int(round(
                    getattr(r, f"historical_{side}_td_attempts")
                ))

                landed = int(round(
                    getattr(r, f"historical_{side}_td_landed")
                ))

                if attempts <= 0:
                    continue

                off, baseline = lookup[
                    (fid, attacker, "takedown_offense")
                ]

                defense, _ = lookup[
                    (fid, defender, "takedown_defense")
                ]

                base = logit(baseline)

                po = float(logistic(base + off))
                pd_ = float(logistic(base - defense))
                pf = float(logistic(base + off - defense))

                labels = (
                    [1] * landed
                    + [0] * (attempts-landed)
                )

                ys.extend(labels)
                p_off.extend([po] * attempts)
                p_def.extend([pd_] * attempts)
                p_full.extend([pf] * attempts)

        print(
            f"{prior:7d}"
            f"{auc(ys,p_off):15.3f}"
            f"{auc(ys,p_def):15.3f}"
            f"{auc(ys,p_full):12.3f}"
        )


if __name__ == "__main__":
    main()
