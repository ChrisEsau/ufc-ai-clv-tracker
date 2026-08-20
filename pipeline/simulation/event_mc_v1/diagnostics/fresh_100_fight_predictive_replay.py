"""Fresh 100-fight predictive replay for the calibrated FSR V2 EVENT MC.

Runs the canonical EVENT MC directly through build_engine(...).run().
No diagnostic rate wrappers and no fighter-profile shifting.

Outputs:
- per-fight model moneyline probabilities
- winner/method diagnostics
- historical prefight market comparison when available
- heartbeat/progress output
- global fight-mechanics diagnostics
- JSON and CSV artifacts
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
    MASTER_PATH,
)

from ..calibration import DEFAULT_CALIBRATION
from ..flow_stats import FlowStatsSink
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import (
    METHODS,
    _fight,
    normalize_method,
    observed_duration_seconds,
)


CUTOFF = pd.Timestamp("2025-03-22")
MARKET_PATH = Path("data/market/historical_moneyline_odds.parquet")

JOINT_CLASSES = tuple(
    f"{side}_{method}"
    for side in ("red", "blue")
    for method in METHODS
)

STRIKE_FAMILIES = {
    "standing": "standing_strike",
    "ground": "ground_strike",
}


def select_fresh_cohort(
    limit: int = 100,
    offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select chronologically without using outcomes except eligibility."""

    master = (
        pd.read_parquet(MASTER_PATH)
        .drop_duplicates("fight_id")
        .copy()
    )
    master["event_date"] = pd.to_datetime(master["date"])

    candidates = master[
        master["event_date"] > CUTOFF
    ].sort_values(["event_date", "fight_id"])

    fsr = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH).copy()

    fsr["fight_id"] = fsr["fight_id"].astype(str)

    valid_fsr = fsr.groupby("fight_id").size()
    valid_fsr_ids = set(valid_fsr[valid_fsr == 2].index)

    # Preserve separation from the 100/50 calibration cohorts.
    train, holdout, _ = temporal_cohorts(100, 50)
    calibration_ids = (
        set(train["fight_id"].astype(str))
        | set(holdout["fight_id"].astype(str))
    )

    selected = []
    eligible_seen = 0
    missing_fsr = 0
    unsupported = 0

    for _, row in candidates.iterrows():
        fight_id = str(row["fight_id"])
        method = normalize_method(row["method"])

        supported = (
            row["winner"] in {row["r_name"], row["b_name"]}
            and method in METHODS
            and int(row["total_rounds"]) in {3, 5}
            and pd.notna(row["match_time_sec"])
        )

        if not supported:
            unsupported += 1
            continue

        if fight_id in calibration_ids:
            continue

        if fight_id not in valid_fsr_ids:
            missing_fsr += 1
            continue

        if eligible_seen < offset:
            eligible_seen += 1
            continue

        selected.append(row)
        eligible_seen += 1

        if len(selected) == limit:
            break

    if len(selected) != limit:
        raise RuntimeError(
            f"required {limit} eligible fresh fights, found {len(selected)}"
        )

    cohort = pd.DataFrame(selected).reset_index(drop=True)

    metadata = {
        "cutoff_exclusive": str(CUTOFF.date()),
        "first_event_date": str(cohort["event_date"].min().date()),
        "last_event_date": str(cohort["event_date"].max().date()),
        "bout_ids": cohort["fight_id"].astype(str).tolist(),
        "excluded_missing_fsr_before_cohort_completed": missing_fsr,
        "excluded_unsupported_or_incomplete_before_cohort_completed": unsupported,
        "calibration_overlap_count": int(
            cohort["fight_id"]
            .astype(str)
            .isin(calibration_ids)
            .sum()
        ),
    }

    return cohort, fsr, metadata


def build_simulation_inputs(
    cohort: pd.DataFrame,
    fsr: pd.DataFrame,
):
    return [_fight(row, fsr) for _, row in cohort.iterrows()]


def _simulate_one(args):
    fight_index, fight, paths, seed = args

    joint = Counter()
    strike_attempts = Counter()
    strike_landed = Counter()
    phase_seconds = Counter()
    ko_finish_rounds = Counter()

    td_attempts = []
    td_landed = []
    kds = []
    subs = []
    ground_entries = []
    exposures = []
    nondecision = []

    for path_index in range(paths):
        path_seed = seed + fight_index * 100000 + path_index

        result = build_engine(
            fight,
            path_seed,
            FlowStatsSink(),
        )[0].run()

        stats = result.sink_result
        winner = result.state.winner
        method = result.state.finish_method
        elapsed = float(result.state.fight_time_seconds)

        joint[f"{winner}_{method}"] += 1
        exposures.append(elapsed)

        if method != "DEC":
            nondecision.append(elapsed)

        if method == "KO_TKO":
            round_no = int(max(elapsed - 1e-12, 0.0) // 300) + 1
            ko_finish_rounds[round_no] += 1

        phase_seconds.update(stats["phase_seconds"])

        for phase, family in STRIKE_FAMILIES.items():
            strike_attempts[phase] += sum(
                stats["attempts"][side].get(family, 0)
                for side in ("red", "blue")
            )
            strike_landed[phase] += sum(
                stats["outcomes"][side].get(f"{family}_landed", 0)
                for side in ("red", "blue")
            )

        path_td_attempts = sum(
            stats["attempts"][side].get("takedown", 0)
            for side in ("red", "blue")
        )
        path_td_landed = sum(
            stats["outcomes"][side].get("takedown_landed", 0)
            for side in ("red", "blue")
        )

        td_attempts.append(path_td_attempts)
        td_landed.append(path_td_landed)

        kds.append(
            sum(
                int(item.knockdown)
                for item in stats["physiology"]
            )
        )

        subs.append(
            sum(
                stats["attempts"][side].get(
                    "submission_attempt",
                    0,
                )
                for side in ("red", "blue")
            )
        )

        ground_entries.append(
            sum(
                1
                for transition in stats["transitions"]
                if transition["to_phase"] == "ground"
                and transition["from_phase"] != "ground"
            )
        )

    return {
        "fight_index": fight_index,
        "joint_counts": dict(joint),
        "ko_finish_rounds": dict(ko_finish_rounds),
        "paths": paths,
        "mean_elapsed": float(np.mean(exposures)),
        "exposures": exposures,
        "nondecision": nondecision,
        "strike_attempts": dict(strike_attempts),
        "strike_landed": dict(strike_landed),
        "td_attempts": td_attempts,
        "td_landed": td_landed,
        "kds": kds,
        "sub_attempts": subs,
        "ground_entries": ground_entries,
        "phase_seconds": dict(phase_seconds),
    }


def probabilities_from_counts(
    counts: dict,
    paths: int,
) -> dict:
    joint = {
        key: counts.get(key, 0) / paths
        for key in JOINT_CLASSES
    }

    red = sum(
        joint[f"red_{method}"]
        for method in METHODS
    )

    methods = {
        method: (
            joint[f"red_{method}"]
            + joint[f"blue_{method}"]
        )
        for method in METHODS
    }

    return {
        "joint": joint,
        "red": red,
        "blue": 1.0 - red,
        "methods": methods,
    }


def _safe_log(value):
    return math.log(max(float(value), 1e-12))


def _confidence_buckets(
    rows: pd.DataFrame,
) -> list[dict]:
    bounds = (
        (.50, .55),
        (.55, .60),
        (.60, .65),
        (.65, .70),
        (.70, .75),
        (.75, .80),
        (.80, .90),
        (.90, 1.0000001),
    )

    output = []

    for low, high in bounds:
        mask = (
            (rows["predicted_winner_probability"] >= low)
            & (rows["predicted_winner_probability"] < high)
        )
        group = rows.loc[mask]

        output.append(
            {
                "bucket": (
                    f"{low:.0%}-{min(high, 1):.0%}"
                ),
                "fights": len(group),
                "average_confidence": (
                    float(
                        group[
                            "predicted_winner_probability"
                        ].mean()
                    )
                    if len(group)
                    else None
                ),
                "accuracy": (
                    float(group["winner_correct"].mean())
                    if len(group)
                    else None
                ),
            }
        )

    return output


def score_rows(rows: pd.DataFrame) -> dict:
    actual_red = (
        rows["actual_side"] == "red"
    ).astype(float).to_numpy()

    pred_red = rows["P_red_win"].to_numpy(float)

    winner_log_loss = -float(
        np.mean(
            actual_red
            * np.log(np.clip(pred_red, 1e-12, 1))
            + (1 - actual_red)
            * np.log(
                np.clip(
                    1 - pred_red,
                    1e-12,
                    1,
                )
            )
        )
    )

    method_log_loss = -float(
        np.mean(
            [
                _safe_log(
                    row[f"P_{row.actual_method}"]
                )
                for _, row in rows.iterrows()
            ]
        )
    )

    method_brier = float(
        np.mean(
            [
                sum(
                    (
                        row[f"P_{method}"]
                        - (
                            method
                            == row.actual_method
                        )
                    )
                    ** 2
                    for method in METHODS
                )
                for _, row in rows.iterrows()
            ]
        )
    )

    return {
        "winner": {
            "correct": int(
                rows["winner_correct"].sum()
            ),
            "accuracy": float(
                rows["winner_correct"].mean()
            ),
            "brier": float(
                np.mean(
                    (pred_red - actual_red) ** 2
                )
            ),
            "log_loss": winner_log_loss,
            "confidence_buckets": (
                _confidence_buckets(rows)
            ),
        },
        "method": {
            "correct": int(
                rows["method_correct"].sum()
            ),
            "accuracy": float(
                rows["method_correct"].mean()
            ),
            "brier": method_brier,
            "log_loss": method_log_loss,
            "actual_shares": {
                method: float(
                    (
                        rows["actual_method"]
                        == method
                    ).mean()
                )
                for method in METHODS
            },
            "mean_probabilities": {
                method: float(
                    rows[f"P_{method}"].mean()
                )
                for method in METHODS
            },
        },
    }


def _load_market() -> pd.DataFrame:
    if not MARKET_PATH.exists():
        return pd.DataFrame(
            columns=["bout_id"]
        )

    market = pd.read_parquet(
        MARKET_PATH
    ).copy()

    required = {
        "fight_id",
        "market_key",
        "outcome_side",
        "american_odds",
        "implied_probability",
    }

    missing = required - set(market.columns)
    if missing:
        raise RuntimeError(
            "historical moneyline file missing "
            f"columns: {sorted(missing)}"
        )

    market = market.loc[
        market["market_key"]
        .astype(str)
        .str.lower()
        .eq("moneyline")
    ].copy()

    market["fight_id"] = (
        market["fight_id"].astype(str)
    )
    market["outcome_side"] = (
        market["outcome_side"]
        .astype(str)
        .str.lower()
    )
    market["american_odds"] = pd.to_numeric(
        market["american_odds"],
        errors="coerce",
    )
    market["implied_probability"] = pd.to_numeric(
        market["implied_probability"],
        errors="coerce",
    )

    sort_cols = [
        c
        for c in (
            "date",
            "historical_market_timestamp",
            "legacy_row_number",
        )
        if c in market.columns
    ]

    if sort_cols:
        market = market.sort_values(sort_cols)

    market = market.drop_duplicates(
        ["fight_id", "outcome_side"],
        keep="last",
    )

    pivot = market[
        [
            "fight_id",
            "outcome_side",
            "american_odds",
            "implied_probability",
        ]
    ].pivot(
        index="fight_id",
        columns="outcome_side",
    )

    pivot.columns = [
        f"market_{metric}_{side}"
        for metric, side in pivot.columns
    ]

    pivot = (
        pivot.reset_index()
        .rename(
            columns={"fight_id": "bout_id"}
        )
    )

    for side in ("red", "blue"):
        for metric in (
            "american_odds",
            "implied_probability",
        ):
            col = f"market_{metric}_{side}"
            if col not in pivot.columns:
                pivot[col] = np.nan

    raw_sum = (
        pivot["market_implied_probability_red"]
        + pivot[
            "market_implied_probability_blue"
        ]
    )

    pivot["market_overround"] = (
        raw_sum - 1.0
    )
    pivot["market_novig_p_red"] = (
        pivot[
            "market_implied_probability_red"
        ]
        / raw_sum
    )
    pivot["market_novig_p_blue"] = (
        pivot[
            "market_implied_probability_blue"
        ]
        / raw_sum
    )

    return pivot


def _attach_market(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    market = _load_market()

    if market.empty:
        out = frame.copy()
        out["market_available"] = False
        return out

    out = frame.merge(
        market,
        on="bout_id",
        how="left",
        validate="one_to_one",
    )

    out["market_available"] = (
        out["market_novig_p_red"].notna()
        & out["market_novig_p_blue"].notna()
    )

    out["market_favorite_side"] = np.where(
        out["market_available"],
        np.where(
            out["market_novig_p_red"]
            >= out["market_novig_p_blue"],
            "red",
            "blue",
        ),
        None,
    )

    out["market_favorite"] = np.where(
        out["market_favorite_side"] == "red",
        out["red_fighter"],
        np.where(
            out["market_favorite_side"] == "blue",
            out["blue_fighter"],
            None,
        ),
    )

    out["market_correct"] = np.where(
        out["market_available"],
        out["market_favorite_side"]
        == out["actual_side"],
        np.nan,
    )

    out["model_market_agree"] = np.where(
        out["market_available"],
        out["predicted_side"]
        == out["market_favorite_side"],
        np.nan,
    )

    out["model_minus_market_p_red"] = (
        out["P_red_win"]
        - out["market_novig_p_red"]
    )

    out["model_actual_probability"] = np.where(
        out["actual_side"] == "red",
        out["P_red_win"],
        out["P_blue_win"],
    )

    out["market_actual_probability"] = np.where(
        out["actual_side"] == "red",
        out["market_novig_p_red"],
        out["market_novig_p_blue"],
    )

    out["model_edge_predicted_side"] = np.where(
        out["predicted_side"] == "red",
        out["P_red_win"]
        - out["market_novig_p_red"],
        out["P_blue_win"]
        - out["market_novig_p_blue"],
    )

    return out


def _market_metrics(
    frame: pd.DataFrame,
) -> dict:
    matched = frame.loc[
        frame["market_available"].fillna(False)
    ].copy()

    if matched.empty:
        return {
            "market_matched_fights": 0,
            "market_coverage": 0.0,
        }

    actual_red = (
        matched["actual_side"] == "red"
    ).astype(float).to_numpy()

    model_red = matched[
        "P_red_win"
    ].to_numpy(float)

    market_red = matched[
        "market_novig_p_red"
    ].to_numpy(float)

    def log_loss(p):
        return -float(
            np.mean(
                actual_red
                * np.log(
                    np.clip(p, 1e-12, 1)
                )
                + (1 - actual_red)
                * np.log(
                    np.clip(
                        1 - p,
                        1e-12,
                        1,
                    )
                )
            )
        )

    disagreements = matched.loc[
        matched["model_market_agree"] == 0
    ]

    return {
        "market_matched_fights": len(matched),
        "market_coverage": (
            len(matched) / len(frame)
        ),
        "model_accuracy_market_matched": float(
            matched["winner_correct"].mean()
        ),
        "market_favorite_accuracy": float(
            matched["market_correct"].mean()
        ),
        "model_market_agreement_rate": float(
            matched["model_market_agree"].mean()
        ),
        "model_brier_market_matched": float(
            np.mean(
                (model_red - actual_red) ** 2
            )
        ),
        "market_brier": float(
            np.mean(
                (market_red - actual_red) ** 2
            )
        ),
        "model_log_loss_market_matched": (
            log_loss(model_red)
        ),
        "market_log_loss": (
            log_loss(market_red)
        ),
        "mean_abs_model_market_probability_gap": float(
            np.mean(
                np.abs(model_red - market_red)
            )
        ),
        "model_market_disagreements": len(
            disagreements
        ),
        "model_right_market_wrong": int(
            (
                (disagreements["winner_correct"] == 1)
                & (disagreements["market_correct"] == 0)
            ).sum()
        ),
        "market_right_model_wrong": int(
            (
                (disagreements["winner_correct"] == 0)
                & (disagreements["market_correct"] == 1)
            ).sum()
        ),
    }


def _global_sim_diagnostics(
    raw: list[dict],
) -> dict:
    joint = Counter()
    attempts = Counter()
    landed = Counter()
    phases = Counter()
    ko_rounds = Counter()

    exposures = []
    td_attempts = []
    td_landed = []
    kds = []
    subs = []

    for item in raw:
        joint.update(item["joint_counts"])
        attempts.update(item["strike_attempts"])
        landed.update(item["strike_landed"])
        phases.update(item["phase_seconds"])
        ko_rounds.update(
            item["ko_finish_rounds"]
        )

        exposures.extend(item["exposures"])
        td_attempts.extend(
            item["td_attempts"]
        )
        td_landed.extend(item["td_landed"])
        kds.extend(item["kds"])
        subs.extend(item["sub_attempts"])

    paths = len(exposures)
    exposure_seconds = float(
        sum(exposures)
    )

    total_strike_attempts = sum(
        attempts.values()
    )
    total_strike_landed = sum(
        landed.values()
    )

    ratio = lambda n, d: (
        float(n / d) if d else 0.0
    )

    return {
        "paths": paths,
        "method_shares": {
            method: ratio(
                joint[f"red_{method}"]
                + joint[f"blue_{method}"],
                paths,
            )
            for method in METHODS
        },
        "ko_tko_round_share_of_all_paths": {
            f"R{round_no}": ratio(
                ko_rounds[round_no],
                paths,
            )
            for round_no in sorted(
                ko_rounds
            )
        },
        "strike_attempts_per_15min": ratio(
            total_strike_attempts * 900,
            exposure_seconds,
        ),
        "strike_landed_per_15min": ratio(
            total_strike_landed * 900,
            exposure_seconds,
        ),
        "strike_accuracy": ratio(
            total_strike_landed,
            total_strike_attempts,
        ),
        "strike_phase": {
            phase: {
                "attempts_per_15min": ratio(
                    attempts[phase] * 900,
                    exposure_seconds,
                ),
                "landed_per_15min": ratio(
                    landed[phase] * 900,
                    exposure_seconds,
                ),
                "accuracy": ratio(
                    landed[phase],
                    attempts[phase],
                ),
            }
            for phase in STRIKE_FAMILIES
        },
        "td_attempts_per_15min": ratio(
            sum(td_attempts) * 900,
            exposure_seconds,
        ),
        "td_landed_per_15min": ratio(
            sum(td_landed) * 900,
            exposure_seconds,
        ),
        "td_success": ratio(
            sum(td_landed),
            sum(td_attempts),
        ),
        "kd_per_15min": ratio(
            sum(kds) * 900,
            exposure_seconds,
        ),
        "submission_attempts_per_15min": ratio(
            sum(subs) * 900,
            exposure_seconds,
        ),
        "mean_fight_duration_seconds": (
            float(np.mean(exposures))
        ),
        "phase_seconds_per_path": {
            phase: ratio(
                phases[phase],
                paths,
            )
            for phase in (
                "standing",
                "ground",
            )
        },
    }


def _historical_diagnostics(
    cohort: pd.DataFrame,
) -> dict:
    methods = cohort["method"].map(
        normalize_method
    )

    finish_round = pd.to_numeric(
        cohort["finish_round"],
        errors="coerce",
    )

    ko_mask = methods.eq("KO_TKO")

    return {
        "fights": len(cohort),
        "method_shares": {
            method: float(
                methods.eq(method).mean()
            )
            for method in METHODS
        },
        "ko_tko_round_share_of_all_fights": {
            f"R{round_no}": float(
                (
                    ko_mask
                    & finish_round.eq(round_no)
                ).mean()
            )
            for round_no in (1, 2, 3, 4, 5)
            if (
                ko_mask
                & finish_round.eq(round_no)
            ).any()
        },
    }


def _heartbeat(
    done: int,
    total: int,
    started: float,
):
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0
    remaining = (
        (total - done) / rate
        if rate
        else 0.0
    )

    print(
        f"[heartbeat] fights {done}/{total} "
        f"({done/total:.0%}) | "
        f"elapsed={elapsed:.1f}s | "
        f"eta={remaining:.1f}s",
        flush=True,
    )


def run(
    paths=250,
    seed=20260813,
    workers=2,
    heartbeat_every=10,
    offset=0,
    output=Path(
        "/tmp/event_mc_fresh_100_replay.json"
    ),
    csv=Path(
        "/tmp/event_mc_fresh_100_replay.csv"
    ),
):
    cohort, fsr, selection = (
        select_fresh_cohort(100, offset)
    )

    fights = build_simulation_inputs(
        cohort,
        fsr,
    )

    print("=" * 100)
    print(
        "CALIBRATED FSR V2 EVENT MC — "
        "FRESH 100-FIGHT REPLAY"
    )
    print("=" * 100)
    print(
        json.dumps(
            selection,
            indent=2,
        )
    )
    print(
        "FSR V2:",
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
    )
    print(
        "calibration fingerprint:",
        DEFAULT_CALIBRATION.fingerprint,
    )
    print(
        "FSR V2 calibration:",
        dict(
            DEFAULT_CALIBRATION.section(
                "fsr_v2_calibration"
            )
        ),
    )
    print(
        f"paths/fight={paths} | "
        f"workers={workers} | "
        f"heartbeat_every={heartbeat_every}"
    )

    started = time.perf_counter()

    raw = [None] * len(fights)
    tasks = [
        (i, fight, paths, seed)
        for i, fight in enumerate(fights)
    ]

    done = 0

    if workers == 1:
        for task in tasks:
            raw[task[0]] = _simulate_one(task)
            done += 1

            if (
                done == 1
                or done % heartbeat_every == 0
                or done == len(tasks)
            ):
                _heartbeat(
                    done,
                    len(tasks),
                    started,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {
                pool.submit(
                    _simulate_one,
                    task,
                ): task[0]
                for task in tasks
            }

            for future in as_completed(
                futures
            ):
                index = futures[future]
                raw[index] = future.result()
                done += 1

                if (
                    done == 1
                    or done % heartbeat_every == 0
                    or done == len(tasks)
                ):
                    _heartbeat(
                        done,
                        len(tasks),
                        started,
                    )

    rows = []

    for index, (_, actual) in enumerate(
        cohort.iterrows()
    ):
        probs = probabilities_from_counts(
            raw[index]["joint_counts"],
            paths,
        )

        actual_side = (
            "red"
            if actual["winner"]
            == actual["r_name"]
            else "blue"
        )

        predicted_side = (
            "red"
            if probs["red"] >= probs["blue"]
            else "blue"
        )

        predicted_method = max(
            METHODS,
            key=probs["methods"].get,
        )

        predicted_joint = max(
            JOINT_CLASSES,
            key=probs["joint"].get,
        )

        actual_method = normalize_method(
            actual["method"]
        )

        row = {
            "event_date": str(
                actual["event_date"].date()
            ),
            "bout_id": str(
                actual["fight_id"]
            ),
            "red_fighter": str(
                actual["r_name"]
            ),
            "blue_fighter": str(
                actual["b_name"]
            ),
            "P_red_win": probs["red"],
            "P_blue_win": probs["blue"],
            "predicted_winner": (
                actual["r_name"]
                if predicted_side == "red"
                else actual["b_name"]
            ),
            "predicted_winner_probability": (
                probs[predicted_side]
            ),
            "predicted_side": predicted_side,
            **{
                f"P_{method}": (
                    probs["methods"][method]
                )
                for method in METHODS
            },
            "predicted_method": predicted_method,
            "predicted_method_probability": (
                probs["methods"][
                    predicted_method
                ]
            ),
            **{
                f"P_{joint}": (
                    probs["joint"][joint]
                )
                for joint in JOINT_CLASSES
            },
            "predicted_winner_method": (
                predicted_joint
            ),
            "predicted_joint_probability": (
                probs["joint"][
                    predicted_joint
                ]
            ),
            "actual_winner": str(
                actual["winner"]
            ),
            "actual_side": actual_side,
            "actual_method": actual_method,
            "actual_finish_round": int(
                actual["finish_round"]
            ),
            "actual_elapsed_seconds": (
                observed_duration_seconds(
                    actual
                )
            ),
            "simulated_mean_elapsed_seconds": (
                raw[index]["mean_elapsed"]
            ),
            "simulated_mean_submission_attempts": (
                float(np.mean(raw[index]["sub_attempts"]))
            ),
            "simulated_mean_ground_seconds": (
                float(raw[index]["phase_seconds"].get("ground", 0)) / paths
            ),
            "simulated_mean_td_attempts": (
                float(np.mean(raw[index]["td_attempts"]))
            ),
            "simulated_mean_td_landed": (
                float(np.mean(raw[index]["td_landed"]))
            ),
            "simulated_mean_ground_entries": (
                float(np.mean(raw[index]["ground_entries"]))
            ),
            "simulated_nondecision_mean_seconds": (
                float(
                    np.mean(
                        raw[index]["nondecision"]
                    )
                )
                if raw[index]["nondecision"]
                else np.nan
            ),
        }

        row.update(
            {
                "winner_correct": (
                    row["predicted_winner"]
                    == row["actual_winner"]
                ),
                "method_correct": (
                    predicted_method
                    == actual_method
                ),
                "winner_method_correct": (
                    predicted_joint
                    == (
                        f"{actual_side}_"
                        f"{actual_method}"
                    )
                ),
                "winner_probability_assigned_to_actual": (
                    probs[actual_side]
                ),
            }
        )

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = _attach_market(frame)

    metrics = score_rows(frame)
    market_metrics = _market_metrics(frame)

    misses = frame.loc[
        ~frame["winner_correct"]
    ].sort_values(
        "predicted_winner_probability",
        ascending=False,
    )

    report = {
        "selection": selection,
        "paths_per_fight": paths,
        "seed": seed,
        "workers": workers,
        "runtime_seconds": (
            time.perf_counter() - started
        ),
        "fsr_v2_path": str(
            FSR_V2_PREFIGHT_SNAPSHOTS_PATH
        ),
        "calibration_fingerprint": (
            DEFAULT_CALIBRATION.fingerprint
        ),
        "fsr_v2_calibration": dict(
            DEFAULT_CALIBRATION.section(
                "fsr_v2_calibration"
            )
        ),
        "performance": metrics,
        "market": market_metrics,
        "historical_diagnostics": (
            _historical_diagnostics(cohort)
        ),
        "simulated_diagnostics": (
            _global_sim_diagnostics(raw)
        ),
        "fight_predictions": (
            frame.replace(
                {np.nan: None}
            ).to_dict("records")
        ),
        "winner_misses": (
            misses.replace(
                {np.nan: None}
            ).to_dict("records")
        ),
    }

    output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )
    frame.to_csv(
        csv,
        index=False,
    )

    print("\n" + "=" * 100)
    print("WINNER PERFORMANCE")
    print("=" * 100)
    print(
        json.dumps(
            metrics["winner"],
            indent=2,
            sort_keys=True,
        )
    )

    print("\n" + "=" * 100)
    print("MODEL vs PREFIGHT MARKET")
    print("=" * 100)

    if market_metrics.get(
        "market_matched_fights",
        0,
    ):
        print(
            f"coverage: "
            f"{market_metrics['market_matched_fights']}"
            f"/{len(frame)} "
            f"= "
            f"{market_metrics['market_coverage']:.1%}"
        )
        print(
            "MODEL accuracy: "
            f"{market_metrics['model_accuracy_market_matched']:.1%}"
        )
        print(
            "MARKET favorite accuracy: "
            f"{market_metrics['market_favorite_accuracy']:.1%}"
        )
        print(
            "MODEL Brier: "
            f"{market_metrics['model_brier_market_matched']:.4f}"
        )
        print(
            "MARKET Brier: "
            f"{market_metrics['market_brier']:.4f}"
        )
        print(
            "MODEL log loss: "
            f"{market_metrics['model_log_loss_market_matched']:.4f}"
        )
        print(
            "MARKET log loss: "
            f"{market_metrics['market_log_loss']:.4f}"
        )
        print(
            "model/market agreement: "
            f"{market_metrics['model_market_agreement_rate']:.1%}"
        )
        print(
            "disagreements: "
            f"{market_metrics['model_market_disagreements']}"
            " | model right/market wrong="
            f"{market_metrics['model_right_market_wrong']}"
            " | market right/model wrong="
            f"{market_metrics['market_right_model_wrong']}"
        )
    else:
        print(
            "No historical moneyline rows matched "
            "this cohort."
        )

    print("\n" + "=" * 100)
    print("HISTORICAL FIGHT DIAGNOSTICS")
    print("=" * 100)
    print(
        json.dumps(
            report["historical_diagnostics"],
            indent=2,
            sort_keys=True,
        )
    )

    print("\n" + "=" * 100)
    print("SIMULATED FIGHT DIAGNOSTICS")
    print("=" * 100)
    print(
        json.dumps(
            report["simulated_diagnostics"],
            indent=2,
            sort_keys=True,
        )
    )

    print("\n" + "=" * 100)
    print("WINNER MISSES")
    print("=" * 100)

    miss_cols = [
        "event_date",
        "bout_id",
        "red_fighter",
        "blue_fighter",
        "actual_winner",
        "predicted_winner",
        "predicted_winner_probability",
    ]

    if "market_favorite" in misses:
        miss_cols += [
            "market_favorite",
            "model_edge_predicted_side",
        ]

    print(
        misses[miss_cols].to_string(
            index=False
        )
        if len(misses)
        else "none"
    )

    print(f"\nwrote JSON: {output}")
    print(f"wrote CSV : {csv}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paths",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260813,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/tmp/event_mc_fresh_100_replay.json"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/tmp/event_mc_fresh_100_replay.csv"
        ),
    )

    run(**vars(parser.parse_args()))
