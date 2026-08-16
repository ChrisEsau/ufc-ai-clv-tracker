from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import load_round_stats, build_paired_rounds
from pipeline.fsr_v2.replay.engine import aggregate_fights


SIM_PATH = "/tmp/event_mc_fresh_100_replay.csv"
OUTPUT_PATH = "/tmp/ground_pipeline_100_fights.csv"


def ratio(mc: pd.Series, hist: pd.Series) -> float:
    historical = float(hist.sum())
    return float(mc.sum() / historical) if historical else np.nan


def main() -> None:
    # ------------------------------------------------------------------
    # Load the latest fresh-100 EVENT MC run.
    # ------------------------------------------------------------------
    sim = pd.read_csv(SIM_PATH)
    sim["bout_id"] = sim["bout_id"].astype(str)

    required = {
        "simulated_mean_td_attempts",
        "simulated_mean_td_landed",
        "simulated_mean_ground_entries",
        "simulated_mean_ground_seconds",
        "simulated_mean_submission_attempts",
    }

    missing = required - set(sim.columns)
    if missing:
        raise SystemExit(
            f"{SIM_PATH} is missing required columns: {sorted(missing)}\n"
            "Rerun fresh_100_fight_predictive_replay first."
        )

    # ------------------------------------------------------------------
    # Build canonical historical fighter-fight observations.
    # ------------------------------------------------------------------
    fights = aggregate_fights(
        build_paired_rounds(rounds=load_round_stats())
    ).copy()

    fights["fight_id"] = fights["fight_id"].astype(str)

    # TD attempts/landed and ground entries are fighter-specific:
    # sum both fighter rows.
    #
    # modeled_ground_exposure_seconds is shared fight opportunity:
    # use MAX rather than SUM to avoid double counting.
    hist = (
        fights.groupby("fight_id", as_index=False)
        .agg(
            historical_td_attempts=("td_attempted", "sum"),
            historical_td_landed=("td_landed", "sum"),
            historical_ground_entries=("ground_entries", "sum"),
            historical_ground_seconds=("modeled_ground_exposure_seconds", "max"),
            actual_sub_attempts=("effective_submission_attempts", "sum"),
        )
        .rename(columns={"fight_id": "bout_id"})
    )

    x = sim.merge(
        hist,
        on="bout_id",
        how="left",
        validate="one_to_one",
    )

    if x["historical_ground_seconds"].isna().any():
        missing_ids = x.loc[
            x["historical_ground_seconds"].isna(), "bout_id"
        ].tolist()
        raise SystemExit(
            f"Historical observations missing for bout IDs: {missing_ids}"
        )

    # ------------------------------------------------------------------
    # Residence time per ground entry.
    # ------------------------------------------------------------------
    x["historical_sec_per_entry"] = np.where(
        x["historical_ground_entries"] > 0,
        x["historical_ground_seconds"] / x["historical_ground_entries"],
        np.nan,
    )

    x["mc_sec_per_entry"] = np.where(
        x["simulated_mean_ground_entries"] > 0,
        x["simulated_mean_ground_seconds"]
        / x["simulated_mean_ground_entries"],
        np.nan,
    )

    # ------------------------------------------------------------------
    # Population calibration.
    # ------------------------------------------------------------------
    hist_td_attempts = float(x["historical_td_attempts"].sum())
    mc_td_attempts = float(x["simulated_mean_td_attempts"].sum())

    hist_td_landed = float(x["historical_td_landed"].sum())
    mc_td_landed = float(x["simulated_mean_td_landed"].sum())

    hist_entries = float(x["historical_ground_entries"].sum())
    mc_entries = float(x["simulated_mean_ground_entries"].sum())

    hist_ground = float(x["historical_ground_seconds"].sum())
    mc_ground = float(x["simulated_mean_ground_seconds"].sum())

    hist_sec_entry = hist_ground / hist_entries if hist_entries else np.nan
    mc_sec_entry = mc_ground / mc_entries if mc_entries else np.nan

    hist_td_success = (
        hist_td_landed / hist_td_attempts
        if hist_td_attempts
        else np.nan
    )

    mc_td_success = (
        mc_td_landed / mc_td_attempts
        if mc_td_attempts
        else np.nan
    )

    print("=" * 110)
    print("GROUND PIPELINE — HISTORICAL vs EVENT MC — SAME 100 FIGHTS")
    print("=" * 110)

    print("\nPOPULATION TOTAL / MEAN CALIBRATION")

    print(
        f"TD attempts/fight        "
        f"hist={x.historical_td_attempts.mean():.3f}  "
        f"MC={x.simulated_mean_td_attempts.mean():.3f}  "
        f"ratio={ratio(x.simulated_mean_td_attempts, x.historical_td_attempts):.3f}"
    )

    print(
        f"TD landed/fight          "
        f"hist={x.historical_td_landed.mean():.3f}  "
        f"MC={x.simulated_mean_td_landed.mean():.3f}  "
        f"ratio={ratio(x.simulated_mean_td_landed, x.historical_td_landed):.3f}"
    )

    print(
        f"ground entries/fight     "
        f"hist={x.historical_ground_entries.mean():.3f}  "
        f"MC={x.simulated_mean_ground_entries.mean():.3f}  "
        f"ratio={ratio(x.simulated_mean_ground_entries, x.historical_ground_entries):.3f}"
    )

    print(
        f"ground seconds/fight     "
        f"hist={x.historical_ground_seconds.mean():.1f}  "
        f"MC={x.simulated_mean_ground_seconds.mean():.1f}  "
        f"ratio={ratio(x.simulated_mean_ground_seconds, x.historical_ground_seconds):.3f}"
    )

    print(
        f"seconds/ground entry     "
        f"hist={hist_sec_entry:.1f}  "
        f"MC={mc_sec_entry:.1f}  "
        f"ratio={mc_sec_entry / hist_sec_entry:.3f}"
    )

    print(
        f"TD success               "
        f"hist={hist_td_success:.3f}  "
        f"MC={mc_td_success:.3f}"
    )

    # ------------------------------------------------------------------
    # Fight-level historical-vs-simulated discrimination.
    # ------------------------------------------------------------------
    print("\nFIGHT-LEVEL CORRELATION WITH HISTORICAL VALUE")

    pairs = [
        (
            "TD attempts",
            "historical_td_attempts",
            "simulated_mean_td_attempts",
        ),
        (
            "TD landed",
            "historical_td_landed",
            "simulated_mean_td_landed",
        ),
        (
            "ground entries",
            "historical_ground_entries",
            "simulated_mean_ground_entries",
        ),
        (
            "ground seconds",
            "historical_ground_seconds",
            "simulated_mean_ground_seconds",
        ),
    ]

    for name, historical, modeled in pairs:
        pearson = x[historical].corr(x[modeled])
        spearman = x[historical].corr(
            x[modeled],
            method="spearman",
        )

        print(
            f"{name:20s}: "
            f"Pearson={pearson: .3f}  "
            f"Spearman={spearman: .3f}"
        )

    # ------------------------------------------------------------------
    # Which ground variables actually correspond to submission activity?
    # ------------------------------------------------------------------
    print("\nCORRELATION WITH ACTUAL SUBMISSION ATTEMPTS")

    submission_pairs = [
        ("historical TD attempts", "historical_td_attempts"),
        ("MC TD attempts", "simulated_mean_td_attempts"),
        ("historical TD landed", "historical_td_landed"),
        ("MC TD landed", "simulated_mean_td_landed"),
        ("historical entries", "historical_ground_entries"),
        ("MC entries", "simulated_mean_ground_entries"),
        ("historical ground sec", "historical_ground_seconds"),
        ("MC ground sec", "simulated_mean_ground_seconds"),
        ("MC submission attempts", "simulated_mean_submission_attempts"),
    ]

    for name, column in submission_pairs:
        pearson = x["actual_sub_attempts"].corr(x[column])
        spearman = x["actual_sub_attempts"].corr(
            x[column],
            method="spearman",
        )

        print(
            f"{name:25s}: "
            f"Pearson={pearson: .3f}  "
            f"Spearman={spearman: .3f}"
        )

    # ------------------------------------------------------------------
    # Largest ground-time misses.
    # ------------------------------------------------------------------
    x["ground_sec_error"] = (
        x["simulated_mean_ground_seconds"]
        - x["historical_ground_seconds"]
    )

    x["abs_ground_sec_error"] = x["ground_sec_error"].abs()

    print("\n" + "=" * 110)
    print("BIGGEST GROUND-TIME MISSES")
    print("=" * 110)

    cols = [
        "red_fighter",
        "blue_fighter",
        "actual_sub_attempts",
        "historical_td_attempts",
        "simulated_mean_td_attempts",
        "historical_td_landed",
        "simulated_mean_td_landed",
        "historical_ground_entries",
        "simulated_mean_ground_entries",
        "historical_ground_seconds",
        "simulated_mean_ground_seconds",
        "historical_sec_per_entry",
        "mc_sec_per_entry",
    ]

    print(
        x.sort_values(
            "abs_ground_sec_error",
            ascending=False,
        )[cols]
        .head(30)
        .to_string(
            index=False,
            formatters={
                "actual_sub_attempts": "{:.0f}".format,
                "historical_td_attempts": "{:.0f}".format,
                "simulated_mean_td_attempts": "{:.2f}".format,
                "historical_td_landed": "{:.0f}".format,
                "simulated_mean_td_landed": "{:.2f}".format,
                "historical_ground_entries": "{:.0f}".format,
                "simulated_mean_ground_entries": "{:.2f}".format,
                "historical_ground_seconds": "{:.1f}".format,
                "simulated_mean_ground_seconds": "{:.1f}".format,
                "historical_sec_per_entry": (
                    lambda v: "NA" if pd.isna(v) else f"{v:.1f}"
                ),
                "mc_sec_per_entry": (
                    lambda v: "NA" if pd.isna(v) else f"{v:.1f}"
                ),
            },
        )
    )

    x.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
