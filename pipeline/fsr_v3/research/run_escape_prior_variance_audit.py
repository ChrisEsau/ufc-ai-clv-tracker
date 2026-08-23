"""Leakage-safe execution wrapper for the escape/retention prior-variance audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.fsr_v3.research import escape_prior_variance_audit as audit

INITIAL_POPULATION_MEAN_SECONDS = 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=audit.DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fights = audit.build_fighter_fights()
    train = fights[fights["event_date"] < audit.VALIDATION_START]
    training_pool_mean, alpha = audit._fit_alpha(train)
    initial_mean = INITIAL_POPULATION_MEAN_SECONDS

    active = fights[fights["ground_entries"] > 0].copy()
    zero_control = active["qualified_control_inflicted_seconds"].eq(0)
    print("=" * 120)
    print("FSR V3 ACTIVE TRAIT AUDIT — ESCAPE / RETENTION")
    print("=" * 120)
    print(f"fighter-fights total: {len(fights):,}")
    print(f"native rows with >=1 modeled ground entry: {len(active):,}")
    print(f"zero-control share conditional on entry: {zero_control.mean():.2%}")
    print(f"leakage-safe initial population mean: {initial_mean:.3f} sec/entry")
    print(f"pre-2022 pooled mean (dispersion fit diagnostic only): {training_pool_mean:.3f} sec/entry")
    print(f"pre-2022 fitted NB2 alpha: {alpha:.6f}")

    legacy_records = []
    legacy_frames = {}
    for k in audit.K_CANDIDATES:
        frame = audit.replay_legacy_k(fights, k=k, alpha=alpha, initial_pop_mean=initial_mean)
        legacy_frames[k] = frame
        val = audit.summarize_window(
            frame, audit.VALIDATION_START, audit.HOLDOUT_START,
            ll_col="legacy_ll", mae_col="legacy_abs_error_seconds",
        )
        hold = audit.summarize_window(
            frame, audit.HOLDOUT_START, None,
            ll_col="legacy_ll", mae_col="legacy_abs_error_seconds",
        )
        legacy_records += [
            {"k": k, "window": "validation_2022_2023", **val},
            {"k": k, "window": "holdout_2024plus", **hold},
        ]
    legacy_summary = pd.DataFrame(legacy_records)
    val_legacy = legacy_summary[legacy_summary["window"] == "validation_2022_2023"]
    best_k = float(val_legacy.sort_values(["total_ll", "k"], ascending=[False, True]).iloc[0]["k"])

    model_records = []
    for sigma_retention in audit.SIGMA_CANDIDATES:
        for sigma_escape in audit.SIGMA_CANDIDATES:
            rows = audit.replay_paired(
                fights,
                sigma_retention=sigma_retention,
                sigma_escape=sigma_escape,
                alpha=alpha,
                initial_pop_mean=initial_mean,
            ).rows
            val = audit.summarize_window(
                rows, audit.VALIDATION_START, audit.HOLDOUT_START,
                ll_col="predictive_ll_c_1", mae_col="plugin_abs_error_seconds",
            )
            model_records.append({
                "sigma_retention": sigma_retention,
                "sigma_escape": sigma_escape,
                "window": "validation_2022_2023",
                **val,
            })
    model_summary = pd.DataFrame(model_records)
    best = model_summary.sort_values(
        ["total_ll", "sigma_retention", "sigma_escape"], ascending=[False, True, True]
    ).iloc[0]
    best_sr = float(best["sigma_retention"])
    best_se = float(best["sigma_escape"])
    selected = audit.replay_paired(
        fights,
        sigma_retention=best_sr,
        sigma_escape=best_se,
        alpha=alpha,
        initial_pop_mean=initial_mean,
    ).rows

    c_records = []
    for c in audit.C_CANDIDATES:
        for label, start, end in (
            ("validation_2022_2023", audit.VALIDATION_START, audit.HOLDOUT_START),
            ("holdout_2024plus", audit.HOLDOUT_START, None),
        ):
            s = audit.summarize_window(
                selected, start, end,
                ll_col=f"predictive_ll_c_{c:g}", mae_col="plugin_abs_error_seconds",
            )
            c_records.append({
                "sigma_retention": best_sr, "sigma_escape": best_se,
                "c": c, "window": label, **s,
            })
    c_summary = pd.DataFrame(c_records)
    val_c = c_summary[c_summary["window"] == "validation_2022_2023"]
    best_c = float(val_c.sort_values(["total_ll", "c"], ascending=[False, True]).iloc[0]["c"])

    hold = selected[selected["event_date"] >= audit.HOLDOUT_START].copy()
    legacy_hold = legacy_frames[best_k][legacy_frames[best_k]["event_date"] >= audit.HOLDOUT_START][
        ["fight_id", "fighter_id", "legacy_ll", "legacy_abs_error_seconds"]
    ]
    hold = hold.merge(legacy_hold, on=["fight_id", "fighter_id"], how="inner", validate="one_to_one")
    hold["selected_ll"] = hold[f"predictive_ll_c_{best_c:g}"]
    hold["selected_vs_population"] = hold["selected_ll"] - hold["population_ll"]
    hold["selected_vs_legacy"] = hold["selected_ll"] - hold["legacy_ll"]

    b_pop = audit.bootstrap_delta(
        hold, "selected_ll", "population_ll", draws=args.bootstrap_draws
    )
    b_legacy = audit.bootstrap_delta(
        hold, "selected_ll", "legacy_ll", draws=args.bootstrap_draws, seed=audit.SEED + 1
    )

    bucket_rows = []
    for bucket, part in hold.groupby("prior_bucket", sort=False):
        bucket_rows.append({
            "prior_bucket": bucket,
            "rows": len(part),
            "fights": part["fight_id"].nunique(),
            "delta_ll_vs_population": part["selected_vs_population"].sum(),
            "delta_ll_vs_legacy": part["selected_vs_legacy"].sum(),
            "plugin_mae_seconds": part["plugin_abs_error_seconds"].mean(),
            "legacy_mae_seconds": part["legacy_abs_error_seconds"].mean(),
            "population_mae_seconds": part["population_abs_error_seconds"].mean(),
        })
    bucket_summary = pd.DataFrame(bucket_rows)

    legacy_summary.to_csv(args.output_dir / "legacy_k_sweep.csv", index=False)
    model_summary.to_csv(args.output_dir / "sigma_sweep.csv", index=False)
    c_summary.to_csv(args.output_dir / "variance_multiplier_sweep.csv", index=False)
    bucket_summary.to_csv(args.output_dir / "holdout_prior_buckets.csv", index=False)
    hold.to_csv(args.output_dir / "holdout_rows.csv", index=False)

    print("\nLEGACY PRIOR K — VALIDATION")
    print(val_legacy.sort_values("total_ll", ascending=False).to_string(index=False))
    print(f"selected K={best_k:g}; frozen V2 K=5")
    print("\nPAIRED PRIOR SD — TOP VALIDATION")
    print(model_summary.sort_values("total_ll", ascending=False).head(12).to_string(index=False))
    print(f"selected sigma_retention={best_sr:g}, sigma_escape={best_se:g}")
    print("\nEPISTEMIC C")
    print(c_summary.to_string(index=False))
    print(f"selected c={best_c:g}")
    print("\nUNTOUCHED HOLDOUT 2024+")
    print(
        f"selected vs population LL={b_pop['delta']:+.3f} "
        f"CI[{b_pop['ci_low']:+.3f},{b_pop['ci_high']:+.3f}] P(>0)={b_pop['p_gt_0']:.3f}"
    )
    print(
        f"selected vs best legacy LL={b_legacy['delta']:+.3f} "
        f"CI[{b_legacy['ci_low']:+.3f},{b_legacy['ci_high']:+.3f}] P(>0)={b_legacy['p_gt_0']:.3f}"
    )
    print(bucket_summary.to_string(index=False))
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
