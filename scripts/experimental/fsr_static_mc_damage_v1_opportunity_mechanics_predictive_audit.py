"""Audit matchup-level opportunity mechanics using existing 300-bout phase diagnostic.

Research-only. Reuses the previously generated 30,000-path phase-mix artifact;
no simulator constants or mechanics are changed and no MC rerun is required.

Questions:
- Does MC rank which bouts will have high takedown-attempt volume?
- Does MC rank which bouts will have high control share?
- Does MC rank which bouts will have high clinch / ground strike opportunity?
- Are failures mostly calibration (wrong average level) or discrimination
  (wrong matchup ordering)?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score


INPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_historical_300_phase_mix_diagnostic.parquet"
)
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "fsr_static_mc_damage_v1_opportunity_mechanics_predictive_audit.parquet"
)


def _safe_auc(actual: pd.Series, score: pd.Series) -> float:
    work = pd.DataFrame({"actual": actual, "score": score}).dropna()
    if len(work) < 4:
        return float("nan")
    cutoff = float(work["actual"].quantile(0.75))
    y = (work["actual"] >= cutoff).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, work["score"]))


def _quartiles(df: pd.DataFrame, actual_col: str, sim_col: str, label: str) -> pd.DataFrame:
    work = df[[actual_col, sim_col]].dropna().copy()
    if work[sim_col].nunique() < 4:
        return pd.DataFrame()
    work["mc_quartile"] = pd.qcut(
        work[sim_col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    out = (
        work.groupby("mc_quartile", observed=True, as_index=False)
        .agg(
            bouts=(actual_col, "size"),
            mc_mean=(sim_col, "mean"),
            actual_mean=(actual_col, "mean"),
        )
    )
    out.insert(0, "metric", label)
    return out


def _metric_summary(df: pd.DataFrame, actual_col: str, sim_col: str, label: str) -> dict[str, object]:
    work = df[[actual_col, sim_col]].dropna().copy()
    a = work[actual_col].astype(float)
    s = work[sim_col].astype(float)
    return {
        "metric": label,
        "bouts": len(work),
        "actual_mean": float(a.mean()),
        "mc_mean": float(s.mean()),
        "mc_minus_actual": float(s.mean() - a.mean()),
        "actual_over_mc": float(a.mean() / s.mean()) if abs(float(s.mean())) > 1e-12 else np.nan,
        "pearson": float(a.corr(s, method="pearson")),
        "spearman": float(a.corr(s, method="spearman")),
        "mae": float(mean_absolute_error(a, s)),
        "top_quartile_auc": _safe_auc(a, s),
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "bout_id", "actual_elapsed_sec", "td_attempted", "sim_td_att",
        "ctrl_sec", "sim_control_sec",
        "actual_clinch_att_share", "sim_clinch_att_share",
        "actual_ground_att_share", "sim_ground_att_share",
        "actual_distance_att_share", "sim_distance_att_share",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Phase diagnostic artifact missing columns: {missing}")

    out = df.copy()
    elapsed_sec = pd.to_numeric(out["actual_elapsed_sec"], errors="coerce").clip(lower=1.0)
    elapsed_min = elapsed_sec / 60.0

    out["actual_td_att_per_min"] = pd.to_numeric(out["td_attempted"], errors="coerce") / elapsed_min
    out["sim_td_att_per_min"] = pd.to_numeric(out["sim_td_att"], errors="coerce") / elapsed_min
    out["actual_ctrl_share"] = pd.to_numeric(out["ctrl_sec"], errors="coerce") / elapsed_sec
    out["sim_ctrl_share"] = pd.to_numeric(out["sim_control_sec"], errors="coerce") / elapsed_sec

    # Treat phase-specific strike-attempt shares as UFCStats-observable proxies
    # for clinch/ground opportunity. Exact historical phase residence time is
    # unavailable, so these are not literal time-share targets.
    out["actual_clinch_opportunity"] = pd.to_numeric(out["actual_clinch_att_share"], errors="coerce")
    out["sim_clinch_opportunity"] = pd.to_numeric(out["sim_clinch_att_share"], errors="coerce")
    out["actual_ground_opportunity"] = pd.to_numeric(out["actual_ground_att_share"], errors="coerce")
    out["sim_ground_opportunity"] = pd.to_numeric(out["sim_ground_att_share"], errors="coerce")
    out["actual_distance_opportunity"] = pd.to_numeric(out["actual_distance_att_share"], errors="coerce")
    out["sim_distance_opportunity"] = pd.to_numeric(out["sim_distance_att_share"], errors="coerce")
    return out


def _print_summary(df: pd.DataFrame) -> None:
    metrics = [
        ("actual_td_att_per_min", "sim_td_att_per_min", "TD attempts/min"),
        ("actual_ctrl_share", "sim_ctrl_share", "control share"),
        ("actual_clinch_opportunity", "sim_clinch_opportunity", "clinch opportunity proxy"),
        ("actual_ground_opportunity", "sim_ground_opportunity", "ground opportunity proxy"),
        ("actual_distance_opportunity", "sim_distance_opportunity", "distance opportunity proxy"),
    ]

    summaries = [_metric_summary(df, a, s, label) for a, s, label in metrics]
    summary_df = pd.DataFrame(summaries)

    print("\n" + "=" * 118)
    print("300-BOUT OPPORTUNITY MECHANICS PREDICTIVE AUDIT")
    print("=" * 118)
    print(f"bouts: {len(df):,}")
    print("No simulator rerun; reusing the existing time-matched 30,000-path phase diagnostic.")
    print("\nMATCHUP-LEVEL PREDICTIVE VALUE")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nACTUAL OPPORTUNITY BY MC-PREDICTED QUARTILE")
    for actual_col, sim_col, label in metrics:
        q = _quartiles(df, actual_col, sim_col, label)
        print(f"\n{label}")
        if q.empty:
            print("insufficient distinct MC values for quartiles")
        else:
            print(q.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # A compact composite ranking diagnostic. Each component is percentile-ranked
    # first so unlike units do not dominate the composite.
    components = [
        ("actual_td_att_per_min", "sim_td_att_per_min"),
        ("actual_ctrl_share", "sim_ctrl_share"),
        ("actual_clinch_opportunity", "sim_clinch_opportunity"),
        ("actual_ground_opportunity", "sim_ground_opportunity"),
    ]
    actual_rank = pd.concat([df[a].rank(pct=True) for a, _ in components], axis=1).mean(axis=1)
    sim_rank = pd.concat([df[s].rank(pct=True) for _, s in components], axis=1).mean(axis=1)
    composite_spearman = float(actual_rank.corr(sim_rank, method="spearman"))
    print("\nCOMPOSITE NON-DISTANCE OPPORTUNITY RANKING")
    print(f"Spearman={composite_spearman:.4f}")

    print("\nINTERPRETATION GUIDE")
    print("- Low mean but decent Spearman/AUC => mostly calibration problem; ranking has value.")
    print("- Low mean and weak Spearman/AUC => both opportunity level and matchup selection are wrong.")
    print("- Good mean but weak ranking => population average hides matchup-level errors.")
    print("- Do not tune a mechanic from mean bias alone; preserve moneyline/prop regression gates.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit MC opportunity-mechanics predictive value")
    ap.add_argument("--input", type=Path, default=INPUT_PATH)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Phase diagnostic not found: {args.input}. Run the 300-bout phase-mix diagnostic first."
        )
    raw = pd.read_parquet(args.input)
    df = _prepare(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    _print_summary(df)
    print(f"\n[opportunity audit] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
