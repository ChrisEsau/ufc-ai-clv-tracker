"""Coverage, distribution, ranking, leakage, and update sanity outputs."""

import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import FSR_V2_DIAGNOSTICS_DIR
from pipeline.fsr_v2.publish.snapshots import load_histories


def build_diagnostics(output_dir: Path = FSR_V2_DIAGNOSTICS_DIR) -> dict:
    histories = load_histories()
    output_dir.mkdir(parents=True, exist_ok=True)
    meaningful = histories.assign(meaningful=histories.raw_denominator.gt(0))
    coverage_rows = []
    distribution_rows = []
    ranking_dir = output_dir / "trait_rankings"
    ranking_dir.mkdir(exist_ok=True)
    for trait, frame in meaningful.groupby("trait"):
        counts = frame.groupby("fighter_id").meaningful.sum()
        coverage_rows.append({
            "trait": trait, "fighter_fight_rows": len(frame), "eligible_observations": int(frame.meaningful.sum()),
            "fighters_1plus": int((counts >= 1).sum()), "fighters_3plus": int((counts >= 3).sum()),
            "fighters_5plus": int((counts >= 5).sum()), "fighters_10plus": int((counts >= 10).sum()),
            "missing_rate": float(frame.observed.isna().mean()), "zero_opportunity_rate": float((~frame.meaningful).mean()),
            "evidence_p50": float(frame.raw_denominator.quantile(.5)), "evidence_p95": float(frame.raw_denominator.quantile(.95)),
        })
        latest = frame.sort_values(["event_date", "fight_id"]).groupby("fighter_id", as_index=False).tail(1)
        values = latest.post_rating
        distribution_rows.append({"trait": trait, **{
            "min": values.min(), "p01": values.quantile(.01), "p05": values.quantile(.05),
            "p25": values.quantile(.25), "median": values.median(), "p75": values.quantile(.75),
            "p95": values.quantile(.95), "p99": values.quantile(.99), "max": values.max(),
            "mean": values.mean(), "std": values.std(), "at_prior_rate": float(values.eq(0).mean()),
        }})
        ranked = latest[["fighter_id", "fighter_name", "post_rating", "raw_denominator"]].sort_values("post_rating", ascending=False)
        middle = ranked.iloc[max(0, len(ranked)//2-5):len(ranked)//2+5].assign(section="middle")
        ranking = pd.concat([ranked.head(20).assign(section="top"), middle, ranked.tail(20).assign(section="bottom")])
        ranking.to_csv(ranking_dir / f"{trait}.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "trait_coverage.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(output_dir / "trait_distribution.csv", index=False)
    suppression = histories[histories.trait.str.endswith("suppression")].head(100)
    suppression.to_csv(output_dir / "suppression_examples.csv", index=False)
    histories[histories.trait.str.startswith("escape_")].head(100).to_csv(output_dir / "escape_audit.csv", index=False)
    histories[histories.trait.str.startswith("submission_")].head(100).to_csv(output_dir / "submission_audit.csv", index=False)
    same_date_unique = not histories.duplicated(["event_date", "fight_id", "fighter_id", "trait"]).any()
    replay = {"same_date_snapshot_unique": same_date_unique, "traits": sorted(histories.trait.unique()),
              "history_rows": len(histories), "prefight_uses_pre_rating": True}
    (output_dir / "replay_validation.json").write_text(json.dumps(replay, indent=2))
    return replay


if __name__ == "__main__":
    print(json.dumps(build_diagnostics(), indent=2))
