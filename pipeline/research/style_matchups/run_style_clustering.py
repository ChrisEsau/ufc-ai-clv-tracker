"""Run standalone UFC style clustering research.

Run from the repository root after building the style dataset:

    python -m pipeline.research.style_matchups.run_style_clustering

This runner is intentionally research-only. It reads
``style_fighter_snapshots.parquet``, scales configured numeric style columns,
evaluates KMeans clusters for k=4..8, writes the best cluster assignments, and
summarizes cluster profiles for manual style labeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_K_VALUES = [4, 5, 6, 7, 8]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the clustering runner."""

    parser = argparse.ArgumentParser(description="Run UFC style clustering research.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the style-matchup research YAML config.",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated KMeans cluster counts to evaluate.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the style-matchup research config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Style research config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {path}")
    return config


def parse_k_values(value: str) -> list[int]:
    """Parse and validate comma-separated k values."""

    k_values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(k < 2 for k in k_values):
        raise ValueError(f"KMeans k values must be >= 2: {k_values}")
    return k_values


def _available_style_columns(df: pd.DataFrame, configured_columns: list[str]) -> list[str]:
    """Return configured style columns present in the dataset."""

    return [column for column in configured_columns if column in df.columns]


def _identity_columns(df: pd.DataFrame, configured_columns: list[str]) -> list[str]:
    """Return configured identity columns present in the dataset."""

    return [column for column in configured_columns if column in df.columns]


def build_feature_matrix(style_df: pd.DataFrame, style_columns: list[str]) -> tuple[pd.DataFrame, Pipeline]:
    """Build the scaled numeric clustering matrix.

    Median imputation keeps the clustering runner tolerant of sparse columns while
    preserving the missingness review as a separate research concern.
    """

    matrix_df = style_df.loc[:, style_columns].apply(pd.to_numeric, errors="coerce")
    usable_columns = [column for column in matrix_df.columns if matrix_df[column].notna().any()]
    if len(usable_columns) < 2:
        raise ValueError(f"Need at least two usable style columns for clustering, observed: {usable_columns}")

    matrix_df = matrix_df.loc[:, usable_columns]
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaled = pipeline.fit_transform(matrix_df)
    scaled_df = pd.DataFrame(scaled, columns=usable_columns, index=style_df.index)
    return scaled_df, pipeline


def evaluate_kmeans(scaled_df: pd.DataFrame, k_values: list[int]) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Evaluate KMeans for each k and return metrics plus labels."""

    metrics: list[dict[str, Any]] = []
    labels_by_k: dict[int, pd.Series] = {}
    sample_count = len(scaled_df)

    for k in k_values:
        if sample_count <= k:
            print(f"Skipping k={k}: sample count {sample_count} is not greater than k")
            continue

        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = pd.Series(model.fit_predict(scaled_df), index=scaled_df.index, name=f"style_cluster_k{k}")
        counts = labels.value_counts().sort_index()
        min_cluster_size = int(counts.min())
        min_cluster_share = float(min_cluster_size / sample_count)
        score = float(silhouette_score(scaled_df, labels))

        metrics.append(
            {
                "k": k,
                "silhouette_score": score,
                "sample_count": int(sample_count),
                "min_cluster_size": min_cluster_size,
                "min_cluster_share": min_cluster_share,
                "max_cluster_size": int(counts.max()),
            }
        )
        labels_by_k[k] = labels

    if not metrics:
        raise ValueError("No valid KMeans configurations were evaluated.")

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["silhouette_score", "min_cluster_share"], ascending=[False, False]
    )
    return metrics_df, labels_by_k


def build_cluster_profiles(
    *,
    assignments_df: pd.DataFrame,
    style_columns: list[str],
    selected_cluster_column: str,
) -> pd.DataFrame:
    """Summarize numeric style features by selected cluster."""

    numeric = assignments_df.loc[:, style_columns].apply(pd.to_numeric, errors="coerce")
    profile_df = numeric.groupby(assignments_df[selected_cluster_column]).agg(["mean", "median", "std"])
    profile_df.columns = [f"{feature}_{stat}" for feature, stat in profile_df.columns]

    counts = assignments_df[selected_cluster_column].value_counts().sort_index().rename("snapshot_count")
    fighter_counts = (
        assignments_df.groupby(selected_cluster_column)["fighter_id"].nunique().rename("fighter_count")
        if "fighter_id" in assignments_df.columns
        else pd.Series(dtype="int64", name="fighter_count")
    )
    profile_df = profile_df.join(counts).join(fighter_counts)
    profile_df = profile_df.reset_index().rename(columns={selected_cluster_column: "style_cluster"})

    ordered_columns = ["style_cluster", "snapshot_count", "fighter_count"]
    ordered_columns += [column for column in profile_df.columns if column not in ordered_columns]
    return profile_df.loc[:, ordered_columns]


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    k_values = parse_k_values(args.k_values)

    outputs = config.get("outputs") or {}
    research_dir = Path(str(outputs.get("research_dir", "data/research/style_matchups")))
    input_path = Path(str(outputs.get("style_fighter_snapshots_path", research_dir / "style_fighter_snapshots.parquet")))
    assignments_path = research_dir / "style_cluster_assignments.parquet"
    profiles_path = research_dir / "style_cluster_profiles.csv"
    metrics_path = research_dir / "style_cluster_metrics.csv"
    summary_path = research_dir / "style_cluster_summary.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Style fighter snapshots not found: {input_path}. "
            "Run build_style_dataset before clustering."
        )

    print("=" * 80)
    print("RUN UFC STYLE CLUSTERING RESEARCH")
    print("=" * 80)
    print(f"Input path: {input_path}")
    print(f"K values  : {k_values}")

    style_df = pd.read_parquet(input_path)
    print(f"Input shape: {style_df.shape}")

    configured_identity_columns = [str(column) for column in config.get("identity_columns", [])]
    configured_style_columns = [str(column) for column in config.get("style_columns", [])]
    identity_columns = _identity_columns(style_df, configured_identity_columns)
    style_columns = _available_style_columns(style_df, configured_style_columns)
    if not style_columns:
        raise ValueError("No configured style columns found in style dataset.")

    scaled_df, _ = build_feature_matrix(style_df, style_columns)
    usable_style_columns = list(scaled_df.columns)
    print(f"Usable style columns: {len(usable_style_columns)}")

    metrics_df, labels_by_k = evaluate_kmeans(scaled_df, k_values)
    selected_k = int(metrics_df.iloc[0]["k"])
    selected_cluster_column = f"style_cluster_k{selected_k}"
    print(f"Selected k: {selected_k}")
    print(metrics_df.to_string(index=False))

    assignments_df = style_df.loc[:, list(dict.fromkeys([*identity_columns, *usable_style_columns]))].copy()
    for k, labels in labels_by_k.items():
        assignments_df[f"style_cluster_k{k}"] = labels.astype(int)
    assignments_df["style_cluster"] = assignments_df[selected_cluster_column].astype(int)
    assignments_df["style_cluster_label"] = "cluster_" + assignments_df["style_cluster"].astype(str)

    profiles_df = build_cluster_profiles(
        assignments_df=assignments_df,
        style_columns=usable_style_columns,
        selected_cluster_column="style_cluster",
    )

    research_dir.mkdir(parents=True, exist_ok=True)
    assignments_df.to_parquet(assignments_path, index=False)
    profiles_df.to_csv(profiles_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "selected_k": selected_k,
                "selected_cluster_column": selected_cluster_column,
                "sample_count": int(len(assignments_df)),
                "usable_style_columns": usable_style_columns,
                "k_values_evaluated": [int(k) for k in labels_by_k],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved assignments: {assignments_path}")
    print(f"Saved profiles   : {profiles_path}")
    print(f"Saved metrics    : {metrics_path}")
    print(f"Saved summary    : {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
