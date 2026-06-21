"""Generate transparent style-score weights from k-cluster profiles.

Research-only runner. It reads style_cluster_profiles.csv from the style matchup
research directory, derives feature weights from each cluster's standardized
feature distinctiveness, and writes:

- style_score_weights.yaml
- style_score_weight_audit.csv
- style_score_weight_summary.json

The intent is to turn hard clusters into interpretable continuous style scores.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = "pipeline/research/style_matchups/style_config.yaml"
DEFAULT_MIN_ABS_Z = 0.50
DEFAULT_MAX_FEATURES = 8

# Locked k=5 semantic labels from profile inspection and persistent ROI analysis.
DEFAULT_CLUSTER_LABELS = {
    0: "control_wrestler",
    1: "ko_finisher",
    2: "submission_grappler",
    3: "decision_technician",
    4: "all_round_finisher",
}

EXCLUDE_BASE_FEATURES = {
    "snapshot_count",
    "fighter_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate style-score weights from cluster profiles.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--min-abs-z", type=float, default=DEFAULT_MIN_ABS_Z)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style research config not found: {p}")
    config = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Style research config must be a mapping: {p}")
    return config


def base_feature_from_mean_col(column: str) -> str | None:
    match = re.fullmatch(r"(.+)_mean", column)
    if not match:
        return None
    feature = match.group(1)
    if feature in EXCLUDE_BASE_FEATURES:
        return None
    return feature


def mean_columns(profile: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in profile.columns:
        feature = base_feature_from_mean_col(col)
        if feature is not None:
            cols.append(col)
    return cols


def derive_global_stats(profile: pd.DataFrame, mean_cols: list[str]) -> dict[str, dict[str, float]]:
    weights = pd.to_numeric(profile.get("snapshot_count", pd.Series([1] * len(profile))), errors="coerce").fillna(1)
    stats: dict[str, dict[str, float]] = {}
    for col in mean_cols:
        values = pd.to_numeric(profile[col], errors="coerce")
        mask = values.notna() & weights.notna() & weights.gt(0)
        if not mask.any():
            continue
        global_mean = float((values[mask] * weights[mask]).sum() / weights[mask].sum())
        variance = float((((values[mask] - global_mean) ** 2) * weights[mask]).sum() / weights[mask].sum())
        global_std = variance ** 0.5
        if global_std <= 0:
            continue
        stats[col] = {"global_mean": global_mean, "global_std": global_std}
    return stats


def build_weight_rows(
    profile: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    min_abs_z: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, cluster_row in profile.iterrows():
        cluster = int(cluster_row["style_cluster"])
        style_name = DEFAULT_CLUSTER_LABELS.get(cluster, f"cluster_{cluster}_style")
        cluster_candidates: list[dict[str, Any]] = []
        for col, col_stats in stats.items():
            feature = base_feature_from_mean_col(col)
            if feature is None:
                continue
            cluster_mean = pd.to_numeric(pd.Series([cluster_row[col]]), errors="coerce").iloc[0]
            if pd.isna(cluster_mean):
                continue
            z = (float(cluster_mean) - col_stats["global_mean"]) / col_stats["global_std"]
            abs_z = abs(z)
            if abs_z < min_abs_z:
                continue
            cluster_candidates.append({
                "style_name": style_name,
                "source_cluster": cluster,
                "feature": feature,
                "cluster_mean": float(cluster_mean),
                "global_mean": col_stats["global_mean"],
                "global_std": col_stats["global_std"],
                "z_score": float(z),
                "abs_z_score": float(abs_z),
                "direction": 1 if z >= 0 else -1,
            })
        cluster_candidates = sorted(cluster_candidates, key=lambda r: r["abs_z_score"], reverse=True)[:max_features]
        denom = sum(r["abs_z_score"] for r in cluster_candidates)
        for rank, row in enumerate(cluster_candidates, start=1):
            row["rank"] = rank
            row["weight_abs"] = row["abs_z_score"] / denom if denom else 0.0
            row["signed_weight"] = row["weight_abs"] * row["direction"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_yaml_payload(audit: pd.DataFrame, min_abs_z: float, max_features: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "source": "data/research/style_matchups/style_cluster_profiles.csv",
            "method": "feature weight = abs(cluster_mean - global_mean) / global_std, normalized within style",
            "min_abs_z": float(min_abs_z),
            "max_features_per_style": int(max_features),
            "allow_negative_weights": True,
            "normalize_abs_weights_to_1": True,
            "locked_k": 5,
        },
        "style_score_weights": {},
    }
    if audit.empty:
        return payload
    for style_name, group in audit.groupby("style_name", sort=False):
        g = group.sort_values("rank")
        source_cluster = int(g["source_cluster"].iloc[0])
        weights = {
            str(row["feature"]): float(row["signed_weight"])
            for _, row in g.iterrows()
        }
        payload["style_score_weights"][style_name] = {
            "source_cluster": source_cluster,
            "weights": weights,
        }
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    research_dir = Path(str((config.get("outputs") or {}).get("research_dir", "data/research/style_matchups")))
    profiles_path = research_dir / "style_cluster_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Style cluster profiles not found: {profiles_path}")

    profile = pd.read_csv(profiles_path)
    required = {"style_cluster", "snapshot_count"}
    missing = sorted(required.difference(profile.columns))
    if missing:
        raise ValueError(f"Cluster profiles missing required columns: {missing}")

    mean_cols = mean_columns(profile)
    stats = derive_global_stats(profile, mean_cols)
    audit = build_weight_rows(
        profile=profile,
        stats=stats,
        min_abs_z=args.min_abs_z,
        max_features=args.max_features,
    )
    payload = build_yaml_payload(audit, args.min_abs_z, args.max_features)

    weights_path = research_dir / "style_score_weights.yaml"
    audit_path = research_dir / "style_score_weight_audit.csv"
    summary_path = research_dir / "style_score_weight_summary.json"

    audit.to_csv(audit_path, index=False)
    weights_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "profiles_path": str(profiles_path),
                "weights_path": str(weights_path),
                "audit_path": str(audit_path),
                "style_count": int(audit["style_name"].nunique()) if not audit.empty else 0,
                "feature_rows": int(len(audit)),
                "min_abs_z": float(args.min_abs_z),
                "max_features_per_style": int(args.max_features),
                "styles": payload["style_score_weights"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved style score weights: {weights_path}")
    print(f"Saved weight audit       : {audit_path}")
    print(f"Saved summary            : {summary_path}")
    print("DONE")


if __name__ == "__main__":
    main()
