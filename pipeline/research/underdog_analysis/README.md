# Underdog Analysis Research Workspace

This workspace is intentionally isolated from production model training, Model Lab, feature registries, live prediction outputs, and CLV artifacts.

## Purpose

Diagnose why underdog selections underperform relative to favorite selections.

The first study answers:

- Are underdogs systematically overestimated?
- Do underdogs produce inflated edge signals?
- Which odds buckets are profitable or unprofitable?
- Is ROI weakness concentrated in small, medium, or large underdogs?

## Run

From repo root:

```bash
python -m pipeline.research.underdog_analysis.run_underdog_audit \
  --model-id moneyline_xgboost_v11
```

Optional filters:

```bash
python -m pipeline.research.underdog_analysis.run_underdog_audit \
  --model-id moneyline_xgboost_v11 \
  --min-edge 0.03 \
  --min-confidence 0.55 \
  --min-odds -300 \
  --max-odds 500
```

## Inputs

Default inputs:

- `data/predictions/model_outcomes.parquet`
- `data/market/historical_market_outcomes.parquet`

## Outputs

Each run writes to:

```text
data/research/underdog_audit/<run_id>/
```

Generated files:

- `underdog_predictions.parquet`
- `underdog_roi.csv`
- `underdog_calibration.csv`
- `underdog_edge_distribution.csv`
- `underdog_odds_buckets.csv`
- `underdog_summary.json`

A run registry is also maintained at:

```text
data/research/underdog_audit/underdog_audit_registry.parquet
```

## Safety Boundary

This runner only reads historical model/market artifacts and writes research outputs under `data/research/`.

It does not modify:

- `configs/`
- `models/`
- `data/features/`
- `data/predictions/`
- `data/clv/`
- Model Lab registry artifacts
- production model artifacts
