"""Orchestrates production rolling UFC base feature generation.

Future responsibility:
- Read data/master/ufc_master.parquet via MASTER_PATH
- Build chronological fighter-state features
- Add EWM/recent-form features
- Write data/features/UFC_enhanced_rolling_features_EWM.parquet via ROLLING_FEATURES_PATH
- Preserve the 483-column rolling feature contract

Migration status:
- Skeleton only. No notebook logic has been moved yet.
"""
