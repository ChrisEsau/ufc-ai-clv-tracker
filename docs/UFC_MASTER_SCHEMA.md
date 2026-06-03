# UFC Master Schema

## Purpose

This document defines the authoritative schema for:

```text
data/master/ufc_master.parquet
```

The master dataset is the canonical historical UFC fight database used by:

* Data Maintenance
* Feature Engineering
* Model Training
* Live Prediction
* CLV Tracking
* Betting Board

The current canonical schema contains:

```text
128 columns
```

---

## Schema Rules

### 1. Master schema is authoritative

All staged rows must match the master schema exactly before append.

Required checks:

```text
column count match
column order match
no duplicate columns
no missing columns
no extra columns
```

Validated by:

```text
pipeline.data_maintenance.run_master_column_validation
```

---

### 2. Append is blocked unless schema validation and final review pass

No staged rows may be appended to master unless:

```text
validation_pass == True
append_ready == True
final_review_pass == True
```

Validated by:

```text
pipeline.data_maintenance.run_append_precheck_validation
pipeline.data_maintenance.run_staged_final_review
```

---

### 3. URLs are not stored in master

URLs are allowed in:

```text
scrapers
staging artifacts
audit artifacts
```

URLs are removed at the mapper boundary.

Master stores IDs only:

```text
event_id
fight_id
r_id
b_id
winner_id
```

---

### 4. Date format

Canonical date format in the master dataset:

```text
M/D/YYYY
```

Example:

```text
9/6/2025
```

---

## Identity Columns

| Column         | Description                            |
| -------------- | -------------------------------------- |
| event_id       | UFCStats event identifier              |
| event_name     | UFC event name                         |
| date           | Event date                             |
| location       | Event location                         |
| fight_id       | UFCStats fight identifier              |
| division       | Weight class                           |
| title_fight    | Whether fight was a title fight (`1` yes, `0` no) |
| method         | Fight result method                    |
| finish_round   | Round fight ended                      |
| match_time_sec | Fight-ending time converted to seconds |
| total_rounds   | Scheduled round count (`3` or `5`)     |
| referee        | Referee name                           |

---

## Red Fighter Columns

| Column              | Description                       |
| ------------------- | --------------------------------- |
| r_name              | Red corner fighter name           |
| r_id                | Red fighter UFCStats ID           |
| r_kd                | Red knockdowns                    |
| r_sig_str_landed    | Red significant strikes landed    |
| r_sig_str_atmpted   | Red significant strikes attempted |
| r_sig_str_acc       | Red significant strike accuracy   |
| r_total_str_landed  | Red total strikes landed          |
| r_total_str_atmpted | Red total strikes attempted       |
| r_total_str_acc     | Red total strike accuracy         |
| r_td_landed         | Red takedowns landed              |
| r_td_atmpted        | Red takedowns attempted           |
| r_td_acc            | Red takedown accuracy             |
| r_sub_att           | Red submission attempts           |
| r_rev               | Red reversals                     |
| r_ctrl              | Red control time                  |

---

## Blue Fighter Columns

| Column              | Description                        |
| ------------------- | ---------------------------------- |
| b_name              | Blue corner fighter name           |
| b_id                | Blue fighter UFCStats ID           |
| b_kd                | Blue knockdowns                    |
| b_sig_str_landed    | Blue significant strikes landed    |
| b_sig_str_atmpted   | Blue significant strikes attempted |
| b_sig_str_acc       | Blue significant strike accuracy   |
| b_total_str_landed  | Blue total strikes landed          |
| b_total_str_atmpted | Blue total strikes attempted       |
| b_total_str_acc     | Blue total strike accuracy         |
| b_td_landed         | Blue takedowns landed              |
| b_td_atmpted        | Blue takedowns attempted           |
| b_td_acc            | Blue takedown accuracy             |
| b_sub_att           | Blue submission attempts           |
| b_rev               | Blue reversals                     |
| b_ctrl              | Blue control time                  |

---

## Red Zone Striking Columns

| Column           | Description                    |
| ---------------- | ------------------------------ |
| r_head_landed    | Red head strikes landed        |
| r_head_atmpted   | Red head strikes attempted     |
| r_head_acc       | Red head strike accuracy       |
| r_body_landed    | Red body strikes landed        |
| r_body_atmpted   | Red body strikes attempted     |
| r_body_acc       | Red body strike accuracy       |
| r_leg_landed     | Red leg strikes landed         |
| r_leg_atmpted    | Red leg strikes attempted      |
| r_leg_acc        | Red leg strike accuracy        |
| r_dist_landed    | Red distance strikes landed    |
| r_dist_atmpted   | Red distance strikes attempted |
| r_dist_acc       | Red distance strike accuracy   |
| r_clinch_landed  | Red clinch strikes landed      |
| r_clinch_atmpted | Red clinch strikes attempted   |
| r_clinch_acc     | Red clinch strike accuracy     |
| r_ground_landed  | Red ground strikes landed      |
| r_ground_atmpted | Red ground strikes attempted   |
| r_ground_acc     | Red ground strike accuracy     |

---

## Blue Zone Striking Columns

| Column           | Description                     |
| ---------------- | ------------------------------- |
| b_head_landed    | Blue head strikes landed        |
| b_head_atmpted   | Blue head strikes attempted     |
| b_head_acc       | Blue head strike accuracy       |
| b_body_landed    | Blue body strikes landed        |
| b_body_atmpted   | Blue body strikes attempted     |
| b_body_acc       | Blue body strike accuracy       |
| b_leg_landed     | Blue leg strikes landed         |
| b_leg_atmpted    | Blue leg strikes attempted      |
| b_leg_acc        | Blue leg strike accuracy        |
| b_dist_landed    | Blue distance strikes landed    |
| b_dist_atmpted   | Blue distance strikes attempted |
| b_dist_acc       | Blue distance strike accuracy   |
| b_clinch_landed  | Blue clinch strikes landed      |
| b_clinch_atmpted | Blue clinch strikes attempted   |
| b_clinch_acc     | Blue clinch strike accuracy     |
| b_ground_landed  | Blue ground strikes landed      |
| b_ground_atmpted | Blue ground strikes attempted   |
| b_ground_acc     | Blue ground strike accuracy     |

---

## Red Strike Distribution Columns

| Column              | Description                                           |
| ------------------- | ----------------------------------------------------- |
| r_landed_head_per   | Percent of red significant strikes landed to head     |
| r_landed_body_per   | Percent of red significant strikes landed to body     |
| r_landed_leg_per    | Percent of red significant strikes landed to leg      |
| r_landed_dist_per   | Percent of red significant strikes landed at distance |
| r_landed_clinch_per | Percent of red significant strikes landed in clinch   |
| r_landed_ground_per | Percent of red significant strikes landed on ground   |

---

## Blue Strike Distribution Columns

| Column              | Description                                            |
| ------------------- | ------------------------------------------------------ |
| b_landed_head_per   | Percent of blue significant strikes landed to head     |
| b_landed_body_per   | Percent of blue significant strikes landed to body     |
| b_landed_leg_per    | Percent of blue significant strikes landed to leg      |
| b_landed_dist_per   | Percent of blue significant strikes landed at distance |
| b_landed_clinch_per | Percent of blue significant strikes landed in clinch   |
| b_landed_ground_per | Percent of blue significant strikes landed on ground   |

---

## Red Fighter Profile Columns

| Column       | Description                                 |
| ------------ | ------------------------------------------- |
| r_nick_name  | Red fighter nickname                        |
| r_wins       | Red fighter career wins                     |
| r_losses     | Red fighter career losses                   |
| r_draws      | Red fighter career draws                    |
| r_height     | Red fighter height                          |
| r_weight     | Red fighter weight                          |
| r_reach      | Red fighter reach                           |
| r_stance     | Red fighter stance                          |
| r_dob        | Red fighter date of birth                   |
| r_splm       | Red significant strikes landed per minute   |
| r_str_acc    | Red career striking accuracy                |
| r_sapm       | Red significant strikes absorbed per minute |
| r_str_def    | Red striking defense                        |
| r_td_avg     | Red takedown average                        |
| r_td_avg_acc | Red takedown accuracy                       |
| r_td_def     | Red takedown defense                        |
| r_sub_avg    | Red submission average                      |

---

## Blue Fighter Profile Columns

| Column       | Description                                  |
| ------------ | -------------------------------------------- |
| b_nick_name  | Blue fighter nickname                        |
| b_wins       | Blue fighter career wins                     |
| b_losses     | Blue fighter career losses                   |
| b_draws      | Blue fighter career draws                    |
| b_height     | Blue fighter height                          |
| b_weight     | Blue fighter weight                          |
| b_reach      | Blue fighter reach                           |
| b_stance     | Blue fighter stance                          |
| b_dob        | Blue fighter date of birth                   |
| b_splm       | Blue significant strikes landed per minute   |
| b_str_acc    | Blue career striking accuracy                |
| b_sapm       | Blue significant strikes absorbed per minute |
| b_str_def    | Blue striking defense                        |
| b_td_avg     | Blue takedown average                        |
| b_td_avg_acc | Blue takedown accuracy                       |
| b_td_def     | Blue takedown defense                        |
| b_sub_avg    | Blue submission average                      |

---

## Result Columns

| Column    | Description                 |
| --------- | --------------------------- |
| winner    | Winning fighter name        |
| winner_id | Winning fighter UFCStats ID |

---

## Merge Artifact Columns

The current 128-column schema still includes cleanup-target merge artifact columns:

```text
r_name_x
b_name_x
r_name_y
b_name_y
```

These columns are retained for compatibility with the current master schema.

Future schema refactor should remove these only through a deliberate migration process.

---

## Metadata and Derived Value Rules

* New staged rows must populate `location`, `division`, `title_fight`, and `total_rounds` before append.
* `title_fight` must use numeric flags: `1` for yes and `0` for no.
* `total_rounds` must be the scheduled round count, normally `3` or `5`.
* Accuracy and percentage-derived fields should use `0` when an attempted/denominator value is zero or missing, rather than storing `NA` for a zero-attempt calculation.

---

## Validation Artifacts

Schema validation output:

```text
data/audits/ufc_master_column_validation.parquet
```

Append precheck output:

```text
data/audits/ufc_append_precheck.parquet
```

Required field audit:

```text
data/audits/ufc_append_required_field_audit.parquet
```

Duplicate check audit:

```text
data/audits/ufc_append_duplicate_check.parquet
```

Final staged review output:

```text
data/audits/ufc_staged_final_review.parquet
```

---

## Schema Change Rules

Any master schema change must update:

1. `docs/UFC_MASTER_SCHEMA.md`
2. `pipeline/data_maintenance/run_staged_master_mapper.py`
3. `pipeline/data_maintenance/run_master_column_validation.py`
4. `pipeline/data_maintenance/run_append_precheck_validation.py`
5. Downstream feature engineering
6. Prediction model feature registry
7. Dashboard readers

Schema changes should be versioned and tested before appending to master.
