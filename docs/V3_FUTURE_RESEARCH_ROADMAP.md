# UFC Platform V3 Future Research Roadmap

_Last updated: 2026-06-09_

## Purpose

This document captures future research directions, feature ideas, model expansions, and long-term platform vision beyond the current V2 architecture.

The goal is not to implement everything immediately. The goal is to maintain a prioritized list of high-value research opportunities that can be evaluated once the V2 migration is complete and stable.

---

# Guiding Principle

The long-term objective is not simply:

"Predict UFC fight winners"

The long-term objective is:

"Build a UFC quantitative research platform capable of identifying profitable betting opportunities across multiple market types while continuously measuring model quality through CLV, calibration, and bankroll performance."

---

# Tier 1 – Highest Expected ROI

## Market-Aware Features

Potential features:
- opening implied probability
- current implied probability
- line movement percentage
- line movement velocity
- sportsbook disagreement
- market consensus probability
- best available line
- steam move indicators

Expected value: Very High.

---

## Expanded Recency Architecture

Track multiple horizons:
- last 3 fights
- last 5 fights
- last 8 fights
- career

For:
- striking
- grappling
- finish rates
- ELO
- recent form

Expected value: High.

---

## Opponent Quality Adjustments

Examples:
- quality-adjusted striking
- quality-adjusted grappling
- quality-adjusted ELO
- quality-adjusted win rate

Equivalent to strength-of-schedule metrics.

Expected value: Very High.

---

# Tier 2 – Model Expansion

## Prop Betting Models

Future model families:
- Moneyline
- KO/TKO
- Submission
- Decision
- Fight Goes Distance
- Round Totals

Long-term architecture:
Moneyline Model
→ Method Model
→ Distance Model
→ Unified Betting Engine

---

## Ensemble Architecture

Potential members:
- XGBoost
- LightGBM
- Random Forest
- Logistic Baseline

Benefits:
- improved calibration
- reduced variance
- more stable predictions

---

## Confidence Meta-Model

Target:

Probability Primary Model Is Correct.

Inputs:
- model probability
- confidence
- market disagreement
- line movement
- feature completeness
- division
- fighter experience

Purpose:
- improve stake sizing
- improve risk management

---

# Tier 3 – Unique UFC Edges

## Style Matchup Engine

Classify fighters:
- wrestler
- striker
- pressure fighter
- counter striker
- submission specialist

Create matchup features:
- wrestler_vs_striker
- pressure_vs_counter
- southpaw_vs_orthodox

---

## Age Curve Modeling

Move beyond raw age.

Research:
- division-specific aging curves
- decline indicators
- peak-age windows

---

## Camp / Gym Effects

Examples:
- American Top Team
- AKA
- City Kickboxing
- Kill Cliff
- Factory X

Potential features:
- camp win rate
- camp momentum
- camp matchup strengths

---

# Tier 4 – Advanced Quant Research

## Fighter Trajectory Model

Predict whether a fighter is:
- improving
- plateauing
- declining

Potential trajectory metrics:
- ELO trajectory
- striking trajectory
- grappling trajectory
- finish trajectory

---

## Market Inefficiency Scanner

Focus on:
- sportsbook disagreement
- abnormal line movement
- opening line errors
- delayed market corrections

Question:
"Which markets are inefficient?"

instead of:

"Who wins?"

---

## Closing Line Prediction Model

Predict future closing odds.

Applications:
- CLV targeting
- timing optimization
- expected market movement

Expected value: Extremely High.

---

# Additional Research Ideas

## Referee Effects Model

Potential features:
- referee finish rate
- referee submission frequency
- referee early stoppage tendency
- referee average fight duration

Useful for props and totals.

---

## Judge Modeling

Potential features:
- judge aggressiveness
- judge striking preference
- judge grappling preference
- historical scorecard tendencies

Useful for decision markets.

---

## Travel and Elevation Effects

Potential features:
- travel distance
- time zone changes
- altitude adjustment
- international travel burden

Particularly interesting for Mexico City, Salt Lake City, Denver, and international cards.

---

## Short-Notice Replacement Model

Potential features:
- notice days
- camp disruption
- weight-cut disruption
- replacement fighter history

Historically short-notice fights behave differently.

---

## Weight Cut Risk Model

Potential features:
- missed weight history
- large weight cuts
- late replacement cuts
- age-adjusted weight cut stress

Potentially valuable for props and live betting.

---

## Championship Fight Effects

Potential features:
- title fight indicator
- five-round experience
- championship round experience
- championship round win rate

---

## Rematch Effects

Potential features:
- first fight result
- rematch timing
- prior finish method
- performance change since prior meeting

---

## Public Betting Sentiment Model

Potential future inputs:
- betting splits
- social sentiment
- media sentiment
- public-vs-sharp disagreement

---

## Market Microstructure Research

Study:
- line movement patterns
- opening-to-close efficiency
- bookmaker disagreement
- sharp-book leadership

Potential outcome:
Dedicated market model independent of fighter performance.

---

# Long-Term Platform Vision

Fighter State Engine
→ Feature View Layer
→ Moneyline Model
→ Method Model
→ Distance Model
→ Market Model
→ CLV Model
→ Confidence Model
→ Unified Betting Engine
→ Bankroll / CLV Tracking
→ Dashboard

The platform should eventually function like a quantitative trading system whose asset class happens to be UFC betting markets.
