# FSR Monte Carlo Card Validation — Pre-Change Baseline V1

**Baseline date:** 2026-08-11  
**Repository:** `ChrisEsau/ufc-ai-clv-tracker`  
**Branch:** `feature/fsr-32-stamina-shadow`  
**Status:** Research / shadow only  
**Purpose:** Freeze the current fight-by-fight validation results before additional simulator or FSR changes so later versions can be rerun on the exact same bouts and compared directly.

## Simulator Contract Used

- Engine: `scripts/experimental/fsr_static_mc_ko_sub_decision_v1.py`
- Runner: `scripts/experimental/run_single_historical_full_fight_bout.py`
- FSR database: FSR-32 leakage-safe historical profiles
- Eligibility: 2020+ aligned mature cohort; both fighters have at least 3 completed prior UFC fights
- Paths per fight: **1,000**
- Runner seed: default **20260811**
- Simulation horizon: **3 rounds for every run**
- Submission neutral `P(SUB | attempt)`: **34%**
- Decision layer: no draws, 10-9 rounds only
- Stored FSR remains immutable during each path

### Five-round caveat

The current runner simulated only three rounds even when the historical bout was scheduled for five. Those bouts remain useful for winner and R1-R3 mechanics, but their decision / goes-distance / totals output is not a valid full-fight five-round market comparison.

Five-round historical bouts in this baseline:
- Ilia Topuria vs Justin Gaethje
- Alex Pereira vs Ciryl Gane
- Manel Kape vs Kyoji Horiguchi
- Rafael Fiziev vs Manuel Torres
- Conor McGregor vs Max Holloway
- Dricus Du Plessis vs Kamaru Usman

## Baseline Summary

| Card | Eligible fights | MC correct winners | MC hit rate | MC expected KO | Actual KO | MC expected SUB | Actual SUB | MC expected DEC | Actual DEC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Freedom 250 | 7 | 7 | 100.0% | 2.58 | 7 | 0.88 | 0 | 3.54 | 0 |
| Kape vs Horiguchi | 8 | 3 | 37.5% | 0.99 | 3 | 1.23 | 1 | 5.78 | 4 |
| Fiziev vs Torres | 7 | 2 | 28.6% | 2.38 | 1 | 0.76 | 3 | 3.86 | 3 |
| UFC 329: McGregor vs Holloway 2 | 9 | 5 | 55.6% | 3.39 | 4 | 1.43 | 3 | 4.19 | 2 |
| Du Plessis vs Usman | 3 | 1 | 33.3% | 0.60 | 0 | 0.37 | 0 | 2.03 | 3 |
| **TOTAL** | **34** | **18** | **52.9%** | **9.93** | **15** | **4.66** | **7** | **19.40** | **12** |

Across all 34 eligible fights:
- MC winner hit rate: **18/34 = 52.9%**
- MC average expected method mix: **29.2% KO/TKO, 13.7% SUB, 57.1% DEC**
- Actual method mix: **44.1% KO/TKO, 20.6% SUB, 35.3% DEC**

## Preliminary Market Comparison

The market comparison gathered during the card-by-card review was not yet normalized into a single per-bout odds artifact, so treat this section as a **preliminary conversation-era checkpoint**, not the authoritative market baseline.

Through Card #5:
- MC actual-winner hit rate: **18/34 = 52.9%**
- Directional market-favorite comparison: **23/33 = 69.7%**
- MC / market directional agreement: **21/33 = 63.6%**
- MC contrarian calls when it disagreed with the market: **3/12 = 25.0%**
- Rafael Fiziev vs Manuel Torres was excluded from market-direction scoring because the reviewed pre-fight price was effectively a pick'em.

Before using market performance for formal calibration, store one consistent sportsbook / closing-price source and no-vig normalization per bout.

## Locked Evaluation Questions

These are **evaluation items, not approved simulator changes**.

1. **Age degradation beyond durability / KD resistance**
   - Determine whether age should reduce additional FSR traits.
   - Candidates include striking pressure/output, precision, defense, wrestling/explosiveness, control and stamina-related traits.
   - Do not apply a blanket age penalty without cohort evidence.

2. **FSR / matchup winner inversions**
   - Investigate bouts where the MC strongly preferred the opposite side from the market and/or actual result.
   - Trace whether the cause is stale FSR ability, matchup formulas, action generation, scoring, or another layer.

3. **KO calibration vs discrimination**
   - Separate whether the engine ranks KO-dangerous matchups correctly from whether absolute KO occurrence/timing is calibrated.
   - Do not solve this with a global KO multiplier based on a single card.

4. **Submission calibration vs discrimination**
   - Overall submission prevalence can be close while the wrong fighters/matchups receive the submission probability.
   - Preserve the locked semantic split: `submission_pressure` generates attempts; `submission_conversion` vs `submission_resistance` converts attempts.

5. **Fight duration / excess decisions**
   - Track whether excessive decisions remain a population-level problem after controlling for matchup type and 3R/5R horizon.

6. **Activity-to-damage bridge**
   - Compare actual UFCStats round activity with simulated attempts, lands, control, damage reservoir, stamina and damaging-contact severity.
   - Determine whether missed finishes arise before the damage layer or inside KO/TKO conversion.

7. **Layoff / recency**
   - Keep available as a diagnostic covariate.
   - The first two age-related misses examined did not show a simple longer-layoff disadvantage.

## Notable Diagnostic Bouts

These should be retained for targeted before/after reruns:

- `30cdb851f6c63444` — Josh Hokit vs Derrick Lewis: winner direction strong; simulation much too long / decision-heavy relative to prior market comparison.
- `5727d5be8c373346` — Alex Pereira vs Ciryl Gane: very strong Gane activity / decision dominance; useful action-generation diagnostic.
- `571a3eabff5c7ba7` — Vinicius Oliveira vs Andre Fili: large winner inversion; older Fili strongly preferred by MC.
- `aaec28fa3c0c576d` — Hyder Amil vs Christian Rodriguez: large winner inversion; older Amil strongly preferred by MC.
- `81cde317c156723b` — Asu Almabayev vs Charles Johnson: MC favored Johnson; actual Almabayev submission.
- `a3ab1979192b647c` — Brandon Royval vs Lone'er Kavanagh: MC gave Royval only 23%; actual Royval submission.
- `52ddf20a10890b41` — Tabatha Ricci vs Fatima Kline: fight-type prediction was strongly decision-heavy and correct, but fighter-side prediction missed.
- `701c97405da76603` — Dricus Du Plessis vs Kamaru Usman: MC essentially pick'em/slight Usman; market strongly favored DDP and DDP won.

## Machine-Readable Baseline

Per-fight results are stored in:

`data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv`

The CSV is the canonical comparison table for rerunning future simulator versions. Match on `bout_id`.

Recommended future comparison columns:
- new `p_red_win`, `p_blue_win`
- delta in actual-winner probability
- favorite flip
- new `p_ko_tko`, `p_sub`, `p_dec`
- method probability deltas
- `p_over_1_5`, `p_over_2_5` deltas
- old vs new MC correctness
- old vs new calibration metrics across the full 34-fight set

## Per-Fight Baseline

| Card | Bout ID | Fight | MC favorite | Actual winner | Actual method | P(KO) | P(SUB) | P(DEC) | MC hit |
|---:|---|---|---|---|---|---:|---:|---:|:---:|
| 1 | `00e8d4b961a65d21` | Bo Nickal vs Kyle Daukaus | Bo Nickal | Bo Nickal | KO/TKO | 9.7% | 30.1% | 60.2% | Y |
| 1 | `127ba4a1ccb3d4a6` | Diego Lopes vs Steve Garcia | Diego Lopes | Diego Lopes | KO/TKO | 14.0% | 17.8% | 68.2% | Y |
| 1 | `30cdb851f6c63444` | Josh Hokit vs Derrick Lewis | Josh Hokit | Josh Hokit | KO/TKO | 28.3% | 7.3% | 64.4% | Y |
| 1 | `5727d5be8c373346` | Alex Pereira vs Ciryl Gane | Ciryl Gane | Ciryl Gane | KO/TKO | 61.4% | 8.0% | 30.6% | Y |
| 1 | `60de0423ae6ed097` | Sean O'Malley vs Aiemann Zahabi | Sean O'Malley | Sean O'Malley | KO/TKO | 57.8% | 5.1% | 37.1% | Y |
| 1 | `7208e40818401e88` | Ilia Topuria vs Justin Gaethje | Justin Gaethje | Justin Gaethje | KO/TKO | 54.8% | 9.5% | 35.7% | Y |
| 1 | `9a9b30b3165b62e4` | Mauricio Ruffy vs Michael Chandler | Mauricio Ruffy | Mauricio Ruffy | KO/TKO | 31.6% | 10.2% | 58.2% | Y |
| 2 | `17d31249c9ab18af` | Andre Lima vs Kevin Borjas | Andre Lima | Kevin Borjas | DEC | 12.3% | 17.8% | 69.9% | N |
| 2 | `33afdd7ad43a2756` | Manel Kape vs Kyoji Horiguchi | Manel Kape | Manel Kape | KO/TKO | 31.7% | 3.0% | 65.3% | Y |
| 2 | `3b4803afa824fea3` | Allan Nascimento vs Mitch Raposo | Allan Nascimento | Mitch Raposo | DEC | 1.8% | 23.5% | 74.7% | N |
| 2 | `571a3eabff5c7ba7` | Vinicius Oliveira vs Andre Fili | Andre Fili | Vinicius Oliveira | KO/TKO | 21.4% | 8.1% | 70.5% | N |
| 2 | `8b809ba1880aded0` | Gaston Bolanos vs Michael Aswell Jr. | Michael Aswell Jr. | Gaston Bolanos | DEC | 7.6% | 16.0% | 76.4% | N |
| 2 | `aaec28fa3c0c576d` | Hyder Amil vs Christian Rodriguez | Hyder Amil | Christian Rodriguez | SUB | 4.8% | 14.8% | 80.4% | N |
| 2 | `afc021700481ac4e` | Ion Cutelaba vs Navajo Stirling | Navajo Stirling | Navajo Stirling | KO/TKO | 16.9% | 15.8% | 67.3% | Y |
| 2 | `b23a1a5d35eb438a` | Karol Rosa vs Luana Santos | Luana Santos | Luana Santos | DEC | 2.7% | 23.7% | 73.6% | Y |
| 3 | `012c307c9d446c4d` | Shara Magomedov vs Michel Pereira | Michel Pereira | Shara Magomedov | DEC | 26.1% | 13.3% | 60.6% | N |
| 3 | `35985895d4ef321e` | Abus Magomedov vs Michal Oleksiejczuk | Michal Oleksiejczuk | Abus Magomedov | SUB | 55.8% | 10.4% | 33.8% | N |
| 3 | `5e7f4ba1210f433d` | Nursulton Ruziboev vs Andrey Pulyaev | Andrey Pulyaev | Nursulton Ruziboev | SUB | 9.4% | 12.5% | 78.1% | N |
| 3 | `81cde317c156723b` | Asu Almabayev vs Charles Johnson | Charles Johnson | Asu Almabayev | SUB | 23.5% | 10.2% | 66.3% | N |
| 3 | `ae0a91c8f92f4758` | Bekzat Almakhan vs Jean Matsumoto | Jean Matsumoto | Jean Matsumoto | DEC | 7.4% | 12.7% | 79.9% | Y |
| 3 | `c13dc0cccef263f7` | Rafael Fiziev vs Manuel Torres | Rafael Fiziev | Rafael Fiziev | KO/TKO | 73.6% | 5.5% | 20.9% | Y |
| 3 | `d83294b031502177` | Ikram Aliskerov vs Brunno Ferreira | Brunno Ferreira | Ikram Aliskerov | DEC | 41.9% | 11.7% | 46.4% | N |
| 4 | `130b325610c54a64` | Benoit Saint Denis vs Paddy Pimblett | Benoit Saint Denis | Paddy Pimblett | SUB | 15.4% | 41.4% | 43.2% | N |
| 4 | `1b6a3b0d27654d3e` | Alessandro Costa vs Cody Durden | Cody Durden | Alessandro Costa | SUB | 13.1% | 16.4% | 70.5% | N |
| 4 | `7765e5e779dd7dc9` | Nikita Krylov vs Robert Whittaker | Robert Whittaker | Robert Whittaker | KO/TKO | 51.8% | 14.6% | 33.6% | Y |
| 4 | `989760fa75321d69` | Conor McGregor vs Max Holloway | Max Holloway | Max Holloway | KO/TKO | 95.0% | 1.7% | 3.3% | Y |
| 4 | `a26ea74f2e908842` | King Green vs Terrance McKinney | King Green | King Green | KO/TKO | 46.0% | 21.7% | 32.3% | Y |
| 4 | `a3ab1979192b647c` | Brandon Royval vs Lone'er Kavanagh | Lone'er Kavanagh | Brandon Royval | SUB | 16.5% | 13.1% | 70.4% | N |
| 4 | `c572daedf2012a49` | Cory Sandhagen vs Mario Bautista | Cory Sandhagen | Mario Bautista | DEC | 18.9% | 19.2% | 61.9% | N |
| 4 | `e84c3f1f9a33fa90` | Tracy Cortez vs Wang Cong | Wang Cong | Wang Cong | DEC | 12.1% | 11.6% | 76.3% | Y |
| 4 | `e9f389ba3a1e33c4` | Cody Garbrandt vs Adrian Yanez | Adrian Yanez | Adrian Yanez | KO/TKO | 69.9% | 3.1% | 27.0% | Y |
| 5 | `4eff5a845db17572` | Jared Cannonier vs Christian Leroy Duncan | Christian Leroy Duncan | Christian Leroy Duncan | DEC | 35.4% | 5.3% | 59.3% | Y |
| 5 | `52ddf20a10890b41` | Tabatha Ricci vs Fatima Kline | Tabatha Ricci | Fatima Kline | DEC | 8.8% | 11.7% | 79.5% | N |
| 5 | `701c97405da76603` | Dricus Du Plessis vs Kamaru Usman | Kamaru Usman | Dricus Du Plessis | DEC | 15.9% | 19.6% | 64.5% | N |

## Change-Control Rule

When simulator or FSR changes are made:
1. Do not overwrite this baseline.
2. Rerun the same 34 `bout_id`s with the new candidate.
3. Save a new versioned CSV / Markdown snapshot.
4. Compare new vs this baseline before deciding whether the change is an improvement.
5. Keep population validation separate from anecdotal fight improvements.

This file is intended to remain a frozen pre-change reference.
