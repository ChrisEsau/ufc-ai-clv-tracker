"""Run the frozen nine-fight KO V3 cohort with S50 KO prior strength.

Research-only wrapper. All mechanics and inputs come from the existing cohort
module; this file changes only KO_PRIOR_STRENGTH from 400 to 50 significant
strikes before invoking main(). Production remains untouched.
"""
from pipeline.simulation.event_clock_mc_v2.diagnostics import ko_v3_age_total_ko_nine_fight_cohort as cohort

cohort.KO_PRIOR_STRENGTH = 50.0

if __name__ == "__main__":
    cohort.main()
