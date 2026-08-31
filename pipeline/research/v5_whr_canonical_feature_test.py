from pathlib import Path

src = Path('pipeline/research/v5_whr_cpp_feature_test.py').read_text(encoding='utf-8')
src = src.replace("base=whr.Base();", "base=whr.Base(config={'w2':7.5625});")
src = src.replace("'experiment':'frozen_v5_plus_leakage_safe_compiled_whr_v1'", "'experiment':'frozen_v5_plus_canonical_whr_w2_7_5625_v1'")
src = src.replace("'whr_protocol':'same-date blocked expanding history; 50 Newton iterations after each event date; default compiled WHR config; zero handicap'", "'whr_protocol':'same-date blocked expanding history; 50 Newton iterations after each event date; canonical UFC WHR w=2.75 (w2=7.5625); zero handicap'")
src = src.replace("v5_plus_whr_cpp_oof.csv", "v5_plus_whr_canonical_oof.csv")
src = src.replace("v5_plus_whr_cpp_summary.json", "v5_plus_whr_canonical_summary.json")
exec(compile(src, 'v5_whr_canonical_feature_test_generated.py', 'exec'), {})
