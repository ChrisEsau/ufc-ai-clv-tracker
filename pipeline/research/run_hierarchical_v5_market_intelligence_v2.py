from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.research import run_hierarchical_v5_market_intelligence as base


def build_score_rows(chosen, fv):
    fv = fv.copy()
    fv["fight_id"] = fv["fight_id"].astype(str)
    red_col = base.find_col(fv.columns, ["red_fighter", "r_fighter", "red_fighter_name", "r_fighter_name", "fighter_red"])
    blue_col = base.find_col(fv.columns, ["blue_fighter", "b_fighter", "blue_fighter_name", "b_fighter_name", "fighter_blue"])
    fmeta = chosen.sort_values("refresh_timestamp").groupby("fight_id", as_index=False).last()[["fight_id", "event_name", "fight_display", "refresh_timestamp"]]
    merged = fmeta.merge(fv, on="fight_id", how="left", indicator="merge_status")
    rows, skips = [], []

    for _, r in merged.iterrows():
        fid = str(r["fight_id"])
        z = chosen[chosen["fight_id"].eq(fid)].copy()
        if r["merge_status"] != "both":
            skips.append({"fight_id":fid,"event_name":r.get("event_name"),"fight_display":r.get("fight_display"),"reason":"no_feature_view_match"})
            continue
        red_name = r.get(red_col, "") if red_col else ""
        blue_name = r.get(blue_col, "") if blue_col else ""
        if not base.norm_name(red_name) or not base.norm_name(blue_name):
            a,b = base.split_display(r.get("fight_display"))
            red_name = red_name if base.norm_name(red_name) else a
            blue_name = blue_name if base.norm_name(blue_name) else b
        if not base.norm_name(red_name) or not base.norm_name(blue_name):
            skips.append({"fight_id":fid,"event_name":r.get("event_name"),"fight_display":r.get("fight_display"),"reason":"cannot_resolve_fighter_orientation"})
            continue

        raw = {}
        ok = True
        for mk, suffix in [("moneyline","ml"),("win_by_ko_tko_dq","ko"),("win_by_submission","sub"),("win_by_decision","dec")]:
            zz = z[z["market_key"].eq(mk)].copy()
            vals = {"red":[], "blue":[]}
            for _, rr in zz.iterrows():
                side = base.classify_side(rr, red_name, blue_name)
                if side:
                    vals[side].append(rr)
            if len(vals["red"]) < 1 or len(vals["blue"]) < 1:
                ok = False
                skips.append({"fight_id":fid,"event_name":r.get("event_name"),"fight_display":r.get("fight_display"),"reason":f"cannot_map_{mk}_to_red_blue","red_fighter":red_name,"blue_fighter":blue_name})
                break
            for side in ["red","blue"]:
                rr = vals[side][-1]
                raw[f"{side}_{suffix}_raw_p"] = float(rr["implied_probability"])
                raw[f"{side}_{suffix}_american"] = float(rr["american_odds"]) if pd.notna(rr["american_odds"]) else np.nan
        if not ok:
            continue

        ml_sum = raw["red_ml_raw_p"] + raw["blue_ml_raw_p"]
        raw["market_overround"] = ml_sum
        raw["market_p_red_ml"] = raw["red_ml_raw_p"] / ml_sum
        method_raw = np.array([raw["red_ko_raw_p"],raw["red_sub_raw_p"],raw["red_dec_raw_p"],raw["blue_ko_raw_p"],raw["blue_sub_raw_p"],raw["blue_dec_raw_p"]], float)
        method_norm = method_raw / method_raw.sum()
        out = {"fight_id":fid,"event_name":r.get("event_name"),"fight_display":r.get("fight_display"),"refresh_timestamp":r.get("refresh_timestamp"),"red_fighter":str(red_name),"blue_fighter":str(blue_name),**raw}
        for j,slug in enumerate(base.SLUGS):
            out[f"market_{slug}"] = float(method_norm[j])
        for c in fv.columns:
            if c != "fight_id" and c in r.index:
                out[c] = r[c]
        rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(skips), red_col, blue_col


base.build_score_rows = build_score_rows
base.main()
