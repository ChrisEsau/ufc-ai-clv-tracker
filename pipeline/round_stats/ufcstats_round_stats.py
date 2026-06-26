from __future__ import annotations

from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from scrapers.selenium_core import fetch_html
from scrapers.ufcstats_fight_details import (
    clean_text,
    parse_red_blue_pair,
    parse_two_stat_pairs,
    time_to_seconds,
)


def scrape_round_stats_for_queue_row(queue_row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    row = dict(queue_row)
    fight_url = row["fight_url"]

    html = fetch_html(fight_url)
    soup = BeautifulSoup(html, "html.parser")

    tables = soup.select("table.b-fight-details__table")
    if len(tables) < 2:
        raise RuntimeError(f"Expected 2 fight-detail tables, found {len(tables)} for {fight_url}")

    total_rows = tables[0].select("tbody tr")
    sig_rows = tables[1].select("tbody tr")

    red_meta = _fighter_meta(row, corner="red")
    blue_meta = _fighter_meta(row, corner="blue")

    rows: list[dict[str, Any]] = []

    for idx, total_row in enumerate(total_rows):
        round_num = idx + 1

        total_cols = [
            clean_text(c.get_text(" ", strip=True))
            for c in total_row.select("td.b-fight-details__table-col")
        ]

        sig_cols = [
            clean_text(c.get_text(" ", strip=True))
            for c in sig_rows[idx].select("td.b-fight-details__table-col")
        ] if idx < len(sig_rows) else []

        if len(total_cols) < 10:
            continue

        red = dict(red_meta)
        blue = dict(blue_meta)
        red["round"] = round_num
        blue["round"] = round_num

        _parse_total_cols(total_cols, red, blue)
        _parse_sig_cols(sig_cols, red, blue)

        rows.extend([red, blue])

    return pd.DataFrame(rows)


def _fighter_meta(row: dict[str, Any], *, corner: str) -> dict[str, Any]:
    if corner == "red":
        fighter_side = "red"
        opponent_side = "blue"
    elif corner == "blue":
        fighter_side = "blue"
        opponent_side = "red"
    else:
        raise ValueError(f"Invalid corner: {corner}")

    return {
        "event_id": row.get("event_id"),
        "event_name": row.get("event_name"),
        "event_date": row.get("date") or row.get("event_date"),
        "fight_id": row.get("fight_id"),
        "fight_url": row.get("fight_url"),
        "event_url": row.get("event_url"),
        "location": row.get("location"),
        "division": row.get("division"),
        "title_fight": row.get("title_fight"),
        "total_rounds": row.get("total_rounds"),
        "fight_order": row.get("fight_order"),
        "corner": corner,
        "fighter_name": row.get(f"{fighter_side}_fighter"),
        "fighter_id": row.get(f"{fighter_side}_fighter_id"),
        "fighter_url": row.get(f"{fighter_side}_fighter_url"),
        "opponent_name": row.get(f"{opponent_side}_fighter"),
        "opponent_id": row.get(f"{opponent_side}_fighter_id"),
        "opponent_url": row.get(f"{opponent_side}_fighter_url"),
    }


def _parse_total_cols(total_cols: list[str], red: dict[str, Any], blue: dict[str, Any]) -> None:
    rk, bk = parse_red_blue_pair(total_cols[1])
    red["kd"] = _to_int(rk)
    blue["kd"] = _to_int(bk)

    rl, ra, bl, ba = parse_two_stat_pairs(total_cols[2])
    red["sig_str_landed"] = _to_int(rl)
    red["sig_str_attempted"] = _to_int(ra)
    blue["sig_str_landed"] = _to_int(bl)
    blue["sig_str_attempted"] = _to_int(ba)

    rl, ra, bl, ba = parse_two_stat_pairs(total_cols[4])
    red["total_str_landed"] = _to_int(rl)
    red["total_str_attempted"] = _to_int(ra)
    blue["total_str_landed"] = _to_int(bl)
    blue["total_str_attempted"] = _to_int(ba)

    rl, ra, bl, ba = parse_two_stat_pairs(total_cols[5])
    red["td_landed"] = _to_int(rl)
    red["td_attempted"] = _to_int(ra)
    blue["td_landed"] = _to_int(bl)
    blue["td_attempted"] = _to_int(ba)

    rs, bs = parse_red_blue_pair(total_cols[7])
    red["sub_att"] = _to_int(rs)
    blue["sub_att"] = _to_int(bs)

    rr, br = parse_red_blue_pair(total_cols[8])
    red["rev"] = _to_int(rr)
    blue["rev"] = _to_int(br)

    rc, bc = parse_red_blue_pair(total_cols[9])
    red["ctrl_sec"] = time_to_seconds(rc)
    blue["ctrl_sec"] = time_to_seconds(bc)


def _parse_sig_cols(sig_cols: list[str], red: dict[str, Any], blue: dict[str, Any]) -> None:
    if len(sig_cols) < 9:
        return

    for col_idx, prefix in [
        (3, "head"),
        (4, "body"),
        (5, "leg"),
        (6, "distance"),
        (7, "clinch"),
        (8, "ground"),
    ]:
        rl, ra, bl, ba = parse_two_stat_pairs(sig_cols[col_idx])
        red[f"{prefix}_landed"] = _to_int(rl)
        red[f"{prefix}_attempted"] = _to_int(ra)
        blue[f"{prefix}_landed"] = _to_int(bl)
        blue[f"{prefix}_attempted"] = _to_int(ba)


def _to_int(value: Any) -> int:
    if value is None or pd.isna(value) or str(value).strip() in {"", "---"}:
        return 0
    return int(value)
