import pandas as pd
from bs4 import BeautifulSoup

from scrapers.selenium_core import fetch_html


def clean_text(x):

    if x is None:
        return None

    return (
        str(x)
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        .strip()
    )


def parse_red_blue_pair(text):

    text = clean_text(text)

    if not text:
        return None, None

    parts = text.split()

    if len(parts) < 2:
        return text, None

    return parts[0], parts[1]


def parse_two_stat_pairs(text):

    text = clean_text(text)

    if not text:
        return None, None, None, None

    parts = text.split()

    if len(parts) >= 6 and parts[1] == "of" and parts[4] == "of":

        return (
            parts[0],
            parts[2],
            parts[3],
            parts[5],
        )

    return None, None, None, None


def clean_percent(x):

    x = clean_text(x)

    if not x or x == "---":
        return None

    return x.replace("%", "")


def time_to_seconds(x):

    x = clean_text(x)

    if not x or ":" not in x:
        return 0

    mins, secs = x.split(":")

    return int(mins) * 60 + int(secs)


def debug_print_table(table, table_name):

    print()
    print("=" * 20)
    print(table_name)
    print("=" * 20)

    rows = table.select("tr")

    for idx, row in enumerate(rows):

        cols = [
            clean_text(
                c.get_text(" ", strip=True)
            )
            for c in row.select("th, td")
        ]

        print(f"ROW {idx}")
        print(cols)


def parse_method_details(soup):

    method = None
    round_num = None
    fight_time = None
    time_format = None
    referee = None

    text = clean_text(
        soup.get_text(" ", strip=True)
    )

    if "Method:" in text:
        method = (
            text.split("Method:")[1]
            .split("Round:")[0]
            .strip()
        )

    if "Round:" in text:
        round_num = (
            text.split("Round:")[1]
            .split("Time:")[0]
            .strip()
        )

    if "Time:" in text:
        fight_time = (
            text.split("Time:")[1]
            .split("Time format:")[0]
            .strip()
        )

    if "Time format:" in text:
        time_format = (
            text.split("Time format:")[1]
            .split("Referee:")[0]
            .strip()
        )

    if "Referee:" in text:
        referee = (
            text.split("Referee:")[1]
            .split("Details:")[0]
            .strip()
        )

    return (
        method,
        round_num,
        fight_time,
        time_format,
        referee,
    )


def parse_totals_table(soup):

    parsed = {}

    tables = soup.select(
        "table.b-fight-details__table"
    )

    if len(tables) < 1:
        return parsed

    totals_table = tables[0]

    debug_print_table(
        totals_table,
        "TOTALS TABLE DEBUG",
    )

    rows = totals_table.select(
        "tr.b-fight-details__table-row"
    )

    parsed = {

        "red_kd": 0,
        "blue_kd": 0,

        "red_sig_str_landed": 0,
        "red_sig_str_attempted": 0,
        "blue_sig_str_landed": 0,
        "blue_sig_str_attempted": 0,

        "red_total_str_landed": 0,
        "red_total_str_attempted": 0,
        "blue_total_str_landed": 0,
        "blue_total_str_attempted": 0,

        "red_td_landed": 0,
        "red_td_attempted": 0,
        "blue_td_landed": 0,
        "blue_td_attempted": 0,

        "red_sub_att": 0,
        "blue_sub_att": 0,

        "red_rev": 0,
        "blue_rev": 0,

        "red_ctrl": 0,
        "blue_ctrl": 0,
    }

    for row in rows:

        cols = [
            clean_text(
                c.get_text(" ", strip=True)
            )
            for c in row.select(
                "td.b-fight-details__table-col"
            )
        ]

        if len(cols) < 10:
            continue

        # KD
        rk, bk = parse_red_blue_pair(
            cols[1]
        )

        parsed["red_kd"] += int(rk)
        parsed["blue_kd"] += int(bk)

        # SIG STR
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[2]
        )

        parsed["red_sig_str_landed"] += int(rl)
        parsed["red_sig_str_attempted"] += int(ra)
        parsed["blue_sig_str_landed"] += int(bl)
        parsed["blue_sig_str_attempted"] += int(ba)

        # SIG STR %
        rp, bp = parse_red_blue_pair(
            cols[3]
        )

        parsed["red_sig_str_pct"] = clean_percent(rp)
        parsed["blue_sig_str_pct"] = clean_percent(bp)

        # TOTAL STR
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[4]
        )

        parsed["red_total_str_landed"] += int(rl)
        parsed["red_total_str_attempted"] += int(ra)
        parsed["blue_total_str_landed"] += int(bl)
        parsed["blue_total_str_attempted"] += int(ba)

        # TD
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[5]
        )

        parsed["red_td_landed"] += int(rl)
        parsed["red_td_attempted"] += int(ra)
        parsed["blue_td_landed"] += int(bl)
        parsed["blue_td_attempted"] += int(ba)

        # TD %
        rtp, btp = parse_red_blue_pair(
            cols[6]
        )

        parsed["red_td_pct"] = clean_percent(rtp)
        parsed["blue_td_pct"] = clean_percent(btp)

        # SUB ATT
        rs, bs = parse_red_blue_pair(
            cols[7]
        )

        parsed["red_sub_att"] += int(rs)
        parsed["blue_sub_att"] += int(bs)

        # REV
        rr, br = parse_red_blue_pair(
            cols[8]
        )

        parsed["red_rev"] += int(rr)
        parsed["blue_rev"] += int(br)

        # CTRL
        rc, bc = parse_red_blue_pair(
            cols[9]
        )

        parsed["red_ctrl"] += time_to_seconds(rc)
        parsed["blue_ctrl"] += time_to_seconds(bc)

    return parsed


def parse_sig_str_breakdown_table(soup):

    parsed = {}

    tables = soup.select(
        "table.b-fight-details__table"
    )

    if len(tables) < 2:
        return parsed

    sig_table = tables[1]

    debug_print_table(
        sig_table,
        "SIG STR BREAKDOWN DEBUG",
    )

    rows = sig_table.select(
        "tr.b-fight-details__table-row"
    )

    parsed = {

        "r_head_landed": 0,
        "r_head_atmpted": 0,
        "b_head_landed": 0,
        "b_head_atmpted": 0,

        "r_body_landed": 0,
        "r_body_atmpted": 0,
        "b_body_landed": 0,
        "b_body_atmpted": 0,

        "r_leg_landed": 0,
        "r_leg_atmpted": 0,
        "b_leg_landed": 0,
        "b_leg_atmpted": 0,

        "r_dist_landed": 0,
        "r_dist_atmpted": 0,
        "b_dist_landed": 0,
        "b_dist_atmpted": 0,

        "r_clinch_landed": 0,
        "r_clinch_atmpted": 0,
        "b_clinch_landed": 0,
        "b_clinch_atmpted": 0,

        "r_ground_landed": 0,
        "r_ground_atmpted": 0,
        "b_ground_landed": 0,
        "b_ground_atmpted": 0,
    }

    for row in rows:

        cols = [
            clean_text(
                c.get_text(" ", strip=True)
            )
            for c in row.select(
                "td.b-fight-details__table-col"
            )
        ]

        if len(cols) < 9:
            continue

        # HEAD
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[3]
        )

        parsed["r_head_landed"] += int(rl)
        parsed["r_head_atmpted"] += int(ra)
        parsed["b_head_landed"] += int(bl)
        parsed["b_head_atmpted"] += int(ba)

        # BODY
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[4]
        )

        parsed["r_body_landed"] += int(rl)
        parsed["r_body_atmpted"] += int(ra)
        parsed["b_body_landed"] += int(bl)
        parsed["b_body_atmpted"] += int(ba)

        # LEG
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[5]
        )

        parsed["r_leg_landed"] += int(rl)
        parsed["r_leg_atmpted"] += int(ra)
        parsed["b_leg_landed"] += int(bl)
        parsed["b_leg_atmpted"] += int(ba)

        # DIST
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[6]
        )

        parsed["r_dist_landed"] += int(rl)
        parsed["r_dist_atmpted"] += int(ra)
        parsed["b_dist_landed"] += int(bl)
        parsed["b_dist_atmpted"] += int(ba)

        # CLINCH
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[7]
        )

        parsed["r_clinch_landed"] += int(rl)
        parsed["r_clinch_atmpted"] += int(ra)
        parsed["b_clinch_landed"] += int(bl)
        parsed["b_clinch_atmpted"] += int(ba)

        # GROUND
        rl, ra, bl, ba = parse_two_stat_pairs(
            cols[8]
        )

        parsed["r_ground_landed"] += int(rl)
        parsed["r_ground_atmpted"] += int(ra)
        parsed["b_ground_landed"] += int(bl)
        parsed["b_ground_atmpted"] += int(ba)

    return parsed


def scrape_fight_details(
    fight_url,
    event_name=None,
    event_date=None,
    fight_order=None,
):

    html = fetch_html(
        fight_url
    )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    print(
        f"Page title: {soup.title.text}"
    )

    print(
        f"HTML length: {len(html)}"
    )

    fighter_names = [
        x.get_text(strip=True)
        for x in soup.select(
            "a.b-link.b-fight-details__person-link"
        )
    ]

    red_fighter = (
        fighter_names[0]
        if len(fighter_names) > 0
        else None
    )

    blue_fighter = (
        fighter_names[1]
        if len(fighter_names) > 1
        else None
    )

    status_elements = soup.select(
        "i.b-fight-details__person-status"
    )

    red_result = (
        status_elements[0].get_text(strip=True)
        if len(status_elements) > 0
        else None
    )

    blue_result = (
        status_elements[1].get_text(strip=True)
        if len(status_elements) > 1
        else None
    )

    (
        method,
        round_num,
        fight_time,
        time_format,
        referee,
    ) = parse_method_details(soup)

    row = {

        "event_name": event_name,
        "event_date": event_date,
        "fight_order": fight_order,
        "fight_url": fight_url,

        "red_fighter": red_fighter,
        "blue_fighter": blue_fighter,

        "red_result": red_result,
        "blue_result": blue_result,

        "method": method,
        "round": round_num,
        "time": fight_time,
        "time_format": time_format,
        "referee": referee,

        "parse_status": "parsed",
    }

    totals = parse_totals_table(soup)
    row.update(totals)

    sig_breakdown = parse_sig_str_breakdown_table(soup)
    row.update(sig_breakdown)

    return pd.DataFrame([row])
