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
    """
    Parses simple red/blue paired values.

    Examples:
    '1 0'
    '64% 39%'
    '4:38 0:34'
    """

    text = clean_text(text)

    if not text:
        return None, None

    parts = text.split()

    if len(parts) < 2:
        return text, None

    return parts[0], parts[1]


def parse_two_stat_pairs(text):
    """
    Parses UFCStats paired values like:
    '17 of 38 5 of 17'

    Returns:
    red_landed, red_attempted, blue_landed, blue_attempted
    """

    text = clean_text(text)

    if not text:
        return None, None, None, None

    parts = text.split()

    if len(parts) >= 6 and parts[1] == "of" and parts[4] == "of":
        return parts[0], parts[2], parts[3], parts[5]

    return None, None, None, None


def clean_percent(x):
    x = clean_text(x)

    if not x or x == "---":
        return None

    return x.replace("%", "")


def time_to_seconds(x):
    x = clean_text(x)

    if not x or ":" not in x:
        return None

    mins, secs = x.split(":")
    return int(mins) * 60 + int(secs)


def parse_method_details(soup):
    method = None
    round_num = None
    fight_time = None
    time_format = None
    referee = None

    detail_items = soup.select("i.b-fight-details__text-item")

    for item in detail_items:
        text = clean_text(item.get_text(" ", strip=True))

        if text.startswith("Method:"):
            method = clean_text(text.replace("Method:", ""))

        elif text.startswith("Round:"):
            round_num = clean_text(text.replace("Round:", ""))

        elif text.startswith("Time:"):
            fight_time = clean_text(text.replace("Time:", ""))

        elif text.startswith("Time format:"):
            time_format = clean_text(text.replace("Time format:", ""))

        elif text.startswith("Referee:"):
            referee = clean_text(text.replace("Referee:", ""))

    return method, round_num, fight_time, time_format, referee


def parse_totals_table(soup):
    parsed = {}

    tables = soup.select("table.b-fight-details__table")

    if len(tables) < 1:
        return parsed

    totals_table = tables[0]

    rows = totals_table.select("tr.b-fight-details__table-row")

    if not rows:
        return parsed

    # First row is fight TOTAL
    row = rows[0]

    cols = [
        clean_text(c.get_text(" ", strip=True))
        for c in row.select("td.b-fight-details__table-col")
    ]

    if len(cols) < 10:
        return parsed

    # KD
    red_kd, blue_kd = parse_red_blue_pair(cols[1])
    parsed["red_kd"] = red_kd
    parsed["blue_kd"] = blue_kd

    # SIG STR
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[2])
    parsed["red_sig_str_landed"] = r_land
    parsed["red_sig_str_attempted"] = r_att
    parsed["blue_sig_str_landed"] = b_land
    parsed["blue_sig_str_attempted"] = b_att

    # SIG STR %
    red_pct, blue_pct = parse_red_blue_pair(cols[3])
    parsed["red_sig_str_pct"] = clean_percent(red_pct)
    parsed["blue_sig_str_pct"] = clean_percent(blue_pct)

    # TOTAL STR
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[4])
    parsed["red_total_str_landed"] = r_land
    parsed["red_total_str_attempted"] = r_att
    parsed["blue_total_str_landed"] = b_land
    parsed["blue_total_str_attempted"] = b_att

    # TD
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[5])
    parsed["red_td_landed"] = r_land
    parsed["red_td_attempted"] = r_att
    parsed["blue_td_landed"] = b_land
    parsed["blue_td_attempted"] = b_att

    # TD %
    red_td_pct, blue_td_pct = parse_red_blue_pair(cols[6])
    parsed["red_td_pct"] = clean_percent(red_td_pct)
    parsed["blue_td_pct"] = clean_percent(blue_td_pct)

    # SUB ATT
    red_sub, blue_sub = parse_red_blue_pair(cols[7])
    parsed["red_sub_att"] = red_sub
    parsed["blue_sub_att"] = blue_sub

    # REV
    red_rev, blue_rev = parse_red_blue_pair(cols[8])
    parsed["red_rev"] = red_rev
    parsed["blue_rev"] = blue_rev

    # CTRL
    red_ctrl, blue_ctrl = parse_red_blue_pair(cols[9])
    parsed["red_ctrl"] = time_to_seconds(red_ctrl)
    parsed["blue_ctrl"] = time_to_seconds(blue_ctrl)

    parsed["raw_col_count"] = len(cols)

    for i, value in enumerate(cols):
        parsed[f"raw_detail_col_{i}"] = value

    return parsed


def parse_sig_str_breakdown_table(soup):
    parsed = {}

    tables = soup.select("table.b-fight-details__table")

    if len(tables) < 2:
        return parsed

    sig_table = tables[1]

    rows = sig_table.select("tr.b-fight-details__table-row")

    if not rows:
        return parsed

    # First row is fight TOTAL
    row = rows[0]

    cols = [
        clean_text(c.get_text(" ", strip=True))
        for c in row.select("td.b-fight-details__table-col")
    ]

    if len(cols) < 9:
        return parsed

    # HEAD
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[3])
    parsed["r_head_landed"] = r_land
    parsed["r_head_atmpted"] = r_att
    parsed["b_head_landed"] = b_land
    parsed["b_head_atmpted"] = b_att

    # BODY
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[4])
    parsed["r_body_landed"] = r_land
    parsed["r_body_atmpted"] = r_att
    parsed["b_body_landed"] = b_land
    parsed["b_body_atmpted"] = b_att

    # LEG
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[5])
    parsed["r_leg_landed"] = r_land
    parsed["r_leg_atmpted"] = r_att
    parsed["b_leg_landed"] = b_land
    parsed["b_leg_atmpted"] = b_att

    # DISTANCE
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[6])
    parsed["r_dist_landed"] = r_land
    parsed["r_dist_atmpted"] = r_att
    parsed["b_dist_landed"] = b_land
    parsed["b_dist_atmpted"] = b_att

    # CLINCH
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[7])
    parsed["r_clinch_landed"] = r_land
    parsed["r_clinch_atmpted"] = r_att
    parsed["b_clinch_landed"] = b_land
    parsed["b_clinch_atmpted"] = b_att

    # GROUND
    r_land, r_att, b_land, b_att = parse_two_stat_pairs(cols[8])
    parsed["r_ground_landed"] = r_land
    parsed["r_ground_atmpted"] = r_att
    parsed["b_ground_landed"] = b_land
    parsed["b_ground_atmpted"] = b_att

    return parsed


def scrape_fight_details(
    fight_url,
    event_name=None,
    event_date=None,
    fight_order=None,
):
    html = fetch_html(fight_url)

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    fighter_names = [
        x.get_text(strip=True)
        for x in soup.select("a.b-link.b-fight-details__person-link")
    ]

    red_fighter = fighter_names[0] if len(fighter_names) > 0 else None
    blue_fighter = fighter_names[1] if len(fighter_names) > 1 else None

    status_elements = soup.select("i.b-fight-details__person-status")

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

    method, round_num, fight_time, time_format, referee = parse_method_details(soup)

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

    if not totals:
        row["parse_status"] = "no_totals_table"

    return pd.DataFrame([row])
