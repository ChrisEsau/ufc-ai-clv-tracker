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


def parse_stat_pair(text):
    """
    UFCStats often shows values like:
    '10 of 25'
    Returns landed/attempted.
    """

    text = clean_text(text)

    if not text:
        return None, None

    if "of" not in text:
        return text, None

    parts = text.split("of")

    landed = clean_text(parts[0])
    attempted = clean_text(parts[1])

    return landed, attempted

def parse_red_blue_pair(text):
    """
    Parses UFCStats paired values.

    Examples:
    "1 0"
    "64% 39%"
    "9:28 1:45"

    Returns:
    red_value, blue_value
    """

    text = clean_text(text)

    if not text:
        return None, None

    parts = text.split()

    if len(parts) < 2:
        return text, None

    red_value = parts[0]
    blue_value = parts[1]

    return red_value, blue_value


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

    rows = []

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

    totals_rows = soup.select("tbody.b-fight-details__table-body tr.b-fight-details__table-row")

    if not totals_rows:
        return pd.DataFrame(
            [
                {
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
                    "parse_status": "no_stats_table",
                }
            ]
        )

    total_row = totals_rows[0]

    cols = [
        clean_text(c.get_text(" ", strip=True))
        for c in total_row.select("td.b-fight-details__table-col")
    ]

    # UFCStats detail total table usually alternates red/blue values
    # Exact positions can vary, so this stores raw parsed columns first.
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
        "raw_col_count": len(cols),
    }

    for i, value in enumerate(cols):
        row[f"raw_detail_col_{i}"] = value


    # ============================================================
    # PARSED TOTAL STATS TABLE
    # ============================================================
    
    # KD
    red_kd, blue_kd = parse_red_blue_pair(
        cols[1] if len(cols) > 1 else None
    )
    
    row["red_kd"] = red_kd
    row["blue_kd"] = blue_kd
    
    
    # SIG STR
    sig_red, sig_blue = parse_red_blue_pair(
        cols[2] if len(cols) > 2 else None
    )
    
    red_sig_landed, red_sig_attempted = parse_stat_pair(
        sig_red
    )
    
    blue_sig_landed, blue_sig_attempted = parse_stat_pair(
        sig_blue
    )
    
    row["red_sig_str_landed"] = red_sig_landed
    row["red_sig_str_attempted"] = red_sig_attempted
    
    row["blue_sig_str_landed"] = blue_sig_landed
    row["blue_sig_str_attempted"] = blue_sig_attempted
    
    
    # SIG STR %
    red_sig_pct, blue_sig_pct = parse_red_blue_pair(
        cols[3] if len(cols) > 3 else None
    )
    
    row["red_sig_str_pct"] = red_sig_pct
    row["blue_sig_str_pct"] = blue_sig_pct
    
    
    # TOTAL STR
    tot_red, tot_blue = parse_red_blue_pair(
        cols[4] if len(cols) > 4 else None
    )
    
    red_total_landed, red_total_attempted = parse_stat_pair(
        tot_red
    )
    
    blue_total_landed, blue_total_attempted = parse_stat_pair(
        tot_blue
    )
    
    row["red_total_str_landed"] = red_total_landed
    row["red_total_str_attempted"] = red_total_attempted
    
    row["blue_total_str_landed"] = blue_total_landed
    row["blue_total_str_attempted"] = blue_total_attempted
    
    
    # TD
    td_red, td_blue = parse_red_blue_pair(
        cols[5] if len(cols) > 5 else None
    )
    
    red_td_landed, red_td_attempted = parse_stat_pair(
        td_red
    )
    
    blue_td_landed, blue_td_attempted = parse_stat_pair(
        td_blue
    )
    
    row["red_td_landed"] = red_td_landed
    row["red_td_attempted"] = red_td_attempted
    
    row["blue_td_landed"] = blue_td_landed
    row["blue_td_attempted"] = blue_td_attempted
    
    
    # TD %
    red_td_pct, blue_td_pct = parse_red_blue_pair(
        cols[6] if len(cols) > 6 else None
    )
    
    row["red_td_pct"] = red_td_pct
    row["blue_td_pct"] = blue_td_pct
    
    
    # SUB ATT
    red_sub_att, blue_sub_att = parse_red_blue_pair(
        cols[7] if len(cols) > 7 else None
    )
    
    row["red_sub_att"] = red_sub_att
    row["blue_sub_att"] = blue_sub_att
    
    
    # REV
    red_rev, blue_rev = parse_red_blue_pair(
        cols[8] if len(cols) > 8 else None
    )
    
    row["red_rev"] = red_rev
    row["blue_rev"] = blue_rev
    
    
    # CTRL
    red_ctrl, blue_ctrl = parse_red_blue_pair(
        cols[9] if len(cols) > 9 else None
    )
    
    row["red_ctrl"] = red_ctrl
    row["blue_ctrl"] = blue_ctrl
        
        
    return pd.DataFrame([row])