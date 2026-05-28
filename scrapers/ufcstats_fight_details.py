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

    return pd.DataFrame([row])