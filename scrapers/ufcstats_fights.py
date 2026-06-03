import pandas as pd
from bs4 import BeautifulSoup

from scrapers.selenium_core import fetch_html


def clean_event_detail_text(value):
    return " ".join(str(value).replace("\n", " ").split())


def parse_event_location(soup):
    detail_items = soup.select("li.b-list__box-list-item")

    for item in detail_items:
        text = clean_event_detail_text(item.get_text(" ", strip=True))

        if text.lower().startswith("location:"):
            return text.split(":", 1)[1].strip() or None

    return None


def scrape_event_fights(
    event_url,
    event_name=None,
    event_date=None,
):

    html = fetch_html(event_url)

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    event_location = parse_event_location(soup)

    fight_rows = []

    rows = soup.select(
        "tr.b-fight-details__table-row"
    )

    for idx, row in enumerate(rows):

        fight_link = row.select_one(
            "a.b-flag.b-flag_style_green"
        )

        fight_url = (
            fight_link.get("href")
            if fight_link
            else None
        )

        fighter_links = row.select(
            "a.b-link.b-link_style_black"
        )

        red_fighter = None
        blue_fighter = None

        red_fighter_url = None
        blue_fighter_url = None

        if len(fighter_links) >= 2:

            red_fighter = fighter_links[0].get_text(
                strip=True
            )

            blue_fighter = fighter_links[1].get_text(
                strip=True
            )

            red_fighter_url = fighter_links[0].get(
                "href"
            )

            blue_fighter_url = fighter_links[1].get(
                "href"
            )

        cols = row.select(
            "td.b-fight-details__table-col"
        )

        winner = None
        method = None
        round_num = None
        fight_time = None
        weight_class = None

        try:
            weight_class = cols[6].get_text(
                strip=True
            )

            method = cols[7].get_text(
                strip=True
            )

            round_num = cols[8].get_text(
                strip=True
            )

            fight_time = cols[9].get_text(
                strip=True
            )

        except Exception:
            pass

        fight_rows.append(
            {
                "event_name": event_name,
                "event_date": event_date,
                "event_url": event_url,
                "event_location": event_location,
                "fight_order": idx + 1,
                "fight_url": fight_url,
                "red_fighter": red_fighter,
                "blue_fighter": blue_fighter,
                "red_fighter_url": red_fighter_url,
                "blue_fighter_url": blue_fighter_url,
                "winner": winner,
                "method": method,
                "round": round_num,
                "time": fight_time,
                "weight_class": weight_class,
            }
        )

    fights = pd.DataFrame(
        fight_rows
    )

    return fights