import pandas as pd
from bs4 import BeautifulSoup

from scrapers.selenium_core import fetch_html
from scrapers.ufcstats_utils import normalize_event_name


UFCSTATS_COMPLETED_EVENTS_URL = (
    "http://ufcstats.com/statistics/events/completed?page=all"
)
UFCSTATS_UPCOMING_EVENTS_URL = (
    "http://ufcstats.com/statistics/events/upcoming?page=all"
)


def extract_event_id_from_url(url):
    if not url:
        return None
    return str(url).rstrip("/").split("/")[-1]


def _extract_event_location(row):
    location_el = row.select_one("td.b-statistics__table-col_style_big-top-padding")
    if location_el is None:
        return None
    location = location_el.get_text(" ", strip=True)
    return location or None


def _scrape_events(events_url, event_state, run_id=None, run_timestamp=None):
    html = fetch_html(events_url)

    soup = BeautifulSoup(html, "html.parser")

    event_rows = []

    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one("a.b-link.b-link_style_black")

        if link is None:
            continue

        event_name = link.get_text(strip=True)
        event_url = link.get("href")

        date_el = row.select_one("span.b-statistics__date")
        event_date = date_el.get_text(strip=True) if date_el else None

        if not event_name or not event_url:
            continue

        event_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "event_state": event_state,
                "ufcstats_event_id": extract_event_id_from_url(event_url),
                "ufcstats_event_name": event_name,
                "ufcstats_event_url": event_url,
                "ufcstats_event_date": event_date,
                "ufcstats_event_location": _extract_event_location(row),
                "ufcstats_event_name_norm": normalize_event_name(event_name),
            }
        )

    events = pd.DataFrame(event_rows)

    if events.empty:
        raise ValueError(f"No UFCStats {event_state} events scraped.")

    events["ufcstats_event_date"] = pd.to_datetime(
        events["ufcstats_event_date"],
        errors="coerce",
    ).dt.date

    return events


def scrape_completed_events(run_id=None, run_timestamp=None):
    return _scrape_events(
        UFCSTATS_COMPLETED_EVENTS_URL,
        event_state="completed",
        run_id=run_id,
        run_timestamp=run_timestamp,
    )


def scrape_upcoming_events(run_id=None, run_timestamp=None):
    return _scrape_events(
        UFCSTATS_UPCOMING_EVENTS_URL,
        event_state="upcoming",
        run_id=run_id,
        run_timestamp=run_timestamp,
    )
