import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

from scrapers.selenium_core import fetch_html


def clean_text(x):
    if x is None:
        return None

    x = (
    str(x)
    .replace("\n", " ")
    .replace("\t", " ")
    .replace("  ", " ")
    .strip()
    )

if x in ["", "--", "---", "nan", "None"]:
    return None

return x


def clean_percent(x):
    x = clean_text(x)

    if not x or x == "---":
        return None

    return x.replace("%", "")


def height_to_cm(x):
    x = clean_text(x)

    if not x or x == "--":
        return None

    try:
        feet, inches = x.replace('"', "").split("'")
        return round((int(feet) * 12 + int(inches.strip())) * 2.54, 2)
    except Exception:
        return None


def weight_to_kg(x):
    x = clean_text(x)

    if not x or x == "--":
        return None

    try:
        lbs = float(x.replace("lbs.", "").strip())
        return round(lbs * 0.453592, 2)
    except Exception:
        return None


def reach_to_cm(x):
    x = clean_text(x)

    if not x or x == "--":
        return None

    try:
        inches = float(x.replace('"', "").strip())
        return round(inches * 2.54, 2)
    except Exception:
        return None


def parse_record(record_text):
    record_text = clean_text(record_text)

    if not record_text:
        return None, None, None

    try:
        parts = record_text.split("-")
        wins = int(parts[0])
        losses = int(parts[1])
        draws = int(parts[2]) if len(parts) > 2 else 0
        return wins, losses, draws
    except Exception:
        return None, None, None

def normalize_dob(x):

    x = clean_text(x)

    if not x or x == "--":
        return None

    try:
        dt = datetime.strptime(x, "%b %d, %Y")
        return dt.strftime("%Y/%m/%d")

    except Exception:
        return x
        
def scrape_fighter_profile(fighter_url, fighter_id=None):
    html = fetch_html(fighter_url)

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    name = clean_text(
        soup.select_one("span.b-content__title-highlight").get_text(strip=True)
        if soup.select_one("span.b-content__title-highlight")
        else None
    )

    nickname = clean_text(
        soup.select_one("p.b-content__Nickname").get_text(strip=True)
        if soup.select_one("p.b-content__Nickname")
        else None
    )

    record_text = clean_text(
        soup.select_one("span.b-content__title-record").get_text(strip=True)
        if soup.select_one("span.b-content__title-record")
        else None
    )

    if record_text:
        record_text = record_text.replace("Record:", "").strip()

    wins, losses, draws = parse_record(record_text)

    profile = {
        "fighter_id": fighter_id,
        "fighter_url": fighter_url,
        "fighter_name": name,
        "nick_name": nickname,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height": None,
        "weight": None,
        "reach": None,
        "stance": None,
        "dob": None,
        "splm": None,
        "str_acc": None,
        "sapm": None,
        "str_def": None,
        "td_avg": None,
        "td_avg_acc": None,
        "td_def": None,
        "sub_avg": None,
    }

    stat_items = soup.select("li.b-list__box-list-item")

    for item in stat_items:
        text = clean_text(item.get_text(" ", strip=True))

        if not text or ":" not in text:
            continue

        label, value = text.split(":", 1)
        label = clean_text(label)
        value = clean_text(value)

        if label == "Height":
            profile["height"] = height_to_cm(value)

        elif label == "Weight":
            profile["weight"] = weight_to_kg(value)

        elif label == "Reach":
            profile["reach"] = reach_to_cm(value)

        elif label == "STANCE":
            profile["stance"] = value

        elif label == "DOB":
            profile["dob"] = normalize_dob(value)

        elif label == "SLpM":
            profile["splm"] = value

        elif label == "Str. Acc.":
            profile["str_acc"] = clean_percent(value)

        elif label == "SApM":
            profile["sapm"] = value

        elif label == "Str. Def":
            profile["str_def"] = clean_percent(value)

        elif label == "TD Avg.":
            profile["td_avg"] = value

        elif label == "TD Acc.":
            profile["td_avg_acc"] = clean_percent(value)

        elif label == "TD Def.":
            profile["td_def"] = clean_percent(value)

        elif label == "Sub. Avg.":
            profile["sub_avg"] = value

    return pd.DataFrame([profile])
