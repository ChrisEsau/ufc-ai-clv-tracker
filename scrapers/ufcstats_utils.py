import pandas as pd


def normalize_event_name(name):
    if pd.isna(name):
        return ""

    return (
        str(name)
        .lower()
        .replace(":", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )