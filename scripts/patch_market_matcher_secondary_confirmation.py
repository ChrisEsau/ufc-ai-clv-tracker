from pathlib import Path

path = Path("pipeline/market/market_matcher.py")
s = path.read_text()

# Add market outcome columns.
if '    "matching_strategy",\n' not in s:
    s = s.replace(
        '    "market_key",\n',
        '    "market_key",\n    "matching_strategy",\n',
        1,
    )

if '    "outcome_join_key",\n' not in s:
    s = s.replace(
        '    "outcome_fighter_id",\n',
        '    "outcome_fighter_id",\n    "outcome_join_key",\n',
        1,
    )

# Add audit columns.
for col in [
    '    "matching_strategy",\n',
    '    "matchup_secondary_confirmed",\n',
    '    "provider_matchup_text",\n',
    '    "red_last_name_token",\n',
    '    "blue_last_name_token",\n',
]:
    if col not in s:
        # Second occurrence of market_key is MARKET_MATCH_AUDIT_COLUMNS.
        first = s.find('    "market_key",\n')
        second = s.find('    "market_key",\n', first + 1)
        if second >= 0:
            s = s[:second] + s[second:].replace('    "market_key",\n', '    "market_key",\n' + col, 1)

# Import shared outcome-join helper if missing.
if "from pipeline.common.outcome_join import build_outcome_join_key" not in s:
    s = s.replace(
        "import pandas as pd\n\n",
        "import pandas as pd\n\nfrom pipeline.common.outcome_join import build_outcome_join_key\n",
    )

# Add helper functions before _event_score.
marker = 'def _event_score(catalog_row: pd.Series, live_row: pd.Series) -> float:\n'
helpers = '''def _last_name_token(value: Any) -> str:
    text = _safe_str(value).lower().replace(".", "").replace(",", "")
    parts = [part for part in text.split() if part]
    return parts[-1] if parts else ""


def _provider_matchup_text(catalog_row: pd.Series) -> str:
    return " ".join(
        part
        for part in [
            _safe_str(catalog_row.get("event_name")),
            _safe_str(catalog_row.get("provider_market_name")),
            _safe_str(catalog_row.get("provider_selection_name")),
        ]
        if part
    ).lower()


def _matchup_confirmation_payload(catalog_row: pd.Series, live_row: pd.Series) -> dict[str, Any]:
    provider_text = _provider_matchup_text(catalog_row)
    red_last = _last_name_token(live_row.get("red_fighter"))
    blue_last = _last_name_token(live_row.get("blue_fighter"))
    confirmed = bool(red_last and blue_last and red_last in provider_text and blue_last in provider_text)
    return {
        "matchup_secondary_confirmed": confirmed,
        "provider_matchup_text": provider_text,
        "red_last_name_token": red_last,
        "blue_last_name_token": blue_last,
    }


def _matching_strategy(catalog_row: pd.Series) -> str:
    strategy = _safe_str(catalog_row.get("matching_strategy")).lower()
    return strategy or "fighter_name"


'''
if "def _matchup_confirmation_payload(" not in s:
    s = s.replace(marker, helpers + marker)

# Initialize strategy and confirmation payload.
s = s.replace(
    '    event_score = _event_score(catalog_row, live_row)\n    red_score, blue_score = _fighter_scores(catalog_row, live_row)\n\n    has_fighter = not np.isnan(red_score) and not np.isnan(blue_score)\n',
    '    event_score = _event_score(catalog_row, live_row)\n    red_score, blue_score = _fighter_scores(catalog_row, live_row)\n    strategy = _matching_strategy(catalog_row)\n    matchup_payload = _matchup_confirmation_payload(catalog_row, live_row)\n\n    has_fighter = not np.isnan(red_score) and not np.isnan(blue_score)\n',
)

# Add strategy branch if missing.
s = s.replace(
    '    has_fighter = not np.isnan(red_score) and not np.isnan(blue_score)\n    if has_fighter:\n',
    '    has_fighter = not np.isnan(red_score) and not np.isnan(blue_score)\n\n    if strategy == "matchup_name":\n        if not matchup_payload["matchup_secondary_confirmed"]:\n            match_score = 0.0\n            min_single_score = 0.0\n        else:\n            match_score = event_score\n            min_single_score = event_score\n    elif strategy == "event_name":\n        match_score = event_score\n        min_single_score = event_score\n    elif has_fighter:\n',
)

# Candidate payload.
s = s.replace(
    '        "event_score": event_score,\n    }\n',
    '        "event_score": event_score,\n        "matching_strategy": strategy,\n        **matchup_payload,\n    }\n',
    1,
)

# Accepted match payload.
for line in [
    '        "matching_strategy": best.get("matching_strategy"),\n',
    '        "matchup_secondary_confirmed": best.get("matchup_secondary_confirmed"),\n',
    '        "provider_matchup_text": best.get("provider_matchup_text"),\n',
    '        "red_last_name_token": best.get("red_last_name_token"),\n',
    '        "blue_last_name_token": best.get("blue_last_name_token"),\n',
]:
    if line not in s:
        s = s.replace('        "event_score": best["event_score"],\n', '        "event_score": best["event_score"],\n' + line, 1)

# Build outcome_join_key in market outcome rows.
s = s.replace(
    'def build_market_outcome_row(catalog_row: pd.Series, match: dict[str, Any]) -> dict[str, Any]:\n    """Build one production market outcome row from a matched canonical row."""\n\n    return {\n',
    'def build_market_outcome_row(catalog_row: pd.Series, match: dict[str, Any]) -> dict[str, Any]:\n    """Build one production market outcome row from a matched canonical row."""\n\n    outcome_label = _outcome_label(catalog_row)\n    outcome_fighter_id = _outcome_fighter_id(catalog_row, match)\n    outcome_join_key = build_outcome_join_key(\n        market_key=catalog_row.get("market_key"),\n        outcome_label=outcome_label,\n        outcome_fighter_id=outcome_fighter_id,\n        outcome_key=catalog_row.get("outcome_key"),\n        side=catalog_row.get("side"),\n        line=catalog_row.get("line"),\n    )\n\n    return {\n',
)

s = s.replace('        "outcome_label": _outcome_label(catalog_row),\n', '        "outcome_label": outcome_label,\n')
s = s.replace(
    '        "outcome_fighter_id": _outcome_fighter_id(catalog_row, match),\n',
    '        "outcome_fighter_id": outcome_fighter_id,\n        "outcome_join_key": outcome_join_key,\n',
)
if '        "matching_strategy": match.get("matching_strategy", catalog_row.get("matching_strategy")),\n' not in s:
    s = s.replace(
        '        "market_key": catalog_row.get("market_key"),\n',
        '        "market_key": catalog_row.get("market_key"),\n        "matching_strategy": match.get("matching_strategy", catalog_row.get("matching_strategy")),\n',
        1,
    )

# Audit unmatched defaults.
default_audit = '        "market_key": catalog_row.get("market_key"),\n        "outcome_key": catalog_row.get("outcome_key"),\n'
replacement_audit = '        "market_key": catalog_row.get("market_key"),\n        "matching_strategy": catalog_row.get("matching_strategy"),\n        "matchup_secondary_confirmed": False,\n        "provider_matchup_text": _provider_matchup_text(catalog_row),\n        "red_last_name_token": None,\n        "blue_last_name_token": None,\n        "outcome_key": catalog_row.get("outcome_key"),\n'
if replacement_audit not in s:
    s = s.replace(default_audit, replacement_audit, 1)

# Audit matched values.
matched_audit = '                "event_score": match.get("event_score"),\n                "is_matched": True,\n'
matched_replacement = '                "event_score": match.get("event_score"),\n                "matching_strategy": match.get("matching_strategy"),\n                "matchup_secondary_confirmed": match.get("matchup_secondary_confirmed"),\n                "provider_matchup_text": match.get("provider_matchup_text"),\n                "red_last_name_token": match.get("red_last_name_token"),\n                "blue_last_name_token": match.get("blue_last_name_token"),\n                "is_matched": True,\n'
if matched_replacement not in s:
    s = s.replace(matched_audit, matched_replacement, 1)

path.write_text(s)
print("Patched", path)
