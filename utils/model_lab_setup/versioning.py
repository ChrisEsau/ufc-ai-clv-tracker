from __future__ import annotations

import re


def safe_model_id(value: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value or '').strip())
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_')


def safe_path_key(value: str) -> str:
    return safe_model_id(value) or 'unknown_market'


def artifact_dir_for_market(model_id: str, market_key: str) -> str:
    return f"models/{safe_path_key(market_key)}/{safe_model_id(model_id)}"


def parse_model_version(model_id: str) -> dict:
    match = re.search(r'_v(\d+)$', model_id)
    return {'base': re.sub(r'_v\d+$', '', model_id), 'version': int(match.group(1)) if match else None}
