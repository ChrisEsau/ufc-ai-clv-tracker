from pipeline.research.score_ufc_paris_hierarchical_v5_live import FIGHTS, METHOD_SUBCATEGORY, fetch_events, match_event
from pipeline.market.providers.draftkings_public import build_event_subcategory_markets_url, fetch_public_json


def main():
    events = fetch_events()
    complete = 0
    for a, b in FIGHTS:
        ev = match_event(events, a, b)
        payload = fetch_public_json(build_event_subcategory_markets_url(str(ev['id']), METHOD_SUBCATEGORY))
        markets = payload.get('markets', [])
        selections = payload.get('selections', [])
        names = sorted({str(m.get('name', '')) for m in markets})
        print(f"{a} vs {b} | event={ev['id']} | markets={len(markets)} | selections={len(selections)} | names={names}")
        if {'KO/TKO/DQ','Submission','Decision'}.issubset(set(names)) and len(selections) >= 6:
            complete += 1
    print(f"COMPLETE_METHOD_FIGHTS={complete}/{len(FIGHTS)}")


if __name__ == '__main__':
    main()
