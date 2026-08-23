"""One-shot Selenium probe for SofaScore MMA JSON.

Uses the repository's existing Selenium helper. This is a research-only access test:
open SofaScore in a normal headless Chrome session, then navigate to the public JSON
endpoint and report whether Chrome receives JSON or an access-denied page.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from selenium.common.exceptions import WebDriverException

from scrapers.selenium_core import get_driver


def _body_text(driver) -> str:
    try:
        return driver.find_element("tag name", "body").text.strip()
    except Exception:
        return ""


def _extract_json_text(driver) -> str:
    # Chrome renders application/json either as body text or inside a <pre> element.
    try:
        pre = driver.find_element("tag name", "pre")
        text = pre.text.strip()
        if text:
            return text
    except Exception:
        pass
    return _body_text(driver)


def run_probe(date_text: str, output_path: Path) -> int:
    api_url = f"https://api.sofascore.com/api/v1/sport/mma/scheduled-events/{date_text}"
    driver = get_driver()
    result: dict[str, object] = {
        "date": date_text,
        "api_url": api_url,
        "home_loaded": False,
        "api_loaded": False,
        "json_valid": False,
    }

    try:
        driver.get("https://www.sofascore.com/")
        time.sleep(4)
        result["home_title"] = driver.title
        result["home_url"] = driver.current_url
        result["home_loaded"] = True
        print(f"Home title: {driver.title!r}")
        print(f"Home URL: {driver.current_url}")

        driver.get(api_url)
        time.sleep(3)
        result["api_title"] = driver.title
        result["api_current_url"] = driver.current_url
        result["api_loaded"] = True

        text = _extract_json_text(driver)
        result["body_preview"] = text[:1000]
        print(f"API title: {driver.title!r}")
        print(f"API URL: {driver.current_url}")
        print(f"Body preview: {text[:500]!r}")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            result["json_error"] = str(exc)
            print("RESULT: NOT_JSON")
            return_code = 2
        else:
            result["json_valid"] = True
            result["top_level_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
            events = payload.get("events") if isinstance(payload, dict) else None
            result["event_count"] = len(events) if isinstance(events, list) else None
            result["payload"] = payload
            print(f"RESULT: JSON_OK events={result['event_count']}")
            return_code = 0

        return return_code

    except WebDriverException as exc:
        result["webdriver_error"] = str(exc)
        print(f"RESULT: WEBDRIVER_ERROR {exc}")
        return 3
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/audits/regional_mma/sofascore/selenium_probe.json"),
    )
    args = parser.parse_args()
    raise SystemExit(run_probe(args.date, args.output))


if __name__ == "__main__":
    main()
