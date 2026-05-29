import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def get_driver():
    chrome_options = Options()

    #chrome_options.binary_location = "/usr/bin/google-chrome"

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--remote-debugging-port=9222")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )


def fetch_html(url, wait_seconds=8):
    driver = get_driver()

    try:
        driver.get(url)
        time.sleep(wait_seconds)

        print("Page title:", driver.title)
        print("HTML length:", len(driver.page_source))

        return driver.page_source

    finally:
        driver.quit()