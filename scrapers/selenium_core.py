from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def get_driver():
    chrome_options = Options()

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0")

    service = Service(
        ChromeDriverManager().install()
    )

    return webdriver.Chrome(
        service=service,
        options=chrome_options,
    )


def fetch_html(url, wait_seconds=10):
    driver = get_driver()

    try:
        driver.get(url)
        driver.implicitly_wait(wait_seconds)
        return driver.page_source

    finally:
        driver.quit()