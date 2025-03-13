import json
import time
import os
import requests
from handlers import *

class APIManager:
    """Tracks API request counts to enforce rate limits and monthly caps.

    This class ensures that API requests do not exceed the given
    rate limit (30 requests per minute) or the monthly cap (10,000 requests).

    Attributes:
        REQUEST_LOG (str): File where request counts are stored.
        RATE_LIMIT (int): Maximum API calls per minute.
        RATE_LIMIT_INTERVAL (int): Interval in seconds for rate limit checks.
        MONTHLY_LIMIT (int): Maximum API calls allowed per month.
    """

    REQUEST_LOG = "request_log.json"
    RATE_LIMIT = 30  # Max calls per minute
    RATE_LIMIT_INTERVAL = 60  # 60 second limit
    MONTHLY_LIMIT = 10_000  # Monthly cap

    def __init__(self, cache_manager=None):
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the API_KEY environment variable.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": self.api_key,
        })
        
        self.cache_manager = cache_manager or CacheManager()
        self.request_log = self.load_request_log()

    def load_request_log(self):
        if os.path.exists(self.REQUEST_LOG):
            with open(self.REQUEST_LOG, "r") as file:
                try:
                    log = json.load(file)
                    if not isinstance(log, dict):
                        log = {"timestamps": [], "monthly_count": 0}
                except json.JSONDecodeError:
                    log = {"timestamps": [], "monthly_count": 0}
        else:
            log = {"timestamps": [], "monthly_count": 0}
        return log


    def save_request_log(self, log):
        with open(self.REQUEST_LOG, "w") as file:
            json.dump(log, file)

    def enforce_rate_limits(self):
        log = self.load_request_log()
        current_time = time.time()
        log["timestamps"] = [t for t in log["timestamps"] if current_time - t < self.RATE_LIMIT_INTERVAL]

        if len(log["timestamps"]) >= self.RATE_LIMIT:
            sleep_time = self.RATE_LIMIT_INTERVAL - (current_time - log["timestamps"][0])
            print(f"Rate limit reached. Sleeping for {int(sleep_time)} seconds...")
            time.sleep(sleep_time)

        if log["monthly_count"] >= self.MONTHLY_LIMIT:
            raise Exception("Monthly API limit reached. No further requests allowed.")

        log["timestamps"].append(current_time)
        log["monthly_count"] += 1
        self.save_request_log(log)