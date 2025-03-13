import json
import os
import time

class CacheManager:
    """Handles caching of API responses to reduce redundant API calls.

    This class loads and saves cached cryptocurrency data to prevent
    unnecessary API calls while considering expiry time.

    Attributes:
        CACHE_FILE (str): The file where cache data is stored.
        CACHE_EXPIRY (int): Time (in seconds) after which the cache expires.
    """

    CACHE_FILE = "crypto_cache.json"
    CACHE_EXPIRY = 60  # Cache expiry time in seconds

    def load_cache(self):
        if os.path.exists(self.CACHE_FILE) and os.path.getsize(self.CACHE_FILE) > 0:
            with open(self.CACHE_FILE, "r") as file:
                cached_data = json.load(file)
                time_since_cached = time.time() - cached_data["timestamp"]
                if time_since_cached < self.CACHE_EXPIRY:
                    print(f"Cache hit: Using cached data (Age: {int(time_since_cached)}s)")
                    return cached_data["data"]
                else:
                    print(f"Cache expired: Data is {int(time_since_cached)}s old, fetching new data")
        return None

    def save_cache(self, data):
        with open(self.CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
        print("Cache updated with new API data")
        """Saves data to the cache with a timestamp.

        Args:
            data (dict): API response data to cache.
        """
        with open(self.CACHE_FILE, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
        print("Cache updated with new API data")
