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

    def __init__(self, cache_file: str | None = None, cache_expiry: int | None = None):
        if cache_file is not None:
            self.CACHE_FILE = cache_file
        if cache_expiry is not None:
            self.CACHE_EXPIRY = cache_expiry

    def _get_cache_path(self, cache_key: str | None = None) -> str:
        if not cache_key:
            return self.CACHE_FILE
        base, ext = os.path.splitext(self.CACHE_FILE)
        safe_key = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in cache_key)
        return f"{base}.{safe_key}{ext or '.json'}"

    def load_cache(self, cache_key: str | None = None):
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "r") as file:
                cached_data = json.load(file)
                time_since_cached = time.time() - cached_data["timestamp"]
                if time_since_cached < self.CACHE_EXPIRY:
                    print(f"Cache hit: Using cached data (Age: {int(time_since_cached)}s)")
                    return cached_data["data"]
                else:
                    print(f"Cache expired: Data is {int(time_since_cached)}s old, fetching new data")
        return None

    def save_cache(self, data, cache_key: str | None = None):
        """Saves data to the cache with a timestamp."""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
        print("Cache updated with new API data")
