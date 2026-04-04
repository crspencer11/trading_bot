import requests
import os
import pandas as pd
from handlers import APIManager

class CoinMarketData:
    """Fetches and processes cryptocurrency data from the CoinMarketCap API.

    This class integrates API requests, caching, and rate limit enforcement
    to efficiently retrieve and handle cryptocurrency data.

    Attributes:
        base_url (str): The CoinMarketCap API endpoint.
        parameters (dict): API request parameters.
        API_KEY (str): The authentication key for API access.
    """

    base_url = os.getenv(
        "CMC_BASE_URL",
        "https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
    )
    parameters = {
        "start": os.getenv("CMC_START", "1"),
        "limit": os.getenv("CMC_LIMIT", "10"),
        "convert": os.getenv("CMC_CONVERT", "USD"),
    }

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.data = None

    def get_live_data(self):
        cache_key = "cmc.listings.latest"
        cached_data = self.api_manager.cache_manager.load_cache(cache_key=cache_key)
        if cached_data:
            self.data = cached_data
            return self.data

        self.api_manager.enforce_rate_limits()
        try:
            response = self.api_manager.session.get(self.base_url, params=self.parameters)
            response.raise_for_status()
            data = response.json()["data"]
            self.api_manager.cache_manager.save_cache(data, cache_key=cache_key)
            self.data = data
            return self.data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def dataframe_transform(self):
        try:
            if self.data is None:
                return None

            df = pd.DataFrame(self.data)
            # CoinMarketCap listings API nests price at quote.<convert>.price
            convert = str(self.parameters.get("convert", "USD"))
            quote_col = "quote"
            if quote_col in df.columns:
                df["price"] = df[quote_col].apply(
                    lambda q: (q or {}).get(convert, {}).get("price")
                )
            return df
        except Exception as e:
            print(f"Error transforming data to DataFrame: {e}")
            return None
