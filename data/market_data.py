import requests
import os
import pandas as pd
from handlers import *

class CoinMarketData:
    """Fetches and processes cryptocurrency data from the CoinMarketCap API.

    This class integrates API requests, caching, and rate limit enforcement
    to efficiently retrieve and handle cryptocurrency data.

    Attributes:
        base_url (str): The CoinMarketCap API endpoint.
        parameters (dict): API request parameters.
        API_KEY (str): The authentication key for API access.
    """

    base_url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {'start': '1', 'limit': '10', 'convert': 'USD'}

    def __init__(self, api_manager):
        self.api_manager = api_manager

    def get_data(self):
        cached_data = self.api_manager.cache_manager.load_cache()
        if cached_data:
            return cached_data

        self.api_manager.enforce_rate_limits()
        try:
            response = self.api_manager.session.get(self.base_url, params=self.parameters)
            response.raise_for_status()
            data = response.json()["data"]
            self.api_manager.cache_manager.save_cache(data)
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def dataframe_transform(self, data):
        try:
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error transforming data to DataFrame: {e}")
            return None