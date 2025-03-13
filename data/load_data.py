import os
import requests
import json
import pandas as pd
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

class CoinMarketData:
    base_url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start': '1',
        'limit': '10',
        'convert': 'USD'
    }
    API_KEY = os.getenv("API_KEY")

    CACHE_FILE = "crypto_cache.json"
    CACHE_EXPIRY = 60

    def __init__(self):
        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.API_KEY,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.weekly_data = self.get_data()
    
    def get_data(self):
        """Fetches weekly market data"""
        try:
            response = requests.get(self.base_url, headers=self.headers, params=self.parameters)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weekly data: {e}")

    def dataframe_transform(self, data):
        """Transform the API data into a pandas DataFrame."""
        try:
            df = pd.DataFrame(data)
            df.head() 
        except Exception as e:
            print(f"Error transforming data to dataframe: {e}")
