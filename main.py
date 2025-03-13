import os
import requests
import time
import json
import pandas as pd
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

base_url = 'https://sandbox-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
    'start': '1',
    'limit': '10',
    'convert': 'USD'
}
API_KEY = os.getenv("API_KEY")

CACHE_FILE = "crypto_cache.json"
CACHE_EXPIRY = 60

headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY,
}
session = requests.Session()
session.headers.update(headers)

def load_cache():
    """Load cached data if available and not expired."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as file:
            cached_data = json.load(file)
            time_since_cached = time.time() - cached_data["timestamp"]
            if time_since_cached < CACHE_EXPIRY:
                print(f"Cache hit: Using cached data (Age: {int(time_since_cached)} seconds)")
                return cached_data["data"]
            else:
                print(f"Cache expired: Data is {int(time_since_cached)}s old, fetching new data")
    else:
        print("No cache file found, fetching new data")
    return None

def save_cache(data):
    """Save data to cache with a timestamp."""
    with open(CACHE_FILE, "w") as f:
        json.dump({"timestamp": time.time(), "data": data}, f)
    print("Cache updated with new API data")

def get_data():
    """Fetch crypto data from API or return cached data if available."""
    cached_data = load_cache()
    if cached_data:
        return cached_data

    try:
        response = session.get(base_url, params=parameters)
        response.raise_for_status()
        data = response.json()["data"]
        
        # Save data to cache
        save_cache(data)
        print("Fetched new data from API")

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def dataframe_transform(data):
    """Transform the API data into a pandas DataFrame."""
    try:
        df = pd.DataFrame(data)
        df.head() 
    except Exception as e:
        print(f"Error transforming data to dataframe: {e}")

def main():
    data = get_data()
    if data:
        # Print the data (you can process it as needed)
        for coin in data['data']:
            print(f"Name: {coin['name']}, Price: ${coin['quote']['USD']['price']:.2f}")

if __name__ == "__main__":
    main()
