from handlers import *
from data import CoinMarketData

def main():
    cache = CacheManager()
    api_manager = APIManager(cache_manager=cache)
    market_data = CoinMarketData(api_manager)

    data = market_data.get_data()
    
    if data:
        df = dataframe_transform(data)
        if df is not None:
            print(df.head())
        else:
            print("Failed to transform data to DataFrame.")
    else:
        print("No data retrieved.")

if __name__ == "__main__":
    main()
