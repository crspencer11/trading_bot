import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import CoinMarketData
from handlers import APIManager, CacheManager
from models import ModelLSTM
from dotenv import load_dotenv


def fetch_and_prepare_data(seq_length: int):
    """Fetch cryptocurrency market data and prepare it for LSTM training."""
    cache = CacheManager()
    api_manager = APIManager(cache_manager=cache)
    market_data = CoinMarketData(api_manager)

    data = market_data.get_live_data()

    if data:
        df = market_data.dataframe_transform()
        if df is not None:
            print("Data Sample:\n", df.head())

            # Baseline example: train over extracted price values.
            if "price" in df.columns:
                prices = df["price"].dropna().values.astype(np.float32)
                if len(prices) <= seq_length:
                    print("Not enough price points to train.")
                    return None, None

                min_price, max_price = prices.min(), prices.max()
                if max_price == min_price:
                    print("Price series is constant; cannot normalize/train.")
                    return None, None
                prices = (prices - min_price) / (max_price - min_price)

                X, y = [], []
                for i in range(len(prices) - seq_length):
                    X.append(prices[i : i + seq_length])
                    y.append(prices[i + seq_length])

                return torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)
            else:
                print("Price column not found in data.")
                return None, None
        else:
            print("Failed to transform data to DataFrame.")
            return None, None
    else:
        print("No data retrieved.")
        return None, None

def train_model(X_train, y_train):
    """Train an LSTM model using prepared market data."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ModelLSTM(input_size=1, hidden_size=10, output_size=1).to(device)

    X_train, y_train = X_train.to(device), y_train.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print("Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Run baseline LSTM training on live market listing data.")
    parser.add_argument("--seq-length", type=int, default=5, help="Input sequence length")
    args = parser.parse_args()

    load_dotenv()
    X_train, y_train = fetch_and_prepare_data(seq_length=args.seq_length)
    if X_train is not None and y_train is not None:
        train_model(X_train, y_train)
    else:
        print("Training aborted due to missing data.")


if __name__ == "__main__":
    main()
