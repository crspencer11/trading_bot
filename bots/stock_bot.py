# import pandas as pd
# import numpy as np
# import json
# import requests
# import config
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# class StockBot:
#     api_key = config.api_key
#     base_url = "jasbndjkbnsdjbadjbaduij"
#     df = None
#     audit_trail = []

#     def __init__(self, ticker: str):
#         self.ticker = ticker

#     def load_dataframe(self):
#         self.self.df = pd.DataFrame.from_dict()
#         self.audit_trail.append(
#             f"initialized df: {self.df.head()}"
#         )
#         return self.df
    
#     def moving_average_convergance_divergance(self):
#         """MACD indicator creation"""

#         # Assuming 'self.df' is your DataFrame with 'Close' prices
#         self.df['ShortEMA'] = self.df['Close'].ewm(span=12, adjust=False).mean()
#         self.df['LongEMA'] = self.df['Close'].ewm(span=26, adjust=False).mean()
#         self.df['MACD'] = self.df['ShortEMA'] - self.df['LongEMA']
#         self.df['Signal Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
#         self.audit_trail.append(
#             f"MACD altered df: {self.df.head()}"
#         )
#         return self.df
    
#     def relative_strength_index(self):
#         """RSI indicator creation"""
#         # Assuming 'df' is your DataFrame with 'Close' prices
#         delta = self.df['Close'].diff(1)

#         # Define the lookback period (e.g., 14 days)
#         lookback = 14

#         # Calculate gains and losses
#         gains = delta.where(delta > 0, 0)
#         losses = -delta.where(delta < 0, 0)

#         # Calculate average gains and losses
#         avg_gains = gains.rolling(window=lookback, min_periods=1).mean()
#         avg_losses = losses.rolling(window=lookback, min_periods=1).mean()

#         # Calculate Relative Strength (RS) and RSI
#         relative_strength = avg_gains / avg_losses
#         self.df['RSI'] = 100 - (100 / (1 + relative_strength))
#         self.audit_trail.append(
#             f"RSI altered df: {self.df.head()}"
#         )
#         return self.df

#     def train_model(self):
#         # Create features and labels
#         X = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']]
#         y = self.df['Target']  # Assuming 'Target' is already defined as in your previous code

#         # Split the dataset into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Feature Scaling
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # Train the Model
#         model = DecisionTreeClassifier(random_state=42)
#         model.fit(X_train_scaled, y_train)

#         # Evaluate the Model
#         predictions = model.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, predictions)
#         print(f"Model Accuracy: {accuracy:.2%}")


import sklearn
print(sklearn.__version__)