import pandas as pd
import numpy as np
import json
import requests
import config
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class StockBot:
    api_key = config.api_key
    base_url = "jasbndjkbnsdjbadjbaduij"
    df = None
    audit_trail = []

    def __init__(self, ticker: str, company: str):
        self.ticker = ticker
        self.company = company
        self.model = RandomForestClassifier()
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
        self.target = 'target'

    def load_dataframe(self):
        """Load into data"""
        if self.df is None:
            self.df = pd.DataFrame.from_dict()
            self.audit_trail.append(
                f"Initialized dataframe:{self.df.head()}"
            )
        return self.df
    
    def preprocess_data(self, data):
        """Perform PCA for dimensionality reduction of redundant features"""
        principle_components = PCA(n_components=10)
        reduced_data = principle_components.fit_transform(data)
        return reduced_data
    
    def perform_action(self, data_path):
        if self.df is None:
            data = self.load_data(data_path)
            processed_data = self.preprocess_data(data)
            self.df = pd.DataFrame(processed_data, columns=[f'feature_{i}' for i in range(processed_data.shape[1])])
        self.model.partial_fit(self.df[self.features], self.df[self.target])
    
    def moving_average_convergance_divergance(self):
        """MACD indicator creation"""

        self.df['ShortEMA'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['LongEMA'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = self.df['ShortEMA'] - self.df['LongEMA']
        self.df['Signal Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.audit_trail.append(
            f"MACD altered df: {self.df.head()}"
        )
        return self.df
    
    def relative_strength_index(self):
        """RSI indicator creation"""
        delta = self.df['Close'].diff(1)

        # Define the lookback period (e.g., 14 days)
        lookback = 14

        # Calculate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=lookback, min_periods=1).mean()
        avg_losses = losses.rolling(window=lookback, min_periods=1).mean()

        # Calculate Relative Strength and RSI
        relative_strength = avg_gains / avg_losses
        self.df['RSI'] = 100 - (100 / (1 + relative_strength))
        self.audit_trail.append(
            f"RSI altered df: {self.df.head()}"
        )
        return self.df

    def train_model(self):
        # Create features and labels
        X = self.df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']]
        y = self.df['Target']  # Assuming 'Target' is already defined as in your previous code

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the Model
        model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))

        model.fit(X_train_scaled, y_train)

        # Evaluate the Model
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy}")
