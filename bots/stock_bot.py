import pandas as pd
import numpy as np
import json
import requests
import config

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

class StockBot:
    api_key = config.api_key
    base_url = "jasbndjkbnsdjbadjbaduij"
    df = None
    audit_trail = []

    def __init__(self, ticker: str, company: str):
        self.ticker = ticker
        self.company = company
        self.features = ['open', 'high', 'low', 'volume', 'RSI', 'MACD']
        self.trained_model = None

    def load_and_split_data(self) -> pd.DataFrame:
        """Load into data"""
        if self.df is None:
            # API call HERE!!!!!
            self.df = pd.DataFrame.from_dict()
            self.audit_trail.append(
                f"Initialized dataframe:{self.df.head()}"
            )
        return self.df
    
    def moving_average_convergance_divergance(self) -> pd.DataFrame:
        """MACD indicator creation"""
        self.df['ShortEMA'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['LongEMA'] = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = self.df['ShortEMA'] - self.df['LongEMA']
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.audit_trail.append(
            f"MACD altered df: {self.df.head()}"
        )
        return self.df
    
    def relative_strength_index(self) -> pd.DataFrame:
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
    
    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # create 20 day window for std of closing price and use as y(prediction) value
        rolling_std = self.df['close'].pct_change().rolling(window=20).std()
        self.df['target'] = (rolling_std.shift(-1) > rolling_std).astype(int)

        # split data in prep for ML feed
        X = self.df[self.features]
        y = self.df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess_data(X_train, X_test) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform PCA for dimensionality reduction of redundant features"""
        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train)
        X_test_standardized = scaler.transform(X_test)
        return X_train_standardized, X_test_standardized
    
    def train_model(self, X_train, y_train):
        # Create a pipeline with preprocessing and the model
        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=10),
            GradientBoostingRegressor()
        )
        pipeline.fit(X_train, y_train)
        self.trained_model = pipeline

    
    @staticmethod
    def perform_pca(X_train_standardized, X_test_standardized):
        principle_components = PCA(n_components=10)
        X_train_pca = principle_components.fit_transform(X_train_standardized)
        X_test_pca = principle_components.transform(X_test_standardized)
        return X_train_pca, X_test_pca
    
    def find_best_regressor(self, X_train_pca, X_test_pca, y_train, y_test):
        param_grid = {
            'max_depth': [2, 3, 4],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 0.01, 0.001]
        }
        pipeline = make_pipeline(GradientBoostingRegressor(), memory="cachedir")

        # Perform grid search with cross-validation
        self.trained_model = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        self.trained_model.fit(X_train_pca, y_train)

        best_regressor = self.trained_model.best_estimator_

        y_predictions = best_regressor.predict(X_test_pca)

        # Evaluate the performance
        mae = mean_absolute_error(y_test, y_predictions)
        print(f'Mean Absolute Error: {mae}')

        mse = mean_squared_error(y_test, y_predictions)
        print(f'Mean Squared Error: {mse}')

        print('Best Hyperparameters:', self.trained_model.best_params_)

        feature_importance = best_regressor.feature_importances_
        print("Feature Importances:")
        for feature, importance in zip(X_train_pca.columns, feature_importance):
            print(f"{feature}: {importance:.4f}")


    def launch_bot(self):

