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

from concurrent.futures import ThreadPoolExecutor

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

    def load_data(self) -> pd.DataFrame:
        """Load into data"""
        if self.df is None:
            # API call HERE!!!!!
            self.df = pd.DataFrame.from_dict()
            self.audit_trail.append(
                f"Initialized dataframe:{self.df.head()}"
            )
        return self.df
    
    def calculate_moving_avgs(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACD indicator creation"""
        data['ShortEMA'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['LongEMA'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['buy_sell_signal'] = np.where(data['Signal'] <= data['MACD'], "buy", "sell")
        self.audit_trail.append(
            f"MACD altered df: {data.head()}"
        )
        return data
    
    def relative_strength_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """RSI indicator creation"""
        delta = data['Close'].diff(1)

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
        data['RSI'] = 100 - (100 / (1 + relative_strength))
        self.audit_trail.append(
            f"RSI altered df: {data.head()}"
        )
        return data
    
    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # create 20 day window for std of closing price and use as y(prediction) value
        rolling_std = data['close'].pct_change().rolling(window=20).std()
        data['target'] = (rolling_std.shift(-1) > rolling_std).astype(int)

        # split data in prep for ML feed
        X = data[self.features]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        # Define a function to train the model
        def train():
            # Create a pipeline with preprocessing and the model
            pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=10),
                GradientBoostingRegressor(),
                memory="cachedir"
            )
            pipeline.fit(X_train, y_train)
            return pipeline

        # Create a ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Execute the training function concurrently
            future = executor.submit(train)
            self.trained_model = future.result()
        
    @staticmethod
    def huber_loss(y_actual, y_predicted, delta: float):
        huber_mse = 0.5*(y_actual - y_predicted)**2
        huber_mae = delta * (np.abs(y_actual - y_predicted) - 0.5 * delta)
        return np.where(np.abs(y_actual - y_predicted) <= delta, huber_mse, huber_mae)
    
    def find_best_regressor(self, X_train, X_test, y_train, y_test):
        param_grid = {
            'max_depth': [2, 3, 4],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 0.01, 0.001]
        }

        # Perform grid search with cross-validation
        searched_model = GridSearchCV(self.trained_model, param_grid, cv=5, scoring='neg_mean_squared_error')
        searched_model.fit(X_train, y_train)

        print(f"best hyperparams: {searched_model.best_params_}")

        best_regressor = searched_model.best_estimator_
        y_predictions = best_regressor.predict(X_test)

        # Evaluate the performance
        mae = mean_absolute_error(y_test, y_predictions)
        print(f'Mean Absolute Error: {mae}')

        mse = mean_squared_error(y_test, y_predictions)
        print(f'Mean Squared Error: {mse}')

        huber_loss = self.huber_loss(y_test, y_predictions, 0.1)
        print(f"Huber Loss: {huber_loss}")

        print('Best Hyperparameters:', searched_model.best_params_)

        feature_importances = best_regressor.feature_importances_
        print("Feature Importances:")
        for feature, importance in zip(X_train.columns, feature_importances):
            print(f"{feature}: {importance:.4f}")

    def launch_bot(self):
        """Implement the execution of statistical calculations to be done concurrently"""
        stock_data = self.load_data()
        with ThreadPoolExecutor() as executor:
            # Execute calculate_moving_avgs and relative_strength_index concurrently
            future1 = executor.submit(self.calculate_moving_avgs, stock_data.copy())
            future2 = executor.submit(self.relative_strength_index, stock_data.copy())
            # wait for completion
            final_df = future1.result()
            final_df = future2.result()
        X_train, X_test, y_train, y_test = self.split_data(final_df)
        # also has concurency implemented in the train_model method
        self.train_model(X_train, y_train)
        self.find_best_regressor(X_train, X_test, y_train, y_test)

