import pandas as pd
import numpy as np
import requests
import config
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from concurrent.futures import ThreadPoolExecutor

class StockBot:
    api_key = config.api_key
    base_url = "your_api_url"
    df = None
    audit_trail = []

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.features = ['open', 'dayHigh', 'dayLow', 'volume', 'priceAvg50', 'priceAvg200', 'pe']
        self.trained_model = None

    def load_data(self) -> pd.DataFrame:
        """Load and parse data"""
        try:
            response = requests.get(f"{self.base_url}/stock_data/{self.ticker}?apikey={self.api_key}")
            if response.status_code == 200:
                data = response.json()
                self.df = pd.DataFrame(data)
                self.audit_trail.append(f"Initialized dataframe: {self.df.head()}")
                return self.df
            else:
                raise Exception("Failed to fetch data")
        except Exception as e:
            self.audit_trail.append(f"An error occurred with load_data: {e}")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional features"""
        data['50_day_avg'] = data['priceAvg50']
        data['200_day_avg'] = data['priceAvg200']
        data['pe_ratio'] = data['pe']
        data['price_change'] = data['price'] - data['previousClose']
        data['volume_change'] = data['volume'] - data['avgVolume']
        self.features.extend(['50_day_avg', '200_day_avg', 'pe_ratio', 'price_change', 'volume_change'])
        return data

    def calculate_moving_avgs(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACD indicator creation"""
        data['ShortEMA'] = data['price'].ewm(span=12, adjust=False).mean()
        data['LongEMA'] = data['price'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['buy_sell_signal'] = np.where(data['Signal'] <= data['MACD'], "buy", "sell")
        self.audit_trail.append(f"MACD altered df: {data.head()}")
        return data

    def relative_strength_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """RSI indicator creation"""
        delta = data['price'].diff(1)
        lookback = 14
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=lookback, min_periods=1).mean()
        avg_losses = losses.rolling(window=lookback, min_periods=1).mean()
        relative_strength = avg_gains / avg_losses
        data['RSI'] = 100 - (100 / (1 + relative_strength))
        self.audit_trail.append(f"RSI altered df: {data.head()}")
        return data

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        data = data.dropna()
        data['target'] = data['price'].pct_change(20).shift(-20)
        data['target'] = (data['target'] > 0).astype(int)
        X = data[self.features].dropna()
        y = data['target'][X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        def train():
            pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(10, len(self.features))),
                GradientBoostingClassifier()
            )
            pipeline.fit(X_train, y_train)
            return pipeline

        with ThreadPoolExecutor() as executor:
            future = executor.submit(train)
            self.trained_model = future.result()

    def find_best_regressor(self, X_train, X_test, y_train, y_test):
        param_grid = {
            'gradientboostingclassifier__max_depth': [2, 3, 4],
            'gradientboostingclassifier__n_estimators': [50, 100, 150],
            'gradientboostingclassifier__learning_rate': [0.1, 0.01, 0.001]
        }

        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(10, len(self.features))),
            GradientBoostingClassifier()
        )

        searched_model = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        searched_model.fit(X_train, y_train)

        best_regressor = searched_model.best_estimator_
        y_predictions = best_regressor.predict(X_test)

        accuracy = accuracy_score(y_test, y_predictions)
        mse = mean_squared_error(y_test, y_predictions)

        print(f'Best Hyperparameters: {searched_model.best_params_}')
        print(f'Accuracy: {accuracy}')
        print(f'Mean Squared Error: {mse}')

        feature_importances = best_regressor.named_steps['gradientboostingclassifier'].feature_importances_
        print("Feature Importances:")
        for feature, importance in zip(self.features, feature_importances):
            print(f"{feature}: {importance:.4f}")

    def launch_bot(self):
        try:
            # Load and prepare the data
            stock_data = self.load_data()
            if stock_data is None or stock_data.empty:
                raise ValueError("No data loaded. Please check the data source and try again.")

            # Calculate additional features
            stock_data = self.calculate_features(stock_data)

            with ThreadPoolExecutor() as executor:
                # Execute calculate_moving_avgs and relative_strength_index concurrently
                future1 = executor.submit(self.calculate_moving_avgs, stock_data.copy())
                future2 = executor.submit(self.relative_strength_index, stock_data.copy())

                # Get the results
                macd_data = future1.result()
                rsi_data = future2.result()

                # Merge the results if necessary (assuming both calculations modify the same dataframe)
                stock_data = macd_data.copy()
                stock_data['RSI'] = rsi_data['RSI']

            # Split the data
            X_train, X_test, y_train, y_test = self.split_data(stock_data)

            # Train the model
            self.train_model(X_train, y_train)

            # Find the best regressor and evaluate
            self.find_best_regressor(X_train, X_test, y_train, y_test)

            # Log completion
            self.audit_trail.append("Bot launched successfully.")
            return self.audit_trail

        except Exception as e:
            self.audit_trail.append(f"An error occurred in launch_bot: {e}")
            return self.audit_trail
