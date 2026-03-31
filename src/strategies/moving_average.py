from __future__ import annotations

import pandas as pd

from .base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """Simple baseline strategy for experimentation."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must include a 'close' column")
        short_ma = df["close"].rolling(self.short_window).mean()
        long_ma = df["close"].rolling(self.long_window).mean()
        return (short_ma > long_ma).astype(int).replace({0: -1}).fillna(0)
