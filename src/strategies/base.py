from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base contract for all strategy implementations."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a signal series aligned with `df.index`.
        Convention: 1 = long, -1 = short, 0 = flat.
        """
