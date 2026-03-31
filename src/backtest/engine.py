from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_capital: float = 10_000.0


def run_backtest(df: pd.DataFrame, signals: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """
    Minimal vectorized backtest skeleton.
    TODO: Replace with an event-driven engine once fills/execution rules are finalized.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must include a 'close' column")

    returns = df["close"].pct_change().fillna(0.0)
    shifted_signal = signals.shift(1).fillna(0.0)
    gross_returns = shifted_signal * returns

    trade_cost = (signals.diff().abs().fillna(0.0) > 0).astype(float)
    total_cost_bps = config.fee_bps + config.slippage_bps
    net_returns = gross_returns - trade_cost * (total_cost_bps / 10_000.0)

    equity_curve = config.initial_capital * (1.0 + net_returns).cumprod()
    return pd.DataFrame({"returns": net_returns, "equity": equity_curve}, index=df.index)
