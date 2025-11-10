"""
Backtest Engine for ML Edge Strategy

Handles trade simulation with proper TP/SL prioritization.
"""

from typing import Tuple, Optional
import pandas as pd


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    max_bars: int = 60
) -> Tuple[float, int]:
    """
    Simulate a trade from entry to exit.

    Args:
        df: DataFrame with OHLC data
        entry_idx: Index of entry bar
        entry_price: Entry price
        take_profit: TP level
        stop_loss: SL level
        max_bars: Max bars to hold

    Returns:
        exit_price, bars_held
    """
    for i in range(1, max_bars + 1):
        if entry_idx + i >= len(df):
            return df.iloc[-1]['close'], i - 1

        bar = df.iloc[entry_idx + i]
        high = bar['high']
        low = bar['low']
        open_price = bar['open']

        hit_tp = high >= take_profit
        hit_sl = low <= stop_loss

        if hit_tp and hit_sl:
            # Prioritize based on candle direction
            if open_price < take_profit:
                return take_profit, i
            else:
                return stop_loss, i
        elif hit_tp:
            return take_profit, i
        elif hit_sl:
            return stop_loss, i

    return df.iloc[min(entry_idx + max_bars, len(df) - 1)]['close'], max_bars


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    is_long: bool = True
) -> float:
    """
    Calculate PnL for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size in units
        is_long: True for long, False for short

    Returns:
        PnL in USD
    """
    if is_long:
        return (exit_price - entry_price) * size
    else:
        return (entry_price - exit_price) * size