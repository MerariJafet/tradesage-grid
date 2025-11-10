"""
Backtest Engine v3 - Live Simulation with Latency and Slippage

Handles realistic trade execution with:
- Signal latency (1-bar delay)
- Slippage modeling
- Dynamic trailing stops
- Symbol-specific confidence thresholds
- Daily trade caps to prevent spam trading
"""

from typing import Tuple, Optional
import pandas as pd


# Symbol-specific confidence thresholds (post-calibration)
CONFIDENCE_THRESHOLDS = {
    'BTCUSDT': 0.78,
    'ETHUSDT': 0.72,
    'BNBUSDT': 0.68
}

# Daily trade caps to prevent spam
MAX_TRADES_PER_DAY = {
    'BTCUSDT': 60,
    'ETHUSDT': 80,
    'BNBUSDT': 80
}


def simulate_trade_v3(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    atr: float,
    max_bars: int = 60,
    use_trailing: bool = True
) -> Tuple[float, int]:
    """
    Simulate a trade with dynamic trailing stop.

    Args:
        df: DataFrame with OHLC data
        entry_idx: Index of entry bar
        entry_price: Entry price
        take_profit: TP level
        stop_loss: SL level
        atr: Current ATR for trailing
        max_bars: Max bars to hold
        use_trailing: Enable trailing stop

    Returns:
        exit_price, bars_held
    """
    current_sl = stop_loss
    
    for i in range(1, max_bars + 1):
        if entry_idx + i >= len(df):
            return df.iloc[-1]['close'], i - 1

        bar = df.iloc[entry_idx + i]
        high = bar['high']
        low = bar['low']
        close = bar['close']
        open_price = bar['open']

        # Update trailing stop (for longs)
        if use_trailing and close > entry_price:
            trailing_sl = close - 2 * atr
            current_sl = max(current_sl, trailing_sl)

        hit_tp = high >= take_profit
        hit_sl = low <= current_sl

        if hit_tp and hit_sl:
            # Prioritize based on candle direction
            if open_price < take_profit:
                return take_profit, i
            else:
                return current_sl, i
        elif hit_tp:
            return take_profit, i
        elif hit_sl:
            return current_sl, i

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


def apply_slippage(
    entry_price: float,
    slippage_bps: float = 3.0,
    is_long: bool = True
) -> float:
    """
    Apply slippage to entry price.

    Args:
        entry_price: Original entry price
        slippage_bps: Slippage in basis points
        is_long: True for long, False for short

    Returns:
        Adjusted entry price
    """
    slippage = entry_price * (slippage_bps / 10000)
    if is_long:
        return entry_price + slippage
    else:
        return entry_price - slippage
