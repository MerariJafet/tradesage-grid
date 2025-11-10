"""
Label generation module - Create targets without look-ahead bias
Implements significant move detection using ATR threshold
"""
import numpy as np
import pandas as pd


def calculate_atr(df, period=14):
    """
    Calculate Average True Range.
    
    Args:
        df: pd.DataFrame with high, low, close columns
        period: int, ATR period
        
    Returns:
        pd.Series: ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr


def label_move_significant(df, horizon=60, atr_threshold=2.0, atr_period=14):
    """
    Create binary labels for significant price movements.
    
    A move is "significant" if it exceeds atr_threshold * ATR within the horizon.
    This avoids labeling noise and focuses on tradable moves.
    
    Args:
        df: pd.DataFrame with OHLCV data
        horizon: int, number of bars to look ahead (e.g., 60 for 60 minutes)
        atr_threshold: float, multiple of ATR to define significance (default 2.0)
        atr_period: int, ATR calculation period (default 14)
        
    Returns:
        pd.DataFrame with added columns:
            - ATR_{period}: Average True Range
            - y: Binary label (1 if significant move, 0 otherwise)
            - y_sign: Direction label (1=long, -1=short, 0=no significant move)
            - future_return: Actual return for analysis (not used in training)
    """
    # Calculate ATR
    if f'ATR_{atr_period}' not in df.columns:
        df[f'ATR_{atr_period}'] = calculate_atr(df, atr_period)
    
    atr = df[f'ATR_{atr_period}']
    
    # Future price (horizon bars ahead)
    future_close = df['close'].shift(-horizon)
    
    # Return calculation
    ret = (future_close - df['close']) / df['close']
    
    # Significance threshold (normalized by current price)
    threshold = atr_threshold * (atr / df['close'])
    
    # Binary label: 1 if |return| > threshold, else 0
    df['y'] = (ret.abs() > threshold).astype(int)
    
    # Direction label: 1 (long), -1 (short), 0 (no trade)
    df['y_sign'] = np.where(ret > threshold, 1,
                            np.where(ret < -threshold, -1, 0))
    
    # Store future return for analysis (NOT to be used as feature)
    df['future_return'] = ret
    
    # Drop last 'horizon' rows (no valid labels)
    df = df.iloc[:-horizon].copy()
    
    return df


def label_move_directional(df, horizon=60, min_return_pct=0.5):
    """
    Alternative labeling: Simple directional move with minimum return threshold.
    
    Args:
        df: pd.DataFrame with OHLCV data
        horizon: int, bars to look ahead
        min_return_pct: float, minimum return % to consider (default 0.5%)
        
    Returns:
        pd.DataFrame with y (1=up, 0=down) and y_sign (1/-1)
    """
    future_close = df['close'].shift(-horizon)
    ret = (future_close - df['close']) / df['close']
    
    # Only label if return magnitude exceeds threshold
    significant = ret.abs() > (min_return_pct / 100)
    
    df['y'] = ((ret > 0) & significant).astype(int)
    df['y_sign'] = np.where(ret > 0, 1, -1) * significant.astype(int)
    df['future_return'] = ret
    
    df = df.iloc[:-horizon].copy()
    return df


def purge_labels(df, train_end_idx, test_start_idx, purge_bars=7*24*60):
    """
    Remove labels in the gap between train and test to prevent leakage.
    
    Args:
        df: pd.DataFrame with labels
        train_end_idx: int, last index of training set
        test_start_idx: int, first index of test set
        purge_bars: int, number of bars to purge (default 1 week = 7*24*60 minutes)
        
    Returns:
        tuple: (train_df, test_df) with purged gap
    """
    # Ensure gap is at least purge_bars
    actual_gap = test_start_idx - train_end_idx
    
    if actual_gap < purge_bars:
        # Adjust test_start_idx forward
        test_start_idx = train_end_idx + purge_bars
    
    train_df = df.iloc[:train_end_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    return train_df, test_df


def get_walk_forward_splits(df, train_months=3, test_months=1, purge_weeks=1):
    """
    Generate walk-forward splits with purging.
    
    Args:
        df: pd.DataFrame with timestamp column
        train_months: int, training window in months
        test_months: int, test window in months
        purge_weeks: int, purge period in weeks between train/test
        
    Returns:
        list of tuples: [(train_start, train_end, test_start, test_end), ...]
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate period lengths in bars (assuming 1-minute bars)
    bars_per_month = 30 * 24 * 60  # approximate
    train_bars = train_months * bars_per_month
    test_bars = test_months * bars_per_month
    purge_bars = purge_weeks * 7 * 24 * 60
    
    splits = []
    total_bars = len(df)
    
    # Rolling window
    start_idx = 0
    
    while start_idx + train_bars + purge_bars + test_bars <= total_bars:
        train_start = start_idx
        train_end = start_idx + train_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_bars
        
        splits.append((train_start, train_end, test_start, test_end))
        
        # Roll forward by test period
        start_idx += test_bars
    
    return splits
