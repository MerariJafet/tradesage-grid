"""
Data streams module - Load klines, orderbook, and funding rate data
Ensures time-aligned data without look-ahead bias
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def load_klines(symbol, interval="1m", start=None, end=None):
    """
    Load OHLCV kline data from CSV files.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        interval: str, timeframe (default '1m')
        start: datetime or None
        end: datetime or None
        
    Returns:
        pd.DataFrame with columns: timestamp, open, high, low, close, volume
    """
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    # Try to find the data file
    possible_files = [
        data_dir / f"{symbol.lower()}_{interval}_12months.csv",
        data_dir / f"{symbol.lower()}_1m_12months.csv",
        data_dir / f"btc_{interval}_12months.csv" if symbol == "BTCUSDT" else None,
    ]
    
    csv_file = None
    for f in possible_files:
        if f and f.exists():
            csv_file = f
            break
    
    if csv_file is None:
        raise FileNotFoundError(f"Could not find data file for {symbol}")
    
    df = pd.read_csv(csv_file)
    
    # Normalize column names
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # Filter by date range
    if start:
        df = df[df['timestamp'] >= start]
    if end:
        df = df[df['timestamp'] <= end]
    
    df = df.reset_index(drop=True)
    return df


def load_orderbook(symbol, depth=5, start=None, end=None):
    """
    Load orderbook (DOM) data with top-N levels.
    
    For now, this is a MOCK implementation that generates synthetic orderbook data
    based on OHLCV. In production, you would load real orderbook snapshots.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        depth: int, number of levels (default 5)
        start: datetime or None
        end: datetime or None
        
    Returns:
        pd.DataFrame with columns: timestamp, bid_px_1..N, bid_sz_1..N, ask_px_1..N, ask_sz_1..N
    """
    # Load klines as base
    df = load_klines(symbol, start=start, end=end)
    
    # Generate synthetic orderbook data (REPLACE WITH REAL DATA IN PRODUCTION)
    # This is for development/testing only
    spread_bps = 5  # 5 basis points
    
    for i in range(1, depth + 1):
        # Bid prices decrease from mid
        df[f'bid_px_{i}'] = df['close'] * (1 - spread_bps/10000 * i)
        # Ask prices increase from mid
        df[f'ask_px_{i}'] = df['close'] * (1 + spread_bps/10000 * i)
        
        # Synthetic sizes (larger at level 1, decreasing)
        base_size = df['volume'] / 100  # arbitrary scaling
        df[f'bid_sz_{i}'] = base_size / i * np.random.uniform(0.8, 1.2, len(df))
        df[f'ask_sz_{i}'] = base_size / i * np.random.uniform(0.8, 1.2, len(df))
    
    # Add taker volume proxy (for CVD calculation)
    # In production, this comes from aggTrades data
    df['taker_buy_vol'] = df['volume'] * np.random.uniform(0.4, 0.6, len(df))
    df['taker_sell_vol'] = df['volume'] - df['taker_buy_vol']
    
    return df


def load_funding(symbol, start=None, end=None):
    """
    Load real funding rate data.
    
    For now, this is a MOCK implementation. In production, fetch from Binance API
    or pre-downloaded funding rate history.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        start: datetime or None
        end: datetime or None
        
    Returns:
        pd.DataFrame with columns: timestamp, funding_rate
    """
    # Load klines for timestamp alignment
    df = load_klines(symbol, start=start, end=end)
    
    # Generate synthetic funding rates (REPLACE WITH REAL DATA IN PRODUCTION)
    # Real funding rates update every 8 hours on Binance
    # Values typically range from -0.05% to +0.05% (annualized ~= daily*3)
    
    # Create 8-hour intervals
    df['funding_rate'] = 0.0001  # placeholder
    
    # Add some variation (mock - real data needed)
    # In production: merge with actual funding rate timestamps
    hours = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
    funding_updates = (hours // 8).astype(int)
    
    # Random walk for mock funding
    unique_updates = funding_updates.unique()
    funding_values = np.random.normal(0.0001, 0.0002, len(unique_updates))
    funding_map = dict(zip(unique_updates, funding_values))
    df['funding_rate'] = funding_updates.map(funding_map)
    
    return df[['timestamp', 'funding_rate']]


def merge_data_sources(symbol, interval="1m", start=None, end=None, orderbook_depth=5):
    """
    Merge klines, orderbook, and funding data into single aligned DataFrame.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        interval: str, timeframe
        start: datetime or None
        end: datetime or None
        orderbook_depth: int, number of orderbook levels
        
    Returns:
        pd.DataFrame with all features aligned by timestamp
    """
    # Load all sources
    klines = load_klines(symbol, interval, start, end)
    orderbook = load_orderbook(symbol, orderbook_depth, start, end)
    funding = load_funding(symbol, start, end)
    
    # Merge on timestamp
    df = klines.copy()
    
    # Add orderbook columns
    ob_cols = [c for c in orderbook.columns if c != 'timestamp' and c not in df.columns]
    for col in ob_cols:
        df[col] = orderbook[col]
    
    # Add funding rate (forward fill for 8-hour intervals)
    df = df.merge(funding, on='timestamp', how='left')
    df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0001)
    
    return df
