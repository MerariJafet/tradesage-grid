"""
Microstructure features - Advanced order flow and market microstructure indicators
"""
import numpy as np
import pandas as pd


def features_microstructure(row):
    """
    Calculate microstructure features from orderbook and volume data.
    
    Features:
    - imbalance5: Top-5 levels bid/ask volume imbalance
    - micro_price_dist: Distance from microprice to close (normalized)
    - cvd_delta: Cumulative Volume Delta per bar
    
    Args:
        row: pd.Series with orderbook columns (bid_px_1..5, bid_sz_1..5, ask_px_1..5, ask_sz_1..5)
              and volume columns (taker_buy_vol, taker_sell_vol)
              
    Returns:
        dict: Microstructure features
    """
    # Top-5 orderbook imbalance
    bid_vol = sum(row.get(f"bid_sz_{i}", 0) for i in range(1, 6))
    ask_vol = sum(row.get(f"ask_sz_{i}", 0) for i in range(1, 6))
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    
    # MicroPrice: weighted mid using top level sizes
    # MicroPrice = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)
    bid_px = row.get("bid_px_1", row["close"])
    ask_px = row.get("ask_px_1", row["close"])
    bid_sz1 = row.get("bid_sz_1", 1)
    ask_sz1 = row.get("ask_sz_1", 1)
    
    micro_price = (bid_px * ask_sz1 + ask_px * bid_sz1) / (bid_sz1 + ask_sz1 + 1e-9)
    micro_price_dist = (micro_price - row["close"]) / (row["close"] + 1e-9)
    
    # CVD (Cumulative Volume Delta) - difference between taker buy and sell
    cvd_delta = row.get("taker_buy_vol", 0) - row.get("taker_sell_vol", 0)
    
    return {
        "imbalance5": imbalance,
        "micro_price_dist": micro_price_dist,
        "cvd_delta": cvd_delta
    }


def add_microstructure_features(df):
    """
    Add microstructure features to dataframe.
    
    Args:
        df: pd.DataFrame with orderbook and volume columns
        
    Returns:
        pd.DataFrame with added microstructure columns
    """
    micro_features = df.apply(features_microstructure, axis=1)
    micro_df = pd.DataFrame(list(micro_features))
    
    for col in micro_df.columns:
        df[col] = micro_df[col]
    
    # Add rolling aggregations for microstructure
    windows = [5, 10, 20]
    
    for window in windows:
        # Rolling imbalance mean/std
        df[f'imbalance5_ma{window}'] = df['imbalance5'].rolling(window).mean()
        df[f'imbalance5_std{window}'] = df['imbalance5'].rolling(window).std()
        
        # Rolling CVD cumsum (actual cumulative delta)
        df[f'cvd_cumsum{window}'] = df['cvd_delta'].rolling(window).sum()
        
        # Microprice distance trend
        df[f'micro_price_dist_ma{window}'] = df['micro_price_dist'].rolling(window).mean()
    
    return df


def calculate_vwap(df, window=20):
    """
    Calculate Volume-Weighted Average Price.
    
    Args:
        df: pd.DataFrame with 'close' and 'volume' columns
        window: int, rolling window
        
    Returns:
        pd.Series: VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
    return vwap


def calculate_order_flow_imbalance(df, window=10):
    """
    Calculate Order Flow Imbalance (OFI) indicator.
    
    OFI measures the aggressiveness of buyers vs sellers.
    
    Args:
        df: pd.DataFrame with bid/ask columns
        window: int, rolling window
        
    Returns:
        pd.Series: OFI values
    """
    # Calculate price changes
    mid_price = (df['bid_px_1'] + df['ask_px_1']) / 2
    mid_change = mid_price.diff()
    
    # Volume-weighted direction
    total_vol = df['taker_buy_vol'] + df['taker_sell_vol']
    vol_imbalance = (df['taker_buy_vol'] - df['taker_sell_vol']) / (total_vol + 1e-9)
    
    # OFI: direction * volume imbalance
    ofi = np.sign(mid_change) * vol_imbalance
    ofi_rolling = ofi.rolling(window).mean()
    
    return ofi_rolling
