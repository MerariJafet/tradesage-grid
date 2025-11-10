"""
Feature Engineering V2 - Real Microstructure Features

New predictive variables:
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Roll Spread (effective spread measure)
- Funding Momentum (dF/dt)
- ΔCVD/Δt (CVD acceleration)
- Depth Imbalance Ratio
"""
import numpy as np
import pandas as pd
from scipy import stats


def calculate_vpin(df, window=50, buckets=50):
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
    
    VPIN measures order flow toxicity by comparing buy vs sell volume
    in volume-time buckets.
    
    Args:
        df: pd.DataFrame with 'taker_buy_vol' and 'taker_sell_vol'
        window: int, number of buckets for rolling calculation
        buckets: int, number of volume buckets
        
    Returns:
        pd.Series: VPIN values
    """
    # Calculate volume imbalance per bar
    total_vol = df['taker_buy_vol'] + df['taker_sell_vol']
    vol_imbalance = abs(df['taker_buy_vol'] - df['taker_sell_vol'])
    
    # VPIN = rolling average of |VI| / total volume
    vpin = (vol_imbalance / (total_vol + 1e-9)).rolling(window).mean()
    
    return vpin


def calculate_roll_spread(df, window=20):
    """
    Calculate Roll Spread - effective spread estimator.
    
    Roll (1984) spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    
    Args:
        df: pd.DataFrame with 'close' prices
        window: int, rolling window
        
    Returns:
        pd.Series: Roll spread values
    """
    price_changes = df['close'].diff()
    
    # Rolling covariance of price changes with lag-1
    cov = price_changes.rolling(window).cov(price_changes.shift(1))
    
    # Roll spread (handle negative covariances)
    roll_spread = 2 * np.sqrt(np.maximum(-cov, 0))
    
    # Normalize by price
    roll_spread_pct = roll_spread / df['close']
    
    return roll_spread_pct


def calculate_funding_momentum(df, window=3):
    """
    Calculate funding rate momentum (dF/dt).
    
    Args:
        df: pd.DataFrame with 'funding_rate' column
        window: int, periods for derivative calculation
        
    Returns:
        pd.Series: Funding rate momentum
    """
    # First derivative (change rate)
    funding_change = df['funding_rate'].diff(window)
    
    # Normalize by time interval (window periods)
    funding_momentum = funding_change / window
    
    return funding_momentum


def calculate_cvd_acceleration(df, window=10):
    """
    Calculate CVD acceleration (ΔCVD/Δt).
    
    Args:
        df: pd.DataFrame with 'cvd_delta' column
        window: int, periods for acceleration calculation
        
    Returns:
        pd.Series: CVD acceleration
    """
    # CVD velocity (first derivative)
    cvd_velocity = df['cvd_delta'].rolling(window).mean()
    
    # CVD acceleration (second derivative)
    cvd_acceleration = cvd_velocity.diff(window) / window
    
    return cvd_acceleration


def calculate_depth_imbalance_ratio(row, levels=5):
    """
    Calculate depth imbalance ratio from orderbook.
    
    DIR = (Σ bid_vol - Σ ask_vol) / (Σ bid_vol + Σ ask_vol)
    
    Args:
        row: pd.Series with bid_sz_1..N and ask_sz_1..N columns
        levels: int, number of levels to aggregate
        
    Returns:
        float: Depth imbalance ratio
    """
    total_bid_vol = sum(row.get(f'bid_sz_{i}', 0) for i in range(1, levels + 1))
    total_ask_vol = sum(row.get(f'ask_sz_{i}', 0) for i in range(1, levels + 1))
    
    dir_ratio = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-9)
    
    return dir_ratio


def calculate_weighted_mid_price(row, levels=5):
    """
    Calculate volume-weighted mid price from orderbook.
    
    Args:
        row: pd.Series with bid/ask prices and sizes
        levels: int, number of levels
        
    Returns:
        float: Weighted mid price
    """
    bid_prices = [row.get(f'bid_px_{i}', 0) for i in range(1, levels + 1)]
    bid_sizes = [row.get(f'bid_sz_{i}', 0) for i in range(1, levels + 1)]
    ask_prices = [row.get(f'ask_px_{i}', 0) for i in range(1, levels + 1)]
    ask_sizes = [row.get(f'ask_sz_{i}', 0) for i in range(1, levels + 1)]
    
    # Weighted mid = (Σ bid_p × bid_v + Σ ask_p × ask_v) / (Σ bid_v + Σ ask_v)
    total_bid_value = sum(p * v for p, v in zip(bid_prices, bid_sizes))
    total_ask_value = sum(p * v for p, v in zip(ask_prices, ask_sizes))
    total_volume = sum(bid_sizes) + sum(ask_sizes)
    
    weighted_mid = (total_bid_value + total_ask_value) / (total_volume + 1e-9)
    
    return weighted_mid


def add_microstructure_features_v2(df):
    """
    Add all V2 microstructure features to dataframe.
    
    Args:
        df: pd.DataFrame with OHLCV + orderbook + trades data
        
    Returns:
        pd.DataFrame with added feature columns
    """
    print("  Calculating V2 microstructure features...")
    
    # VPIN - Order flow toxicity
    if 'taker_buy_vol' in df.columns and 'taker_sell_vol' in df.columns:
        df['vpin'] = calculate_vpin(df, window=50)
        print("    ✓ VPIN")
    
    # Roll Spread - Effective spread
    df['roll_spread'] = calculate_roll_spread(df, window=20)
    print("    ✓ Roll Spread")
    
    # Funding Momentum
    if 'funding_rate' in df.columns:
        df['funding_momentum'] = calculate_funding_momentum(df, window=3)
        df['funding_momentum_ma5'] = df['funding_momentum'].rolling(5).mean()
        print("    ✓ Funding Momentum")
    
    # CVD Acceleration
    if 'cvd_delta' in df.columns:
        df['cvd_acceleration'] = calculate_cvd_acceleration(df, window=10)
        df['cvd_acceleration_ma5'] = df['cvd_acceleration'].rolling(5).mean()
        print("    ✓ CVD Acceleration")
    
    # Depth Imbalance Ratio (per row)
    if 'bid_sz_1' in df.columns:
        df['depth_imbalance_ratio'] = df.apply(
            lambda row: calculate_depth_imbalance_ratio(row, levels=5), 
            axis=1
        )
        df['dir_ma5'] = df['depth_imbalance_ratio'].rolling(5).mean()
        df['dir_std5'] = df['depth_imbalance_ratio'].rolling(5).std()
        print("    ✓ Depth Imbalance Ratio")
    
    # Weighted Mid Price
    if 'bid_px_1' in df.columns:
        df['weighted_mid_price'] = df.apply(
            lambda row: calculate_weighted_mid_price(row, levels=5), 
            axis=1
        )
        df['wmp_dist'] = (df['weighted_mid_price'] - df['close']) / df['close']
        print("    ✓ Weighted Mid Price")
    
    # Additional derived features
    
    # Volume pressure (buy vs sell intensity)
    if 'taker_buy_vol' in df.columns:
        total_taker_vol = df['taker_buy_vol'] + df['taker_sell_vol']
        df['buy_pressure'] = df['taker_buy_vol'] / (total_taker_vol + 1e-9)
        df['buy_pressure_ma10'] = df['buy_pressure'].rolling(10).mean()
        df['buy_pressure_std10'] = df['buy_pressure'].rolling(10).std()
        print("    ✓ Buy Pressure")
    
    # Price impact proxy (volatility × volume)
    if 'ATR_14' in df.columns:
        df['price_impact'] = df['ATR_14'] * np.log1p(df['volume'])
        df['price_impact_ma5'] = df['price_impact'].rolling(5).mean()
        print("    ✓ Price Impact")
    
    # Funding-Price divergence
    if 'funding_rate' in df.columns:
        price_return = df['close'].pct_change(8)  # 8-hour return
        df['funding_price_div'] = df['funding_rate'] - price_return
        print("    ✓ Funding-Price Divergence")
    
    # Z-score normalization for selected features
    features_to_normalize = [
        'vpin', 'roll_spread', 'funding_momentum', 'cvd_acceleration',
        'depth_imbalance_ratio', 'buy_pressure', 'price_impact'
    ]
    
    print("  Normalizing features (z-score)...")
    for feat in features_to_normalize:
        if feat in df.columns:
            # Rolling z-score (250-bar window)
            mean = df[feat].rolling(250, min_periods=50).mean()
            std = df[feat].rolling(250, min_periods=50).std()
            df[f'{feat}_zscore'] = (df[feat] - mean) / (std + 1e-9)
    
    print(f"  ✓ Added {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features")
    
    return df


def validate_features(df):
    """
    Validate feature quality (check for inf/nan/extreme values).
    
    Args:
        df: pd.DataFrame with features
        
    Returns:
        dict: Validation report
    """
    report = {
        'total_features': len(df.columns),
        'nan_counts': {},
        'inf_counts': {},
        'extreme_values': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # NaN count
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report['nan_counts'][col] = nan_count
        
        # Inf count
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            report['inf_counts'][col] = inf_count
        
        # Extreme values (beyond 10 std)
        if df[col].std() > 0:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            extreme_count = (z_scores > 10).sum()
            if extreme_count > 0:
                report['extreme_values'][col] = extreme_count
    
    return report
