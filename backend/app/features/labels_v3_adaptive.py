"""
Binary Adaptive Labeling for Sprint 14
Volatility-regime based thresholds to respect market structure
"""

import pandas as pd
import numpy as np


def calculate_volatility_regime(df, window=250):
    """
    Classify market into low/mid/high volatility regimes.
    
    Uses rolling standard deviation of returns to determine regime.
    Bottom tercile = low vol, middle = mid vol, top = high vol.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'close' prices
    window : int
        Rolling window for volatility calculation (default 250 bars ≈ 20h)
    
    Returns:
    --------
    pd.Series
        Regime labels: 'low', 'mid', 'high'
    """
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(window=window).std()
    
    # Classify into terciles
    regime = pd.qcut(
        rolling_vol, 
        q=3, 
        labels=['low', 'mid', 'high'],
        duplicates='drop'
    )
    
    return regime


def label_adaptive_binary(df, atr_col='atr', forward_bars=60, vol_window=250):
    """
    Create binary labels with volatility-adaptive thresholds.
    
    Adaptive thresholds:
    - Low volatility:  1.5 × ATR
    - Mid volatility:  2.0 × ATR
    - High volatility: 2.5 × ATR
    
    Logic:
    - UP (1):   forward_return > threshold
    - DOWN (0): forward_return < -threshold
    - Neutral:  Ignored (no label)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'close' and ATR column
    atr_col : str
        Name of ATR column
    forward_bars : int
        Lookahead period for returns
    vol_window : int
        Window for volatility regime calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns:
        - 'vol_regime': low/mid/high
        - 'adaptive_threshold': ATR multiplier
        - 'forward_return': forward price change
        - 'label': 0 (DOWN), 1 (UP), NaN (neutral)
    """
    df = df.copy()
    
    # 1. Calculate volatility regime
    print("  Calculating volatility regime...")
    df['vol_regime'] = calculate_volatility_regime(df, window=vol_window)
    
    # 2. Calculate forward returns
    df['forward_return'] = df['close'].pct_change(forward_bars).shift(-forward_bars)
    
    # 3. Set adaptive threshold based on regime
    print("  Setting adaptive thresholds...")
    threshold_map = {
        'low': 1.5,
        'mid': 2.0,
        'high': 2.5
    }
    
    # Convert category to string first for mapping
    df['adaptive_threshold'] = df['vol_regime'].astype(str).map(threshold_map)
    
    # 4. Calculate absolute threshold in price terms
    df['threshold_value'] = df['adaptive_threshold'] * df[atr_col]
    
    # 5. Create binary labels
    print("  Creating binary labels...")
    df['label'] = np.nan
    
    # UP: forward_return > threshold
    up_mask = df['forward_return'] > (df['threshold_value'] / df['close'])
    df.loc[up_mask, 'label'] = 1
    
    # DOWN: forward_return < -threshold
    down_mask = df['forward_return'] < -(df['threshold_value'] / df['close'])
    df.loc[down_mask, 'label'] = 0
    
    # Count labels
    n_up = (df['label'] == 1).sum()
    n_down = (df['label'] == 0).sum()
    n_neutral = df['label'].isna().sum()
    total = len(df)
    
    print(f"  Label distribution:")
    print(f"    UP (1):      {n_up:,} ({n_up/total*100:.1f}%)")
    print(f"    DOWN (0):    {n_down:,} ({n_down/total*100:.1f}%)")
    print(f"    Neutral:     {n_neutral:,} ({n_neutral/total*100:.1f}%)")
    print(f"    UP/DOWN ratio: {n_up/n_down if n_down > 0 else 0:.2f}")
    
    # Analyze by regime
    print(f"\n  Distribution by volatility regime:")
    for regime in ['low', 'mid', 'high']:
        regime_mask = df['vol_regime'] == regime
        regime_up = ((df['label'] == 1) & regime_mask).sum()
        regime_down = ((df['label'] == 0) & regime_mask).sum()
        regime_total = regime_mask.sum()
        
        if regime_total > 0:
            print(f"    {regime.upper():5s}: UP={regime_up:,} ({regime_up/regime_total*100:.1f}%), "
                  f"DOWN={regime_down:,} ({regime_down/regime_total*100:.1f}%)")
    
    return df


def analyze_label_quality_adaptive(df):
    """
    Analyze quality metrics for adaptive binary labels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'label', 'vol_regime', 'adaptive_threshold' columns
    
    Returns:
    --------
    dict
        Quality metrics
    """
    # Filter valid labels
    valid_labels = df.dropna(subset=['label'])
    
    if len(valid_labels) == 0:
        return {
            'error': 'No valid labels found',
            'total_rows': len(df)
        }
    
    # Basic metrics
    n_up = (valid_labels['label'] == 1).sum()
    n_down = (valid_labels['label'] == 0).sum()
    total_valid = len(valid_labels)
    
    # Balance ratio (ideally close to 1.0)
    balance_ratio = min(n_up, n_down) / max(n_up, n_down) if max(n_up, n_down) > 0 else 0
    
    # Label density (% of rows with labels vs neutral)
    label_density = total_valid / len(df)
    
    # Transition rate (how often label changes)
    transitions = (valid_labels['label'].diff() != 0).sum()
    transition_rate = transitions / len(valid_labels) if len(valid_labels) > 0 else 0
    
    # Regime distribution
    regime_dist = {}
    for regime in ['low', 'mid', 'high']:
        regime_count = (valid_labels['vol_regime'] == regime).sum()
        regime_dist[regime] = {
            'count': int(regime_count),
            'percentage': regime_count / total_valid * 100 if total_valid > 0 else 0
        }
    
    metrics = {
        'total_rows': len(df),
        'valid_labels': total_valid,
        'label_density': label_density,
        'up_count': int(n_up),
        'down_count': int(n_down),
        'up_percentage': n_up / total_valid * 100 if total_valid > 0 else 0,
        'down_percentage': n_down / total_valid * 100 if total_valid > 0 else 0,
        'balance_ratio': balance_ratio,
        'transition_rate': transition_rate,
        'regime_distribution': regime_dist
    }
    
    return metrics


def build_dataset_adaptive(df, atr_col='atr', forward_bars=60, vol_window=250):
    """
    Complete pipeline to build dataset with adaptive binary labels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLCV data with ATR
    atr_col : str
        ATR column name
    forward_bars : int
        Forward lookahead period
    vol_window : int
        Volatility regime window
    
    Returns:
    --------
    pd.DataFrame
        Dataset with adaptive labels ready for training
    """
    print("Building dataset with adaptive binary labels...")
    
    # Create labels
    df = label_adaptive_binary(
        df, 
        atr_col=atr_col,
        forward_bars=forward_bars,
        vol_window=vol_window
    )
    
    # Analyze quality
    print("\nAnalyzing label quality...")
    quality = analyze_label_quality_adaptive(df)
    
    print(f"\nQuality Metrics:")
    print(f"  Label density:   {quality['label_density']:.1%}")
    print(f"  Balance ratio:   {quality['balance_ratio']:.3f}")
    print(f"  Transition rate: {quality['transition_rate']:.1%}")
    
    # Drop rows without labels (neutral)
    df_labeled = df.dropna(subset=['label']).copy()
    
    print(f"\nFinal dataset:")
    print(f"  Total rows:   {len(df_labeled):,}")
    print(f"  UP labels:    {(df_labeled['label']==1).sum():,}")
    print(f"  DOWN labels:  {(df_labeled['label']==0).sum():,}")
    
    return df_labeled


if __name__ == "__main__":
    """
    Test adaptive labeling with sample data
    """
    import sys
    sys.path.append('/Users/merari/Desktop/bot de scalping')
    
    # Load sample data
    print("Loading sample data...")
    df = pd.read_csv('data/btcusdt_5m.csv', parse_dates=['timestamp'])
    
    # Ensure ATR exists
    if 'atr' not in df.columns:
        print("Calculating ATR...")
        from backend.app.features.technical import add_technical_indicators
        df = add_technical_indicators(df)
    
    # Build dataset
    df_adaptive = build_dataset_adaptive(
        df,
        atr_col='atr',
        forward_bars=60,
        vol_window=250
    )
    
    print(f"\nSample rows:")
    print(df_adaptive[['close', 'vol_regime', 'adaptive_threshold', 'label']].head(20))
    
    print("\nAdaptive binary labeling test complete!")
