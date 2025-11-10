"""
Label Generation V2 - Multi-class Target Based on Quantiles

5-class classification based on return/ATR quantiles:
- Class 0: No movement (neutral)
- Class 1: Weak up
- Class 2: Strong up  
- Class 3: Weak down
- Class 4: Strong down
"""
import numpy as np
import pandas as pd


def label_multiclass_quantiles(df, horizon=60, atr_period=14, n_classes=5):
    """
    Create multi-class labels based on quantiles of return/ATR ratio.
    
    Args:
        df: pd.DataFrame with OHLCV data
        horizon: int, forward-looking bars
        atr_period: int, ATR calculation period
        n_classes: int, number of classes (default 5)
        
    Returns:
        pd.DataFrame with added label columns
    """
    print(f"  Creating {n_classes}-class labels (horizon={horizon})...")
    
    # Calculate ATR if not present
    if f'ATR_{atr_period}' not in df.columns:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'ATR_{atr_period}'] = tr.rolling(atr_period).mean()
    
    # Future price
    future_close = df['close'].shift(-horizon)
    
    # Raw return
    raw_return = future_close - df['close']
    
    # Normalized return (by ATR)
    normalized_return = raw_return / (df[f'ATR_{atr_period}'] + 1e-9)
    
    # Create quantile-based labels
    # Use 5 bins: [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
    df['y_multiclass'] = pd.qcut(
        normalized_return, 
        q=n_classes, 
        labels=False,
        duplicates='drop'
    )
    
    # Map to semantic classes
    # 0: strong down, 1: weak down, 2: neutral, 3: weak up, 4: strong up
    class_mapping = {
        0: 'strong_down',
        1: 'weak_down', 
        2: 'neutral',
        3: 'weak_up',
        4: 'strong_up'
    }
    
    df['y_class_name'] = df['y_multiclass'].map(class_mapping)
    
    # Binary labels for compatibility
    # 1 if return > 0, else 0
    df['y_binary'] = (normalized_return > 0).astype(int)
    
    # Direction for trading
    # 1 (long), -1 (short), 0 (no trade)
    # Trade only on strong signals (class 0 or 4)
    df['y_direction'] = 0
    df.loc[df['y_multiclass'] == 4, 'y_direction'] = 1   # Strong up -> Long
    df.loc[df['y_multiclass'] == 0, 'y_direction'] = -1  # Strong down -> Short
    
    # Store raw normalized return for analysis
    df['normalized_return'] = normalized_return
    
    # Drop last horizon rows (no valid labels)
    df = df.iloc[:-horizon].copy()
    
    # Validate class balance
    class_dist = df['y_multiclass'].value_counts(normalize=True).sort_index()
    
    print(f"  Class distribution:")
    for cls, pct in class_dist.items():
        class_name = class_mapping.get(cls, f'class_{cls}')
        print(f"    {cls} ({class_name}): {pct*100:.1f}%")
    
    # Check for severe imbalance
    if class_dist.max() > 0.35 or class_dist.min() < 0.10:
        print(f"  ⚠ WARNING: Class imbalance detected!")
        print(f"    Max: {class_dist.max()*100:.1f}%, Min: {class_dist.min()*100:.1f}%")
    
    return df


def label_multiclass_thresholds(df, horizon=60, atr_period=14):
    """
    Alternative: Create multi-class labels using fixed ATR thresholds.
    
    Classes:
    - 0: Strong down (return < -2.0 ATR)
    - 1: Weak down (-2.0 ATR < return < -0.5 ATR)
    - 2: Neutral (-0.5 ATR < return < 0.5 ATR)
    - 3: Weak up (0.5 ATR < return < 2.0 ATR)
    - 4: Strong up (return > 2.0 ATR)
    
    Args:
        df: pd.DataFrame with OHLCV data
        horizon: int, forward-looking bars
        atr_period: int, ATR calculation period
        
    Returns:
        pd.DataFrame with added label columns
    """
    print(f"  Creating threshold-based labels (horizon={horizon})...")
    
    # Calculate ATR if not present
    if f'ATR_{atr_period}' not in df.columns:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'ATR_{atr_period}'] = tr.rolling(atr_period).mean()
    
    # Future price and return
    future_close = df['close'].shift(-horizon)
    raw_return = future_close - df['close']
    normalized_return = raw_return / (df[f'ATR_{atr_period}'] + 1e-9)
    
    # Define thresholds
    strong_threshold = 2.0
    weak_threshold = 0.5
    
    # Create labels
    df['y_multiclass'] = 2  # Default: neutral
    
    # Strong down
    df.loc[normalized_return < -strong_threshold, 'y_multiclass'] = 0
    # Weak down
    df.loc[(normalized_return >= -strong_threshold) & (normalized_return < -weak_threshold), 'y_multiclass'] = 1
    # Weak up
    df.loc[(normalized_return > weak_threshold) & (normalized_return <= strong_threshold), 'y_multiclass'] = 3
    # Strong up
    df.loc[normalized_return > strong_threshold, 'y_multiclass'] = 4
    
    # Map to names
    class_mapping = {
        0: 'strong_down',
        1: 'weak_down',
        2: 'neutral',
        3: 'weak_up',
        4: 'strong_up'
    }
    
    df['y_class_name'] = df['y_multiclass'].map(class_mapping)
    
    # Binary and direction labels
    df['y_binary'] = (normalized_return > 0).astype(int)
    
    df['y_direction'] = 0
    df.loc[df['y_multiclass'] == 4, 'y_direction'] = 1
    df.loc[df['y_multiclass'] == 0, 'y_direction'] = -1
    
    df['normalized_return'] = normalized_return
    
    # Drop last horizon rows
    df = df.iloc[:-horizon].copy()
    
    # Class distribution
    class_dist = df['y_multiclass'].value_counts(normalize=True).sort_index()
    
    print(f"  Class distribution:")
    for cls, pct in class_dist.items():
        class_name = class_mapping.get(cls, f'class_{cls}')
        print(f"    {cls} ({class_name}): {pct*100:.1f}%")
    
    return df


def get_walk_forward_splits_v2(df, train_months=3, test_months=1, purge_weeks=1):
    """
    Generate walk-forward splits with purging (same as V1).
    
    Args:
        df: pd.DataFrame with timestamp column
        train_months: int, training window in months
        test_months: int, test window in months
        purge_weeks: int, purge period in weeks
        
    Returns:
        list of tuples: [(train_start, train_end, test_start, test_end), ...]
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Bars per period (1-minute bars)
    bars_per_month = 30 * 24 * 60
    train_bars = train_months * bars_per_month
    test_bars = test_months * bars_per_month
    purge_bars = purge_weeks * 7 * 24 * 60
    
    splits = []
    total_bars = len(df)
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


def analyze_label_quality(df, label_col='y_multiclass'):
    """
    Analyze label distribution and quality metrics.
    
    Args:
        df: pd.DataFrame with labels
        label_col: str, label column name
        
    Returns:
        dict: Quality metrics
    """
    metrics = {}
    
    # Class distribution
    class_dist = df[label_col].value_counts(normalize=True).sort_index()
    metrics['class_distribution'] = class_dist.to_dict()
    
    # Balance metrics
    metrics['max_class_pct'] = class_dist.max()
    metrics['min_class_pct'] = class_dist.min()
    metrics['balance_ratio'] = class_dist.min() / class_dist.max()
    
    # Entropy (higher = more balanced)
    from scipy.stats import entropy
    metrics['entropy'] = entropy(class_dist.values)
    metrics['max_entropy'] = np.log(len(class_dist))  # Maximum possible entropy
    metrics['entropy_ratio'] = metrics['entropy'] / metrics['max_entropy']
    
    # Transition frequency (how often labels change)
    transitions = (df[label_col].diff() != 0).sum()
    metrics['transition_rate'] = transitions / len(df)
    
    return metrics
