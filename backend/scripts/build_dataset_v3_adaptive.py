"""
Build Dataset V3 - Adaptive Binary Labeling
Sprint 14 - Edge Reconstruction
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.data.streams import load_klines
from app.features.microstructure_v2 import add_microstructure_features_v2
from app.features.labels_v3_adaptive import build_dataset_adaptive


def load_raw_data(symbol, data_dir='data'):
    """Load existing dataset V2 as base."""
    print(f"Loading existing dataset V2 for {symbol}...")
    
    # Try loading V2 dataset (quantile or threshold)
    v2_path_quantile = Path(data_dir) / 'datasets_v2' / f"{symbol}_v2_quantile.parquet"
    v2_path_threshold = Path(data_dir) / 'datasets_v2' / f"{symbol}_v2_threshold.parquet"
    
    if v2_path_quantile.exists():
        df = pd.read_parquet(v2_path_quantile)
        print(f"  Loaded from V2 quantile dataset: {len(df):,} rows")
    elif v2_path_threshold.exists():
        df = pd.read_parquet(v2_path_threshold)
        print(f"  Loaded from V2 threshold dataset: {len(df):,} rows")
    else:
        raise FileNotFoundError(
            f"No V2 dataset found for {symbol}. "
            f"Run build_dataset_v2.py first."
        )
    
    # Drop V2 label columns
    label_cols_to_drop = ['label', 'y_multiclass', 'y_class_name', 
                          'y_binary', 'y_direction', 'normalized_return']
    existing_cols = [c for c in label_cols_to_drop if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"  Dropped V2 label columns: {existing_cols}")
    
    return df


def calculate_technical_indicators(df):
    """
    Calculate standard technical indicators.
    
    Args:
        df: pd.DataFrame with OHLCV columns
        
    Returns:
        pd.DataFrame with added indicators
    """
    print("  Calculating technical indicators...")
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['ATR_14'] = df['atr']  # Alias
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr_smooth)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX_14'] = dx.rolling(14).mean()
    
    # Bollinger Bands
    df['BB_middle'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 1e-9)
    df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-9)
    
    # Volume features
    df['volume_sma_20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-9)
    
    # VWAP
    typical_price = (high + low + close) / 3
    df['vwap_20'] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    df['vwap_dist'] = (close - df['vwap_20']) / (df['vwap_20'] + 1e-9)
    
    # Price momentum
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['return_20'] = close.pct_change(20)
    
    print(f"    ✓ Added 14 technical indicators")
    
    return df


def prepare_features(df):
    """V2 features already exist, just ensure ATR column."""
    print("\n1. Verifying features...")
    
    # Check if we have ATR
    if 'atr' not in df.columns and 'ATR_14' in df.columns:
        df['atr'] = df['ATR_14']
        print("  ✓ Created 'atr' alias from 'ATR_14'")
    elif 'atr' in df.columns:
        print("  ✓ ATR column exists")
    else:
        raise ValueError("No ATR column found in dataset")
    
    # Verify we have timestamp
    if 'timestamp' not in df.columns:
        raise ValueError("No timestamp column found")
    
    feature_count = len([c for c in df.columns if c not in ['timestamp', 'open_time', 'close_time']])
    print(f"  ✓ Dataset has {feature_count} features")
    
    return df


def create_adaptive_labels(df, forward_bars=60, vol_window=250):
    """Create adaptive binary labels."""
    print(f"\n3. Creating adaptive binary labels (horizon={forward_bars})...")
    df = build_dataset_adaptive(
        df,
        atr_col='atr',
        forward_bars=forward_bars,
        vol_window=vol_window
    )
    return df


def shift_features(df, shift_bars=1):
    """
    V2 dataset already has 1-bar shift applied.
    Just verify and drop NaN.
    """
    print(f"\n4. Features already shifted in V2 dataset...")
    
    # Drop rows with NaN
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    
    print(f"  Dropped {dropped:,} rows with NaN")
    print(f"  Final dataset: {len(df):,} rows")
    
    return df


def validate_dataset(df):
    """Validate dataset quality."""
    print("\n5. Validating dataset...")
    
    # Check for NaN/Inf
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  WARNING: Found {nan_counts.sum()} NaN values")
        top_nan = nan_counts[nan_counts > 0].head(5)
        print(f"  Top NaN columns:\n{top_nan}")
    
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print(f"  WARNING: Found {inf_counts.sum()} Inf values")
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print(f"\n  Label distribution:")
    print(f"    DOWN (0): {label_counts.get(0, 0):,}")
    print(f"    UP (1):   {label_counts.get(1, 0):,}")
    
    # Regime distribution
    regime_counts = df['vol_regime'].value_counts()
    print(f"\n  Volatility regime distribution:")
    for regime in ['low', 'mid', 'high']:
        count = regime_counts.get(regime, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"    {regime:5s}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  ✓ Dataset validated")


def generate_walk_forward_splits(df, train_months=3, test_months=1, purge_days=7):
    """
    Generate walk-forward train/test splits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with timestamp
    train_months : int
        Training period in months
    test_months : int
        Test period in months
    purge_days : int
        Purge period between train/test in days
    
    Returns:
    --------
    list of dict
        List of split configurations
    """
    print(f"\n6. Generating walk-forward splits...")
    print(f"  Train: {train_months}m, Test: {test_months}m, Purge: {purge_days}d")
    
    # Calculate period lengths
    train_delta = pd.Timedelta(days=train_months * 30)
    test_delta = pd.Timedelta(days=test_months * 30)
    purge_delta = pd.Timedelta(days=purge_days)
    window_delta = train_delta + test_delta
    
    # Get date range
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    splits = []
    window_id = 0
    current_start = start_date
    
    while current_start + window_delta <= end_date:
        train_end = current_start + train_delta
        purge_end = train_end + purge_delta
        test_start = purge_end
        test_end = test_start + test_delta
        
        # Get indices
        train_mask = (df['timestamp'] >= current_start) & (df['timestamp'] < train_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        train_indices = df[train_mask].index.tolist()
        test_indices = df[test_mask].index.tolist()
        
        if len(train_indices) > 0 and len(test_indices) > 0:
            split = {
                'window_id': window_id,
                'train_start': current_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            }
            splits.append(split)
            
            print(f"  Window {window_id}: "
                  f"Train {split['train_start']} to {split['train_end']} ({len(train_indices):,}), "
                  f"Test {split['test_start']} to {split['test_end']} ({len(test_indices):,})")
            
            window_id += 1
        
        # Move to next window
        current_start += window_delta
    
    print(f"  Generated {len(splits)} windows")
    return splits


def save_dataset(df, splits, symbol, output_dir='data/datasets_v3'):
    """Save dataset and splits to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    dataset_file = output_path / f"{symbol}_v3_adaptive.parquet"
    df.to_parquet(dataset_file, index=False)
    print(f"\n  Saved: {dataset_file}")
    
    # Save splits
    splits_file = output_path / f"{symbol}_v3_splits.csv"
    pd.DataFrame(splits).to_csv(splits_file, index=False)
    print(f"  Saved: {splits_file}")
    
    return dataset_file, splits_file


def main():
    parser = argparse.ArgumentParser(description='Build Dataset V3 - Adaptive Binary')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], 
                       help='Symbols to process')
    parser.add_argument('--span', type=str, default='12m',
                       help='Data span (e.g., 12m)')
    parser.add_argument('--horizon', type=int, default=60,
                       help='Forward bars for labels')
    parser.add_argument('--vol-window', type=int, default=250,
                       help='Volatility regime window')
    parser.add_argument('--output-dir', type=str, default='data/datasets_v3',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("############################################################")
    print("# Dataset Builder V3 - Adaptive Binary Labeling")
    print("############################################################")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Span: {args.span}")
    print(f"Horizon: {args.horizon} bars")
    print(f"Volatility window: {args.vol_window} bars")
    print(f"Output: {args.output_dir}")
    print()
    
    for symbol in args.symbols:
        print("=" * 60)
        print(f"Building Dataset V3 for {symbol}")
        print("=" * 60)
        
        try:
            # Load data
            df = load_raw_data(symbol)
            
            # Add features
            df = prepare_features(df)
            
            # Create adaptive labels
            df = create_adaptive_labels(
                df,
                forward_bars=args.horizon,
                vol_window=args.vol_window
            )
            
            # Shift features (1-bar latency)
            df = shift_features(df, shift_bars=1)
            
            # Validate
            validate_dataset(df)
            
            # Generate splits
            splits = generate_walk_forward_splits(df)
            
            # Save
            dataset_file, splits_file = save_dataset(
                df, splits, symbol, args.output_dir
            )
            
            print(f"\n✓ {symbol} dataset V3 complete!")
            
        except Exception as e:
            print(f"\n✗ ERROR building {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Dataset V3 building complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
