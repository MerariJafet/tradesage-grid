#!/usr/bin/env python3
"""
Build Dataset V2 - Real Binance Data with Multiclass Labels

Integrates:
- Real funding rates, orderbook, aggTrades
- V2 microstructure features (VPIN, Roll spread, etc.)
- Multi-class quantile labels
- Walk-forward splits with purging

Usage:
    python build_dataset_v2.py --symbols BTCUSDT ETHUSDT BNBUSDT --mode quantile
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.data.streams import load_klines, merge_data_sources
from app.features.microstructure_v2 import add_microstructure_features_v2, validate_features
from app.features.labels_v2 import (
    label_multiclass_quantiles, 
    label_multiclass_thresholds,
    get_walk_forward_splits_v2,
    analyze_label_quality
)


def calculate_technical_indicators(df):
    """
    Calculate standard technical indicators (same as V1).
    
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
    df['ATR_14'] = tr.rolling(14).mean()
    
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
    
    print(f"    ✓ Added {14} technical indicators")
    
    return df


def build_dataset_v2(symbol, span_months=12, horizon=60, labeling_mode='quantile'):
    """
    Build complete ML dataset V2 with real microstructure data.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        span_months: int, total data span in months
        horizon: int, prediction horizon in bars
        labeling_mode: str, 'quantile' or 'threshold'
        
    Returns:
        pd.DataFrame: Complete dataset
    """
    print(f"\n{'='*60}")
    print(f"Building Dataset V2 for {symbol}")
    print(f"{'='*60}")
    
    # 1. Load klines + microstructure data
    print("1. Loading data (klines + orderbook + funding + trades)...")
    df = merge_data_sources(symbol, interval="1m", orderbook_depth=5)
    print(f"   Loaded {len(df):,} bars")
    
    # 2. Technical indicators
    df = calculate_technical_indicators(df)
    
    # 3. Microstructure features V2
    print("2. Calculating V2 microstructure features...")
    df = add_microstructure_features_v2(df)
    
    # 4. Create labels
    print(f"3. Creating {labeling_mode} labels (horizon={horizon})...")
    if labeling_mode == 'quantile':
        df = label_multiclass_quantiles(df, horizon=horizon, n_classes=5)
    else:
        df = label_multiclass_thresholds(df, horizon=horizon)
    
    # 5. Shift features by 1 bar (prevent look-ahead)
    print("4. Shifting features by 1 bar...")
    feature_cols = [c for c in df.columns if c not in [
        'timestamp', 'y_multiclass', 'y_class_name', 'y_binary', 'y_direction',
        'normalized_return', 'open', 'high', 'low', 'close', 'volume'
    ]]
    
    for col in feature_cols:
        df[col] = df[col].shift(1)
    
    # 6. Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    print(f"   Final dataset: {len(df):,} rows, {len(feature_cols)} features")
    
    # 7. Validate features
    print("5. Validating feature quality...")
    validation_report = validate_features(df)
    
    if validation_report['nan_counts']:
        print(f"   ⚠ NaN found in: {list(validation_report['nan_counts'].keys())}")
    if validation_report['inf_counts']:
        print(f"   ⚠ Inf found in: {list(validation_report['inf_counts'].keys())}")
    
    # 8. Analyze labels
    print("6. Analyzing label quality...")
    label_metrics = analyze_label_quality(df, 'y_multiclass')
    print(f"   Balance ratio: {label_metrics['balance_ratio']:.3f}")
    print(f"   Entropy ratio: {label_metrics['entropy_ratio']:.3f}")
    print(f"   Transition rate: {label_metrics['transition_rate']:.3f}")
    
    return df


def save_dataset_splits_v2(df, symbol, output_dir, labeling_mode):
    """
    Save dataset with walk-forward splits.
    
    Args:
        df: pd.DataFrame
        symbol: str
        output_dir: Path
        labeling_mode: str
    """
    print("\n7. Generating walk-forward splits...")
    
    splits = get_walk_forward_splits_v2(df, train_months=3, test_months=1, purge_weeks=1)
    print(f"   Generated {len(splits)} windows")
    
    # Save full dataset
    full_path = output_dir / f"{symbol}_v2_{labeling_mode}.parquet"
    df.to_parquet(full_path, index=False)
    print(f"   Saved: {full_path}")
    
    # Save splits metadata
    splits_meta = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        meta = {
            'window': i,
            'train_start_idx': train_start,
            'train_end_idx': train_end,
            'test_start_idx': test_start,
            'test_end_idx': test_end,
            'train_start_time': str(df.iloc[train_start]['timestamp']),
            'train_end_time': str(df.iloc[train_end - 1]['timestamp']),
            'test_start_time': str(df.iloc[test_start]['timestamp']),
            'test_end_time': str(df.iloc[test_end - 1]['timestamp']),
            'train_samples': train_end - train_start,
            'test_samples': test_end - test_start
        }
        splits_meta.append(meta)
    
    splits_df = pd.DataFrame(splits_meta)
    splits_path = output_dir / f"{symbol}_v2_splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"   Saved: {splits_path}")


def main():
    parser = argparse.ArgumentParser(description='Build Dataset V2 with real microstructure')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    parser.add_argument('--span', type=str, default='12m')
    parser.add_argument('--horizon', type=int, default=60)
    parser.add_argument('--mode', type=str, default='quantile', 
                        choices=['quantile', 'threshold'])
    
    args = parser.parse_args()
    
    span_months = int(args.span.replace('m', ''))
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "datasets_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Dataset Builder V2 - Real Microstructure + Multiclass")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Span: {span_months} months")
    print(f"Horizon: {args.horizon} bars")
    print(f"Labeling mode: {args.mode}")
    print(f"Output: {output_dir}")
    
    # Process each symbol
    for symbol in args.symbols:
        try:
            df = build_dataset_v2(
                symbol=symbol,
                span_months=span_months,
                horizon=args.horizon,
                labeling_mode=args.mode
            )
            
            save_dataset_splits_v2(df, symbol, output_dir, args.mode)
            
        except Exception as e:
            print(f"   ERROR processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Dataset V2 building complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
