#!/usr/bin/env python3
"""
Build ML dataset with microstructure features and proper train/test splitting.

Steps:
1. Load klines + orderbook + funding data
2. Generate technical indicators (ATR, RSI, ADX, Bollinger, Volume features)
3. Generate microstructure features (imbalance5, micro_price_dist, cvd_delta)
4. Create labels with significant move threshold (2*ATR)
5. Shift features by 1 bar to prevent look-ahead bias
6. Generate walk-forward splits with 1-week purge
7. Save to parquet files

Usage:
    python build_dataset_v1.py --symbols BTCUSDT ETHUSDT BNBUSDT --span 12m
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.data.streams import merge_data_sources
from app.features.microstructure import add_microstructure_features, calculate_vwap
from app.features.labels import label_move_significant, get_walk_forward_splits


def calculate_technical_indicators(df):
    """
    Calculate standard technical indicators.
    
    Args:
        df: pd.DataFrame with OHLCV columns
        
    Returns:
        pd.DataFrame with added indicator columns
    """
    # ATR (already calculated in labels, but ensure it's here)
    high = df['high']
    low = df['low']
    close = df['close']
    
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
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-9)
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-9)
    
    # VWAP distance
    df['vwap_20'] = calculate_vwap(df, window=20)
    df['vwap_dist'] = (close - df['vwap_20']) / (df['vwap_20'] + 1e-9)
    
    # Price momentum
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['return_20'] = close.pct_change(20)
    
    return df


def build_dataset(symbol, span_months=12, horizon=60, atr_threshold=2.0):
    """
    Build complete ML dataset for a symbol.
    
    Args:
        symbol: str, e.g., 'BTCUSDT'
        span_months: int, total data span in months
        horizon: int, prediction horizon in bars
        atr_threshold: float, ATR multiplier for significant moves
        
    Returns:
        pd.DataFrame: Complete dataset with features and labels
    """
    print(f"\n{'='*60}")
    print(f"Building dataset for {symbol}")
    print(f"{'='*60}")
    
    # 1. Load data
    print("1. Loading klines + orderbook + funding...")
    df = merge_data_sources(symbol, interval="1m", orderbook_depth=5)
    print(f"   Loaded {len(df):,} bars")
    
    # 2. Technical indicators
    print("2. Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # 3. Microstructure features
    print("3. Calculating microstructure features...")
    df = add_microstructure_features(df)
    
    # 4. Create labels
    print(f"4. Creating labels (horizon={horizon}, ATR threshold={atr_threshold})...")
    df = label_move_significant(df, horizon=horizon, atr_threshold=atr_threshold)
    
    # 5. Shift features by 1 bar (CRITICAL: prevents look-ahead bias)
    print("5. Shifting features by 1 bar to prevent look-ahead...")
    feature_cols = [c for c in df.columns if c not in 
                    ['timestamp', 'y', 'y_sign', 'future_return', 'open', 'high', 'low', 'close', 'volume']]
    
    for col in feature_cols:
        df[col] = df[col].shift(1)
    
    # Drop rows with NaN (from rolling windows and shift)
    df = df.dropna().reset_index(drop=True)
    
    print(f"   Final dataset: {len(df):,} rows, {len(feature_cols)} features")
    print(f"   Label distribution: {df['y'].value_counts().to_dict()}")
    print(f"   Direction distribution: {df['y_sign'].value_counts().to_dict()}")
    
    return df


def save_dataset_splits(df, symbol, output_dir):
    """
    Generate walk-forward splits and save to parquet.
    
    Args:
        df: pd.DataFrame with complete dataset
        symbol: str, symbol name
        output_dir: Path, output directory
    """
    print("\n6. Generating walk-forward splits (3m train / 1m test, 1w purge)...")
    
    splits = get_walk_forward_splits(df, train_months=3, test_months=1, purge_weeks=1)
    
    print(f"   Generated {len(splits)} walk-forward windows")
    
    # Save full dataset
    full_path = output_dir / f"{symbol}_full.parquet"
    df.to_parquet(full_path, index=False)
    print(f"   Saved full dataset: {full_path}")
    
    # Save splits metadata
    splits_meta = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
        meta = {
            'window': i,
            'train_start_idx': train_start,
            'train_end_idx': train_end,
            'test_start_idx': test_start,
            'test_end_idx': test_end,
            'train_start_time': df.iloc[train_start]['timestamp'],
            'train_end_time': df.iloc[train_end - 1]['timestamp'],
            'test_start_time': df.iloc[test_start]['timestamp'],
            'test_end_time': df.iloc[test_end - 1]['timestamp'],
            'train_samples': train_end - train_start,
            'test_samples': test_end - test_start
        }
        splits_meta.append(meta)
    
    splits_df = pd.DataFrame(splits_meta)
    splits_path = output_dir / f"{symbol}_splits.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"   Saved splits metadata: {splits_path}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Build ML datasets with microstructure features')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                        help='Symbols to process')
    parser.add_argument('--span', type=str, default='12m', help='Data span (e.g., 12m)')
    parser.add_argument('--horizon', type=int, default=60, help='Prediction horizon in bars')
    parser.add_argument('--atr-threshold', type=float, default=2.0, 
                        help='ATR multiplier for significant moves')
    
    args = parser.parse_args()
    
    # Parse span
    span_months = int(args.span.replace('m', ''))
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Dataset Builder v1 - Microstructure Features")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Span: {span_months} months")
    print(f"Horizon: {args.horizon} bars")
    print(f"ATR Threshold: {args.atr_threshold}x")
    print(f"Output: {output_dir}")
    
    # Process each symbol
    all_splits = {}
    
    for symbol in args.symbols:
        try:
            df = build_dataset(
                symbol=symbol,
                span_months=span_months,
                horizon=args.horizon,
                atr_threshold=args.atr_threshold
            )
            
            splits = save_dataset_splits(df, symbol, output_dir)
            all_splits[symbol] = splits
            
        except Exception as e:
            print(f"   ERROR processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Dataset building complete!")
    print(f"{'='*60}")
    print(f"Processed {len(all_splits)} symbols")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
