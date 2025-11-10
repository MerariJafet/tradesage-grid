#!/usr/bin/env python3
"""
Advanced Feature Engineering - Microstructure Features V2

Implementa features avanzadas para buscar edge monetario:
1. Kyle's Lambda (Price Impact)
2. Trade Intensity Bursts
3. Depth Imbalance L2-L5
4. Roll Impact
5. Quote Stuffing Detector

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15.5 (Advanced Feature Exploration)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """Compute advanced microstructure features."""
    
    def __init__(self, symbol: str, data_path: str = 'data/real_binance_ws_pilot'):
        self.symbol = symbol
        self.data_path = Path(data_path) / symbol
        
    def load_orderbook(self, date: str = None) -> pd.DataFrame:
        """Load orderbook data."""
        orderbook_path = self.data_path / 'orderbook'
        
        if date:
            files = list(orderbook_path.glob(f'date={date}/*.parquet'))
        else:
            files = list(orderbook_path.glob('**/*.parquet'))
        
        if not files:
            raise FileNotFoundError(f"No orderbook files found for {self.symbol}")
        
        logger.info(f"Loading {len(files)} orderbook files...")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def load_trades(self, date: str = None) -> pd.DataFrame:
        """Load trade data."""
        trades_path = self.data_path / 'aggtrades'
        
        if date:
            files = list(trades_path.glob(f'date={date}/*.parquet'))
        else:
            files = list(trades_path.glob('**/*.parquet'))
        
        if not files:
            raise FileNotFoundError(f"No trade files found for {self.symbol}")
        
        logger.info(f"Loading {len(files)} trade files...")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def compute_kyles_lambda(self, trades: pd.DataFrame, orderbook: pd.DataFrame,
                            window_ms: int = 1000) -> pd.Series:
        """
        Kyle's Lambda: Measure of price impact.
        
        λ = Cov(ΔPrice, SignedVolume) / Var(SignedVolume)
        
        Higher lambda = more toxic flow (informed trading).
        """
        logger.info(f"Computing Kyle's Lambda (window={window_ms}ms)...")
        
        # Merge trades with orderbook on nearest timestamp
        df = pd.merge_asof(
            trades.sort_values('timestamp'),
            orderbook[['timestamp', 'midprice']].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        # Signed volume (buy = +1, sell = -1)
        df['signed_volume'] = np.where(df['is_buyer_maker'], -df['quantity'], df['quantity'])
        
        # Price change
        df['price_change'] = df['midprice'].diff()
        
        # Rolling window calculation
        window_size = int(window_ms / 100)  # Assuming 100ms sampling
        
        # Covariance and Variance
        df['cov'] = df['price_change'].rolling(window_size).cov(df['signed_volume'])
        df['var'] = df['signed_volume'].rolling(window_size).var()
        
        # Lambda
        df['lambda'] = df['cov'] / (df['var'] + 1e-10)
        
        # Resample to orderbook timestamps
        result = pd.merge_asof(
            orderbook[['timestamp']].sort_values('timestamp'),
            df[['timestamp', 'lambda']].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        return result['lambda'].fillna(0)
    
    def compute_trade_intensity(self, trades: pd.DataFrame, orderbook: pd.DataFrame,
                                window_ms: int = 1000) -> pd.Series:
        """
        Trade Intensity Bursts: Detect sudden increase in trading activity.
        
        Intensity = N_trades_window / rolling_avg(N_trades_1h)
        
        Values > 2.0 indicate burst activity.
        """
        logger.info(f"Computing Trade Intensity (window={window_ms}ms)...")
        
        # Count trades per window
        trades['timestamp_round'] = trades['timestamp'].dt.floor(f'{window_ms}ms')
        trade_counts = trades.groupby('timestamp_round').size().reset_index(name='n_trades')
        trade_counts = trade_counts.rename(columns={'timestamp_round': 'timestamp'})
        
        # Compute rolling average (1 hour baseline)
        baseline_window = int(3600000 / window_ms)  # 1 hour in windows
        trade_counts['baseline'] = trade_counts['n_trades'].rolling(
            baseline_window, min_periods=10
        ).mean()
        
        # Intensity ratio
        trade_counts['intensity'] = trade_counts['n_trades'] / (trade_counts['baseline'] + 1)
        
        # Merge with orderbook
        result = pd.merge_asof(
            orderbook[['timestamp']].sort_values('timestamp'),
            trade_counts[['timestamp', 'intensity']].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        return result['intensity'].fillna(1.0)
    
    def compute_depth_imbalance_L2L5(self, orderbook: pd.DataFrame) -> pd.Series:
        """
        Depth Imbalance L2-L5: Capture hidden pressure beyond best bid/ask.
        
        Imbalance_L25 = (Bid_vol_L2-5 - Ask_vol_L2-5) / (Bid_vol_L2-5 + Ask_vol_L2-5)
        """
        logger.info("Computing Depth Imbalance L2-L5...")
        
        # Sum volumes from L2 to L5
        bid_vol_L25 = orderbook[['bid_sz_2', 'bid_sz_3', 'bid_sz_4', 'bid_sz_5']].sum(axis=1)
        ask_vol_L25 = orderbook[['ask_sz_2', 'ask_sz_3', 'ask_sz_4', 'ask_sz_5']].sum(axis=1)
        
        # Imbalance
        total_vol = bid_vol_L25 + ask_vol_L25
        imbalance = (bid_vol_L25 - ask_vol_L25) / (total_vol + 1e-10)
        
        return imbalance
    
    def compute_roll_impact(self, trades: pd.DataFrame, orderbook: pd.DataFrame,
                           window_ms: int = 5000) -> pd.Series:
        """
        Roll Impact: Measure of short-term price reversal (microstructure noise).
        
        Roll = -Cov(ΔP_t, ΔP_{t-1})
        
        Positive Roll → Mean reversion (liquidity providing profitable)
        """
        logger.info(f"Computing Roll Impact (window={window_ms}ms)...")
        
        # Use midprice from orderbook
        df = orderbook[['timestamp', 'midprice']].copy()
        
        # Price changes
        df['dp'] = df['midprice'].diff()
        df['dp_lag'] = df['dp'].shift(1)
        
        # Rolling covariance
        window_size = int(window_ms / 100)  # 100ms sampling
        df['roll'] = -df['dp'].rolling(window_size).cov(df['dp_lag'])
        
        return df['roll'].fillna(0)
    
    def compute_quote_stuffing(self, orderbook: pd.DataFrame,
                               window_ms: int = 1000) -> pd.Series:
        """
        Quote Stuffing Detector: Identify HFT manipulation.
        
        Stuffing = (N_updates_window > P99(N_updates_baseline))
        
        Binary indicator of abnormal orderbook activity.
        """
        logger.info(f"Computing Quote Stuffing (window={window_ms}ms)...")
        
        # Count orderbook updates per window
        orderbook['timestamp_round'] = orderbook['timestamp'].dt.floor(f'{window_ms}ms')
        update_counts = orderbook.groupby('timestamp_round').size().reset_index(name='n_updates')
        update_counts = update_counts.rename(columns={'timestamp_round': 'timestamp'})
        
        # Compute P99 baseline (1 hour window)
        baseline_window = int(3600000 / window_ms)
        update_counts['p99'] = update_counts['n_updates'].rolling(
            baseline_window, min_periods=10
        ).quantile(0.99)
        
        # Stuffing indicator
        update_counts['stuffing'] = (
            update_counts['n_updates'] > update_counts['p99']
        ).astype(int)
        
        # Merge with orderbook
        result = pd.merge_asof(
            orderbook[['timestamp']].sort_values('timestamp'),
            update_counts[['timestamp', 'stuffing']].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        return result['stuffing'].fillna(0)
    
    def compute_all_features(self, date: str = None) -> pd.DataFrame:
        """Compute all advanced features."""
        logger.info(f"Computing advanced features for {self.symbol}...")
        
        # Load data
        orderbook = self.load_orderbook(date)
        trades = self.load_trades(date)
        
        logger.info(f"Orderbook: {len(orderbook):,} rows")
        logger.info(f"Trades: {len(trades):,} rows")
        
        # Base features
        orderbook['midprice'] = (orderbook['bid_px_1'] + orderbook['ask_px_1']) / 2
        orderbook['spread'] = orderbook['ask_px_1'] - orderbook['bid_px_1']
        
        # Advanced features
        features_df = orderbook[['timestamp', 'midprice', 'spread']].copy()
        
        # 1. Kyle's Lambda (multiple windows)
        features_df['KyleLambda_500ms'] = self.compute_kyles_lambda(
            trades, orderbook, window_ms=500
        )
        features_df['KyleLambda_1s'] = self.compute_kyles_lambda(
            trades, orderbook, window_ms=1000
        )
        features_df['KyleLambda_5s'] = self.compute_kyles_lambda(
            trades, orderbook, window_ms=5000
        )
        
        # 2. Trade Intensity
        features_df['TradeIntensity_500ms'] = self.compute_trade_intensity(
            trades, orderbook, window_ms=500
        )
        features_df['TradeIntensity_1s'] = self.compute_trade_intensity(
            trades, orderbook, window_ms=1000
        )
        
        # 3. Depth Imbalance L2-L5
        features_df['DepthImb_L2L5'] = self.compute_depth_imbalance_L2L5(orderbook)
        
        # 4. Roll Impact
        features_df['RollImpact_1s'] = self.compute_roll_impact(
            trades, orderbook, window_ms=1000
        )
        features_df['RollImpact_5s'] = self.compute_roll_impact(
            trades, orderbook, window_ms=5000
        )
        
        # 5. Quote Stuffing
        features_df['QuoteStuffing_500ms'] = self.compute_quote_stuffing(
            orderbook, window_ms=500
        )
        features_df['QuoteStuffing_1s'] = self.compute_quote_stuffing(
            orderbook, window_ms=1000
        )
        
        # Derived features
        features_df['KyleLambda_momentum'] = features_df['KyleLambda_1s'].diff()
        features_df['TradeIntensity_spike'] = (
            features_df['TradeIntensity_1s'] > 2.0
        ).astype(int)
        
        # Labels (future returns)
        features_df['return_1s'] = features_df['midprice'].pct_change(10).shift(-10)  # 1s ahead
        features_df['return_3s'] = features_df['midprice'].pct_change(30).shift(-30)  # 3s ahead
        features_df['return_5s'] = features_df['midprice'].pct_change(50).shift(-50)  # 5s ahead
        
        # Binary labels
        features_df['label_1s'] = np.sign(features_df['return_1s'])
        features_df['label_3s'] = np.sign(features_df['return_3s'])
        features_df['label_5s'] = np.sign(features_df['return_5s'])
        
        logger.info(f"Features computed: {len(features_df.columns)} columns")
        
        return features_df


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute advanced features')
    parser.add_argument('--symbol', type=str, required=True,
                       help='Symbol (e.g., BTCUSDT)')
    parser.add_argument('--date', type=str, default=None,
                       help='Date filter (e.g., 2025-11-02)')
    parser.add_argument('--out', type=str, default='data/advanced_features',
                       help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ADVANCED FEATURE COMPUTATION")
    logger.info("="*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date filter: {args.date or 'All dates'}")
    
    # Compute features
    engine = AdvancedFeatureEngine(args.symbol)
    features = engine.compute_all_features(date=args.date)
    
    # Save
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    output_file = out_path / f'{args.symbol}_advanced_features.parquet'
    features.to_parquet(output_file, index=False)
    
    logger.info(f"\n✅ Features saved to: {output_file}")
    logger.info(f"   Total rows: {len(features):,}")
    logger.info(f"   Total features: {len(features.columns)}")
    logger.info(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Summary stats
    logger.info("\n📊 Feature Summary:")
    feature_cols = [c for c in features.columns if c not in [
        'timestamp', 'midprice', 'spread', 'return_1s', 'return_3s', 'return_5s',
        'label_1s', 'label_3s', 'label_5s'
    ]]
    
    for col in feature_cols:
        mean_val = features[col].mean()
        std_val = features[col].std()
        logger.info(f"  {col:30s}: mean={mean_val:10.6f}, std={std_val:10.6f}")


if __name__ == '__main__':
    main()
