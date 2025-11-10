#!/usr/bin/env python3
"""
Feature Engineering para Edge Validation
Calcula OFI, Microprice, VPIN desde datos WebSocket reales.

Features implementados:
- OFI (Order Flow Imbalance): Presión direccional L1-L5
- Microprice: Precio justo ponderado por liquidez
- VPIN: Toxicidad del flujo de órdenes

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15 (Edge Validation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureComputer:
    """Compute microstructure features from WebSocket data."""
    
    def __init__(self, symbol: str, data_path: Path):
        self.symbol = symbol
        self.data_path = Path(data_path)
        
    def load_orderbook(self) -> pd.DataFrame:
        """Load and prepare orderbook data."""
        logger.info(f"Loading orderbook for {self.symbol}...")
        
        ob_path = self.data_path / self.symbol / 'orderbook'
        df = pd.read_parquet(ob_path)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df):,} orderbook snapshots")
        return df
    
    def load_trades(self) -> pd.DataFrame:
        """Load and prepare trade data."""
        logger.info(f"Loading trades for {self.symbol}...")
        
        tr_path = self.data_path / self.symbol / 'aggtrades'
        df = pd.read_parquet(tr_path)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df):,} trades")
        return df
    
    def compute_ofi(self, df: pd.DataFrame) -> pd.Series:
        """
        Order Flow Imbalance (OFI)
        Mide presión compradora vs vendedora en el libro.
        
        OFI = Σ(Δbid_volume - Δask_volume) for levels 1-5
        
        Positivo = presión compradora
        Negativo = presión vendedora
        """
        logger.info("Computing OFI...")
        
        ofi = pd.Series(0.0, index=df.index)
        
        for level in range(1, 6):
            # Cambios en volumen bid
            bid_vol_col = f'bid_sz_{level}'
            ask_vol_col = f'ask_sz_{level}'
            
            if bid_vol_col in df.columns and ask_vol_col in df.columns:
                delta_bid = df[bid_vol_col].diff()
                delta_ask = df[ask_vol_col].diff()
                
                ofi += (delta_bid - delta_ask)
        
        # Normalize by total volume
        total_vol = sum(df[f'bid_sz_{i}'] + df[f'ask_sz_{i}'] 
                       for i in range(1, 6) if f'bid_sz_{i}' in df.columns)
        
        ofi = ofi / (total_vol + 1e-10)
        
        logger.info(f"OFI computed. Mean: {ofi.mean():.6f}, Std: {ofi.std():.6f}")
        return ofi
    
    def compute_microprice(self, df: pd.DataFrame) -> pd.Series:
        """
        Microprice
        Precio justo ponderado por liquidez bid/ask.
        
        MP = (bid_price * ask_volume + ask_price * bid_volume) / (bid_volume + ask_volume)
        
        Refleja el precio de equilibrio instantáneo.
        """
        logger.info("Computing Microprice...")
        
        bid_px = df['bid_px_1']
        ask_px = df['ask_px_1']
        bid_vol = df['bid_sz_1']
        ask_vol = df['ask_sz_1']
        
        microprice = (bid_px * ask_vol + ask_px * bid_vol) / (bid_vol + ask_vol + 1e-10)
        
        logger.info(f"Microprice computed. Mean: {microprice.mean():.2f}")
        return microprice
    
    def compute_vpin(self, df_ob: pd.DataFrame, df_trades: pd.DataFrame, 
                     window: int = 50) -> pd.Series:
        """
        VPIN (Volume-Synchronized Probability of Informed Trading)
        Mide toxicidad del flujo de órdenes.
        
        VPIN = |buy_volume - sell_volume| / total_volume
        
        Alto VPIN = alta toxicidad = posible movimiento brusco inminente
        """
        logger.info("Computing VPIN...")
        
        # Classify trades as buy or sell based on aggressor side
        df_trades['side_volume'] = np.where(
            df_trades['is_buyer_maker'],
            -df_trades['quantity'],  # Sell aggressor
            df_trades['quantity']     # Buy aggressor
        )
        
        # Resample to orderbook timestamps (100ms grid)
        trades_resampled = df_trades.set_index('timestamp').resample('100ms').agg({
            'side_volume': 'sum',
            'quantity': 'sum'
        }).reindex(df_ob['timestamp'], method='nearest')
        
        # Calculate VPIN
        buy_vol = trades_resampled['side_volume'].clip(lower=0).rolling(window).sum()
        sell_vol = (-trades_resampled['side_volume'].clip(upper=0)).rolling(window).sum()
        total_vol = trades_resampled['quantity'].rolling(window).sum()
        
        vpin = np.abs(buy_vol - sell_vol) / (total_vol + 1e-10)
        vpin = vpin.values
        
        logger.info(f"VPIN computed. Mean: {np.nanmean(vpin):.4f}, Std: {np.nanstd(vpin):.4f}")
        return pd.Series(vpin, index=df_ob.index)
    
    def compute_labels(self, df: pd.DataFrame, horizons: list = [1, 3, 5]) -> pd.DataFrame:
        """
        Compute future returns for different time horizons.
        
        Label = sign(midprice[t+h] - midprice[t])
        
        1 = up
        0 = flat
        -1 = down
        """
        logger.info("Computing labels...")
        
        midprice = (df['bid_px_1'] + df['ask_px_1']) / 2
        
        labels = pd.DataFrame(index=df.index)
        
        for h in horizons:
            # h in seconds, convert to 100ms ticks
            ticks = h * 10
            
            future_mid = midprice.shift(-ticks)
            ret = (future_mid - midprice) / midprice
            
            # Classify
            labels[f'label_{h}s'] = np.sign(ret)
            labels[f'return_{h}s'] = ret
        
        logger.info(f"Labels computed for horizons: {horizons}s")
        return labels
    
    def compute_all_features(self) -> pd.DataFrame:
        """Compute all features and labels."""
        logger.info(f"Starting feature computation for {self.symbol}")
        
        # Load data
        df_ob = self.load_orderbook()
        df_trades = self.load_trades()
        
        # Compute features
        features = pd.DataFrame({
            'timestamp': df_ob['timestamp'],
            'bid_px_1': df_ob['bid_px_1'],
            'ask_px_1': df_ob['ask_px_1'],
            'midprice': (df_ob['bid_px_1'] + df_ob['ask_px_1']) / 2,
            'spread': df_ob['ask_px_1'] - df_ob['bid_px_1'],
            'OFI': self.compute_ofi(df_ob),
            'Microprice': self.compute_microprice(df_ob),
            'VPIN': self.compute_vpin(df_ob, df_trades, window=50)
        })
        
        # Compute labels
        labels = self.compute_labels(df_ob, horizons=[1, 3, 5])
        
        # Merge
        result = pd.concat([features, labels], axis=1)
        
        # Remove NaN rows
        result = result.dropna()
        
        logger.info(f"Feature computation complete. Final rows: {len(result):,}")
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute features for Edge validation')
    parser.add_argument('--symbol', required=True, help='Symbol to process')
    parser.add_argument('--data-path', required=True, help='Path to WebSocket data')
    parser.add_argument('--output', required=True, help='Output parquet file')
    
    args = parser.parse_args()
    
    # Compute features
    computer = FeatureComputer(args.symbol, args.data_path)
    features = computer.compute_all_features()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, compression='snappy')
    
    logger.info(f"Features saved to {output_path}")
    logger.info(f"Shape: {features.shape}")
    logger.info(f"Columns: {list(features.columns)}")
    
    # Quick stats
    print("\n" + "="*70)
    print("FEATURE STATISTICS")
    print("="*70)
    print(features[['OFI', 'Microprice', 'VPIN']].describe())
    print("="*70)


if __name__ == '__main__':
    main()
