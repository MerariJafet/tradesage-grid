"""
Step 1: Feature Fusion
Create enhanced features combining OFI, VPIN, Microprice.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureFusion:
    """Creates fused and derived features from base features."""
    
    def __init__(self, data_path: str, symbols: list, base_features: list):
        self.data_path = Path(data_path)
        self.symbols = symbols
        self.base_features = base_features
        
    def create_fused_features(self) -> pd.DataFrame:
        """Create all fused features."""
        logger.info("Creating fused features...")
        
        all_features = []
        
        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")
            
            # Load base features (from edge validation)
            features_path = Path(f'data/edge_validation/{symbol}_features.parquet')
            df = pd.read_parquet(features_path)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Create derived features
            df = self._add_derived_features(df)
            
            all_features.append(df)
        
        # Combine all symbols
        result = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"Total features created: {len(result.columns)}")
        logger.info(f"Total samples: {len(result):,}")
        
        return result
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and fused features."""
        logger.info("  Adding derived features...")
        
        # 1. Normalized OFI
        df['OFI_norm'] = df['OFI'] / (df['OFI'].rolling(100).std() + 1e-10)
        
        # 2. OFI * VPIN interaction
        df['OFI_VPIN_interaction'] = df['OFI_norm'] * df['VPIN']
        
        # 3. Microprice change
        df['ΔMicroprice'] = df['Microprice'].diff()
        
        # 4. ΔMicroprice / Spread (normalized price movement)
        df['ΔMicroprice_spread_ratio'] = df['ΔMicroprice'] / (df['spread'] + 1e-10)
        
        # 5. Rolling OFI (500ms = 5 ticks @ 100ms)
        df['Rolling_OFI_500ms'] = df['OFI'].rolling(5).mean()
        
        # 6. Rolling VPIN (500ms)
        df['Rolling_VPIN_500ms'] = df['VPIN'].rolling(5).mean()
        
        # 7. OFI momentum (change in OFI)
        df['OFI_momentum'] = df['OFI'].diff()
        
        # 8. VPIN momentum
        df['VPIN_momentum'] = df['VPIN'].diff()
        
        # 9. Spread percentage
        df['spread_pct'] = df['spread'] / df['midprice']
        
        # 10. Microprice vs Midprice deviation
        df['micro_mid_deviation'] = (df['Microprice'] - df['midprice']) / df['midprice']
        
        # 11. OFI acceleration (second derivative)
        df['OFI_acceleration'] = df['OFI_momentum'].diff()
        
        # 12. Rolling volatility of returns
        if 'return_1s' in df.columns:
            df['volatility_1s'] = df['return_1s'].rolling(10).std()
        
        logger.info(f"  Added {len([c for c in df.columns if c not in self.base_features])} new features")
        
        return df
