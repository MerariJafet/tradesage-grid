#!/usr/bin/env python3
"""
Edge Validation Framework
Pruebas estadísticas para confirmar Edge antes de colección 90d.

Tests implementados:
1. Predictibilidad: Correlación, Mutual Information
2. Persistencia: Rolling Window, Autocorrelación
3. Monetización: Backtest simple con costos

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15 (Edge Validation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import logging

from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeValidator:
    """Validates trading edge existence through statistical tests."""
    
    def __init__(self, features_df: pd.DataFrame, symbol: str):
        self.df = features_df
        self.symbol = symbol
        self.results = {}
        
    def test_1_predictability(self, horizon: str = '1s') -> Dict:
        """
        Test 1: PREDICTIBILIDAD
        ¿El feature anticipa el movimiento del precio?
        
        Tests:
        - Correlación Pearson/Spearman
        - Mutual Information
        - KS-Test (distribución de returns cuando feature > 0 vs < 0)
        """
        logger.info(f"=== TEST 1: PREDICTIBILIDAD (horizon={horizon}) ===")
        
        results = {}
        label_col = f'label_{horizon}'
        return_col = f'return_{horizon}'
        
        if label_col not in self.df.columns:
            logger.warning(f"Label column {label_col} not found")
            return results
        
        # Clean data
        df_clean = self.df[[label_col, return_col, 'OFI', 'Microprice', 'VPIN']].dropna()
        
        for feature in ['OFI', 'Microprice', 'VPIN']:
            logger.info(f"\nTesting {feature}:")
            
            X = df_clean[feature].values
            y_ret = df_clean[return_col].values
            y_label = df_clean[label_col].values
            
            # 1.1 Correlación
            corr_pearson, p_pearson = pearsonr(X, y_ret)
            corr_spearman, p_spearman = spearmanr(X, y_ret)
            
            logger.info(f"  Pearson correlation: {corr_pearson:.4f} (p={p_pearson:.4f})")
            logger.info(f"  Spearman correlation: {corr_spearman:.4f} (p={p_spearman:.4f})")
            
            # 1.2 Mutual Information
            # Discretize feature and label for MI
            X_discrete = pd.qcut(X, q=10, labels=False, duplicates='drop')
            y_discrete = (y_label + 1).astype(int)  # -1,0,1 -> 0,1,2
            
            mi = mutual_info_score(X_discrete, y_discrete)
            logger.info(f"  Mutual Information: {mi:.4f} bits")
            
            # 1.3 KS-Test (feature > 0 vs feature < 0)
            returns_positive = y_ret[X > 0]
            returns_negative = y_ret[X < 0]
            
            if len(returns_positive) > 0 and len(returns_negative) > 0:
                ks_stat, ks_p = ks_2samp(returns_positive, returns_negative)
                logger.info(f"  KS-Test: stat={ks_stat:.4f}, p={ks_p:.4f}")
            else:
                ks_stat, ks_p = 0, 1
            
            # Store results
            results[feature] = {
                'pearson_corr': corr_pearson,
                'pearson_p': p_pearson,
                'spearman_corr': corr_spearman,
                'spearman_p': p_spearman,
                'mutual_info': mi,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'edge_detected': (
                    abs(corr_pearson) > 0.02 and p_pearson < 0.05
                ) or (
                    mi > 0.01
                ) or (
                    ks_p < 0.05
                )
            }
        
        return results
    
    def test_2_persistence(self, n_blocks: int = 6) -> Dict:
        """
        Test 2: PERSISTENCIA
        ¿El patrón se repite en distintos períodos?
        
        Divide la muestra en n bloques y verifica:
        - Consistencia del signo de correlación
        - Autocorrelación temporal
        """
        logger.info(f"\n=== TEST 2: PERSISTENCIA (blocks={n_blocks}) ===")
        
        results = {}
        
        # Divide en bloques
        block_size = len(self.df) // n_blocks
        
        for feature in ['OFI', 'Microprice', 'VPIN']:
            logger.info(f"\nTesting {feature}:")
            
            correlations = []
            signs = []
            
            for i in range(n_blocks):
                start_idx = i * block_size
                end_idx = start_idx + block_size if i < n_blocks - 1 else len(self.df)
                
                block = self.df.iloc[start_idx:end_idx]
                
                if 'return_1s' in block.columns and feature in block.columns:
                    df_clean = block[[feature, 'return_1s']].dropna()
                    
                    if len(df_clean) > 10:
                        corr, _ = pearsonr(df_clean[feature], df_clean['return_1s'])
                        correlations.append(corr)
                        signs.append(np.sign(corr))
            
            if len(signs) > 0:
                # Consistency: % de bloques con mismo signo que la mayoría
                sign_consistency = np.mean(signs == np.sign(np.sum(signs)))
                
                # Autocorrelación del feature
                autocorr = self.df[feature].autocorr(lag=10) if feature in self.df else 0
                
                logger.info(f"  Correlations: {correlations}")
                logger.info(f"  Sign consistency: {sign_consistency:.2%}")
                logger.info(f"  Autocorrelation (lag=10): {autocorr:.4f}")
                
                results[feature] = {
                    'correlations': correlations,
                    'sign_consistency': sign_consistency,
                    'autocorrelation': autocorr,
                    'persistent': sign_consistency >= 0.7  # 70% de bloques con mismo signo
                }
        
        return results
    
    def test_3_classification(self) -> Dict:
        """
        Test 3: CLASIFICACIÓN
        ¿Podemos predecir dirección del movimiento?
        
        - Logistic Regression
        - Random Forest feature importance
        """
        logger.info("\n=== TEST 3: CLASIFICACIÓN ===")
        
        results = {}
        
        # Prepare data
        df_clean = self.df[['OFI', 'Microprice', 'VPIN', 'label_1s']].dropna()
        
        # Convert label to binary (up/down, ignore flat)
        df_binary = df_clean[df_clean['label_1s'] != 0].copy()
        df_binary['label_binary'] = (df_binary['label_1s'] > 0).astype(int)
        
        if len(df_binary) < 100:
            logger.warning("Not enough data for classification")
            return results
        
        X = df_binary[['OFI', 'Microprice', 'VPIN']].values
        y = df_binary['label_binary'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        # 3.1 Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        auc_lr = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"  Logistic Regression AUC: {auc_lr:.4f}")
        logger.info(f"  Coefficients: OFI={lr.coef_[0][0]:.4f}, "
                   f"Microprice={lr.coef_[0][1]:.4f}, VPIN={lr.coef_[0][2]:.4f}")
        
        # 3.2 Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        rf.fit(X_train, y_train)
        
        y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
        
        feature_importance = dict(zip(['OFI', 'Microprice', 'VPIN'], rf.feature_importances_))
        
        logger.info(f"  Random Forest AUC: {auc_rf:.4f}")
        logger.info(f"  Feature importance: {feature_importance}")
        
        results = {
            'logistic_auc': auc_lr,
            'logistic_coefficients': {
                'OFI': lr.coef_[0][0],
                'Microprice': lr.coef_[0][1],
                'VPIN': lr.coef_[0][2]
            },
            'rf_auc': auc_rf,
            'rf_importance': feature_importance,
            'edge_detected': auc_lr > 0.52 or auc_rf > 0.52
        }
        
        return results
    
    def test_4_monetization(self, commission: float = 0.0001, slippage: float = 0.0003) -> Dict:
        """
        Test 4: MONETIZACIÓN
        ¿El pattern produce ganancias netas?
        
        Simple backtest:
        - Long if OFI > 0, Short if OFI < 0
        - Take-profit/Stop = 0.1%
        - Commission + Slippage
        """
        logger.info("\n=== TEST 4: MONETIZACIÓN ===")
        
        results = {}
        
        df_clean = self.df[['OFI', 'midprice', 'timestamp']].dropna().copy()
        
        if len(df_clean) < 100:
            return results
        
        # Simple strategy: long if OFI > 0
        df_clean['signal'] = np.where(df_clean['OFI'] > 0, 1, -1)
        df_clean['returns'] = df_clean['midprice'].pct_change()
        
        # Strategy returns
        df_clean['strategy_returns'] = df_clean['signal'].shift(1) * df_clean['returns']
        
        # Apply costs
        df_clean['net_returns'] = df_clean['strategy_returns'] - (commission + slippage)
        
        # Calculate metrics
        total_return = df_clean['net_returns'].sum()
        sharpe = df_clean['net_returns'].mean() / (df_clean['net_returns'].std() + 1e-10) * np.sqrt(252 * 24 * 60 * 6)  # annualized
        
        # Profit factor
        wins = df_clean[df_clean['net_returns'] > 0]['net_returns'].sum()
        losses = abs(df_clean[df_clean['net_returns'] < 0]['net_returns'].sum())
        profit_factor = wins / (losses + 1e-10)
        
        # Hit rate
        hit_rate = (df_clean['net_returns'] > 0).mean()
        
        # Max drawdown
        cumulative = (1 + df_clean['net_returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        logger.info(f"  Total Return: {total_return:.4%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
        logger.info(f"  Profit Factor: {profit_factor:.4f}")
        logger.info(f"  Hit Rate: {hit_rate:.2%}")
        logger.info(f"  Max Drawdown: {max_dd:.2%}")
        
        results = {
            'total_return': total_return,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'hit_rate': hit_rate,
            'max_drawdown': max_dd,
            'profitable': profit_factor > 1.05 and sharpe > 0.5
        }
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Execute all edge validation tests."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EDGE VALIDATION: {self.symbol}")
        logger.info(f"Data points: {len(self.df):,}")
        logger.info(f"{'='*70}\n")
        
        self.results['predictability'] = self.test_1_predictability(horizon='1s')
        self.results['persistence'] = self.test_2_persistence(n_blocks=6)
        self.results['classification'] = self.test_3_classification()
        self.results['monetization'] = self.test_4_monetization()
        
        # Overall verdict
        self.results['verdict'] = self._compute_verdict()
        
        return self.results
    
    def _compute_verdict(self) -> Dict:
        """Compute overall edge verdict."""
        logger.info(f"\n{'='*70}")
        logger.info("VEREDICTO FINAL")
        logger.info(f"{'='*70}\n")
        
        # Count edge signals
        pred_edges = sum(
            1 for f in self.results.get('predictability', {}).values()
            if isinstance(f, dict) and f.get('edge_detected', False)
        )
        
        persist_edges = sum(
            1 for f in self.results.get('persistence', {}).values()
            if isinstance(f, dict) and f.get('persistent', False)
        )
        
        class_edge = self.results.get('classification', {}).get('edge_detected', False)
        profit_edge = self.results.get('monetization', {}).get('profitable', False)
        
        # Decision logic
        total_signals = pred_edges + persist_edges + int(class_edge) + int(profit_edge)
        
        if total_signals >= 3:
            decision = "GO"
            confidence = "HIGH"
            reason = f"Edge detectado en {total_signals}/4 categorías"
        elif total_signals == 2:
            decision = "CONDITIONAL GO"
            confidence = "MEDIUM"
            reason = "Edge parcial detectado, requiere validación extendida"
        else:
            decision = "NO-GO"
            confidence = "LOW"
            reason = f"Insuficiente evidencia de edge ({total_signals}/4)"
        
        logger.info(f"Predictability edges: {pred_edges}/3")
        logger.info(f"Persistence edges: {persist_edges}/3")
        logger.info(f"Classification edge: {class_edge}")
        logger.info(f"Monetization edge: {profit_edge}")
        logger.info(f"\nDECISION: {decision}")
        logger.info(f"CONFIDENCE: {confidence}")
        logger.info(f"REASON: {reason}")
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reason': reason,
            'signals': total_signals,
            'pred_edges': pred_edges,
            'persist_edges': persist_edges,
            'class_edge': class_edge,
            'profit_edge': profit_edge
        }


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Validate trading edge')
    parser.add_argument('--features', required=True, help='Features parquet file')
    parser.add_argument('--symbol', required=True, help='Symbol name')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load features
    logger.info(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    
    # Run validation
    validator = EdgeValidator(df, args.symbol)
    results = validator.run_all_tests()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print verdict
    print("\n" + "="*70)
    print("EDGE VALIDATION SUMMARY")
    print("="*70)
    verdict = results['verdict']
    print(f"DECISION: {verdict['decision']}")
    print(f"CONFIDENCE: {verdict['confidence']}")
    print(f"REASON: {verdict['reason']}")
    print("="*70)


if __name__ == '__main__':
    main()
