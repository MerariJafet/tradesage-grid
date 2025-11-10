#!/usr/bin/env python3
"""
Edge Probe V2 - Advanced Feature Validation

Valida edge estadístico con features avanzadas.
CRITERIO DE ABANDONO: Si correlation < 0.15 AND AUC < 0.65 → STOP

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15.5 (Advanced Feature Exploration)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_advanced_features(symbol: str) -> pd.DataFrame:
    """Load advanced features for symbol."""
    features_path = Path(f'data/advanced_features/{symbol}_advanced_features.parquet')
    df = pd.read_parquet(features_path)
    df['symbol'] = symbol
    return df


def test_predictability_advanced(df: pd.DataFrame) -> dict:
    """
    Test 1: Predictability with Advanced Features.
    
    Tests: Correlation, Mutual Information, Classification AUC
    Target: correlation > 0.15 OR AUC > 0.65
    """
    logger.info("Test 1: Predictability (Advanced Features)")
    
    results = {}
    
    # Advanced feature columns
    advanced_cols = [
        'KyleLambda_500ms', 'KyleLambda_1s', 'KyleLambda_5s',
        'TradeIntensity_500ms', 'TradeIntensity_1s',
        'DepthImb_L2L5', 'RollImpact_1s', 'RollImpact_5s',
        'QuoteStuffing_500ms', 'QuoteStuffing_1s',
        'KyleLambda_momentum', 'TradeIntensity_spike'
    ]
    
    # Test each feature
    for col in advanced_cols:
        if col not in df.columns:
            continue
            
        df_clean = df[[col, 'return_1s']].dropna()
        
        if len(df_clean) < 100:
            continue
        
        x = df_clean[col].values
        y = df_clean['return_1s'].values
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(x, y)
        
        results[col] = {
            'pearson_corr': float(pearson_corr),
            'pearson_p': float(pearson_p),
            'spearman_corr': float(spearman_corr),
            'spearman_p': float(spearman_p),
            'edge_detected': bool(abs(pearson_corr) > 0.15 or abs(spearman_corr) > 0.20)
        }
        
        logger.info(f"  {col:30s}: Pearson={pearson_corr:7.4f} (p={pearson_p:.2e}), "
                   f"Spearman={spearman_corr:7.4f}")
    
    return results


def test_classification_advanced(df: pd.DataFrame) -> dict:
    """
    Test 2: Classification with Advanced Features.
    
    Models: Logistic Regression, Random Forest, LightGBM
    Target: AUC > 0.65
    """
    logger.info("Test 2: Classification (Advanced Features)")
    
    # Prepare data
    feature_cols = [
        'KyleLambda_500ms', 'KyleLambda_1s', 'KyleLambda_5s',
        'TradeIntensity_500ms', 'TradeIntensity_1s',
        'DepthImb_L2L5', 'RollImpact_1s', 'RollImpact_5s',
        'QuoteStuffing_500ms', 'QuoteStuffing_1s',
        'KyleLambda_momentum', 'TradeIntensity_spike'
    ]
    
    # Only use available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Binary labels (ignore neutral)
    df_class = df[df['label_1s'] != 0].copy()
    df_class['label_binary'] = (df_class['label_1s'] > 0).astype(int)
    
    # Clean data
    df_clean = df_class[feature_cols + ['label_binary']].dropna()
    
    if len(df_clean) < 1000:
        logger.warning("Not enough samples for classification")
        return {}
    
    X = df_clean[feature_cols].values
    y = df_clean['label_binary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    results = {}
    
    # Logistic Regression
    logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    
    results['LogisticRegression'] = {
        'auc': float(lr_auc),
        'coefficients': dict(zip(feature_cols, lr.coef_[0].tolist()))
    }
    logger.info(f"    AUC: {lr_auc:.4f}")
    
    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    results['RandomForest'] = {
        'auc': float(rf_auc),
        'feature_importance': dict(zip(feature_cols, rf.feature_importances_.tolist()))
    }
    logger.info(f"    AUC: {rf_auc:.4f}")
    
    # LightGBM (if available)
    if HAS_LIGHTGBM:
        logger.info("  Training LightGBM...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        lgbm.fit(X_train, y_train)
        lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        lgbm_auc = roc_auc_score(y_test, lgbm_proba)
        
        results['LightGBM'] = {
            'auc': float(lgbm_auc),
            'feature_importance': dict(zip(feature_cols, lgbm.feature_importances_.tolist()))
        }
        logger.info(f"    AUC: {lgbm_auc:.4f}")
    
    # Best AUC
    best_auc = max([r['auc'] for r in results.values() if 'auc' in r])
    results['edge_detected'] = bool(best_auc > 0.65)
    results['best_auc'] = float(best_auc)
    
    return results


def make_decision_v2(predictability: dict, classification: dict) -> dict:
    """
    Make GO/NO-GO decision for advanced features.
    
    CRITERIA:
    - GO: Best correlation > 0.15 OR Best AUC > 0.65
    - STOP: All correlations < 0.15 AND All AUC < 0.65
    """
    logger.info("\nMaking decision...")
    
    # Best correlation
    if predictability:
        best_corr = max([abs(r['pearson_corr']) for r in predictability.values()])
        best_spearman = max([abs(r['spearman_corr']) for r in predictability.values()])
    else:
        best_corr = 0
        best_spearman = 0
    
    # Best AUC
    best_auc = classification.get('best_auc', 0)
    
    # Decision logic
    corr_pass = bool(best_corr > 0.15 or best_spearman > 0.20)
    auc_pass = bool(best_auc > 0.65)
    
    if corr_pass or auc_pass:
        flag = 'GO'
        confidence = 'HIGH' if (corr_pass and auc_pass) else 'MEDIUM'
        reason = []
        if corr_pass:
            reason.append(f"Correlation {best_corr:.4f} > 0.15")
        if auc_pass:
            reason.append(f"AUC {best_auc:.4f} > 0.65")
        reason_str = " AND ".join(reason)
    else:
        flag = 'STOP'
        confidence = 'HIGH'
        reason_str = f"Correlation {best_corr:.4f} < 0.15 AND AUC {best_auc:.4f} < 0.65"
    
    verdict = {
        'decision': flag,
        'confidence': confidence,
        'reason': reason_str,
        'best_correlation': float(best_corr),
        'best_spearman': float(best_spearman),
        'best_auc': float(best_auc),
        'criteria_passed': {
            'correlation': corr_pass,
            'auc': auc_pass
        }
    }
    
    logger.info(f"Decision: {flag} ({confidence} confidence)")
    logger.info(f"Reason: {reason_str}")
    
    return verdict


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Edge Probe V2 - Advanced Features')
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--out', type=str, default='data/advanced_features')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info(f"EDGE PROBE V2 - {args.symbol}")
    logger.info("="*80)
    
    # Load features
    df = load_advanced_features(args.symbol)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Test 1: Predictability
    predictability = test_predictability_advanced(df)
    
    # Test 2: Classification
    classification = test_classification_advanced(df)
    
    # Decision
    verdict = make_decision_v2(predictability, classification)
    
    # Save results
    results = {
        'symbol': args.symbol,
        'predictability': predictability,
        'classification': classification,
        'verdict': verdict,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    output_file = out_path / f'{args.symbol}_edge_results_v2.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    if verdict['decision'] == 'GO':
        print(f"✅ {args.symbol}: EDGE DETECTED - Proceed to monetization")
        print(f"   Best Correlation: {verdict['best_correlation']:.4f}")
        print(f"   Best AUC: {verdict['best_auc']:.4f}")
    else:
        print(f"❌ {args.symbol}: NO EDGE - STOP exploration")
        print(f"   Best Correlation: {verdict['best_correlation']:.4f} (< 0.15)")
        print(f"   Best AUC: {verdict['best_auc']:.4f} (< 0.65)")
    print("="*80)


if __name__ == '__main__':
    main()
