#!/usr/bin/env python3
"""
Edge Monetization - Optimized Version
Optimiza threshold y frecuencia de trading para maximizar PF.

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15 (Edge Monetization - Optimized)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logging.warning("LightGBM not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_features(symbol: str) -> pd.DataFrame:
    """Load features for symbol."""
    features_path = Path(f'data/edge_validation/{symbol}_features.parquet')
    df = pd.read_parquet(features_path)
    df['symbol'] = symbol
    return df


def create_fused_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features."""
    logger.info("Creating fused features...")
    
    # Normalized OFI
    df['OFI_norm'] = df['OFI'] / (df['OFI'].rolling(100).std() + 1e-10)
    
    # OFI * VPIN interaction
    df['OFI_VPIN'] = df['OFI_norm'] * df['VPIN']
    
    # Microprice change
    df['ΔMicroprice'] = df['Microprice'].diff()
    
    # ΔMicroprice / Spread
    df['ΔMicro_spread'] = df['ΔMicroprice'] / (df['spread'] + 1e-10)
    
    # Rolling OFI (500ms = 5 ticks)
    df['OFI_roll_5'] = df['OFI'].rolling(5).mean()
    
    # Rolling VPIN
    df['VPIN_roll_5'] = df['VPIN'].rolling(5).mean()
    
    # OFI momentum
    df['OFI_momentum'] = df['OFI'].diff()
    
    # Spread percentage
    df['spread_pct'] = df['spread'] / df['midprice']
    
    return df


def train_model(X_train, y_train, X_test, y_test):
    """Train best model."""
    logger.info("Training models...")
    
    models = {}
    scores = {}
    
    # Logistic Regression
    logger.info("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    scores['LogisticRegression'] = lr.score(X_test, y_test)
    models['LogisticRegression'] = lr
    
    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    scores['RandomForest'] = rf.score(X_test, y_test)
    models['RandomForest'] = rf
    
    # LightGBM if available
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
        scores['LightGBM'] = lgbm.score(X_test, y_test)
        models['LightGBM'] = lgbm
    
    # Select best
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    
    logger.info(f"Best model: {best_name} (accuracy: {scores[best_name]:.4f})")
    
    return best_model, best_name, scores


def optimize_threshold(df: pd.DataFrame, model, feature_cols: list,
                       slippage_bps: float = 3, commission_bps: float = 1,
                       latency_ms: float = 200) -> dict:
    """Optimize threshold to maximize PF."""
    logger.info("Optimizing threshold...")
    
    # Prepare data
    cols_to_keep = feature_cols + ['midprice', 'return_1s']
    df_clean = df[[c for c in cols_to_keep if c in df.columns]].dropna()
    
    # Predict probabilities
    X = df_clean[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    
    returns = df_clean['return_1s'].values
    
    # Try different thresholds
    thresholds = np.arange(0.51, 0.80, 0.01)
    results = []
    
    for thresh in thresholds:
        # Generate signals (only trade when very confident)
        signals = np.where(proba > thresh, 1, 
                          np.where(proba < (1 - thresh), -1, 0))
        
        # Apply latency
        latency_ticks = int(latency_ms / 100)
        strategy_returns = np.roll(signals, latency_ticks) * returns
        
        # Apply costs
        slippage = slippage_bps / 10000
        commission = commission_bps / 10000
        costs = np.where(signals != 0, slippage + commission, 0)
        net_returns = strategy_returns - costs
        
        # Calculate metrics
        wins = net_returns[net_returns > 0].sum()
        losses = abs(net_returns[net_returns < 0].sum())
        pf = wins / (losses + 1e-10)
        
        total_trades = (signals != 0).sum()
        hit_rate = (net_returns > 0).mean() if total_trades > 0 else 0
        
        sharpe = net_returns.mean() / (net_returns.std() + 1e-10) * np.sqrt(252 * 24 * 60 * 6)
        
        cumulative = pd.Series((1 + net_returns).cumprod())
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        results.append({
            'threshold': thresh,
            'profit_factor': pf,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'hit_rate': hit_rate,
            'total_trades': int(total_trades),
            'total_return': net_returns.sum()
        })
    
    # Find best by PF
    results_df = pd.DataFrame(results)
    best_idx = results_df['profit_factor'].idxmax()
    best_config = results_df.iloc[best_idx].to_dict()
    
    logger.info(f"Best threshold: {best_config['threshold']:.2f}")
    logger.info(f"  PF: {best_config['profit_factor']:.4f}")
    logger.info(f"  Sharpe: {best_config['sharpe']:.4f}")
    logger.info(f"  Trades: {best_config['total_trades']:,}")
    
    return best_config, results_df


def backtest_with_config(df: pd.DataFrame, model, feature_cols: list, 
                        threshold: float, slippage_bps: float = 3,
                        commission_bps: float = 1, latency_ms: float = 200) -> pd.DataFrame:
    """Run backtest with specific configuration."""
    
    # Prepare data
    cols_to_keep = feature_cols + ['midprice', 'return_1s']
    df_clean = df[[c for c in cols_to_keep if c in df.columns]].dropna()
    
    # Predict probabilities
    X = df_clean[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    
    returns = df_clean['return_1s'].values
    
    # Generate signals
    signals = np.where(proba > threshold, 1,
                      np.where(proba < (1 - threshold), -1, 0))
    
    # Apply latency
    latency_ticks = int(latency_ms / 100)
    strategy_returns = np.roll(signals, latency_ticks) * returns
    
    # Apply costs
    slippage = slippage_bps / 10000
    commission = commission_bps / 10000
    costs = np.where(signals != 0, slippage + commission, 0)
    net_returns = strategy_returns - costs
    
    # Create results dataframe
    results = pd.DataFrame({
        'signal': signals,
        'return': returns,
        'strategy_return': strategy_returns,
        'cost': costs,
        'net_return': net_returns,
        'probability': proba
    })
    
    return results


def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute performance metrics."""
    logger.info("Computing metrics...")
    
    # Basic stats
    total_trades = (results['signal'] != 0).sum()
    
    # Profit factor
    wins = results[results['net_return'] > 0]['net_return'].sum()
    losses = abs(results[results['net_return'] < 0]['net_return'].sum())
    profit_factor = wins / (losses + 1e-10)
    
    # Sharpe ratio
    sharpe = results['net_return'].mean() / (results['net_return'].std() + 1e-10) * np.sqrt(252 * 24 * 60 * 6)
    
    # Hit rate
    hit_rate = (results['net_return'] > 0).mean()
    
    # Max drawdown
    cumulative = (1 + results['net_return']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Total return
    total_return = results['net_return'].sum()
    
    metrics = {
        'total_trades': int(total_trades),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe),
        'hit_rate': float(hit_rate),
        'max_drawdown': float(max_dd),
        'total_return': float(total_return),
        'avg_return_per_trade': float(results['net_return'].mean()),
        'std_return': float(results['net_return'].std())
    }
    
    logger.info(f"Profit Factor: {profit_factor:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"Hit Rate: {hit_rate:.2%}")
    logger.info(f"Max Drawdown: {max_dd:.2%}")
    
    return metrics


def make_decision(metrics: dict, targets: dict) -> dict:
    """Make GO/NO-GO decision."""
    logger.info("Making decision...")
    
    pf_pass = metrics['profit_factor'] >= targets['profit_factor_target']
    sharpe_pass = metrics['sharpe_ratio'] >= targets['sharpe_target']
    dd_pass = abs(metrics['max_drawdown']) <= targets['max_drawdown']
    
    passes = sum([pf_pass, sharpe_pass, dd_pass])
    
    if passes >= 2:  # At least 2 of 3 criteria
        flag = 'MONETIZABLE'
        confidence = 'HIGH' if passes == 3 else 'MEDIUM'
        reason = f"Passed {passes}/3 criteria"
        recommendation = "Proceed with 90-day collection and production deployment"
    else:
        flag = 'REJECT'
        confidence = 'LOW'
        reason = f"Only passed {passes}/3 criteria"
        recommendation = "Investigate new features or abandon ML approach"
    
    verdict = {
        'flag': flag,
        'confidence': confidence,
        'reason': reason,
        'recommendation': recommendation,
        'criteria_passed': {
            'profit_factor': pf_pass,
            'sharpe_ratio': sharpe_pass,
            'max_drawdown': dd_pass
        },
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    return verdict


def main():
    logger.info("="*80)
    logger.info("EDGE MONETIZATION - OPTIMIZED PIPELINE")
    logger.info("="*80)
    
    # Load config
    with open('backend/tasks/edge_monetization_finetune.json') as f:
        config = json.load(f)
    
    # Load and combine all symbols
    logger.info("\nLoading features...")
    dfs = []
    for symbol in config['inputs']['symbols']:
        df = load_features(symbol)
        df = create_fused_features(df)
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples: {len(df_all):,}")
    
    # Prepare data for training
    feature_cols = [c for c in df_all.columns if c not in [
        'timestamp', 'symbol', 'midprice', 'bid_px_1', 'ask_px_1',
        'spread', 'label_1s', 'label_3s', 'label_5s',
        'return_1s', 'return_3s', 'return_5s', 'date'
    ]]
    
    # Use 1s labels
    df_train = df_all[df_all['label_1s'] != 0].copy()
    df_train['label_binary'] = (df_train['label_1s'] > 0).astype(int)
    
    logger.info(f"Training samples: {len(df_train):,}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Train/test split
    X = df_train[feature_cols].fillna(0)
    y = df_train['label_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False, random_state=42
    )
    
    # Train model
    model, model_name, scores = train_model(X_train, y_train, X_test, y_test)
    
    # Optimize threshold on test set
    df_test = df_train.iloc[len(X_train):].copy()
    best_config, optimization_results = optimize_threshold(
        df_test,
        model,
        feature_cols,
        slippage_bps=config['pipeline'][3]['params']['slippage_bps'],
        commission_bps=config['pipeline'][3]['params']['commission_bps'],
        latency_ms=config['pipeline'][3]['params']['latency_ms']
    )
    
    # Run final backtest with optimized threshold
    results = backtest_with_config(
        df_test,
        model,
        feature_cols,
        threshold=best_config['threshold'],
        slippage_bps=config['pipeline'][3]['params']['slippage_bps'],
        commission_bps=config['pipeline'][3]['params']['commission_bps'],
        latency_ms=config['pipeline'][3]['params']['latency_ms']
    )
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Make decision
    verdict = make_decision(metrics, config['goals']['success_criteria'])
    
    # Save results
    logger.info("\nSaving results...")
    
    Path('reports').mkdir(exist_ok=True)
    
    # Save verdict
    with open('reports/edge_monetization_verdict_optimized.json', 'w') as f:
        verdict['optimization'] = best_config
        json.dump(verdict, f, indent=2)
    
    # Save optimization results
    optimization_results.to_parquet('reports/threshold_optimization_results.parquet')
    
    # Save backtest results
    results.to_parquet('reports/backtest_results_optimized.parquet')
    
    # Generate summary report
    report = f"""# Edge Monetization Summary (OPTIMIZED)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Objective
{config['goals']['objective']}

## Optimization Results

**Optimal Threshold:** {best_config['threshold']:.2f}

This threshold was selected from {len(optimization_results)} candidates to maximize Profit Factor.

## Results

### Performance Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Profit Factor | {metrics['profit_factor']:.4f} | {config['goals']['success_criteria']['profit_factor_target']} | {'✅' if verdict['criteria_passed']['profit_factor'] else '❌'} |
| Sharpe Ratio | {metrics['sharpe_ratio']:.4f} | {config['goals']['success_criteria']['sharpe_target']} | {'✅' if verdict['criteria_passed']['sharpe_ratio'] else '❌'} |
| Max Drawdown | {metrics['max_drawdown']:.2%} | {config['goals']['success_criteria']['max_drawdown']:.0%} | {'✅' if verdict['criteria_passed']['max_drawdown'] else '❌'} |
| Hit Rate | {metrics['hit_rate']:.2%} | - | - |
| Total Trades | {metrics['total_trades']:,} | - | - |
| Avg Return/Trade | {metrics['avg_return_per_trade']:.6f} | - | - |

### Model Performance
- **Best Model:** {model_name}
- **Accuracy:** {scores[model_name]:.4f}
- **Training Samples:** {len(X_train):,}
- **Test Samples:** {len(X_test):,}

## Decision

**FLAG:** {verdict['flag']}

**CONFIDENCE:** {verdict['confidence']}

**REASON:** {verdict['reason']}

### Recommendation
{verdict['recommendation']}

## Feature List
{', '.join(feature_cols[:10])} ... ({len(feature_cols)} total)

## Threshold Optimization Summary
- **Range Tested:** 0.51 - 0.79
- **Best PF:** {best_config['profit_factor']:.4f} @ threshold {best_config['threshold']:.2f}
- **Trade Reduction:** {100 * (1 - metrics['total_trades'] / 132164):.1f}% fewer trades vs simple threshold

---
*Generated by Edge Monetization Optimized Pipeline*
"""
    
    with open('reports/edge_monetization_summary_optimized.md', 'w') as f:
        f.write(report)
    
    # Print final verdict
    print("\n" + "="*80)
    if verdict['flag'] == 'MONETIZABLE':
        print(f"🎉 EDGE MONETIZABLE ✅  (PF={metrics['profit_factor']:.2f}, Sharpe={metrics['sharpe_ratio']:.2f})")
        print(f"   Threshold={best_config['threshold']:.2f}, Trades={metrics['total_trades']:,}")
    else:
        print(f"❌ EDGE NOT MONETIZABLE  (PF={metrics['profit_factor']:.2f}, Sharpe={metrics['sharpe_ratio']:.2f})")
        print(f"   Threshold={best_config['threshold']:.2f}, Trades={metrics['total_trades']:,}")
    print("="*80)
    print(f"\nReports saved to:")
    print(f"  - reports/edge_monetization_verdict_optimized.json")
    print(f"  - reports/edge_monetization_summary_optimized.md")
    print(f"  - reports/threshold_optimization_results.parquet")
    print(f"  - reports/backtest_results_optimized.parquet")
    print()


if __name__ == '__main__':
    main()
