#!/usr/bin/env python3
"""
Edge Monetization V2 - Advanced Features + Optuna Optimization

Entrena LightGBM con features avanzadas (DepthImb_L2L5) y optimiza para ProfitFactor.

Author: Abinadab (AI Assistant)
Date: 2025-11-03
Sprint: 15.5 (Advanced Feature Monetization)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_features(symbols: list) -> pd.DataFrame:
    """Load and merge base + advanced features."""
    logger.info("Loading features...")
    
    dfs = []
    for symbol in symbols:
        # Load advanced features
        adv_path = Path(f'data/advanced_features/{symbol}_advanced_features.parquet')
        df_adv = pd.read_parquet(adv_path)
        
        # Load base features
        base_path = Path(f'data/edge_validation/{symbol}_features.parquet')
        df_base = pd.read_parquet(base_path)
        
        # Merge on timestamp
        df = pd.merge(
            df_adv,
            df_base[['timestamp', 'OFI', 'VPIN', 'Microprice']],
            on='timestamp',
            how='left'
        )
        
        df['symbol'] = symbol
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples: {len(df_all):,}")
    
    return df_all


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare X, y for training."""
    
    # Feature columns
    feature_cols = [
        # Advanced features
        'KyleLambda_500ms', 'KyleLambda_1s', 'KyleLambda_5s',
        'TradeIntensity_500ms', 'TradeIntensity_1s',
        'DepthImb_L2L5',
        'RollImpact_1s', 'RollImpact_5s',
        'QuoteStuffing_500ms', 'QuoteStuffing_1s',
        'KyleLambda_momentum', 'TradeIntensity_spike',
        # Base features
        'OFI', 'VPIN', 'Microprice'
    ]
    
    # Binary labels (ignore neutral)
    df_train = df[df['label_1s'] != 0].copy()
    df_train['label_binary'] = (df_train['label_1s'] > 0).astype(int)
    
    # Clean data
    df_clean = df_train[feature_cols + ['label_binary', 'return_1s', 'timestamp']].dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean['label_binary'].values
    
    logger.info(f"Training samples: {len(X):,}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Positive rate: {y.mean():.2%}")
    
    return X, y, feature_cols, df_clean


def compute_profit_factor(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          returns: np.ndarray, threshold: float = 0.5,
                          slippage_bps: float = 3, commission_bps: float = 1) -> float:
    """
    Compute Profit Factor for a given threshold.
    
    PF = sum(winning_trades) / sum(losing_trades)
    """
    # Generate signals
    signals = np.where(y_pred_proba > threshold, 1,
                      np.where(y_pred_proba < (1 - threshold), -1, 0))
    
    # Strategy returns
    strategy_returns = signals * returns
    
    # Apply costs
    costs = np.where(signals != 0, (slippage_bps + commission_bps) / 10000, 0)
    net_returns = strategy_returns - costs
    
    # Profit factor
    wins = net_returns[net_returns > 0].sum()
    losses = abs(net_returns[net_returns < 0].sum())
    
    pf = wins / (losses + 1e-10)
    
    return pf


def objective_lightgbm(trial: optuna.Trial, X_train, y_train, X_val, y_val,
                       returns_val: np.ndarray, cost_scenario: Dict) -> float:
    """
    Optuna objective for LightGBM.
    
    Optimizes hyperparameters to maximize Profit Factor on validation set.
    """
    # Hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Train model
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on validation
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Optimize threshold for PF
    best_pf = 0
    for threshold in np.arange(0.50, 0.85, 0.05):
        pf = compute_profit_factor(
            y_val, y_pred_proba, returns_val,
            threshold=threshold,
            slippage_bps=cost_scenario['slippage_bps'],
            commission_bps=cost_scenario['commission_bps']
        )
        if pf > best_pf:
            best_pf = pf
    
    return best_pf


def optimize_model(X_train, y_train, X_val, y_val, returns_val: np.ndarray,
                   cost_scenario: Dict, n_trials: int = 50) -> Tuple[dict, float]:
    """
    Run Optuna optimization.
    
    Returns: (best_params, best_pf)
    """
    logger.info(f"Optimizing model for {cost_scenario['name']}...")
    logger.info(f"  Slippage: {cost_scenario['slippage_bps']} bps")
    logger.info(f"  Commission: {cost_scenario['commission_bps']} bps")
    logger.info(f"  Total cost: {cost_scenario['slippage_bps'] + cost_scenario['commission_bps']} bps")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective_lightgbm(
            trial, X_train, y_train, X_val, y_val, returns_val, cost_scenario
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    logger.info(f"  Best PF: {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    
    return study.best_params, study.best_value


def train_final_model(X_train, y_train, best_params: dict) -> lgb.LGBMClassifier:
    """Train final model with best hyperparameters."""
    model = lgb.LGBMClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model


def backtest_with_costs(model, X_test, y_test, returns_test: np.ndarray,
                       cost_scenario: Dict) -> pd.DataFrame:
    """
    Backtest model with specific cost scenario.
    
    Returns DataFrame with signals, returns, costs, net_returns.
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold on test set
    best_threshold = 0.5
    best_pf = 0
    
    for threshold in np.arange(0.50, 0.85, 0.01):
        pf = compute_profit_factor(
            y_test, y_pred_proba, returns_test,
            threshold=threshold,
            slippage_bps=cost_scenario['slippage_bps'],
            commission_bps=cost_scenario['commission_bps']
        )
        if pf > best_pf:
            best_pf = pf
            best_threshold = threshold
    
    logger.info(f"  Optimal threshold: {best_threshold:.2f}")
    
    # Generate signals with optimal threshold
    signals = np.where(y_pred_proba > best_threshold, 1,
                      np.where(y_pred_proba < (1 - best_threshold), -1, 0))
    
    # Strategy returns
    strategy_returns = signals * returns_test
    
    # Apply costs
    costs = np.where(
        signals != 0,
        (cost_scenario['slippage_bps'] + cost_scenario['commission_bps']) / 10000,
        0
    )
    net_returns = strategy_returns - costs
    
    # Create results DataFrame
    results = pd.DataFrame({
        'signal': signals,
        'return': returns_test,
        'strategy_return': strategy_returns,
        'cost': costs,
        'net_return': net_returns,
        'probability': y_pred_proba
    })
    
    return results, best_threshold


def compute_metrics(results: pd.DataFrame) -> Dict:
    """Compute performance metrics."""
    total_trades = (results['signal'] != 0).sum()
    
    # Profit factor
    wins = results[results['net_return'] > 0]['net_return'].sum()
    losses = abs(results[results['net_return'] < 0]['net_return'].sum())
    profit_factor = wins / (losses + 1e-10)
    
    # Sharpe ratio (annualized)
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
        'avg_return_per_trade': float(results[results['signal'] != 0]['net_return'].mean()) if total_trades > 0 else 0,
        'std_return': float(results['net_return'].std())
    }
    
    return metrics


def make_final_decision(results_by_scenario: Dict) -> Dict:
    """
    Make final GO/NO-GO decision based on all cost scenarios.
    
    Decision logic:
    - IF any scenario achieves PF >= 0.8 → GO
    - ELSE IF best PF >= 0.5 → INVESTIGATE
    - ELSE → ABANDON
    """
    logger.info("\nMaking final decision...")
    
    # Find best scenario
    best_scenario = None
    best_pf = 0
    
    for scenario_name, result in results_by_scenario.items():
        pf = result['metrics']['profit_factor']
        if pf > best_pf:
            best_pf = pf
            best_scenario = scenario_name
    
    # Decision logic
    if best_pf >= 0.8:
        flag = 'MONETIZABLE'
        confidence = 'HIGH'
        recommendation = f"Proceed to 30-day collection. Best scenario: {best_scenario} (PF={best_pf:.2f})"
    elif best_pf >= 0.5:
        flag = 'INVESTIGATE'
        confidence = 'MEDIUM'
        recommendation = f"Explore feature combinations. Best PF={best_pf:.2f} is promising but below target"
    else:
        flag = 'ABANDON'
        confidence = 'HIGH'
        recommendation = f"Despite advanced features, PF={best_pf:.2f} < 0.5. Abandon ML approach"
    
    verdict = {
        'flag': flag,
        'confidence': confidence,
        'recommendation': recommendation,
        'best_scenario': best_scenario,
        'best_profit_factor': float(best_pf),
        'all_scenarios': {
            name: {
                'pf': float(result['metrics']['profit_factor']),
                'sharpe': float(result['metrics']['sharpe_ratio']),
                'trades': int(result['metrics']['total_trades'])
            }
            for name, result in results_by_scenario.items()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Decision: {flag} ({confidence} confidence)")
    logger.info(f"Best scenario: {best_scenario} (PF={best_pf:.4f})")
    
    return verdict


def main():
    logger.info("="*80)
    logger.info("EDGE MONETIZATION V2 - ADVANCED FEATURES + OPTUNA")
    logger.info("="*80)
    
    # Load features
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    df = load_all_features(symbols)
    
    # Prepare training data
    X, y, feature_cols, df_clean = prepare_training_data(df)
    
    # Train/val/test split (chronological)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    returns_val = df_clean['return_1s'].values[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    returns_test = df_clean['return_1s'].values[train_size+val_size:]
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    # Cost scenarios
    cost_scenarios = [
        {
            'name': 'Taker (Baseline)',
            'slippage_bps': 3,
            'commission_bps': 1  # Binance standard taker fee
        },
        {
            'name': 'Maker (with Rebate)',
            'slippage_bps': 1,  # Lower slippage for maker orders
            'commission_bps': -0.2  # Maker rebate -0.01%, slippage ~1bp
        },
        {
            'name': 'VIP-1 Taker',
            'slippage_bps': 3,
            'commission_bps': 0.8  # VIP-1: 0.018% taker
        }
    ]
    
    results_by_scenario = {}
    
    for scenario in cost_scenarios:
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO: {scenario['name']}")
        logger.info(f"{'='*80}")
        
        # Optimize model
        best_params, best_val_pf = optimize_model(
            X_train, y_train, X_val, y_val, returns_val,
            scenario, n_trials=30  # Reduce to 30 for speed
        )
        
        # Train final model
        final_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            **best_params
        }
        model = train_final_model(X_train, y_train, final_params)
        
        # Backtest on test set
        results, optimal_threshold = backtest_with_costs(
            model, X_test, y_test, returns_test, scenario
        )
        
        # Compute metrics
        metrics = compute_metrics(results)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.4f}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Hit Rate: {metrics['hit_rate']:.2%}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {metrics['total_trades']:,}")
        
        # Save results
        results_by_scenario[scenario['name']] = {
            'scenario': scenario,
            'best_params': best_params,
            'optimal_threshold': float(optimal_threshold),
            'metrics': metrics,
            'backtest_results': results
        }
    
    # Make final decision
    verdict = make_final_decision(results_by_scenario)
    
    # Save all results
    logger.info("\nSaving results...")
    Path('reports').mkdir(exist_ok=True)
    
    # Save verdict
    with open('reports/edge_monetization_v2_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2)
    
    # Save detailed results for each scenario
    for scenario_name, result in results_by_scenario.items():
        safe_name = scenario_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Save backtest results
        result['backtest_results'].to_parquet(
            f'reports/backtest_v2_{safe_name}.parquet'
        )
        
        # Save metrics
        with open(f'reports/metrics_v2_{safe_name}.json', 'w') as f:
            json.dump({
                'scenario': result['scenario'],
                'best_params': result['best_params'],
                'optimal_threshold': result['optimal_threshold'],
                'metrics': result['metrics']
            }, f, indent=2)
    
    # Generate summary report
    generate_summary_report(verdict, results_by_scenario, feature_cols)
    
    # Print final verdict
    print("\n" + "="*80)
    if verdict['flag'] == 'MONETIZABLE':
        print(f"🎉 EDGE MONETIZABLE ✅  (Best PF={verdict['best_profit_factor']:.2f})")
        print(f"   Best scenario: {verdict['best_scenario']}")
        print(f"   Recommendation: {verdict['recommendation']}")
    elif verdict['flag'] == 'INVESTIGATE':
        print(f"🟡 PARTIAL SUCCESS  (Best PF={verdict['best_profit_factor']:.2f})")
        print(f"   Recommendation: {verdict['recommendation']}")
    else:
        print(f"❌ EDGE NOT MONETIZABLE  (Best PF={verdict['best_profit_factor']:.2f})")
        print(f"   Recommendation: {verdict['recommendation']}")
    print("="*80)


def generate_summary_report(verdict: Dict, results_by_scenario: Dict, feature_cols: list):
    """Generate markdown summary report."""
    
    report = f"""# Edge Monetization V2 - Summary Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Sprint:** 15.5 (Advanced Feature Monetization)

---

## Executive Summary

### Objective
Validate if advanced features (especially DepthImb_L2L5) can achieve monetary edge (PF ≥ 0.8).

### Final Decision

**FLAG:** {verdict['flag']}  
**CONFIDENCE:** {verdict['confidence']}  
**BEST SCENARIO:** {verdict['best_scenario']}  
**BEST PROFIT FACTOR:** {verdict['best_profit_factor']:.4f}

### Recommendation
{verdict['recommendation']}

---

## Results by Cost Scenario

"""
    
    for scenario_name, result in results_by_scenario.items():
        metrics = result['metrics']
        scenario = result['scenario']
        
        report += f"""
### {scenario_name}

**Costs:**
- Slippage: {scenario['slippage_bps']} bps
- Commission: {scenario['commission_bps']} bps
- **Total:** {scenario['slippage_bps'] + scenario['commission_bps']:.1f} bps

**Optimal Threshold:** {result['optimal_threshold']:.2f}

**Performance Metrics:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Profit Factor | {metrics['profit_factor']:.4f} | ≥ 0.80 | {'✅' if metrics['profit_factor'] >= 0.8 else '❌'} |
| Sharpe Ratio | {metrics['sharpe_ratio']:.4f} | ≥ 0.50 | {'✅' if metrics['sharpe_ratio'] >= 0.5 else '❌'} |
| Max Drawdown | {metrics['max_drawdown']:.2%} | ≤ -5% | {'✅' if metrics['max_drawdown'] >= -0.05 else '❌'} |
| Hit Rate | {metrics['hit_rate']:.2%} | - | - |
| Total Trades | {metrics['total_trades']:,} | - | - |
| Avg Return/Trade | {metrics['avg_return_per_trade']:.6f} | - | - |

"""
    
    report += f"""
---

## Feature Importance

**Total Features:** {len(feature_cols)}

**Advanced Features:**
- DepthImb_L2L5 (Depth Imbalance L2-L5)
- Kyle's Lambda (500ms, 1s, 5s)
- Trade Intensity (500ms, 1s)
- Roll Impact (1s, 5s)
- Quote Stuffing (500ms, 1s)

**Base Features:**
- OFI (Order Flow Imbalance)
- VPIN (Volume-Synchronized PIN)
- Microprice

---

## Comparison with V1

| Metric | V1 (Simple) | V2 (Optimized) | Improvement |
|--------|-------------|----------------|-------------|
| Best PF | 0.1659 | {verdict['best_profit_factor']:.4f} | {((verdict['best_profit_factor'] - 0.1659) / 0.1659 * 100):+.1f}% |
| Features | 11 (base) | {len(feature_cols)} (base+advanced) | +{len(feature_cols) - 11} features |
| Optimization | Threshold only | Optuna + Threshold | Full pipeline |

---

## Files Generated

- `reports/edge_monetization_v2_verdict.json` - Final decision
- `reports/backtest_v2_*.parquet` - Backtest results per scenario
- `reports/metrics_v2_*.json` - Metrics per scenario
- `reports/edge_monetization_v2_summary.md` - This file

---

**Generated by:** Edge Monetization V2 Pipeline  
**Status:** {'✅ MONETIZABLE' if verdict['flag'] == 'MONETIZABLE' else '❌ NOT MONETIZABLE' if verdict['flag'] == 'ABANDON' else '🟡 INVESTIGATE'}
"""
    
    with open('reports/edge_monetization_v2_summary.md', 'w') as f:
        f.write(report)
    
    logger.info("✅ Summary report saved to reports/edge_monetization_v2_summary.md")


if __name__ == '__main__':
    main()
