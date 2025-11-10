#!/usr/bin/env python3
"""
Walk-Forward Evaluation V2 - Multi-class Ensemble with Stacking

Features:
- Multi-class probability predictions
- 1-bar latency + 3 bps slippage
- Dynamic TP/SL based on ATR
- Trailing stops
- Trade only on strong signals (class 0 or 4)
- Symbol-specific confidence thresholds
- Bucket analysis by hour/volatility/imbalance

Usage:
    python eval_walk_forward_v2.py --symbols BTCUSDT ETHUSDT BNBUSDT
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.utils.metrics import calculate_metrics


# Symbol-specific confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'BTCUSDT': 0.65,  # Lowered from 0.75 for quantile retrain
    'ETHUSDT': 0.60,  # Lowered from 0.70
    'BNBUSDT': 0.55   # Lowered from 0.65
}

# Daily trade caps
MAX_TRADES_PER_DAY = {
    'BTCUSDT': 60,
    'ETHUSDT': 80,
    'BNBUSDT': 80
}


def apply_slippage(price, slippage_bps, is_long=True):
    """Apply slippage."""
    slippage_factor = slippage_bps / 10000.0
    if is_long:
        return price * (1 + slippage_factor)
    else:
        return price * (1 - slippage_factor)


def simulate_trade_with_trailing(df, entry_idx, direction, entry_price, atr,
                                   tp_mult, sl_mult, trailing=True):
    """Simulate trade with TP/SL and trailing stop."""
    is_long = (direction == 1)
    
    tp_price = entry_price + (direction * tp_mult * atr)
    sl_price = entry_price - (direction * sl_mult * atr)
    current_sl = sl_price
    
    max_bars = min(200, len(df) - entry_idx - 1)
    
    for i in range(1, max_bars + 1):
        bar_idx = entry_idx + i
        if bar_idx >= len(df):
            break
        
        bar = df.iloc[bar_idx]
        
        # Update trailing stop
        if trailing and is_long:
            potential_sl = bar['close'] - (sl_mult * bar['ATR_14'])
            current_sl = max(current_sl, potential_sl)
        elif trailing and not is_long:
            potential_sl = bar['close'] + (sl_mult * bar['ATR_14'])
            current_sl = min(current_sl, potential_sl)
        
        # Check TP/SL
        if is_long:
            if bar['high'] >= tp_price:
                return {'exit_price': tp_price, 'exit_bar': bar_idx, 
                        'bars_held': i, 'exit_type': 'TP'}
            elif bar['low'] <= current_sl:
                return {'exit_price': current_sl, 'exit_bar': bar_idx,
                        'bars_held': i, 'exit_type': 'SL'}
        else:
            if bar['low'] <= tp_price:
                return {'exit_price': tp_price, 'exit_bar': bar_idx,
                        'bars_held': i, 'exit_type': 'TP'}
            elif bar['high'] >= current_sl:
                return {'exit_price': current_sl, 'exit_bar': bar_idx,
                        'bars_held': i, 'exit_type': 'SL'}
    
    exit_bar = min(entry_idx + max_bars, len(df) - 1)
    return {'exit_price': df.iloc[exit_bar]['close'], 'exit_bar': exit_bar,
            'bars_held': max_bars, 'exit_type': 'TIMEOUT'}


def get_atr_multipliers(atr, price):
    """Get TP/SL multipliers based on volatility."""
    atr_pct = (atr / price) * 100
    
    if atr_pct < 0.5:
        return 8.0, 3.0
    elif atr_pct < 1.0:
        return 6.0, 2.5
    else:
        return 4.5, 2.0


def predict_with_stacking(base_models, meta_learner, X):
    """
    Make predictions with stacked ensemble.
    
    Args:
        base_models: dict with xgb, lgbm, catboost
        meta_learner: calibrated meta-learner
        X: features array
        
    Returns:
        probabilities array
    """
    # Get base predictions
    xgb_pred = base_models['xgb'].predict_proba(X)
    lgbm_pred = base_models['lgbm'].predict_proba(X)
    catboost_pred = base_models['catboost'].predict_proba(X)
    
    # Stack
    meta_features = np.hstack([xgb_pred, lgbm_pred, catboost_pred])
    
    # Meta prediction
    probas = meta_learner.predict_proba(meta_features)
    
    return probas


def evaluate_window_v2(symbol, df, window_start, window_end, ensemble, features,
                        latency_bars=1, slippage_bps=3, confidence_threshold=0.70):
    """
    Evaluate window with multi-class ensemble.
    
    Returns:
        dict: metrics and trades
    """
    test_df = df.iloc[window_start:window_end].copy()
    X_test = test_df[features].values
    
    # Get predictions from stacked ensemble
    probas = predict_with_stacking(
        ensemble['base_models'],
        ensemble['meta_learner'],
        X_test
    )
    
    # Classes: 0=strong_down, 1=weak_down, 2=neutral, 3=weak_up, 4=strong_up
    predicted_classes = probas.argmax(axis=1)
    max_probas = probas.max(axis=1)
    
    # Daily trade tracking
    trades_today = {}
    max_daily = MAX_TRADES_PER_DAY.get(symbol, 80)
    
    trades = []
    
    for i in range(len(test_df) - latency_bars - 200):
        signal_idx = window_start + i
        signal_row = df.iloc[signal_idx]
        
        pred_class = predicted_classes[i]
        confidence = max_probas[i]
        
        # Trade only on strong signals (class 0 or 4)
        if pred_class not in [0, 4]:
            continue
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            continue
        
        # Direction: class 0 = short, class 4 = long
        direction = 1 if pred_class == 4 else -1
        
        # Daily trade cap
        signal_date = signal_row['timestamp'].date() if hasattr(signal_row['timestamp'], 'date') else signal_row['timestamp']
        if signal_date not in trades_today:
            trades_today[signal_date] = 0
        
        if trades_today[signal_date] >= max_daily:
            continue
        
        # Entry at next bar
        entry_idx = signal_idx + latency_bars
        if entry_idx >= len(df):
            continue
        
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['open']
        entry_price = apply_slippage(entry_price, slippage_bps, is_long=(direction == 1))
        
        # ATR and TP/SL
        atr = entry_row['ATR_14']
        tp_mult, sl_mult = get_atr_multipliers(atr, entry_price)
        
        # Simulate trade
        result = simulate_trade_with_trailing(
            df, entry_idx, direction, entry_price, atr, tp_mult, sl_mult, trailing=True
        )
        
        exit_price = result['exit_price']
        exit_price = apply_slippage(exit_price, slippage_bps, is_long=(direction == -1))
        
        # PnL
        position_size = 10000 / entry_price
        pnl = (exit_price - entry_price) * position_size * direction
        
        trades.append({
            'symbol': symbol,
            'signal_bar': signal_idx,
            'entry_bar': entry_idx,
            'exit_bar': result['exit_bar'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'pnl': pnl,
            'bars_held': result['bars_held'],
            'exit_type': result['exit_type'],
            'atr': atr,
            'tp_mult': tp_mult,
            'sl_mult': sl_mult,
            'confidence': confidence,
            'pred_class': int(pred_class),
            'hour': signal_row['timestamp'].hour if hasattr(signal_row['timestamp'], 'hour') else 0
        })
        
        trades_today[signal_date] += 1
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'expectancy': 0.0,
            'total_pnl': 0.0
        }, []
    
    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(trades_df, initial_equity=10000)
    
    return metrics, trades


def calculate_bucket_metrics(trades_df):
    """
    Calculate metrics by buckets (hour, volatility, etc.).
    
    Returns:
        dict: Bucket analysis
    """
    if len(trades_df) == 0:
        return {}
    
    buckets = {}
    
    # Hour buckets
    if 'hour' in trades_df.columns:
        hour_groups = trades_df.groupby('hour')['pnl']
        buckets['by_hour'] = {
            int(h): {
                'trades': len(g),
                'avg_pnl': g.mean(),
                'total_pnl': g.sum(),
                'win_rate': (g > 0).mean()
            }
            for h, g in hour_groups
        }
    
    # ATR terciles (volatility)
    if 'atr' in trades_df.columns:
        trades_df['atr_tercile'] = pd.qcut(trades_df['atr'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
        atr_groups = trades_df.groupby('atr_tercile')['pnl']
        buckets['by_volatility'] = {
            str(t): {
                'trades': len(g),
                'avg_pnl': g.mean(),
                'total_pnl': g.sum(),
                'win_rate': (g > 0).mean()
            }
            for t, g in atr_groups
        }
    
    # Direction
    dir_groups = trades_df.groupby('direction')['pnl']
    buckets['by_direction'] = {
        d: {
            'trades': len(g),
            'avg_pnl': g.mean(),
            'total_pnl': g.sum(),
            'win_rate': (g > 0).mean()
        }
        for d, g in dir_groups
    }
    
    return buckets


def evaluate_symbol_v2(symbol, data_dir, models_dir, latency_bars=1, slippage_bps=3):
    """Evaluate all windows for a symbol."""
    print(f"\n{'='*60}")
    print(f"Evaluating {symbol} V2")
    print(f"{'='*60}")
    
    # Load dataset
    dataset_files = list(data_dir.glob(f"{symbol}_v2_*.parquet"))
    dataset_file = [f for f in dataset_files if 'quantile' in f.name]
    if not dataset_file:
        dataset_file = dataset_files
    dataset_file = dataset_file[0]
    
    df = pd.read_parquet(dataset_file)
    
    # Load splits
    splits_path = data_dir / f"{symbol}_v2_splits.csv"
    splits_df = pd.read_csv(splits_path)
    
    # Confidence threshold
    confidence_threshold = CONFIDENCE_THRESHOLDS.get(symbol, 0.70)
    
    print(f"Dataset: {len(df):,} rows")
    print(f"Windows: {len(splits_df)}")
    print(f"Confidence: {confidence_threshold}")
    
    all_trades = []
    window_metrics = []
    
    for idx, row in splits_df.iterrows():
        window_id = row['window']
        test_start = row['test_start_idx']
        test_end = row['test_end_idx']
        
        # Load model
        model_path = models_dir / symbol / f"ensemble_v2_window_{window_id}.pkl"
        if not model_path.exists():
            print(f"   Window {window_id}: Model not found")
            continue
        
        ensemble = joblib.load(model_path)
        features = ensemble['features']
        
        print(f"\n   Window {window_id}:")
        
        # Evaluate
        metrics, trades = evaluate_window_v2(
            symbol, df, test_start, test_end, ensemble, features,
            latency_bars, slippage_bps, confidence_threshold
        )
        
        metrics['window'] = window_id
        window_metrics.append(metrics)
        all_trades.extend(trades)
        
        print(f"      Trades: {metrics['total_trades']}, "
              f"WinRate: {metrics['win_rate']*100:.1f}%, "
              f"PF: {metrics['profit_factor']:.2f}")
    
    # Combined metrics
    if len(all_trades) == 0:
        print(f"\n   WARNING: No trades for {symbol}")
        return None
    
    all_trades_df = pd.DataFrame(all_trades)
    combined_metrics = calculate_metrics(all_trades_df, initial_equity=10000)
    
    # Bucket analysis
    buckets = calculate_bucket_metrics(all_trades_df)
    
    print(f"\n{'='*60}")
    print(f"COMBINED METRICS - {symbol}")
    print(f"{'='*60}")
    print(f"Total Trades: {combined_metrics['total_trades']:,}")
    print(f"Win Rate: {combined_metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {combined_metrics['profit_factor']:.2f}")
    print(f"Sharpe: {combined_metrics['sharpe']:.2f}")
    print(f"Expectancy: ${combined_metrics['expectancy']:.2f}")
    print(f"Total PnL: ${combined_metrics['total_pnl']:.2f}")
    
    return {
        'symbol': symbol,
        'combined_metrics': combined_metrics,
        'window_metrics': window_metrics,
        'buckets': buckets,
        'trades': all_trades
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ensemble V2')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    parser.add_argument('--latency', type=str, default='1bar')
    parser.add_argument('--slippage', type=str, default='3bps')
    
    args = parser.parse_args()
    
    latency_bars = int(args.latency.replace('bar', ''))
    slippage_bps = float(args.slippage.replace('bps', ''))
    
    # Directories
    data_dir = Path(__file__).parent.parent.parent / "data" / "datasets_v2"
    models_dir = Path(__file__).parent.parent.parent / "models" / "ensemble_v2"
    output_dir = Path(__file__).parent.parent.parent
    
    print(f"\n{'#'*60}")
    print(f"# Walk-Forward Evaluation V2")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Latency: {latency_bars} bar")
    print(f"Slippage: {slippage_bps} bps")
    
    # Evaluate each symbol
    all_results = []
    
    for symbol in args.symbols:
        try:
            result = evaluate_symbol_v2(symbol, data_dir, models_dir, latency_bars, slippage_bps)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR: {symbol} - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if len(all_results) > 0:
        # QA metrics
        qa_metrics = {r['symbol']: r['combined_metrics'] for r in all_results}
        qa_path = output_dir / "qa_metrics_wf_v2.json"
        with open(qa_path, 'w') as f:
            json.dump(qa_metrics, f, indent=2, default=str)
        
        # Buckets
        buckets_all = {r['symbol']: r['buckets'] for r in all_results}
        buckets_path = output_dir / "buckets_wf_v2.json"
        with open(buckets_path, 'w') as f:
            json.dump(buckets_all, f, indent=2, default=str)
        
        # PnL log
        all_trades = []
        for r in all_results:
            all_trades.extend(r['trades'])
        
        pnl_df = pd.DataFrame(all_trades)
        pnl_path = output_dir / "pnl_log_wf_v2.txt"
        pnl_df.to_csv(pnl_path, index=False, sep='\t')
        
        print(f"\n{'='*60}")
        print("Evaluation V2 complete!")
        print(f"{'='*60}")
        print(f"QA Metrics: {qa_path}")
        print(f"Buckets: {buckets_path}")
        print(f"PnL Log: {pnl_path}")
        print(f"Total trades: {len(all_trades):,}")


if __name__ == "__main__":
    main()
