#!/usr/bin/env python3
"""
Evaluate ensemble models using walk-forward protocol with realistic simulation.

Features:
- 1-bar latency (signal at bar i, entry at bar i+1)
- 3 bps slippage on entry and exit
- Dynamic TP/SL based on ATR
- Trailing stops
- Symbol-specific confidence thresholds
- Daily trade caps to prevent spam trading

Usage:
    python eval_walk_forward_v1.py --symbols BTCUSDT ETHUSDT BNBUSDT --latency 1bar --slippage 3bps
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


# Symbol-specific thresholds (post-calibration)
CONFIDENCE_THRESHOLDS = {
    'BTCUSDT': 0.78,
    'ETHUSDT': 0.72,
    'BNBUSDT': 0.68
}

# Daily trade caps to prevent spam
MAX_TRADES_PER_DAY = {
    'BTCUSDT': 60,
    'ETHUSDT': 80,
    'BNBUSDT': 80
}


def apply_slippage(price, slippage_bps, is_long=True):
    """Apply slippage to entry/exit price."""
    slippage_factor = slippage_bps / 10000.0
    if is_long:
        return price * (1 + slippage_factor)  # Buy at higher price
    else:
        return price * (1 - slippage_factor)  # Sell at lower price


def simulate_trade_with_trailing(df, entry_idx, direction, entry_price, atr, 
                                   tp_mult, sl_mult, trailing=True):
    """
    Simulate trade with TP/SL and optional trailing stop.
    
    Returns:
        dict: Trade result with exit_price, exit_bar, pnl, exit_type
    """
    is_long = (direction == 1)
    
    # Initial TP/SL
    tp_price = entry_price + (direction * tp_mult * atr)
    sl_price = entry_price - (direction * sl_mult * atr)
    
    current_sl = sl_price
    
    # Scan forward bars
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
                # TP hit
                return {
                    'exit_price': tp_price,
                    'exit_bar': bar_idx,
                    'bars_held': i,
                    'exit_type': 'TP'
                }
            elif bar['low'] <= current_sl:
                # SL hit
                return {
                    'exit_price': current_sl,
                    'exit_bar': bar_idx,
                    'bars_held': i,
                    'exit_type': 'SL'
                }
        else:  # Short
            if bar['low'] <= tp_price:
                # TP hit
                return {
                    'exit_price': tp_price,
                    'exit_bar': bar_idx,
                    'bars_held': i,
                    'exit_type': 'TP'
                }
            elif bar['high'] >= current_sl:
                # SL hit
                return {
                    'exit_price': current_sl,
                    'exit_bar': bar_idx,
                    'bars_held': i,
                    'exit_type': 'SL'
                }
    
    # Timeout - exit at market
    exit_bar = min(entry_idx + max_bars, len(df) - 1)
    return {
        'exit_price': df.iloc[exit_bar]['close'],
        'exit_bar': exit_bar,
        'bars_held': max_bars,
        'exit_type': 'TIMEOUT'
    }


def get_atr_multipliers(atr, price):
    """Get TP/SL multipliers based on volatility regime."""
    atr_pct = (atr / price) * 100
    
    if atr_pct < 0.5:  # Low volatility
        return 8.0, 3.0
    elif atr_pct < 1.0:  # Medium volatility
        return 6.0, 2.5
    else:  # High volatility
        return 4.5, 2.0


def evaluate_window(symbol, df, window_start, window_end, ensemble, features, 
                     latency_bars=1, slippage_bps=3, confidence_threshold=0.7):
    """
    Evaluate ensemble on test window with realistic simulation.
    
    Args:
        symbol: str
        df: pd.DataFrame, full dataset
        window_start: int, test window start index
        window_end: int, test window end index
        ensemble: trained ensemble model
        features: list of feature names
        latency_bars: int, signal-to-entry delay
        slippage_bps: float, slippage in basis points
        confidence_threshold: float, minimum probability to trade
        
    Returns:
        dict: Evaluation metrics and trade log
    """
    test_df = df.iloc[window_start:window_end].copy()
    X_test = test_df[features].values
    
    # Get predictions from ensemble (dictionary of models)
    if isinstance(ensemble, dict):
        # Manual soft voting
        probas_xgb = ensemble['xgb'].predict_proba(X_test)[:, 1]
        probas_lgbm = ensemble['lgbm'].predict_proba(X_test)[:, 1]
        probas_catboost = ensemble['catboost'].predict_proba(X_test)[:, 1]
        probas = (probas_xgb + probas_lgbm + probas_catboost) / 3
    else:
        # Standard VotingClassifier (shouldn't happen with new code)
        probas = ensemble.predict_proba(X_test)[:, 1]
    
    test_df['proba'] = probas
    
    # Daily trade tracking
    trades_today = {}
    max_daily = MAX_TRADES_PER_DAY.get(symbol, 80)
    
    trades = []
    
    for i in range(len(test_df) - latency_bars - 200):  # Reserve bars for trade execution
        signal_idx = window_start + i
        signal_row = df.iloc[signal_idx]
        
        proba = probas[i]
        
        # Check confidence threshold
        if proba < confidence_threshold:
            continue
        
        # Direction based on probability
        direction = 1 if proba > 0.5 else -1
        
        # Daily trade cap check
        signal_date = signal_row['timestamp'].date()
        if signal_date not in trades_today:
            trades_today[signal_date] = 0
        
        if trades_today[signal_date] >= max_daily:
            continue
        
        # Entry at next bar (latency)
        entry_idx = signal_idx + latency_bars
        if entry_idx >= len(df):
            continue
        
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['open']  # Enter at next bar's open
        
        # Apply slippage
        entry_price = apply_slippage(entry_price, slippage_bps, is_long=(direction == 1))
        
        # Get ATR and TP/SL multipliers
        atr = entry_row['ATR_14']
        tp_mult, sl_mult = get_atr_multipliers(atr, entry_price)
        
        # Simulate trade
        result = simulate_trade_with_trailing(
            df, entry_idx, direction, entry_price, atr, tp_mult, sl_mult, trailing=True
        )
        
        exit_price = result['exit_price']
        
        # Apply exit slippage
        exit_price = apply_slippage(exit_price, slippage_bps, is_long=(direction == -1))
        
        # Calculate PnL (per $10k position)
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
            'confidence': proba
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


def evaluate_symbol(symbol, data_dir, models_dir, latency_bars=1, slippage_bps=3):
    """
    Evaluate all walk-forward windows for a symbol.
    
    Returns:
        dict: Combined metrics and all trades
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {symbol}")
    print(f"{'='*60}")
    
    # Load dataset
    full_path = data_dir / f"{symbol}_full.parquet"
    df = pd.read_parquet(full_path)
    
    # Load splits
    splits_path = data_dir / f"{symbol}_splits.csv"
    splits_df = pd.read_csv(splits_path)
    
    # Confidence threshold
    confidence_threshold = CONFIDENCE_THRESHOLDS.get(symbol, 0.70)
    
    print(f"Dataset: {len(df):,} rows")
    print(f"Windows: {len(splits_df)}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Max trades/day: {MAX_TRADES_PER_DAY.get(symbol, 80)}")
    
    all_trades = []
    window_metrics = []
    
    for idx, row in splits_df.iterrows():
        window_id = row['window']
        test_start = row['test_start_idx']
        test_end = row['test_end_idx']
        
        # Load model for this window
        model_path = models_dir / symbol / f"ensemble_window_{window_id}.pkl"
        if not model_path.exists():
            print(f"   Window {window_id}: Model not found, skipping")
            continue
        
        model_data = joblib.load(model_path)
        ensemble = model_data['ensemble']
        features = model_data['features']
        
        print(f"\n   Window {window_id} ({row['test_start_time']} to {row['test_end_time']}):")
        
        # Evaluate
        metrics, trades = evaluate_window(
            symbol, df, test_start, test_end, ensemble, features,
            latency_bars, slippage_bps, confidence_threshold
        )
        
        metrics['window'] = window_id
        window_metrics.append(metrics)
        all_trades.extend(trades)
        
        print(f"      Trades: {metrics['total_trades']}, "
              f"WinRate: {metrics['win_rate']*100:.1f}%, "
              f"PF: {metrics['profit_factor']:.2f}, "
              f"Expectancy: ${metrics['expectancy']:.2f}")
    
    # Combined metrics
    if len(all_trades) == 0:
        print(f"\n   WARNING: No trades generated for {symbol}")
        return None
    
    all_trades_df = pd.DataFrame(all_trades)
    combined_metrics = calculate_metrics(all_trades_df, initial_equity=10000)
    
    print(f"\n{'='*60}")
    print(f"COMBINED METRICS - {symbol}")
    print(f"{'='*60}")
    print(f"Total Trades: {combined_metrics['total_trades']:,}")
    print(f"Win Rate: {combined_metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {combined_metrics['profit_factor']:.2f}")
    print(f"Sharpe: {combined_metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {combined_metrics['max_drawdown']*100:.1f}%")
    print(f"Expectancy: ${combined_metrics['expectancy']:.2f}")
    print(f"Total PnL: ${combined_metrics['total_pnl']:.2f}")
    
    return {
        'symbol': symbol,
        'combined_metrics': combined_metrics,
        'window_metrics': window_metrics,
        'trades': all_trades
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble with walk-forward protocol')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    parser.add_argument('--latency', type=str, default='1bar', help='Signal latency (e.g., 1bar)')
    parser.add_argument('--slippage', type=str, default='3bps', help='Slippage (e.g., 3bps)')
    
    args = parser.parse_args()
    
    # Parse parameters
    latency_bars = int(args.latency.replace('bar', ''))
    slippage_bps = float(args.slippage.replace('bps', ''))
    
    # Directories
    data_dir = Path(__file__).parent.parent.parent / "data" / "datasets"
    models_dir = Path(__file__).parent.parent.parent / "models" / "ensemble_v1"
    output_dir = Path(__file__).parent.parent.parent
    
    print(f"\n{'#'*60}")
    print(f"# Walk-Forward Evaluation v1")
    print(f"{'#'*60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Latency: {latency_bars} bar(s)")
    print(f"Slippage: {slippage_bps} bps")
    
    # Evaluate each symbol
    all_results = []
    
    for symbol in args.symbols:
        try:
            result = evaluate_symbol(symbol, data_dir, models_dir, latency_bars, slippage_bps)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR evaluating {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if len(all_results) > 0:
        # QA metrics
        qa_metrics = {r['symbol']: r['combined_metrics'] for r in all_results}
        qa_path = output_dir / "qa_metrics_wf_v1.json"
        with open(qa_path, 'w') as f:
            json.dump(qa_metrics, f, indent=2, default=str)
        
        # PnL log
        all_trades = []
        for r in all_results:
            all_trades.extend(r['trades'])
        
        pnl_df = pd.DataFrame(all_trades)
        pnl_path = output_dir / "pnl_log_wf_v1.txt"
        pnl_df.to_csv(pnl_path, index=False, sep='\t')
        
        print(f"\n{'='*60}")
        print("Evaluation complete!")
        print(f"{'='*60}")
        print(f"QA Metrics: {qa_path}")
        print(f"PnL Log: {pnl_path}")
        print(f"Total trades: {len(all_trades):,}")


if __name__ == "__main__":
    main()
