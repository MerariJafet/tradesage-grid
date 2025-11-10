import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.feature_engineering import engineer_features
from backend.app.core.ml_edge_classifier import MLEdgeClassifier
from backend.app.core.backtest_engine import simulate_trade, calculate_pnl

DATA_PATH = Path("data/btc_1m_12months.csv")
MODEL_PATH = Path("models/model.joblib")
REPORT_PATH = Path("walk_forward_report_pnl_fixed.txt")
DEFAULT_IMBALANCE = 0.1
DEFAULT_FUNDING_RATE = 0.0001


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_features(
    df: pd.DataFrame,
    imbalance: float = DEFAULT_IMBALANCE,
    funding_rate: float = DEFAULT_FUNDING_RATE,
) -> pd.DataFrame:
    imbalance_series = pd.Series(imbalance, index=df.index)
    features = engineer_features(df, imbalance_series, funding_rate)
    non_numeric = features.select_dtypes(exclude=["number"]).columns
    if not non_numeric.empty:
        features = features.drop(columns=list(non_numeric))
    return features


def split_data(features: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_len = len(features)
    train_end = int(total_len * 0.5)
    val_end = int(total_len * 0.75)
    train = features.iloc[:train_end]
    val = features.iloc[train_end:val_end]
    forward = features.iloc[val_end:]
    return train, val, forward


def evaluate_with_dynamic_sizing(
    features: pd.DataFrame, 
    df: pd.DataFrame, 
    model: XGBClassifier, 
    pnl_cap: float = 50000.0,  # Increased cap
    confidence_threshold: float = 0.6,
    use_momentum_filter: bool = False,
    use_dynamic_atr: bool = False
) -> dict:
    classifier = MLEdgeClassifier(confidence_threshold=confidence_threshold)
    classifier.model = model  # Already loaded
    preds = classifier.predict(features)
    proba = model.predict_proba(features.drop(columns=["target"]))[:, 1]
    
    trades_pnl = []
    trades_log = []
    cumulative_pnl = 0
    
    for idx in features.index:
        if preds.loc[idx] == 1 and cumulative_pnl <= pnl_cap:
            entry_price = df.loc[idx, 'close']
            atr = features.loc[idx, 'atr']
            adx = features.loc[idx, 'adx']
            volume = features.loc[idx, 'volume']
            volume_sma = features.loc[idx, 'volume_sma']
            
            # === MOMENTUM FILTER ===
            if use_momentum_filter:
                adx_threshold = 25
                volume_filter = 1.2 * volume_sma
                
                if adx < adx_threshold:
                    continue  # evitar mercado sin tendencia
                
                if volume < volume_filter:
                    continue  # evitar baja liquidez
            
            # === ATR DYNAMIC TARGETS ===
            if use_dynamic_atr:
                if atr < 10:
                    tp_mult = 8.0
                    sl_mult = 3.0
                elif atr < 30:
                    tp_mult = 6.0
                    sl_mult = 2.5
                else:
                    tp_mult = 4.5
                    sl_mult = 2.0
            else:
                # Original multipliers
                if atr < 10:
                    tp_mult, sl_mult = 8, 3
                else:
                    tp_mult, sl_mult = 5, 2
            
            sl = entry_price - sl_mult * atr
            tp = entry_price + tp_mult * atr
            confidence = proba[features.index.get_loc(idx)]
            size = 1.0  # Fixed size for validation
            
            exit_price, bars_held = simulate_trade(df, idx, entry_price, tp, sl, 60)
            pnl = calculate_pnl(entry_price, exit_price, size)
            
            trades_pnl.append(pnl)
            trades_log.append({
                'idx': idx,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'atr': atr,
                'adx': adx,
                'bars': bars_held
            })
            cumulative_pnl += pnl
            if cumulative_pnl > pnl_cap:
                break  # Stop trading
    
    if not trades_pnl:
        return {
            "win_rate": 0, 
            "expectancy": 0, 
            "profit_factor": 0, 
            "sharpe_ratio": 0, 
            "trades": 0,
            "max_dd": 0,
            "trades_log": []
        }
    
    trades = pd.Series(trades_pnl)
    win_rate = (trades > 0).mean()
    expectancy = trades.mean()
    
    winning = trades[trades > 0].sum()
    losing = abs(trades[trades < 0].sum())
    profit_factor = winning / losing if losing > 0 else float('inf')
    
    # Sharpe based on trades
    sharpe_ratio = trades.mean() / trades.std() if trades.std() > 0 else 0
    
    # Max DD
    cumsum = trades.cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    return {
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "trades": len(trades),
        "max_dd": max_dd,
        "trades_log": trades_log
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML Edge with dynamic sizing")
    parser.add_argument("--mode", help="Mode, e.g., walk-forward")
    parser.add_argument("--period", help="Period, e.g., 3months")
    parser.add_argument("--symbol", help="Symbol, e.g., BTCUSDT")
    parser.add_argument("--validate-pnl", action="store_true", help="Validate PnL simulation")
    parser.add_argument("--validate-edge", action="store_true", help="Validate with ATR dynamic and momentum filter")
    parser.add_argument("--dynamic-sizing", action="store_true", help="Use dynamic sizing")
    parser.add_argument("--dynamic-atr", action="store_true", help="Use dynamic ATR targets")
    parser.add_argument("--momentum-filter", action="store_true", help="Use momentum filter (ADX/Volume)")
    args = parser.parse_args()
    
    df = load_dataset()
    features = build_features(df)
    
    # For 6months, use last 6 months as forward
    period = args.period or '12months'
    if '6months' in period:
        forward_len = int(len(features) * 0.5)
    else:
        forward_len = int(len(features) * 0.25)
    
    forward = features.iloc[-forward_len:]
    forward_df = df.iloc[-forward_len:]
    
    model = joblib.load(MODEL_PATH)
    
    use_momentum = args.momentum_filter or args.validate_edge
    use_dynamic = args.dynamic_atr or args.validate_edge
    
    if args.validate_pnl or args.dynamic_sizing or args.validate_edge or args.dynamic_atr or args.momentum_filter:
        metrics = evaluate_with_dynamic_sizing(
            forward, 
            forward_df, 
            model, 
            confidence_threshold=0.7,  # Increased from 0.6
            use_momentum_filter=use_momentum,
            use_dynamic_atr=use_dynamic
        )
    else:
        metrics = {
            "win_rate": 0, 
            "expectancy": 0, 
            "profit_factor": 0, 
            "sharpe_ratio": 0, 
            "trades": 0,
            "max_dd": 0,
            "trades_log": []
        }
    
    # Write detailed logs
    pnl_log_path = Path("pnl_log_edge_v2.txt")
    with pnl_log_path.open('w') as f:
        f.write("TRADE LOG - Edge Calibration v2\n")
        f.write("=" * 60 + "\n")
        for t in metrics.get('trades_log', []):
            f.write(f"Idx: {t['idx']}, Entry: {t['entry']:.2f}, Exit: {t['exit']:.2f}, "
                   f"PnL: ${t['pnl']:.2f}, ATR: {t['atr']:.2f}, ADX: {t['adx']:.2f}, Bars: {t['bars']}\n")
    
    # Write QA metrics
    qa_metrics = {
        "strategy": "XGBoost_Edge_Classifier_v2",
        "symbol": args.symbol or "BTCUSDT",
        "period": args.period or "12months",
        "win_rate_pct": float(metrics['win_rate'] * 100),
        "expectancy_usd": float(metrics['expectancy']),
        "sharpe_ratio": float(metrics['sharpe_ratio']),
        "profit_factor": float(metrics['profit_factor']),
        "max_dd_usd": float(metrics['max_dd']),
        "trades": int(metrics['trades']),
        "dynamic_atr": use_dynamic,
        "momentum_filter": use_momentum,
        "final_validation_status": "EDGE_V2_CALIBRATION"
    }
    
    import json
    qa_path = Path("qa_metrics_edge_v2.json")
    qa_path.write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
    
    report = f"""Walk-Forward Report - Edge Calibration v2
Period: {args.period or '12months'}
Symbol: {args.symbol or 'BTCUSDT'}
Validate PnL: {args.validate_pnl}
Validate Edge: {args.validate_edge}
Dynamic ATR: {use_dynamic}
Momentum Filter: {use_momentum}
Confidence Threshold: 0.7
Forward Metrics:
Trades: {metrics['trades']}
Win Rate: {metrics['win_rate']:.1%}
Expectancy: ${metrics['expectancy']:.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max DD: ${metrics['max_dd']:.2f}
"""
    
    walk_forward_path = Path("walk_forward_edge_v2.txt")
    walk_forward_path.write_text(report, encoding="utf-8")
    print("Reports saved:")
    print(f"  - {walk_forward_path}")
    print(f"  - {pnl_log_path}")
    print(f"  - {qa_path}")
    print(report)


if __name__ == "__main__":
    main()