import argparse
import sys
from pathlib import Path

import joblib
import json
import pandas as pd
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.app.ml.feature_engineering import engineer_features
from backend.app.core.ml_edge_classifier import MLEdgeClassifier
from backend.app.core.backtest_engine import simulate_trade, calculate_pnl

DEFAULT_IMBALANCE = 0.1
DEFAULT_FUNDING_RATE = 0.0001


def load_dataset(symbol: str) -> pd.DataFrame:
    # Handle btc special case
    if symbol == "btcusdt":
        path = Path("data/btc_1m_12months.csv")
    else:
        path = Path(f"data/{symbol}_1m_12months.csv")
    
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


def evaluate_multi_asset(
    features: pd.DataFrame,
    df: pd.DataFrame,
    model: XGBClassifier,
    symbol: str,
    confidence_threshold: float = 0.7,
    use_momentum_filter: bool = False,
    use_dynamic_atr: bool = True
) -> dict:
    classifier = MLEdgeClassifier(confidence_threshold=confidence_threshold)
    classifier.model = model
    preds = classifier.predict(features)
    proba = model.predict_proba(features.drop(columns=["target"]))[:, 1]
    
    trades_pnl = []
    trades_log = []
    cumulative_pnl = 0
    pnl_cap = 50000.0
    
    for idx in features.index:
        if preds.loc[idx] == 1 and cumulative_pnl <= pnl_cap:
            entry_price = df.loc[idx, 'close']
            atr = features.loc[idx, 'atr']
            adx = features.loc[idx, 'adx']
            volume = features.loc[idx, 'volume']
            volume_sma = features.loc[idx, 'volume_sma']
            
            # Momentum filter
            if use_momentum_filter:
                if adx < 25 or volume < 1.2 * volume_sma:
                    continue
            
            # Dynamic ATR
            if use_dynamic_atr:
                if atr < 10:
                    tp_mult, sl_mult = 8.0, 3.0
                elif atr < 30:
                    tp_mult, sl_mult = 6.0, 2.5
                else:
                    tp_mult, sl_mult = 4.5, 2.0
            else:
                tp_mult, sl_mult = 5, 2
            
            sl = entry_price - sl_mult * atr
            tp = entry_price + tp_mult * atr
            size = 1.0
            
            exit_price, bars_held = simulate_trade(df, idx, entry_price, tp, sl, 60)
            pnl = calculate_pnl(entry_price, exit_price, size)
            
            trades_pnl.append(pnl)
            trades_log.append({
                'symbol': symbol,
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
                break
    
    if not trades_pnl:
        return {
            "symbol": symbol,
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
    
    sharpe_ratio = trades.mean() / trades.std() if trades.std() > 0 else 0
    
    cumsum = trades.cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    return {
        "symbol": symbol,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "trades": len(trades),
        "max_dd": max_dd,
        "trades_log": trades_log
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset backtest for Edge Calibration v2")
    parser.add_argument("--period", default="6months", help="Period")
    parser.add_argument("--dynamic-atr", action="store_true", help="Use dynamic ATR")
    parser.add_argument("--momentum-filter", action="store_true", help="Use momentum filter")
    args = parser.parse_args()
    
    symbols = ["btcusdt", "ethusdt", "bnbusdt"]
    model = joblib.load(Path("models/model.joblib"))
    
    all_results = []
    all_trades_log = []
    
    for symbol in symbols:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {symbol.upper()}...")
            print(f"{'='*60}")
            
            df = load_dataset(symbol)
            features = build_features(df)
            
            # Use last 6 months
            forward_len = int(len(features) * 0.5)
            forward = features.iloc[-forward_len:]
            forward_df = df.iloc[-forward_len:]
            
            metrics = evaluate_multi_asset(
                forward,
                forward_df,
                model,
                symbol.upper(),
                confidence_threshold=0.7,
                use_momentum_filter=args.momentum_filter,
                use_dynamic_atr=args.dynamic_atr
            )
            
            all_results.append(metrics)
            all_trades_log.extend(metrics['trades_log'])
            
            print(f"{symbol.upper()} Results:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Expectancy: ${metrics['expectancy']:.2f}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max DD: ${metrics['max_dd']:.2f}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Aggregate results
    total_trades = sum(r['trades'] for r in all_results)
    
    if total_trades > 0:
        all_pnl = [t['pnl'] for t in all_trades_log]
        combined_trades = pd.Series(all_pnl)
        
        combined_win_rate = (combined_trades > 0).mean()
        combined_expectancy = combined_trades.mean()
        
        winning = combined_trades[combined_trades > 0].sum()
        losing = abs(combined_trades[combined_trades < 0].sum())
        combined_pf = winning / losing if losing > 0 else float('inf')
        
        combined_sharpe = combined_trades.mean() / combined_trades.std() if combined_trades.std() > 0 else 0
        
        cumsum = combined_trades.cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        combined_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    else:
        combined_win_rate = 0
        combined_expectancy = 0
        combined_pf = 0
        combined_sharpe = 0
        combined_dd = 0
    
    # Write detailed log
    log_path = Path("pnl_log_multi_asset_v2.txt")
    with log_path.open('w') as f:
        f.write("MULTI-ASSET TRADE LOG - Edge Calibration v2\n")
        f.write("=" * 80 + "\n")
        for t in all_trades_log:
            f.write(f"{t['symbol']}: Idx {t['idx']}, Entry {t['entry']:.2f}, Exit {t['exit']:.2f}, "
                   f"PnL ${t['pnl']:.2f}, ATR {t['atr']:.2f}, ADX {t['adx']:.2f}, Bars {t['bars']}\n")
    
    # Write QA metrics
    qa_metrics = {
        "strategy": "XGBoost_Edge_Multi_Asset_v2",
        "symbols": [r['symbol'] for r in all_results],
        "period": args.period,
        "combined_metrics": {
            "total_trades": total_trades,
            "win_rate_pct": float(combined_win_rate * 100),
            "expectancy_usd": float(combined_expectancy),
            "sharpe_ratio": float(combined_sharpe),
            "profit_factor": float(combined_pf) if combined_pf != float('inf') else "Infinity",
            "max_dd_usd": float(combined_dd)
        },
        "per_asset_metrics": [
            {
                "symbol": r['symbol'],
                "trades": r['trades'],
                "win_rate_pct": float(r['win_rate'] * 100),
                "expectancy_usd": float(r['expectancy']),
                "profit_factor": float(r['profit_factor']) if r['profit_factor'] != float('inf') else "Infinity",
                "sharpe_ratio": float(r['sharpe_ratio']),
                "max_dd_usd": float(r['max_dd'])
            }
            for r in all_results
        ],
        "dynamic_atr": args.dynamic_atr,
        "momentum_filter": args.momentum_filter,
        "final_validation_status": "MULTI_ASSET_V2_CALIBRATION"
    }
    
    qa_path = Path("qa_metrics_multi_asset_v2.json")
    qa_path.write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
    
    # Write report
    report = f"""Multi-Asset Backtest Report - Edge Calibration v2
Period: {args.period}
Dynamic ATR: {args.dynamic_atr}
Momentum Filter: {args.momentum_filter}
Confidence Threshold: 0.7

COMBINED METRICS:
Total Trades: {total_trades}
Win Rate: {combined_win_rate:.1%}
Expectancy: ${combined_expectancy:.2f}
Profit Factor: {combined_pf:.2f}
Sharpe Ratio: {combined_sharpe:.2f}
Max DD: ${combined_dd:.2f}

PER-ASSET BREAKDOWN:
"""
    
    for r in all_results:
        report += f"""
{r['symbol']}:
  Trades: {r['trades']}
  Win Rate: {r['win_rate']:.1%}
  Expectancy: ${r['expectancy']:.2f}
  Profit Factor: {r['profit_factor']:.2f}
  Sharpe: {r['sharpe_ratio']:.2f}
  Max DD: ${r['max_dd']:.2f}
"""
    
    report_path = Path("walk_forward_multi_asset_v2.txt")
    report_path.write_text(report, encoding="utf-8")
    
    print(f"\n{'='*80}")
    print("MULTI-ASSET RESULTS SAVED")
    print(f"{'='*80}")
    print(f"  - {report_path}")
    print(f"  - {log_path}")
    print(f"  - {qa_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
