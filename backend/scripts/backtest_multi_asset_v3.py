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
from backend.app.core.backtest_engine_v3 import simulate_trade_v3, calculate_pnl, apply_slippage

DEFAULT_IMBALANCE = 0.1
DEFAULT_FUNDING_RATE = 0.0001

# Confidence thresholds by symbol
CONFIDENCE_THRESHOLDS = {
    'BTCUSDT': 0.80,
    'ETHUSDT': 0.70,
    'BNBUSDT': 0.65
}


def load_dataset(symbol: str) -> pd.DataFrame:
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


def evaluate_live_simulation(
    features: pd.DataFrame,
    df: pd.DataFrame,
    model: XGBClassifier,
    symbol: str,
    use_dynamic_atr: bool = True,
    use_trailing: bool = True,
    slippage_bps: float = 3.0,
    latency_bars: int = 1
) -> dict:
    """
    Evaluate with live simulation: latency, slippage, trailing stops.
    """
    confidence_threshold = CONFIDENCE_THRESHOLDS.get(symbol, 0.70)
    
    classifier = MLEdgeClassifier(confidence_threshold=confidence_threshold)
    classifier.model = model
    
    # Get predictions on features
    X = features.drop(columns=["target"])
    proba = model.predict_proba(X)[:, 1]
    
    trades_pnl = []
    trades_log = []
    cumulative_pnl = 0
    pnl_cap = 50000.0
    
    # Iterate with latency offset
    for i in range(len(features) - latency_bars - 60):
        signal_bar = i
        entry_bar = i + latency_bars
        
        # Check confidence from signal bar
        if proba[signal_bar] < confidence_threshold:
            continue
        
        if cumulative_pnl > pnl_cap:
            break
        
        # === SIGNAL LATENCY EMULATION ===
        # Use features from signal_bar
        atr = features.iloc[signal_bar]['atr']
        adx = features.iloc[signal_bar]['adx']
        
        # Entry on next bar's open
        entry_price = df.iloc[entry_bar]['open']
        
        # === SLIPPAGE MODEL ===
        entry_price = apply_slippage(entry_price, slippage_bps, is_long=True)
        
        # === ATR DYNAMIC TARGETS ===
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
        
        # Simulate with trailing
        exit_price, bars_held = simulate_trade_v3(
            df, 
            entry_bar, 
            entry_price, 
            tp, 
            sl, 
            atr,
            max_bars=60,
            use_trailing=use_trailing
        )
        
        pnl = calculate_pnl(entry_price, exit_price, size)
        
        trades_pnl.append(pnl)
        trades_log.append({
            'symbol': symbol,
            'signal_bar': signal_bar,
            'entry_bar': entry_bar,
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'atr': atr,
            'adx': adx,
            'bars': bars_held,
            'confidence': proba[signal_bar]
        })
        cumulative_pnl += pnl
    
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
    max_dd_pct = (max_dd / running_max.max() * 100) if running_max.max() > 0 else 0
    
    return {
        "symbol": symbol,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "trades": len(trades),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "trades_log": trades_log
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset backtest v3 - Live Simulation")
    parser.add_argument("--period", default="6months", help="Period")
    parser.add_argument("--symbols", nargs='+', default=["BTCUSDT", "ETHUSDT", "BNBUSDT"], help="Symbols")
    parser.add_argument("--validate-live", action="store_true", help="Use live simulation")
    parser.add_argument("--slippage", default="3bps", help="Slippage in bps")
    parser.add_argument("--latency", default="1bar", help="Latency in bars")
    args = parser.parse_args()
    
    # Parse slippage
    slippage_bps = float(args.slippage.replace('bps', ''))
    
    # Parse latency
    latency_bars = int(args.latency.replace('bar', ''))
    
    symbols = [s.lower() for s in args.symbols]
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
            
            metrics = evaluate_live_simulation(
                forward,
                forward_df,
                model,
                symbol.upper(),
                use_dynamic_atr=True,
                use_trailing=args.validate_live,
                slippage_bps=slippage_bps,
                latency_bars=latency_bars
            )
            
            all_results.append(metrics)
            all_trades_log.extend(metrics['trades_log'])
            
            print(f"{symbol.upper()} Results:")
            print(f"  Trades: {metrics['trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Expectancy: ${metrics['expectancy']:.2f}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max DD: ${metrics['max_dd']:.2f} ({metrics['max_dd_pct']:.2f}%)")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
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
        combined_dd_pct = (combined_dd / running_max.max() * 100) if running_max.max() > 0 else 0
    else:
        combined_win_rate = 0
        combined_expectancy = 0
        combined_pf = 0
        combined_sharpe = 0
        combined_dd = 0
        combined_dd_pct = 0
    
    # Write detailed log
    log_path = Path("pnl_log_multi_asset_v3.txt")
    with log_path.open('w') as f:
        f.write("MULTI-ASSET TRADE LOG - Sprint 13 Live Validation\n")
        f.write("=" * 80 + "\n")
        for t in all_trades_log:
            f.write(f"{t['symbol']}: Signal {t['signal_bar']}, Entry {t['entry_bar']}, "
                   f"Price {t['entry']:.2f} → {t['exit']:.2f}, PnL ${t['pnl']:.2f}, "
                   f"ATR {t['atr']:.2f}, Conf {t['confidence']:.3f}, Bars {t['bars']}\n")
    
    # Write QA metrics
    qa_metrics = {
        "strategy": "XGBoost_Edge_Live_Simulation_v3",
        "symbols": [r['symbol'] for r in all_results],
        "period": args.period,
        "latency_bars": latency_bars,
        "slippage_bps": slippage_bps,
        "combined_metrics": {
            "total_trades": total_trades,
            "win_rate_pct": float(combined_win_rate * 100),
            "expectancy_usd": float(combined_expectancy),
            "sharpe_ratio": float(combined_sharpe),
            "profit_factor": float(combined_pf) if combined_pf != float('inf') else "Infinity",
            "max_dd_usd": float(combined_dd),
            "max_dd_pct": float(combined_dd_pct)
        },
        "per_asset_metrics": [
            {
                "symbol": r['symbol'],
                "confidence_threshold": CONFIDENCE_THRESHOLDS.get(r['symbol'], 0.70),
                "trades": r['trades'],
                "win_rate_pct": float(r['win_rate'] * 100),
                "expectancy_usd": float(r['expectancy']),
                "profit_factor": float(r['profit_factor']) if r['profit_factor'] != float('inf') else "Infinity",
                "sharpe_ratio": float(r['sharpe_ratio']),
                "max_dd_usd": float(r['max_dd']),
                "max_dd_pct": float(r['max_dd_pct'])
            }
            for r in all_results
        ],
        "final_validation_status": "LIVE_SIM_V3_VALIDATION"
    }
    
    qa_path = Path("qa_metrics_multi_asset_v3.json")
    qa_path.write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
    
    # Write report
    report = f"""Sprint 13 - Live Simulation Validation
Period: {args.period}
Latency: {latency_bars} bar
Slippage: {slippage_bps} bps
Trailing Stop: {'Enabled' if args.validate_live else 'Disabled'}

COMBINED METRICS:
Total Trades: {total_trades}
Win Rate: {combined_win_rate:.1%}
Expectancy: ${combined_expectancy:.2f}
Profit Factor: {combined_pf:.2f}
Sharpe Ratio: {combined_sharpe:.2f}
Max DD: ${combined_dd:.2f} ({combined_dd_pct:.2f}%)

PER-ASSET BREAKDOWN:
"""
    
    for r in all_results:
        threshold = CONFIDENCE_THRESHOLDS.get(r['symbol'], 0.70)
        report += f"""
{r['symbol']} (Confidence ≥ {threshold:.0%}):
  Trades: {r['trades']}
  Win Rate: {r['win_rate']:.1%}
  Expectancy: ${r['expectancy']:.2f}
  Profit Factor: {r['profit_factor']:.2f}
  Sharpe: {r['sharpe_ratio']:.2f}
  Max DD: ${r['max_dd']:.2f} ({r['max_dd_pct']:.2f}%)
"""
    
    # Status determination
    if combined_pf >= 1.8 and combined_sharpe >= 0.7:
        status = "ROBUST"
    elif combined_pf >= 1.2:
        status = "NEEDS TUNING"
    else:
        status = "REQUIRES RETRAINING"
    
    report += f"\nStatus: {status}\n"
    
    report_path = Path("walk_forward_multi_asset_v3.txt")
    report_path.write_text(report, encoding="utf-8")
    
    print(f"\n{'='*80}")
    print("SPRINT 13 - LIVE VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"  - {report_path}")
    print(f"  - {log_path}")
    print(f"  - {qa_path}")
    print(f"\n{report}")
    
    # Print formatted result
    print(f"\n[SPRINT 13 – LIVE VALIDATION]")
    print(f"Trades: {total_trades}")
    print(f"Win Rate: {combined_win_rate:.1%}")
    print(f"Profit Factor: {combined_pf:.2f}")
    print(f"Sharpe: {combined_sharpe:.2f}")
    print(f"Expectancy: ${combined_expectancy:.2f}")
    print(f"Max DD: {combined_dd_pct:.2f}%")
    print(f"Latency: {latency_bars} bar")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
