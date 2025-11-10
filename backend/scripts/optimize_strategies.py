# backend/scripts/optimize_strategies.py

import asyncio
from datetime import datetime, timedelta
from app.backtest.data_loader import BinanceHistoricalDataLoader
from app.backtest.backtest_engine import BacktestEngine
from app.backtest.performance_analytics import PerformanceAnalytics
from app.core.strategies.momentum_scalping import MomentumScalpingStrategy
from app.core.strategies.mean_reversion import MeanReversionStrategy
import json

async def main():
    """Optimización de estrategias para Sprint 10"""

    print("=" * 80)
    print("SPRINT 10: STRATEGY OPTIMIZATION")
    print("=" * 80)

    symbol = "BTCUSDT"

    # 1. Descargar datos de 6 meses para optimización
    print(f"\n1️⃣ Downloading 6 months of historical data for optimization...")
    print("   This may take 5-10 minutes...")

    async with BinanceHistoricalDataLoader() as loader:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=180)  # 6 meses

        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        bars = await loader.download_klines(
            symbol=symbol,
            interval="1m",
            start_date=start_date,
            end_date=end_date
        )

        # Guardar datos
        filename = f"data/{symbol}_1m_6months_{datetime.now().strftime('%Y%m%d')}.csv"
        loader.save_to_csv(bars, symbol, "1m", start_date, end_date)

        print(f"   ✅ Downloaded {len(bars):,} bars")
        print(f"   ✅ Saved to {filename}")

    # 2. Configurar motor de backtest
    engine = BacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.0004,
        slippage_pct=0.02
    )

    analytics = PerformanceAnalytics()

    # 3. Optimizar Momentum Scalping Strategy
    print(f"\n2️⃣ Optimizing Momentum Scalping Strategy...")

    momentum_param_grid = {
        'macd_fast': [8, 10, 12, 14, 16],
        'adx_threshold': [20, 25, 30],
        'profit_target_pct': [0.004, 0.005, 0.006],  # 0.4%, 0.5%, 0.6%
        'stop_loss_pct': [0.002, 0.003, 0.004]  # 0.2%, 0.3%, 0.4%
    }

    from itertools import product
    total_combinations = len(list(product(*momentum_param_grid.values())))
    print(f"   Parameter grid: {total_combinations} combinations")
    print("   Testing up to 100 random combinations...")

    momentum_best = await engine.grid_search_optimization(
        MomentumScalpingStrategy,
        bars,
        symbol,
        momentum_param_grid,
        max_evaluations=100
    )

    if momentum_best:
        print("   ✅ Best Momentum parameters found:")
        print(f"      Sharpe: {momentum_best['sharpe_ratio']:.3f}")
        print(f"      Trades: {momentum_best['total_trades']}")
        print(f"      Win Rate: {momentum_best['win_rate']:.1f}%")
        print(f"      P&L: ${momentum_best['total_pnl']:.2f}")
        print(f"      Params: {momentum_best['params']}")

        # Guardar mejores parámetros
        with open("reports/momentum_best_params.json", 'w') as f:
            json.dump(momentum_best, f, indent=2, default=str)
    else:
        print("   ❌ No valid Momentum parameters found")

    # 4. Optimizar Mean Reversion Strategy
    print(f"\n3️⃣ Optimizing Mean Reversion Strategy...")

    mean_rev_param_grid = {
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80],
        'volume_multiplier': [1.2, 1.5, 1.8],
        'atr_stop_multiplier': [1.0, 1.5, 2.0],
        'profit_target_atr_multiplier': [1.5, 2.0, 2.5]
    }

    total_combinations_mr = len(list(product(*mean_rev_param_grid.values())))
    print(f"   Parameter grid: {total_combinations_mr} combinations")
    print("   Testing up to 100 random combinations...")

    mean_rev_best = await engine.grid_search_optimization(
        MeanReversionStrategy,
        bars,
        symbol,
        mean_rev_param_grid,
        max_evaluations=100
    )

    if mean_rev_best:
        print("   ✅ Best Mean Reversion parameters found:")
        print(f"      Sharpe: {mean_rev_best['sharpe_ratio']:.3f}")
        print(f"      Trades: {mean_rev_best['total_trades']}")
        print(f"      Win Rate: {mean_rev_best['win_rate']:.1f}%")
        print(f"      P&L: ${mean_rev_best['total_pnl']:.2f}")
        print(f"      Params: {mean_rev_best['params']}")

        # Guardar mejores parámetros
        with open("reports/mean_reversion_best_params.json", 'w') as f:
            json.dump(mean_rev_best, f, indent=2, default=str)
    else:
        print("   ❌ No valid Mean Reversion parameters found")

    # 5. Walk-Forward Analysis
    print(f"\n4️⃣ Running Walk-Forward Analysis...")

    wfa_results = {}

    if momentum_best:
        print("   Testing Momentum Scalping robustness...")
        momentum_wfa = await engine.walk_forward_analysis(
            MomentumScalpingStrategy,
            bars,
            symbol,
            momentum_best['params'],
            window_size=0.7,  # 70% in-sample
            step_size=0.15    # 15% step
        )

        wfa_results['momentum'] = momentum_wfa
        print("   ✅ Momentum WFA completed:")
        print(f"      Steps: {momentum_wfa.get('total_steps', 0)}")
        print(".3f")
        print(".3f")
    if mean_rev_best:
        print("   Testing Mean Reversion robustness...")
        mean_rev_wfa = await engine.walk_forward_analysis(
            MeanReversionStrategy,
            bars,
            symbol,
            mean_rev_best['params'],
            window_size=0.7,
            step_size=0.15
        )

        wfa_results['mean_reversion'] = mean_rev_wfa
        print("   ✅ Mean Reversion WFA completed:")
        print(f"      Steps: {mean_rev_wfa.get('total_steps', 0)}")
        print(".3f")
        print(".3f")
    # 6. Generar reporte final
    print(f"\n5️⃣ Generating optimization report...")

    optimization_report = {
        "optimization_info": {
            "symbol": symbol,
            "data_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "total_bars": len(bars),
            "initial_balance": 10000.0,
            "timestamp": datetime.now().isoformat()
        },
        "momentum_scalping": momentum_best,
        "mean_reversion": mean_rev_best,
        "walk_forward_analysis": wfa_results
    }

    # Guardar reporte completo
    report_filename = f"reports/optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(optimization_report, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print("OPTIMIZATION RESULTS SUMMARY")
    print('=' * 80)

    print(f"\n📊 Dataset: {len(bars):,} bars ({symbol})")
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Comparación final
    print(f"\n🏆 OPTIMIZATION RESULTS:")

    if momentum_best and momentum_best.get('sharpe_ratio', 0) > 0:
        print(f"\n✅ Momentum Scalping - OPTIMIZED")
        print(f"   Sharpe Ratio: {momentum_best['sharpe_ratio']:.3f}")
        print(f"   Total Trades: {momentum_best['total_trades']}")
        print(f"   Win Rate: {momentum_best['win_rate']:.1f}%")
        print(f"   Total P&L: ${momentum_best['total_pnl']:.2f}")
        print(f"   Best Params: {momentum_best['params']}")

        if 'momentum' in wfa_results:
            wfa = wfa_results['momentum']
            print(f"   WFA Stability: {wfa.get('sharpe_stability', 0):.3f}")
    else:
        print(f"\n❌ Momentum Scalping - NO VIABLE PARAMETERS")

    if mean_rev_best and mean_rev_best.get('sharpe_ratio', 0) > 0:
        print(f"\n✅ Mean Reversion - OPTIMIZED")
        print(f"   Sharpe Ratio: {mean_rev_best['sharpe_ratio']:.3f}")
        print(f"   Total Trades: {mean_rev_best['total_trades']}")
        print(f"   Win Rate: {mean_rev_best['win_rate']:.1f}%")
        print(f"   Total P&L: ${mean_rev_best['total_pnl']:.2f}")
        print(f"   Best Params: {mean_rev_best['params']}")

        if 'mean_reversion' in wfa_results:
            wfa = wfa_results['mean_reversion']
            print(f"   WFA Stability: {wfa.get('sharpe_stability', 0):.3f}")
    else:
        print(f"\n❌ Mean Reversion - NO VIABLE PARAMETERS")

    print(f"\n💾 Reports saved:")
    print(f"   Optimization report: {report_filename}")
    if momentum_best:
        print("   Momentum best params: reports/momentum_best_params.json")
    if mean_rev_best:
        print("   Mean Reversion best params: reports/mean_reversion_best_params.json")
    print(f"   Historical data: {filename}")

    print(f"\n{'=' * 80}")
    print("✅ STRATEGY OPTIMIZATION COMPLETED")
    print('=' * 80)

    # Recomendaciones finales
    print(f"\n🎯 RECOMMENDATIONS:")

    best_strategy = None
    best_sharpe = float('-inf')

    if momentum_best and momentum_best.get('sharpe_ratio', 0) > best_sharpe:
        best_strategy = "Momentum Scalping"
        best_sharpe = momentum_best['sharpe_ratio']

    if mean_rev_best and mean_rev_best.get('sharpe_ratio', 0) > best_sharpe:
        best_strategy = "Mean Reversion"
        best_sharpe = mean_rev_best['sharpe_ratio']

    if best_strategy and best_sharpe > 0.5:
        print(f"✅ {best_strategy} shows strong potential (Sharpe > 0.5)")
        print("   Ready for live testing with proper risk management")
    elif best_strategy and best_sharpe > 0:
        print(f"⚠️ {best_strategy} shows marginal performance (Sharpe > 0)")
        print("   Consider further parameter tuning or different market conditions")
    else:
        print("❌ No strategy achieved positive Sharpe ratio")
        print("   Consider alternative strategies or market conditions")

if __name__ == "__main__":
    asyncio.run(main())