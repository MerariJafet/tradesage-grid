# backend/scripts/run_extended_backtest.py

import asyncio
from datetime import datetime, timedelta
from app.backtest.data_loader import BinanceHistoricalDataLoader
from app.backtest.backtest_engine import BacktestEngine
from app.backtest.performance_analytics import PerformanceAnalytics
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.breakout_compression import BreakoutCompressionStrategy
from app.core.strategies.mean_reversion import MeanReversionStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
import json

async def main():
    """Backtest extendido de 30 días para validación estadística"""

    print("=" * 80)
    print("EXTENDED BACKTEST - 30 DAYS VALIDATION")
    print("=" * 80)

    symbol = "BTCUSDT"
    days = 30

    # 1. Descargar datos históricos
    print(f"\n1️⃣ Downloading {days} days of historical data...")
    print("   This may take 2-3 minutes...")

    async with BinanceHistoricalDataLoader() as loader:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        bars = await loader.download_klines(
            symbol=symbol,
            interval="1m",
            start_date=start_date,
            end_date=end_date
        )

        # Guardar
        filename = loader.save_to_csv(bars, symbol, "1m", start_date, end_date)

        print(f"   ✅ Downloaded {len(bars):,} bars")
        print(f"   ✅ Saved to {filename}")

    # 2. Crear estrategias
    print(f"\n2️⃣ Initializing strategies...")

    indicator_manager_breakout = IndicatorManager()
    indicator_manager_mean_rev = IndicatorManager()

    position_sizer = PositionSizer(account_balance=10000)
    signal_validator_breakout = SignalValidator(indicator_manager=indicator_manager_breakout)
    signal_validator_mean_rev = SignalValidator(indicator_manager=indicator_manager_mean_rev)

    strategies = [
        {
            "strategy": BreakoutCompressionStrategy(
                symbol=symbol,
                indicator_manager=indicator_manager_breakout,
                position_sizer=position_sizer,
                signal_validator=signal_validator_breakout
            ),
            "name": "BreakoutCompression",
            "description": "Volatility breakout with BB compression"
        },
        {
            "strategy": MeanReversionStrategy(
                symbol=symbol,
                indicator_manager=indicator_manager_mean_rev,
                position_sizer=position_sizer,
                signal_validator=signal_validator_mean_rev,
                rsi_oversold=30,
                rsi_overbought=70
            ),
            "name": "MeanReversion",
            "description": "RSI mean reversion strategy"
        }
    ]

    print(f"   ✅ Created {len(strategies)} strategies")

    # 3. Ejecutar backtests
    print(f"\n3️⃣ Running backtests...")
    print(f"   Processing {len(bars):,} bars per strategy...")
    print(f"   Estimated time: 3-5 minutes\n")

    engine = BacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.0004,  # Binance Futures: 0.04%
        slippage_pct=0.02  # 0.02% slippage
    )

    analytics = PerformanceAnalytics()
    results_summary = []

    for i, strat_config in enumerate(strategies, 1):
        strategy = strat_config["strategy"]

        print(f"   [{i}/{len(strategies)}] Testing {strat_config['name']}...")
        print(f"        {strat_config['description']}")

        # Ejecutar backtest
        result = await engine.run_backtest(
            strategy=strategy,
            bars=bars,
            symbol=symbol
        )

        # Análisis completo
        analysis = analytics.analyze(result)

        # Guardar reporte JSON
        report_filename = f"reports/{strat_config['name']}_{days}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analytics.save_report(analysis, filename=report_filename)

        # Resumen
        roi = ((result.final_balance - result.initial_balance) / result.initial_balance) * 100

        summary = {
            "strategy": strat_config['name'],
            "description": strat_config['description'],
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "roi_pct": roi,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "expectancy": analysis['expectancy']['expectancy'],
            "expectancy_interpretation": analysis['expectancy']['interpretation'],
            "avg_trade_duration_minutes": result.avg_trade_duration_minutes,
            "report_file": report_filename
        }

        results_summary.append(summary)

        print(f"        ✅ Completed: {result.total_trades} trades, ${result.total_pnl:.2f} P&L\n")

    # 4. Generar reporte comparativo
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS - 30 DAYS BACKTEST")
    print("=" * 80)

    print(f"\n📊 Dataset Information:")
    print(f"   Symbol: {symbol}")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: {days} days")
    print(f"   Total Bars: {len(bars):,}")
    print(f"   Initial Balance: $10,000.00")

    # Tabla comparativa
    print(f"\n{'Strategy':<25} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'ROI%':<10} {'Sharpe':<10} {'Max DD%':<10}")
    print("-" * 100)

    for summary in results_summary:
        print(f"{summary['strategy']:<25} "
              f"{summary['total_trades']:<10} "
              f"{summary['win_rate']:<10.1f} "
              f"${summary['total_pnl']:<14.2f} "
              f"{summary['roi_pct']:<10.2f} "
              f"{summary['sharpe_ratio']:<10.2f} "
              f"{summary['max_drawdown_pct']:<10.2f}")

    # Análisis detallado por estrategia
    for summary in results_summary:
        print(f"\n{'=' * 80}")
        print(f"{summary['strategy']} - Detailed Analysis")
        print('=' * 80)

        print(f"\n📝 Description: {summary['description']}")

        print(f"\n💰 Performance Metrics:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.2f}%")
        print(f"   Total P&L: ${summary['total_pnl']:.2f}")
        print(f"   ROI: {summary['roi_pct']:.2f}%")
        print(f"   Profit Factor: {summary['profit_factor']:.2f}")

        print(f"\n📈 Risk-Adjusted Returns:")
        print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {summary['sortino_ratio']:.2f}")

        print(f"\n⚠️ Risk Metrics:")
        print(f"   Max Drawdown: {summary['max_drawdown_pct']:.2f}%")

        print(f"\n🎯 Expectancy Analysis:")
        print(f"   Expectancy: ${summary['expectancy']:.2f} per trade")
        print(f"   {summary['expectancy_interpretation']}")

        print(f"\n⏱️ Trade Characteristics:")
        print(f"   Avg Duration: {summary['avg_trade_duration_minutes']:.1f} minutes")

        print(f"\n📄 Full Report: {summary['report_file']}")

    # Recomendaciones
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print('=' * 80)

    # Identificar mejor estrategia
    best_strategy = max(results_summary, key=lambda x: x['sharpe_ratio'])
    most_trades = max(results_summary, key=lambda x: x['total_trades'])
    best_roi = max(results_summary, key=lambda x: x['roi_pct'])

    print(f"\n🏆 Best Risk-Adjusted Return: {best_strategy['strategy']}")
    print(f"   (Highest Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f})")

    print(f"\n📊 Most Active: {most_trades['strategy']}")
    print(f"   ({most_trades['total_trades']} trades)")

    print(f"\n💰 Highest ROI: {best_roi['strategy']}")
    print(f"   ({best_roi['roi_pct']:.2f}%)")

    # Evaluación general
    print(f"\n📋 Overall Assessment:")

    for summary in results_summary:
        if summary['total_trades'] == 0:
            print(f"\n   ⚠️ {summary['strategy']}: NO TRADES")
            print(f"      No trading opportunities found in this period.")
            print(f"      Consider: Adjusting parameters or testing different market conditions.")

        elif summary['expectancy'] > 0:
            print(f"\n   ✅ {summary['strategy']}: POSITIVE EXPECTANCY")
            print(f"      Expectancy: ${summary['expectancy']:.2f} per trade")
            print(f"      {summary['expectancy_interpretation']}")
            print(f"      Recommendation: Consider for live trading with risk management.")

        else:
            print(f"\n   ⚠️ {summary['strategy']}: NEGATIVE EXPECTANCY")
            print(f"      Expectancy: ${summary['expectancy']:.2f} per trade")
            print(f"      {summary['expectancy_interpretation']}")
            print(f"      Recommendation: Optimize parameters or avoid this strategy in similar market conditions.")

    # Guardar resumen comparativo
    comparison_filename = f"reports/comparison_{days}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_filename, 'w') as f:
        json.dump({
            "backtest_info": {
                "symbol": symbol,
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "duration_days": days,
                "total_bars": len(bars),
                "initial_balance": 10000.0
            },
            "strategies": results_summary
        }, f, indent=2)

    print(f"\n💾 Comparison report saved: {comparison_filename}")

    print("\n" + "=" * 80)
    print("✅ EXTENDED BACKTEST COMPLETED")
    print("=" * 80)

    print("\n📁 Generated Files:")
    print("   Individual reports in: reports/")
    print(f"   Comparison report: {comparison_filename}")
    print(f"   Historical data: {filename}")

if __name__ == "__main__":
    asyncio.run(main())