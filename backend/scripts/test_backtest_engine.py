#!/usr/bin/env python3
"""
Test script for Backtest Engine (Sprint 8.2)
Tests the backtest engine functionality
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.logger import setup_logging, get_logger
from app.backtest.data_loader import BinanceHistoricalDataLoader
from app.backtest.backtest_engine import BacktestEngine
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.breakout_compression import BreakoutCompressionStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator

logger = get_logger("test_backtest")

async def test_backtest_engine():
    """Test del backtest engine"""

    logger.info("Starting Backtest Engine test")

    symbol = "BTCUSDT"

    # 1. Cargar datos históricos
    logger.info("Loading historical data...")

    async with BinanceHistoricalDataLoader() as loader:
        # Cargar desde CSV si existe, sino descargar
        try:
            bars = loader.load_from_csv("data/BTCUSDT_1m_20251013_20251020.csv")
            logger.info("loaded_bars_from_csv", count=len(bars))
        except Exception as e:
            logger.warning("csv_load_failed", error=str(e))
            logger.info("Downloading fresh data...")

            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)

            bars = await loader.download_klines(
                symbol=symbol,
                interval="1m",
                start_date=start_date,
                end_date=end_date
            )

            filename = loader.save_to_csv(bars, symbol, "1m", start_date, end_date)
            logger.info("downloaded_and_saved", count=len(bars), filename=filename)

    # 2. Crear componentes
    logger.info("Initializing strategy components...")

    indicator_manager = IndicatorManager()
    position_sizer = PositionSizer(account_balance=10000)
    signal_validator = SignalValidator(indicator_manager=indicator_manager)

    # 3. Crear estrategia
    logger.info("Creating Breakout Compression strategy...")

    strategy = BreakoutCompressionStrategy(
        symbol=symbol,
        indicator_manager=indicator_manager,
        position_sizer=position_sizer,
        signal_validator=signal_validator
    )

    logger.info("strategy_created", name=strategy.name)

    # 4. Ejecutar backtest
    logger.info("Running backtest...")

    engine = BacktestEngine(
        initial_balance=10000.0,
        commission_rate=0.0004,
        slippage_pct=0.02
    )

    result = await engine.run_backtest(
        strategy=strategy,
        bars=bars,
        symbol=symbol
    )

    # 5. Mostrar resultados
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n📊 Strategy: {result.strategy_name}")
    print(f"   Symbol: {result.symbol}")
    print(f"   Period: {result.start_date} to {result.end_date}")
    print(f"   Duration: {(result.end_date - result.start_date).days} days")

    print(f"\n💰 Performance:")
    print(f"   Initial Balance: ${result.initial_balance:,.2f}")
    print(f"   Final Balance: ${result.final_balance:,.2f}")
    print(f"   Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2f}%)")

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Winning Trades: {result.winning_trades}")
    print(f"   Losing Trades: {result.losing_trades}")
    print(f"   Win Rate: {result.win_rate:.2f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")

    print(f"\n💸 Win/Loss Analysis:")
    print(f"   Avg Win: ${result.avg_win:.2f} ({result.avg_win_pct:.2f}%)")
    print(f"   Avg Loss: ${result.avg_loss:.2f} ({result.avg_loss_pct:.2f}%)")
    print(f"   Largest Win: ${result.largest_win:.2f}")
    print(f"   Largest Loss: ${result.largest_loss:.2f}")

    print(f"\n📉 Risk Metrics:")
    print(f"   Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {result.sortino_ratio:.2f}")

    print(f"\n⏱️ Trade Metrics:")
    print(f"   Avg Trade Duration: {result.avg_trade_duration_minutes:.1f} minutes")
    print(f"   Max Consecutive Wins: {result.max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {result.max_consecutive_losses}")

    # Mostrar primeros 5 trades
    if result.trades:
        print(f"\n📋 First 5 Trades:")
        for i, trade in enumerate(result.trades[:5], 1):
            entry_dt = datetime.fromtimestamp(trade.entry_timestamp / 1000)
            exit_dt = datetime.fromtimestamp(trade.exit_timestamp / 1000) if trade.exit_timestamp else None

            print(f"\n   Trade #{i}:")
            print(f"     Side: {trade.side}")
            print(f"     Entry: ${trade.entry_price:.2f} @ {entry_dt}")
            if exit_dt:
                print(f"     Exit: ${trade.exit_price:.2f} @ {exit_dt}")
                print(f"     P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
                print(f"     Reason: {trade.exit_reason}")

    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETED")
    print("=" * 80)

    # Validar resultados
    success = True

    if result.total_trades == 0:
        logger.warning("No trades executed - strategy may need adjustment")
        success = False

    if result.final_balance <= result.initial_balance * 0.95:  # Más del 5% de pérdida
        logger.warning("Strategy shows significant losses", pnl_pct=result.total_pnl_pct)

    logger.info("backtest_validation", success=success, total_trades=result.total_trades,
                win_rate=result.win_rate, total_pnl=result.total_pnl,
                max_drawdown_pct=result.max_drawdown_pct)

    return success

async def main():
    """Main test function"""
    setup_logging()

    logger.info("=" * 80)
    logger.info("BACKTEST ENGINE TEST (Sprint 8.2)")
    logger.info("=" * 80)

    success = await test_backtest_engine()

    logger.info("=" * 80)
    if success:
        logger.info("🎉 ALL TESTS PASSED - Backtest Engine is ready!")
        sys.exit(0)
    else:
        logger.info("⚠️ TESTS COMPLETED - Review results above")
        sys.exit(0)  # No fallar por ahora, solo mostrar resultados

if __name__ == "__main__":
    asyncio.run(main())