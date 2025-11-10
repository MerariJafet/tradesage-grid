#!/usr/bin/env python3
"""
Test script para estrategias Sprint 4
Ejecuta pruebas manuales de estrategias con datos simulados
"""

import sys
import os
from datetime import datetime, timedelta

# Añadir el directorio backend al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.ws_manager import WebSocketManager
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_strategy")

async def main():
    """Test manual de estrategias con datos reales"""

    symbols = ["BTCUSDT"]
    manager = WebSocketManager(symbols=symbols)

    try:
        logger.info("=" * 80)
        logger.info("SPRINT 4 STRATEGY TEST - Breakout Compression Strategy")
        logger.info("=" * 80)

        # Iniciar WebSocket manager en background
        logger.info("Starting WebSocket Manager...")
        task = asyncio.create_task(manager.start())

        # Esperar 3 minutos para acumular datos
        logger.info("Waiting 180 seconds for data accumulation...")
        logger.info("Indicators need ~30 bars to be ready (ATR=14, BB=20)")

        for i in range(18):  # 18 x 10 seg = 180 seg
            await asyncio.sleep(10)
            elapsed = (i + 1) * 10
            logger.info(f"Elapsed: {elapsed}s / 180s")

            # Obtener estado cada 30 segundos
            if (i + 1) % 3 == 0:
                status = await manager.get_status()

                # Estado de indicadores
                for symbol in symbols:
                    ind_status = status['indicators'].get(symbol, {})
                    logger.info(
                        f"\n{'=' * 60}\n"
                        f"Symbol: {symbol}\n"
                        f"Indicators Ready: {ind_status.get('ready', False)}\n"
                        f"Values: {ind_status.get('values', {})}\n"
                        f"Signals: {ind_status.get('signals', {})}\n"
                    )

                # Estado de estrategias
                strategy_stats = status.get('strategies', {})
                logger.info(
                    f"\n{'=' * 60}\n"
                    f"STRATEGY STATISTICS\n"
                    f"{'=' * 60}\n"
                )

                for key, stats in strategy_stats.items():
                    if key != 'global':
                        logger.info(
                            f"\nStrategy: {key}\n"
                            f"  Enabled: {stats.get('enabled')}\n"
                            f"  Signals Generated: {stats.get('signals_generated', 0)}\n"
                            f"  Signals Executed: {stats.get('signals_executed', 0)}\n"
                            f"  Total Trades: {stats.get('total_trades', 0)}\n"
                            f"  Win Rate: {stats.get('win_rate', 0):.2f}%\n"
                            f"  Has Open Position: {stats.get('has_open_position', False)}\n"
                        )

                global_stats = strategy_stats.get('global', {})
                logger.info(
                    f"\nGlobal Stats:\n"
                    f"  Total Strategies: {global_stats.get('total_strategies', 0)}\n"
                    f"  Enabled: {global_stats.get('enabled_strategies', 0)}\n"
                    f"  Open Positions: {global_stats.get('total_open_positions', 0)}\n"
                    f"  Account Balance: ${global_stats.get('account_balance', 0):.2f}\n"
                )

        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETED - Final Summary")
        logger.info("=" * 80)

        # Resumen final
        final_status = await manager.get_status()

        logger.info("\n📊 FINAL RESULTS:")

        # Data stats
        writer_stats = final_status.get('writer_stats', {})
        logger.info(
            f"\nData Processing:\n"
            f"  ✅ Ticks Written: {writer_stats.get('ticks_written', 0)}\n"
            f"  ✅ Bars Written: {writer_stats.get('bars_written', 0)}\n"
            f"  ✅ Orderbooks Written: {writer_stats.get('orderbooks_written', 0)}\n"
        )

        # Indicator stats
        for symbol in symbols:
            ind_status = final_status['indicators'].get(symbol, {})
            logger.info(
                f"\nIndicators ({symbol}):\n"
                f"  ✅ Ready: {ind_status.get('ready', False)}\n"
            )

            if ind_status.get('ready'):
                values = ind_status.get('values', {})
                logger.info(
                    f"  ATR: {values.get('atr', 0):.2f}\n"
                    f"  BB Bandwidth: {values.get('bb_bandwidth', 0):.6f}\n"
                    f"  RSI(2): {values.get('rsi_2', 0):.2f}\n"
                    f"  RSI(14): {values.get('rsi_14', 0):.2f}\n"
                    f"  VWAP: {values.get('vwap', 0):.2f}\n"
                )

        # Strategy stats
        strategy_stats = final_status.get('strategies', {})

        for key, stats in strategy_stats.items():
            if key != 'global':
                signals_generated = stats.get('signals_generated', 0)
                signals_executed = stats.get('signals_executed', 0)

                logger.info(
                    f"\nStrategy: {key}\n"
                    f"  {'✅' if signals_generated > 0 else '⚠️'} Signals Generated: {signals_generated}\n"
                    f"  {'✅' if signals_executed > 0 else 'ℹ️'} Signals Executed: {signals_executed}\n"
                    f"  Total Trades: {stats.get('total_trades', 0)}\n"
                    f"  Win Rate: {stats.get('win_rate', 0):.2f}%\n"
                )

                if signals_generated == 0:
                    logger.info(
                        f"  ℹ️ No signals generated - Market conditions may not meet strategy criteria\n"
                        f"     (BB compression, volume spike, RSI neutral, etc.)"
                    )

        # Detener
        logger.info("\nStopping WebSocket Manager...")
        await manager.stop()

        logger.info("\n" + "=" * 80)
        logger.info("✅ SPRINT 4 TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        await manager.stop()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        await manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())