# backend/scripts/test_paper_execution.py

import asyncio
from app.core.ws_manager import WebSocketManager
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_paper_execution")

async def main():
    """Test ejecución paper trading con señales reales"""

    symbols = ["BTCUSDT"]
    manager = WebSocketManager(symbols=symbols)

    try:
        logger.info("=" * 80)
        logger.info("PAPER EXECUTION TEST")
        logger.info("=" * 80)

        # Iniciar WebSocket manager
        logger.info("Starting WebSocket Manager with Paper Exchange...")
        task = asyncio.create_task(manager.start())

        # Esperar 3 minutos para que se generen señales y ejecuten órdenes
        logger.info("Waiting 180 seconds for signals and executions...")

        for i in range(18):  # 18 x 10 seg = 180 seg
            await asyncio.sleep(10)
            elapsed = (i + 1) * 10

            # Obtener estado cada 30 segundos
            if (i + 1) % 3 == 0:
                status = await manager.get_status()

                # Estado Paper Exchange
                paper_stats = status.get('paper_exchange', {})
                logger.info(
                    f"\n{'=' * 60}\n"
                    f"PAPER EXCHANGE STATUS (t={elapsed}s)\n"
                    f"{'=' * 60}\n"
                    f"Balance: ${paper_stats.get('current_balance', 0):.2f}\n"
                    f"PnL: ${paper_stats.get('pnl', 0):.2f} ({paper_stats.get('pnl_percent', 0):.2f}%)\n"
                    f"Trades Executed: {paper_stats.get('trades_executed', 0)}\n"
                    f"Commission Paid: ${paper_stats.get('total_commission_paid', 0):.2f}\n"
                    f"Open Positions: {paper_stats.get('open_positions', 0)}\n"
                )

                # Últimas órdenes
                recent_orders = manager.paper_exchange.get_recent_orders(limit=5)
                if recent_orders:
                    logger.info("\nRecent Orders:")
                    for order in recent_orders:
                        logger.info(
                            f"  {order.id}: {order.side} {order.filled_quantity} @ ${order.filled_price:.2f} "
                            f"[{order.status}] (slip: ${order.slippage:.2f}, comm: ${order.commission:.2f})"
                        )

        # Resumen final
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)

        final_status = await manager.get_status()
        final_paper = final_status.get('paper_exchange', {})

        logger.info(
            f"\nPaper Exchange Final Stats:\n"
            f"  Initial Balance: ${final_paper.get('initial_balance', 0):.2f}\n"
            f"  Final Balance: ${final_paper.get('current_balance', 0):.2f}\n"
            f"  Total PnL: ${final_paper.get('pnl', 0):.2f}\n"
            f"  PnL %: {final_paper.get('pnl_percent', 0):.2f}%\n"
            f"  Trades: {final_paper.get('trades_executed', 0)}\n"
            f"  Commission: ${final_paper.get('total_commission_paid', 0):.2f}\n"
        )

        # Detener
        await manager.stop()

        logger.info("✅ TEST COMPLETED")

    except KeyboardInterrupt:
        logger.info("Test interrupted")
        await manager.stop()
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())