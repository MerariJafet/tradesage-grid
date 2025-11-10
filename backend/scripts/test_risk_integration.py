import asyncio
from app.core.ws_manager import WebSocketManager
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test_risk_integration")

async def main():
    """Test integración completa con Risk Manager"""

    logger.info("=" * 80)
    logger.info("RISK MANAGER INTEGRATION TEST")
    logger.info("=" * 80)

    symbols = ["BTCUSDT"]
    manager = WebSocketManager(symbols=symbols)

    try:
        # Iniciar sistema
        logger.info("Starting WebSocket Manager with Risk Manager...")
        task = asyncio.create_task(manager.start())

        # Esperar 3 minutos monitoreando risk manager
        logger.info("Monitoring for 180 seconds...")

        for i in range(18):  # 18 x 10 seg = 180 seg
            await asyncio.sleep(10)
            elapsed = (i + 1) * 10

            # Status cada 30 segundos
            if (i + 1) % 3 == 0:
                status = await manager.get_status()

                # Paper Exchange
                paper = status.get('paper_exchange', {})
                logger.info(
                    f"\n{'=' * 60}\n"
                    f"SYSTEM STATUS (t={elapsed}s)\n"
                    f"{'=' * 60}\n"
                    f"Paper Trading:\n"
                    f"  Balance: ${paper.get('current_balance', 0):.2f}\n"
                    f"  PnL: ${paper.get('pnl', 0):.2f}\n"
                    f"  Trades: {paper.get('trades_executed', 0)}\n"
                )

                # Risk Manager
                risk = status.get('risk_manager', {})
                if risk:
                    risk_status = risk.get('status', {})
                    logger.info(
                        f"Risk Manager:\n"
                        f"  Trading Enabled: {risk_status.get('trading_enabled')}\n"
                        f"  Kill-Switch: {risk_status.get('kill_switch_active')}\n"
                        f"  Consecutive Losses: {risk_status.get('consecutive_losses')}\n"
                        f"  In Cooldown: {risk_status.get('in_cooldown')}\n"
                    )

                    period_pnl = risk.get('period_pnl', {})
                    daily = period_pnl.get('daily', {})
                    logger.info(
                        f"Daily PnL:\n"
                        f"  Amount: ${daily.get('pnl', 0):.2f}\n"
                        f"  Percent: {daily.get('pnl_pct', 0):.2f}%\n"
                        f"  Limit: {daily.get('limit_pct', 0):.2f}%\n"
                    )

                    drawdown = risk.get('drawdown', {})
                    current_dd = drawdown.get('current', {})
                    logger.info(
                        f"Drawdown:\n"
                        f"  Current: {current_dd.get('drawdown_pct', 0):.2f}%\n"
                        f"  In Drawdown: {current_dd.get('in_drawdown')}\n"
                    )

        # Resumen final
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)

        final_status = await manager.get_status()
        final_risk = final_status.get('risk_manager', {})

        if final_risk:
            logger.info("\nRisk Manager Final Stats:")
            logger.info(f"  Status: {final_risk.get('status')}")
            logger.info(f"  Balance: {final_risk.get('balance')}")
            logger.info(f"  Period PnL: {final_risk.get('period_pnl')}")
            logger.info(f"  Drawdown: {final_risk.get('drawdown')}")

            # Eventos de riesgo
            events = final_risk.get('risk_events', [])
            if events:
                logger.info(f"\nRisk Events: {len(events)}")
                for event in events:
                    logger.info(f"  - {event.get('type')}: {event.get('message')}")

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