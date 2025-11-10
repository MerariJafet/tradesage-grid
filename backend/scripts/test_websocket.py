import asyncio
from app.core.ws_manager import WebSocketManager
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test")

async def main():
    symbols = ["BTCUSDT", "ETHUSDT"]
    manager = WebSocketManager(symbols=symbols)

    try:
        logger.info("Starting WebSocket test...")

        # Iniciar en background
        task = asyncio.create_task(manager.start())

        # Dejar correr por 60 segundos
        await asyncio.sleep(60)

        # Obtener stats
        status = await manager.get_status()
        logger.info("ws_status", status=status)

        # Detener
        await manager.stop()

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())