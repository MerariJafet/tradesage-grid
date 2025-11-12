import asyncio
import json
from pathlib import Path

import websocket

try:
    from trading_manager.live_trading_manager import LiveTradingManager
except ModuleNotFoundError:
    import sys

    BASE_DIR = Path(__file__).resolve().parents[1]
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))
    from trading_manager.live_trading_manager import LiveTradingManager


manager = LiveTradingManager()


async def dispatch_trade(data: dict) -> None:
    trade = {
        "symbol": data.get("s", "UNKNOWN"),
        "price": float(data.get("p", 0.0)),
        "qty": float(data.get("q", 0.0)),
        "timestamp": data.get("T"),
    }
    await manager.process_trade(trade)


def on_message(ws, message):
    data = json.loads(message)
    asyncio.run(dispatch_trade(data))


def on_error(ws, error):
    print("[ERROR]", error)


def on_close(ws, close_status_code, close_msg):
    print("[CLOSE]", close_status_code, close_msg)


def on_open(ws):
    print("[OPEN] Connected to Binance stream")


if __name__ == "__main__":
    socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(
        socket,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
