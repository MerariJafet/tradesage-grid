import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK


def _format_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


async def stream_orderbook(symbol: str = "BTCUSDT", depth: int = 5) -> AsyncIterator[Dict[str, float]]:
    """Yield bid/ask depth metrics from Binance every ~100ms."""
    stream_name = f"{symbol.lower()}@depth@100ms"
    uri = f"wss://fstream.binance.com/ws/{stream_name}"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=15, ping_timeout=10) as websocket:
                async for message in websocket:
                    payload = json.loads(message)
                    bids = payload.get("b", [])
                    asks = payload.get("a", [])
                    bid_volume = sum(float(level[1]) for level in bids[:depth])
                    ask_volume = sum(float(level[1]) for level in asks[:depth])
                    total = bid_volume + ask_volume
                    imbalance = 0.0 if total == 0 else (bid_volume - ask_volume) / total
                    yield {
                        "bid_volume": bid_volume,
                        "ask_volume": ask_volume,
                        "imbalance": imbalance,
                    }
        except (ConnectionClosedError, ConnectionClosedOK, OSError):
            await asyncio.sleep(1)


async def run_orderbook_stream(symbol: str, duration: Optional[float], output: Optional[Path], depth: int) -> None:
    sink = output.open("a", encoding="ascii", errors="ignore") if output else None
    connection_line = f"[{_format_timestamp()}] Conectado a {symbol} depth @100ms"
    print(connection_line)
    if sink:
        sink.write(connection_line + "\n")
        sink.flush()
    start_time = time.monotonic()
    try:
        async for snapshot in stream_orderbook(symbol=symbol, depth=depth):
            line = (
                f"[{_format_timestamp()}] IMBALANCE: {snapshot['imbalance']:+.3f} | "
                f"BIDS: {snapshot['bid_volume']:.2f} | ASKS: {snapshot['ask_volume']:.2f}"
            )
            print(line)
            if sink:
                sink.write(line + "\n")
                sink.flush()
            if duration is not None and (time.monotonic() - start_time) >= duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if sink:
            sink.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Binance depth imbalance metrics.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to stream, default BTCUSDT")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to stream")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path")
    parser.add_argument("--depth", type=int, default=5, help="Depth levels to aggregate per side")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        run_orderbook_stream(
            symbol=args.symbol.upper(),
            duration=args.duration,
            output=output_path,
            depth=max(1, args.depth),
        )
    )


if __name__ == "__main__":
    main()
