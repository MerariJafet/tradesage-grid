from app.core.exchanges.base_ws_client import BaseWSClient
from app.core.exchanges.binance.models import (
    BinanceAggTradeMessage,
    BinanceKlineMessage,
    BinanceDepthMessage,
    Tick,
    Bar,
    OrderBookSnapshot
)
from app.config import settings
from app.utils.logger import get_logger
from app.core.monitoring.latency_tracker import latency_tracker
from datetime import datetime, timezone
import time

class BinanceWSClient(BaseWSClient):
    def __init__(self, market_type: str = "spot", symbols: list[str] = None):
        """
        market_type: 'spot' o 'futures'
        symbols: lista de símbolos, ej: ['btcusdt', 'ethusdt']
        """
        self.market_type = market_type
        self.symbols = symbols or []

        # URLs según testnet/mainnet y spot/futures
        if settings.BINANCE_TESTNET:
            if market_type == "spot":
                url = "wss://testnet.binance.vision/ws"
            else:
                url = "wss://stream.binancefuture.com/ws"  # Testnet futures
        else:
            if market_type == "spot":
                url = "wss://stream.binance.com:9443/ws"
            else:
                url = "wss://fstream.binance.com/ws"

        # Construir streams
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            streams.extend([
                f"{symbol_lower}@aggTrade",
                f"{symbol_lower}@kline_1m",
                f"{symbol_lower}@depth@100ms"
            ])

        # Añadir markPrice para futures
        if market_type == "futures":
            for symbol in self.symbols:
                streams.append(f"{symbol.lower()}@markPrice@1s")

        super().__init__(url, streams)
        self.logger = get_logger("binance_ws").bind(market_type=market_type, symbols=symbols)

    async def parse_message(self, message: dict):
        """Parser específico para mensajes de Binance"""
        try:
            event_type = message.get("e")

            if event_type == "aggTrade":
                await self._handle_agg_trade(message)
            elif event_type == "kline":
                await self._handle_kline(message)
            elif event_type == "depthUpdate":
                await self._handle_depth_update(message)
            elif event_type == "markPriceUpdate":
                await self._handle_mark_price(message)
            else:
                self.logger.debug("unknown_event_type", event_type=event_type)

        except Exception as e:
            self.logger.error("parse_error", error=str(e), message=message)

    async def _handle_agg_trade(self, message: dict):
        """Procesar aggTrade y convertir a Tick"""
        try:
            raw = BinanceAggTradeMessage(**message)

            # Calcular latencia
            exchange_time_ms = raw.T
            now_ms = int(time.time() * 1000)
            latency_ms = now_ms - exchange_time_ms
            
            # Registrar en tracker
            latency_tracker.record("binance", raw.s, latency_ms)

            tick = Tick(
                timestamp=datetime.fromtimestamp(raw.T / 1000, tz=timezone.utc),
                exchange="binance",
                symbol=raw.s,
                price=float(raw.p),
                quantity=float(raw.q),
                is_buyer_maker=raw.m,
                trade_id=raw.a,
                latency_ms=latency_ms
            )

            await self.emit("tick", tick.model_dump())

        except Exception as e:
            self.logger.error("agg_trade_parse_error", error=str(e))

    async def _handle_kline(self, message: dict):
        """Procesar kline y convertir a Bar"""
        try:
            raw = BinanceKlineMessage(**message)
            kline = raw.k

            bar = Bar(
                timestamp=datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc),
                exchange="binance",
                symbol=raw.s,
                timeframe="1m",
                open=float(kline["o"]),
                high=float(kline["h"]),
                low=float(kline["l"]),
                close=float(kline["c"]),
                volume=float(kline["v"]),
                is_closed=kline["x"]  # True si vela cerrada
            )

            # Solo emitir velas cerradas para evitar ruido
            if bar.is_closed:
                await self.emit("bar", bar.model_dump())

        except Exception as e:
            self.logger.error("kline_parse_error", error=str(e))

    async def _handle_depth_update(self, message: dict):
        """Procesar depthUpdate y convertir a OrderBookSnapshot"""
        try:
            raw = BinanceDepthMessage(**message)

            # Convertir strings a floats
            bids = [(float(p), float(q)) for p, q in raw.b]
            asks = [(float(p), float(q)) for p, q in raw.a]

            snapshot = OrderBookSnapshot(
                timestamp=datetime.fromtimestamp(raw.E / 1000, tz=timezone.utc),
                exchange="binance",
                symbol=raw.s,
                bids=bids,
                asks=asks,
                last_update_id=raw.u
            )

            # Emitir con OBI calculado
            snapshot_dict = snapshot.model_dump()
            snapshot_dict["obi_5"] = snapshot.calculate_obi(depth=5)
            snapshot_dict["obi_10"] = snapshot.calculate_obi(depth=10)

            await self.emit("orderbook", snapshot_dict)

        except Exception as e:
            self.logger.error("depth_update_parse_error", error=str(e))

    async def _handle_mark_price(self, message: dict):
        """Procesar markPrice (solo futures)"""
        try:
            await self.emit("mark_price", {
                "symbol": message["s"],
                "mark_price": float(message["p"]),
                "funding_rate": float(message.get("r", 0)),
                "timestamp": datetime.fromtimestamp(message["E"] / 1000, tz=timezone.utc)
            })
        except Exception as e:
            self.logger.error("mark_price_parse_error", error=str(e))