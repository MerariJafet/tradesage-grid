from typing import Dict, Any
from app.core.exchanges.binance.models import Tick, Bar, OrderBookSnapshot
from app.utils.logger import get_logger

logger = get_logger("normalizer")

class DataNormalizer:
    """Normaliza datos de diferentes exchanges a formato unificado"""

    @staticmethod
    def validate_tick(tick: Dict[str, Any]) -> bool:
        """Validar que un tick tenga campos requeridos"""
        required = ["timestamp", "symbol", "price", "quantity"]
        return all(field in tick for field in required)

    @staticmethod
    def validate_bar(bar: Dict[str, Any]) -> bool:
        """Validar que una barra tenga campos requeridos"""
        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        return all(field in bar for field in required)

    @staticmethod
    def deduplicate_tick(tick: Dict[str, Any], cache: set) -> bool:
        """
        Deduplicar ticks usando cache en memoria
        Returns: True si es único, False si es duplicado
        """
        key = f"{tick['symbol']}_{tick['trade_id']}_{tick['timestamp']}"
        if key in cache:
            logger.debug("duplicate_tick_detected", key=key)
            return False
        cache.add(key)

        # Limitar tamaño del cache (últimos 10k ticks)
        if len(cache) > 10000:
            cache.pop()

        return True

    @staticmethod
    def sanitize_symbol(symbol: str, exchange: str = "binance") -> str:
        """Normalizar formato de símbolo"""
        # Binance usa BTCUSDT, otros pueden usar BTC/USDT
        return symbol.upper().replace("/", "")

    @staticmethod
    def calculate_spread(orderbook: OrderBookSnapshot) -> float:
        """Calcular spread del order book"""
        if not orderbook.bids or not orderbook.asks:
            return 0.0

        best_bid = orderbook.bids[0][0]
        best_ask = orderbook.asks[0][0]

        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100

        return spread_pct