from typing import Dict, List, Optional
from collections import deque
from datetime import datetime, date
from app.utils.logger import get_logger

logger = get_logger("vwap")

class VWAP:
    """
    Volume Weighted Average Price
    VWAP = sum(price * volume) / sum(volume)

    Se resetea diariamente a las 00:00 UTC
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_date: Optional[date] = None
        self.price_volume_sum = 0.0
        self.volume_sum = 0.0
        self.last_value: Optional[float] = None
        self.is_ready = False

    def update(self, bar: Dict) -> Optional[float]:
        """
        Actualizar VWAP con nueva barra

        Args:
            bar: Dict con keys 'timestamp', 'high', 'low', 'close', 'volume'

        Returns:
            VWAP value o None si no hay volumen
        """
        try:
            # Extraer datos de la barra
            timestamp = bar['timestamp']
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            volume = float(bar['volume'])

            # Determinar fecha
            if isinstance(timestamp, str):
                bar_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
            elif isinstance(timestamp, datetime):
                bar_date = timestamp.date()
            else:
                logger.warning("invalid_timestamp_format", timestamp=timestamp)
                return self.last_value

            # Reset diario
            if self.current_date != bar_date:
                self._reset_daily()
                self.current_date = bar_date

            # Calcular precio típico (OHLC/4)
            typical_price = (high + low + close + close) / 4

            # Actualizar acumuladores
            self.price_volume_sum += typical_price * volume
            self.volume_sum += volume

            # Calcular VWAP
            if self.volume_sum > 0:
                self.last_value = self.price_volume_sum / self.volume_sum
                self.is_ready = True
                return self.last_value
            else:
                return self.last_value

        except Exception as e:
            logger.error("vwap_update_error", symbol=self.symbol, error=str(e))
            return self.last_value

    def _reset_daily(self):
        """Resetear acumuladores para nuevo día"""
        self.price_volume_sum = 0.0
        self.volume_sum = 0.0
        self.last_value = None
        self.is_ready = False

        logger.debug("vwap_daily_reset", symbol=self.symbol, date=self.current_date)

    def get_deviation_pct(self, current_price: float) -> Optional[float]:
        """
        Calcular desviación porcentual del precio actual vs VWAP

        Returns:
            Porcentaje positivo si precio > VWAP, negativo si precio < VWAP
        """
        if not self.is_ready or self.last_value is None:
            return None

        return ((current_price - self.last_value) / self.last_value) * 100

    def is_above_vwap(self, price: float) -> bool:
        """Verificar si precio está por encima del VWAP"""
        return self.is_ready and price > self.last_value

    def is_below_vwap(self, price: float) -> bool:
        """Verificar si precio está por debajo del VWAP"""
        return self.is_ready and price < self.last_value

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "is_ready": self.is_ready,
            "last_value": self.last_value,
            "current_date": str(self.current_date) if self.current_date else None,
            "price_volume_sum": self.price_volume_sum,
            "volume_sum": self.volume_sum
        }

    def reset(self):
        """Resetear indicador"""
        self.current_date = None
        self.price_volume_sum = 0.0
        self.volume_sum = 0.0
        self.last_value = None
        self.is_ready = False