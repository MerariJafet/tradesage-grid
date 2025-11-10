from typing import Dict, List, Optional
from collections import deque
import statistics
from app.utils.logger import get_logger

logger = get_logger("atr")

class ATR:
    """
    Average True Range - mide volatilidad
    ATR = media móvil del True Range
    """

    def __init__(self, symbol: str, period: int = 14):
        self.symbol = symbol
        self.period = period
        self.prices: deque = deque(maxlen=period + 1)  # Necesitamos period + 1 para calcular TR
        self.tr_values: deque = deque(maxlen=period)
        self.last_value: Optional[float] = None
        self.is_ready = False

    def update(self, bar: Dict) -> Optional[float]:
        """
        Actualizar ATR con nueva barra

        Args:
            bar: Dict con keys 'high', 'low', 'close'

        Returns:
            ATR value o None si no está listo
        """
        try:
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])

            # Calcular True Range
            if len(self.prices) == 0:
                # Primera barra
                tr = high - low
            else:
                prev_close = self.prices[-1]['close']
                tr = max(
                    high - low,  # Range actual
                    abs(high - prev_close),  # High vs prev close
                    abs(low - prev_close)    # Low vs prev close
                )

            # Guardar precios para siguiente cálculo
            self.prices.append({
                'high': high,
                'low': low,
                'close': close
            })

            # Calcular ATR
            self.tr_values.append(tr)

            if len(self.tr_values) >= self.period:
                self.last_value = statistics.mean(self.tr_values)
                self.is_ready = True
                return self.last_value
            else:
                self.is_ready = False
                return None

        except Exception as e:
            logger.error("atr_update_error", symbol=self.symbol, error=str(e))
            return None

    def get_atr_multiple(self, multiplier: float) -> Optional[float]:
        """Obtener ATR multiplicado (ej: 1.5x ATR para stop loss)"""
        if self.last_value is None:
            return None
        return self.last_value * multiplier

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "period": self.period,
            "is_ready": self.is_ready,
            "last_value": self.last_value,
            "buffer_size": len(self.tr_values)
        }

    def reset(self):
        """Resetear indicador"""
        self.prices.clear()
        self.tr_values.clear()
        self.last_value = None
        self.is_ready = False