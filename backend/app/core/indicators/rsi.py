from typing import Dict, List, Optional
from collections import deque
import statistics
from app.utils.logger import get_logger

logger = get_logger("rsi")

class RSI:
    """
    Relative Strength Index - indicador de momentum
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """

    def __init__(self, symbol: str, period: int = 14):
        self.symbol = symbol
        self.period = period
        self.prices: deque = deque(maxlen=period + 1)
        self.gains: deque = deque(maxlen=period)
        self.losses: deque = deque(maxlen=period)
        self.last_value: Optional[float] = None
        self.is_ready = False

    def update(self, bar: Dict) -> Optional[float]:
        """
        Actualizar RSI con nueva barra

        Args:
            bar: Dict con key 'close'

        Returns:
            RSI value (0-100) o None si no está listo
        """
        try:
            close = float(bar['close'])
            self.prices.append(close)

            if len(self.prices) < 2:
                return None

            # Calcular cambio de precio
            prev_close = self.prices[-2]
            change = close - prev_close

            # Calcular gain/loss
            gain = max(change, 0)
            loss = max(-change, 0)

            self.gains.append(gain)
            self.losses.append(loss)

            if len(self.gains) >= self.period:
                # Calcular promedio de ganancias y pérdidas
                avg_gain = statistics.mean(self.gains)
                avg_loss = statistics.mean(self.losses)

                if avg_loss == 0:
                    # Evitar división por cero
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                self.last_value = rsi
                self.is_ready = True
                return rsi
            else:
                self.is_ready = False
                return None

        except Exception as e:
            logger.error("rsi_update_error", symbol=self.symbol, error=str(e))
            return None

    def is_overbought(self, threshold: float = 70) -> bool:
        """Verificar si RSI indica sobrecompra"""
        return self.is_ready and self.last_value >= threshold

    def is_oversold(self, threshold: float = 30) -> bool:
        """Verificar si RSI indica sobreventa"""
        return self.is_ready and self.last_value <= threshold

    def is_extreme(self, overbought: float = 80, oversold: float = 20) -> Optional[str]:
        """
        Verificar si RSI está en extremos

        Returns:
            'overbought', 'oversold', or None
        """
        if not self.is_ready:
            return None

        if self.last_value >= overbought:
            return 'overbought'
        elif self.last_value <= oversold:
            return 'oversold'
        else:
            return None

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "period": self.period,
            "is_ready": self.is_ready,
            "last_value": self.last_value,
            "buffer_size": len(self.gains)
        }

    def reset(self):
        """Resetear indicador"""
        self.prices.clear()
        self.gains.clear()
        self.losses.clear()
        self.last_value = None
        self.is_ready = False