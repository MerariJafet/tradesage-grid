from typing import Dict, List, Optional
from collections import deque
import statistics
from app.utils.logger import get_logger

logger = get_logger("bollinger_bands")

class BollingerBands:
    """
    Bollinger Bands - bandas de volatilidad alrededor de la media móvil
    Upper = SMA + (std_dev * num_std)
    Middle = SMA
    Lower = SMA - (std_dev * num_std)
    """

    def __init__(self, symbol: str, period: int = 20, num_std: float = 2.0):
        self.symbol = symbol
        self.period = period
        self.num_std = num_std
        self.prices: deque = deque(maxlen=period)
        self.upper: Optional[float] = None
        self.middle: Optional[float] = None
        self.lower: Optional[float] = None
        self.bandwidth: float = 0.0
        self.last_value: Optional[float] = None
        self.is_ready = False

    def update(self, bar: Dict) -> Optional[Dict]:
        """
        Actualizar Bollinger Bands con nueva barra

        Args:
            bar: Dict con key 'close'

        Returns:
            Dict con upper/middle/lower o None si no está listo
        """
        try:
            close = float(bar['close'])
            self.prices.append(close)

            if len(self.prices) >= self.period:
                # Calcular estadísticas
                prices_list = list(self.prices)
                sma = statistics.mean(prices_list)
                std_dev = statistics.stdev(prices_list)

                self.upper = sma + (std_dev * self.num_std)
                self.middle = sma
                self.lower = sma - (std_dev * self.num_std)

                # Calcular bandwidth (%)
                if sma > 0:
                    self.bandwidth = (self.upper - self.lower) / sma

                self.last_value = self.middle
                self.is_ready = True

                return {
                    'upper': self.upper,
                    'middle': self.middle,
                    'lower': self.lower,
                    'bandwidth': self.bandwidth
                }
            else:
                self.is_ready = False
                return None

        except Exception as e:
            logger.error("bb_update_error", symbol=self.symbol, error=str(e))
            return None

    def get_position(self, price: float) -> str:
        """
        Obtener posición del precio respecto a las bandas

        Returns:
            'above_upper', 'above_middle', 'inside', 'below_middle', 'below_lower'
        """
        if not self.is_ready:
            return 'unknown'

        if price >= self.upper:
            return 'above_upper'
        elif price >= self.middle:
            return 'above_middle'
        elif price >= self.lower:
            return 'inside'
        elif price >= self.lower - (self.middle - self.lower):
            return 'below_middle'
        else:
            return 'below_lower'

    def is_compressed(self, threshold: float = 0.02) -> bool:
        """
        Verificar si las bandas están comprimidas (baja volatilidad)

        Args:
            threshold: Bandwidth máximo para considerar comprimido (2% por defecto)
        """
        return self.is_ready and self.bandwidth <= threshold

    def is_expanded(self, threshold: float = 0.05) -> bool:
        """
        Verificar si las bandas están expandidas (alta volatilidad)
        """
        return self.is_ready and self.bandwidth >= threshold

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "period": self.period,
            "num_std": self.num_std,
            "is_ready": self.is_ready,
            "upper": self.upper,
            "middle": self.middle,
            "lower": self.lower,
            "bandwidth": self.bandwidth,
            "buffer_size": len(self.prices)
        }

    def reset(self):
        """Resetear indicador"""
        self.prices.clear()
        self.upper = None
        self.middle = None
        self.lower = None
        self.bandwidth = 0.0
        self.last_value = None
        self.is_ready = False