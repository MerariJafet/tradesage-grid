from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics
from app.utils.logger import get_logger

logger = get_logger("obi")

class OBI:
    """
    Order Book Imbalance
    Mide el desequilibrio entre bids y asks en el order book

    OBI = (Bid Volume) / (Bid Volume + Ask Volume)
    Rango: 0.0 (solo asks) a 1.0 (solo bids)
    """

    def __init__(self, symbol: str, history_size: int = 100):
        self.symbol = symbol
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.last_value: Optional[float] = None
        self.is_ready = False

    def calculate(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
                  depth: int = 5) -> float:
        """
        Calcular OBI para un depth específico

        Args:
            bids: Lista de tuplas (price, quantity) - ordenadas descendente
            asks: Lista de tuplas (price, quantity) - ordenadas ascendente
            depth: Número de niveles a considerar (default: 5)

        Returns:
            OBI value entre 0.0 y 1.0
        """
        try:
            # Tomar solo los primeros 'depth' niveles
            bids_slice = bids[:depth]
            asks_slice = asks[:depth]

            # Calcular volumen total de bids y asks
            bid_volume = sum(quantity for _, quantity in bids_slice)
            ask_volume = sum(quantity for _, quantity in asks_slice)
            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                obi = 0.5  # Neutral si no hay volumen
            else:
                obi = bid_volume / total_volume

            # Guardar en historial
            self.history.append(obi)
            self.last_value = obi
            self.is_ready = len(self.history) >= 10  # Ready después de 10 muestras

            return obi

        except Exception as e:
            logger.error("obi_calculate_error", symbol=self.symbol, error=str(e))
            return 0.5  # Retornar neutral en caso de error

    def is_buy_pressure(self, threshold: float = 0.62) -> bool:
        """
        Verificar si hay presión compradora fuerte

        Args:
            threshold: Umbral mínimo para considerar presión compradora (default: 0.62)
        """
        return self.is_ready and self.last_value >= threshold

    def is_sell_pressure(self, threshold: float = 0.38) -> bool:
        """
        Verificar si hay presión vendedora fuerte

        Args:
            threshold: Umbral máximo para considerar presión vendedora (default: 0.38)
        """
        return self.is_ready and self.last_value <= threshold

    def is_neutral(self, tolerance: float = 0.05) -> bool:
        """
        Verificar si el order book está balanceado

        Args:
            tolerance: Tolerancia alrededor de 0.5 (default: 0.05 = 5%)
        """
        if not self.is_ready:
            return True  # Considerar neutral si no está listo

        return abs(self.last_value - 0.5) <= tolerance

    def get_trend(self, window: int = 10) -> str:
        """
        Analizar tendencia del OBI en ventana reciente

        Args:
            window: Número de muestras recientes a analizar

        Returns:
            'increasing', 'decreasing', 'stable'
        """
        if len(self.history) < window:
            return 'stable'

        recent = list(self.history)[-window:]

        if len(recent) < 2:
            return 'stable'

        # Calcular pendiente usando regresión lineal simple
        x = list(range(len(recent)))
        y = recent

        try:
            slope = statistics.linear_regression(x, y).slope

            if slope > 0.001:  # Tendencia positiva
                return 'increasing'
            elif slope < -0.001:  # Tendencia negativa
                return 'decreasing'
            else:
                return 'stable'

        except Exception as e:
            logger.error("obi_trend_error", symbol=self.symbol, error=str(e))
            return 'stable'

    def get_volatility(self, window: int = 20) -> Optional[float]:
        """
        Calcular volatilidad del OBI (desviación estándar)

        Args:
            window: Ventana para calcular volatilidad
        """
        if len(self.history) < window:
            return None

        recent = list(self.history)[-window:]
        return statistics.stdev(recent) if len(recent) > 1 else 0.0

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "is_ready": self.is_ready,
            "last_value": self.last_value,
            "history_size": len(self.history),
            "trend": self.get_trend(),
            "volatility": self.get_volatility(),
            "is_buy_pressure": self.is_buy_pressure(),
            "is_sell_pressure": self.is_sell_pressure()
        }

    def reset(self):
        """Resetear indicador"""
        self.history.clear()
        self.last_value = None
        self.is_ready = False