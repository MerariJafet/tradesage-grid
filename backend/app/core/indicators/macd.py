from typing import Dict, List, Optional
from collections import deque
import statistics
from app.utils.logger import get_logger

logger = get_logger("macd")

class MACD:
    """
    Moving Average Convergence Divergence
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line
    """

    def __init__(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9):
        self.symbol = symbol
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal

        # Buffers para EMAs
        self.prices: deque = deque(maxlen=slow + signal)  # Necesitamos suficientes datos
        self.fast_ema: Optional[float] = None
        self.slow_ema: Optional[float] = None
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.histogram: Optional[float] = None
        self.last_value: Optional[float] = None
        self.is_ready = False

        # Multiplicadores para EMA
        self.fast_multiplier = 2 / (fast + 1)
        self.slow_multiplier = 2 / (slow + 1)
        self.signal_multiplier = 2 / (signal + 1)

    def _calculate_ema(self, current_ema: float, price: float, multiplier: float) -> float:
        """Calcular EMA recursivamente"""
        if current_ema is None:
            return price
        return (price * multiplier) + (current_ema * (1 - multiplier))

    def update(self, bar: Dict) -> Optional[Dict]:
        """
        Actualizar MACD con nueva barra

        Args:
            bar: Dict con key 'close'

        Returns:
            Dict con macd/signal/histogram o None si no está listo
        """
        try:
            close = float(bar['close'])
            self.prices.append(close)

            # Calcular EMAs
            self.fast_ema = self._calculate_ema(self.fast_ema, close, self.fast_multiplier)
            self.slow_ema = self._calculate_ema(self.slow_ema, close, self.slow_multiplier)

            # Calcular MACD Line
            if self.fast_ema is not None and self.slow_ema is not None:
                self.macd_line = self.fast_ema - self.slow_ema

                # Calcular Signal Line (EMA del MACD)
                self.signal_line = self._calculate_ema(
                    self.signal_line, self.macd_line, self.signal_multiplier
                )

                # Calcular Histogram
                if self.signal_line is not None:
                    self.histogram = self.macd_line - self.signal_line
                    self.last_value = self.macd_line
                    self.is_ready = True

                    return {
                        'macd_line': self.macd_line,
                        'signal_line': self.signal_line,
                        'histogram': self.histogram
                    }

            return None

        except Exception as e:
            logger.error("macd_update_error", symbol=self.symbol, error=str(e))
            return None

    def is_bullish_crossover(self, prev_histogram: Optional[float] = None) -> bool:
        """
        Verificar si hay cruce alcista (MACD cruza por encima de Signal)

        Args:
            prev_histogram: Valor anterior del histogram para comparación
        """
        if not self.is_ready or prev_histogram is None:
            return False

        # Cruce alcista: histogram cambia de negativo a positivo
        return prev_histogram <= 0 and self.histogram > 0

    def is_bearish_crossover(self, prev_histogram: Optional[float] = None) -> bool:
        """
        Verificar si hay cruce bajista (MACD cruza por debajo de Signal)
        """
        if not self.is_ready or prev_histogram is None:
            return False

        # Cruce bajista: histogram cambia de positivo a negativo
        return prev_histogram >= 0 and self.histogram < 0

    def get_momentum(self) -> str:
        """
        Obtener dirección del momentum

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if not self.is_ready:
            return 'neutral'

        if self.histogram > 0:
            return 'bullish'
        elif self.histogram < 0:
            return 'bearish'
        else:
            return 'neutral'

    def get_state(self) -> Dict:
        """Obtener estado completo del indicador"""
        return {
            "symbol": self.symbol,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "is_ready": self.is_ready,
            "macd_line": self.macd_line,
            "signal_line": self.signal_line,
            "histogram": self.histogram,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "buffer_size": len(self.prices)
        }

    def reset(self):
        """Resetear indicador"""
        self.prices.clear()
        self.fast_ema = None
        self.slow_ema = None
        self.macd_line = None
        self.signal_line = None
        self.histogram = None
        self.last_value = None
        self.is_ready = False