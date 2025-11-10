from collections import deque
from typing import Dict, Optional

from app.utils.logger import get_logger

logger = get_logger("adx")


class ADX:
    """Average Directional Index with Wilder smoothing."""

    def __init__(self, symbol: str, period: int = 14):
        self.symbol = symbol
        self.period = period

        self.last_value: Optional[float] = None
        self.plus_di: Optional[float] = None
        self.minus_di: Optional[float] = None
        self.is_ready = False

        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._smoothed_tr: Optional[float] = None
        self._smoothed_plus_dm: Optional[float] = None
        self._smoothed_minus_dm: Optional[float] = None

        self._dx_seed = deque(maxlen=period)
        self._adx_initialized = False

        self._tr_seed = deque(maxlen=period)
        self._plus_dm_seed = deque(maxlen=period)
        self._minus_dm_seed = deque(maxlen=period)

    def update(self, bar: Dict) -> Optional[float]:
        try:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("adx_invalid_bar", symbol=self.symbol, error=str(exc))
            return None

        if self._prev_high is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = close
            return None

        up_move = high - self._prev_high
        down_move = self._prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        true_range = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )

        if self._smoothed_tr is None:
            self._tr_seed.append(true_range)
            self._plus_dm_seed.append(plus_dm)
            self._minus_dm_seed.append(minus_dm)

            if len(self._tr_seed) < self.period:
                self._prev_high = high
                self._prev_low = low
                self._prev_close = close
                return None

            self._smoothed_tr = sum(self._tr_seed)
            self._smoothed_plus_dm = sum(self._plus_dm_seed)
            self._smoothed_minus_dm = sum(self._minus_dm_seed)
        else:
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + true_range
            self._smoothed_plus_dm = (
                self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            )
            self._smoothed_minus_dm = (
                self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm
            )

        plus_di = 0.0
        minus_di = 0.0
        if self._smoothed_tr and self._smoothed_tr > 0:
            plus_di = (self._smoothed_plus_dm / self._smoothed_tr) * 100.0
            minus_di = (self._smoothed_minus_dm / self._smoothed_tr) * 100.0

        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum > 0 else 0.0

        value: Optional[float] = None
        if not self._adx_initialized:
            self._dx_seed.append(dx)
            if len(self._dx_seed) == self.period:
                value = sum(self._dx_seed) / self.period
                self.last_value = value
                self._adx_initialized = True
        else:
            previous_adx = self.last_value or 0.0
            self.last_value = ((previous_adx * (self.period - 1)) + dx) / self.period
            value = self.last_value

        if value is not None:
            self.plus_di = plus_di
            self.minus_di = minus_di
            self.is_ready = True

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        return value

    def get_state(self) -> Dict:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "is_ready": self.is_ready,
            "adx": self.last_value,
            "plus_di": self.plus_di,
            "minus_di": self.minus_di,
        }

    def reset(self) -> None:
        self.last_value = None
        self.plus_di = None
        self.minus_di = None
        self.is_ready = False

        self._prev_high = None
        self._prev_low = None
        self._prev_close = None

        self._smoothed_tr = None
        self._smoothed_plus_dm = None
        self._smoothed_minus_dm = None

        self._dx_seed.clear()
        self._adx_initialized = False

        self._tr_seed.clear()
        self._plus_dm_seed.clear()
        self._minus_dm_seed.clear()
