from collections import deque
from typing import Dict, Optional

from app.core.strategies.base import BaseStrategy
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.strategies.position import PositionStatus
from app.utils.logger import get_logger

logger = get_logger("strategy.momentum_scalping")


class MomentumScalpingStrategy(BaseStrategy):
    """Momentum scalping strategy for BTCUSDT short-term trades."""

    MIN_CONFIDENCE = 0.65

    def __init__(
        self,
        symbol: str,
        indicator_manager: IndicatorManager,
        position_sizer: PositionSizer,
        signal_validator: SignalValidator,
        execution_engine=None,
        risk_manager=None,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        adx_period: int = 14,
        adx_threshold: float = 30.0,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        rsi_period: int = 7,
        rsi_momentum: float = 55.0,
        rsi_delta: float = 5.0,
        rsi_overbought: float = 65.0,
        rsi_oversold: float = 35.0,
        bollinger_window: int = 20,
        volume_sma_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 10,
        stop_multiplier: float = 2.0,
        atr_take_profit_multiplier: float = 3.0,
        profit_target_pct: float = 0.005,
        stop_loss_pct: float = 0.003,
        enabled: bool = True,
    ):
        super().__init__(
            name="MomentumScalping",
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator,
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            enabled=enabled,
        )

        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.rsi_period = rsi_period
        self.rsi_momentum = rsi_momentum
        self.rsi_delta = rsi_delta
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bollinger_window = bollinger_window
        self.volume_sma_period = volume_sma_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.stop_multiplier = stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct

        self.volume_history = deque(maxlen=volume_sma_period)
        self.prev_macd_histogram: Optional[float] = None
        self.prev_rsi: Optional[float] = None
        self.cooldown_bars = max(5, self.rsi_period)
        self.bars_since_last_signal = self.cooldown_bars
        self.last_signal_side: Optional[str] = None

        self.min_periods = max(
            self.macd_slow,
            self.adx_period,
            self.rsi_period,
            self.volume_sma_period,
            self.atr_period,
            self.bollinger_window,
        )

        self.logger.info(
            "momentum_scalping_initialized",
            symbol=symbol,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            adx_threshold=adx_threshold,
            rsi_momentum=rsi_momentum,
            volume_multiplier=volume_multiplier,
        )

    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        bar = market_data.get("bar", {})
        if not bar:
            return None

        self.logger.debug(
            "momentum_generate_signal_called",
            symbol=self.symbol,
            bar_timestamp=bar.get("timestamp"),
        )
        return await self.check_momentum_setup(market_data)

    async def check_momentum_setup(self, market_data: Dict) -> Optional[TradingSignal]:
        bar = market_data.get("bar", {})
        indicators: Dict = {}

        try:
            indicators = self.indicator_manager.get_all_values(self.symbol)

            macd_line = indicators.get("macd_line")
            macd_signal_value = indicators.get("macd_signal")
            macd_histogram = indicators.get("macd_histogram")
            rsi = indicators.get(f"rsi_{self.rsi_period}") or indicators.get("rsi_14")
            atr = indicators.get("atr")
            adx = indicators.get("adx")
            bb_middle = indicators.get("bb_middle")
            bb_bandwidth = indicators.get("bb_bandwidth")

            current_volume = bar.get("volume", 0)
            self.volume_history.append(current_volume)
            volume_sma = (
                sum(self.volume_history) / len(self.volume_history)
                if len(self.volume_history) == self.volume_sma_period
                else None
            )
            volume_ok = (
                volume_sma is not None
                and current_volume >= volume_sma * self.volume_multiplier
            )
            volume_ratio = (current_volume / volume_sma) if volume_sma else 0.0

            required_missing = (
                macd_line is None
                or macd_signal_value is None
                or macd_histogram is None
                or rsi is None
                or atr is None
                or atr <= 0
                or bb_middle is None
                or volume_sma is None
            )
            if required_missing:
                self.logger.debug(
                    "momentum_missing_indicators",
                    macd_line=macd_line,
                    macd_signal=macd_signal_value,
                    rsi=rsi,
                    atr=atr,
                    bb_middle=bb_middle,
                    volume_sma_ready=volume_sma is not None,
                    symbol=self.symbol,
                )
                self._update_state(rsi, macd_histogram, None)
                return None

            current_price = bar["close"]

            if self.bars_since_last_signal < self.cooldown_bars:
                self.logger.debug(
                    "momentum_cooldown_active",
                    bars_since_last_signal=self.bars_since_last_signal,
                    cooldown_required=self.cooldown_bars,
                    symbol=self.symbol,
                )
                self._update_state(rsi, macd_histogram, None)
                return None

            macd_cross_up = (
                self.prev_macd_histogram is not None
                and self.prev_macd_histogram <= 0 < macd_histogram
            )
            macd_cross_down = (
                self.prev_macd_histogram is not None
                and self.prev_macd_histogram >= 0 > macd_histogram
            )

            rsi_trending_up = self.prev_rsi is None or rsi >= self.prev_rsi
            rsi_trending_down = self.prev_rsi is None or rsi <= self.prev_rsi

            rsi_bullish = rsi >= (self.rsi_momentum + self.rsi_delta)
            rsi_bearish = rsi <= (self.rsi_momentum - self.rsi_delta)

            adx_ok = adx is None or adx >= self.adx_threshold

            long_conditions = (
                macd_line is not None
                and macd_signal_value is not None
                and macd_line > macd_signal_value
                and macd_cross_up
                and rsi_bullish
                and rsi_trending_up
                and adx_ok
                and current_price > bb_middle
                and volume_ok
            )

            short_conditions = (
                macd_line is not None
                and macd_signal_value is not None
                and macd_line < macd_signal_value
                and macd_cross_down
                and rsi_bearish
                and rsi_trending_down
                and adx_ok
                and current_price < bb_middle
                and volume_ok
            )

            self.logger.debug(
                "momentum_conditions_check",
                current_price=current_price,
                macd_line=macd_line,
                macd_signal=macd_signal_value,
                macd_histogram=macd_histogram,
                rsi=rsi,
                volume=current_volume,
                volume_sma=volume_sma,
                volume_ratio=volume_ratio,
                long_conditions=long_conditions,
                short_conditions=short_conditions,
                available_indicators=list(indicators.keys()),
            )

            if long_conditions:
                confidence = self._calculate_confidence(
                    direction="LONG",
                    macd_histogram=macd_histogram,
                    adx=adx,
                    rsi=rsi,
                    volume_ratio=volume_ratio,
                )

                if adx is None or adx < 30:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                if volume_ratio < 1.5:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                if confidence < self.MIN_CONFIDENCE:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                signal = TradingSignal(
                    strategy_name=self.name,
                    symbol=self.symbol,
                    signal_type=SignalType.MOMENTUM,
                    action=SignalAction.BUY,
                    entry_price=current_price,
                    quantity=self.position_sizer.calculate_quantity(
                        self.symbol,
                        current_price,
                        self.get_stop_loss(current_price, "BUY", atr),
                        atr,
                    ),
                    stop_loss=self.get_stop_loss(current_price, "BUY", atr),
                    take_profit=self.get_take_profit(current_price, "BUY", atr),
                    confidence=confidence,
                    indicators={
                        "macd_line": macd_line,
                        "macd_signal": macd_signal_value,
                        "rsi": rsi,
                        "macd_histogram": macd_histogram,
                        "volume": current_volume,
                        "volume_sma": volume_sma,
                        "atr": atr,
                        "adx": adx,
                        "bb_middle": bb_middle,
                        "bb_bandwidth": bb_bandwidth,
                    },
                    reason="MACD bullish crossover + RSI momentum + strong volume",
                )

                is_valid, errors = await self.signal_validator.validate(
                    signal,
                    market_data.get("orderbook"),
                )
                self.logger.debug(
                    "momentum_long_validation",
                    is_valid=is_valid,
                    errors=errors,
                    quantity=signal.quantity,
                    confidence=signal.confidence,
                )
                if is_valid and signal.quantity > 0:
                    if not self._passes_final_quality_filter("LONG", adx, rsi, confidence):
                        self._update_state(rsi, macd_histogram, None)
                        return None
                    self.total_signals_generated += 1
                    self._update_state(rsi, macd_histogram, signal)
                    return signal

            elif short_conditions:
                confidence = self._calculate_confidence(
                    direction="SHORT",
                    macd_histogram=macd_histogram,
                    adx=adx,
                    rsi=rsi,
                    volume_ratio=volume_ratio,
                )

                if adx is None or adx < 30:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                if volume_ratio < 1.5:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                if confidence < self.MIN_CONFIDENCE:
                    self._update_state(rsi, macd_histogram, None)
                    return None

                signal = TradingSignal(
                    strategy_name=self.name,
                    symbol=self.symbol,
                    signal_type=SignalType.MOMENTUM,
                    action=SignalAction.SELL,
                    entry_price=current_price,
                    quantity=self.position_sizer.calculate_quantity(
                        self.symbol,
                        current_price,
                        self.get_stop_loss(current_price, "SELL", atr),
                        atr,
                    ),
                    stop_loss=self.get_stop_loss(current_price, "SELL", atr),
                    take_profit=self.get_take_profit(current_price, "SELL", atr),
                    confidence=confidence,
                    indicators={
                        "macd_line": macd_line,
                        "macd_signal": macd_signal_value,
                        "rsi": rsi,
                        "macd_histogram": macd_histogram,
                        "volume": current_volume,
                        "volume_sma": volume_sma,
                        "atr": atr,
                        "adx": adx,
                        "bb_middle": bb_middle,
                        "bb_bandwidth": bb_bandwidth,
                    },
                    reason="MACD bearish crossover + RSI momentum + strong volume",
                )

                is_valid, errors = await self.signal_validator.validate(
                    signal,
                    market_data.get("orderbook"),
                )
                self.logger.debug(
                    "momentum_short_validation",
                    is_valid=is_valid,
                    errors=errors,
                    quantity=signal.quantity,
                    confidence=signal.confidence,
                )
                if is_valid and signal.quantity > 0:
                    if not self._passes_final_quality_filter("SHORT", adx, rsi, confidence):
                        self._update_state(rsi, macd_histogram, None)
                        return None
                    self.total_signals_generated += 1
                    self._update_state(rsi, macd_histogram, signal)
                    return signal

            self._update_state(rsi, macd_histogram, None)
            return None

        except Exception as exc:
            self.logger.error(
                "momentum_scalping_error",
                error=str(exc),
                symbol=self.symbol,
            )
            if indicators:
                self._update_state(
                    indicators.get(f"rsi_{self.rsi_period}"),
                    indicators.get("macd_histogram"),
                    None,
                )
            return None

    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        if side == "BUY":
            if atr and atr > 0:
                return entry_price - (atr * self.stop_multiplier)
            return entry_price * (1 - self.stop_loss_pct)
        if atr and atr > 0:
            return entry_price + (atr * self.stop_multiplier)
        return entry_price * (1 + self.stop_loss_pct)

    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        if side == "BUY":
            if atr and atr > 0:
                return entry_price + (atr * self.atr_take_profit_multiplier)
            return entry_price * (1 + self.profit_target_pct)
        if atr and atr > 0:
            return entry_price - (atr * self.atr_take_profit_multiplier)
        return entry_price * (1 - self.profit_target_pct)

    def should_exit_position(
        self,
        position,
        current_price: float,
        indicators: Dict,
    ) -> bool:
        if not position or position.status != PositionStatus.OPEN:
            return False

        if position.side == "BUY" and current_price >= position.take_profit:
            return True
        if position.side == "SELL" and current_price <= position.take_profit:
            return True

        if position.side == "BUY" and current_price <= position.stop_loss:
            return True
        if position.side == "SELL" and current_price >= position.stop_loss:
            return True

        atr = indicators.get("atr", {}).get("atr", 0)
        if atr > 0:
            if position.side == "BUY":
                trailing_stop = current_price - (atr * self.stop_multiplier)
                if trailing_stop > position.stop_loss:
                    position.stop_loss = trailing_stop
            else:
                trailing_stop = current_price + (atr * self.stop_multiplier)
                if trailing_stop < position.stop_loss:
                    position.stop_loss = trailing_stop

        return False

    def _update_state(
        self,
        rsi: Optional[float],
        macd_histogram: Optional[float],
        signal: Optional[TradingSignal],
    ) -> None:
        if macd_histogram is not None:
            self.prev_macd_histogram = macd_histogram

        if rsi is not None:
            self.prev_rsi = rsi

        if signal is not None:
            self.bars_since_last_signal = 0
            self.last_signal_side = signal.action
        else:
            self.bars_since_last_signal = min(
                self.bars_since_last_signal + 1,
                self.cooldown_bars + 1,
            )

    def _calculate_confidence(
        self,
        direction: str,
        macd_histogram: Optional[float],
        adx: Optional[float],
        rsi: float,
        volume_ratio: float,
    ) -> float:
        macd_score = 0.0
        if macd_histogram is not None:
            macd_score = min(1.0, max(0.0, abs(macd_histogram) / 0.5))

        adx_score = 0.5
        if adx is not None:
            adx_score = min(1.0, max(0.0, (adx - (self.adx_threshold - 5)) / 15.0))

        if direction == "LONG":
            rsi_score = min(
                1.0,
                max(0.0, (rsi - self.rsi_momentum) / max(1.0, 100 - self.rsi_momentum)),
            )
        else:
            rsi_score = min(
                1.0,
                max(0.0, (self.rsi_momentum - rsi) / max(1.0, self.rsi_momentum)),
            )

        volume_score = min(
            1.0,
            max(0.0, (volume_ratio - 1.0) / max(1e-9, self.volume_multiplier)),
        )

        confidence = (
            0.4 * macd_score
            + 0.25 * adx_score
            + 0.2 * rsi_score
            + 0.15 * volume_score
        )
        return round(min(1.0, confidence), 2)

    def _passes_final_quality_filter(
        self,
        direction: str,
        adx: Optional[float],
        rsi: float,
        confidence: float,
    ) -> bool:
        if adx is None or adx < 35:
            return False

        if direction == "LONG" and rsi < 25:
            return False

        if direction == "SHORT" and rsi > 75:
            return False

        if confidence < 0.50:  # RELAJADO PARA MÁS TRADES
            return False

        return True
