from collections import deque
from typing import Dict, Optional
from app.core.strategies.base import BaseStrategy
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.utils.logger import get_logger

logger = get_logger("strategy.mean_reversion")

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy

    Lógica:
    1. Detectar sobrecompra/sobreventa con RSI
    2. Confirmar con Bollinger Bands (precio fuera de bandas)
    3. Volumen debe ser significativo
    4. Entrar cuando el precio empieza a revertir

    Entry Conditions (LONG):
    - RSI < 30 (oversold)
    - Precio toca o cruza lower Bollinger Band
    - Volumen > promedio 20 períodos
    - Precio empieza a subir (close > open)

    Entry Conditions (SHORT):
    - RSI > 70 (overbought)
    - Precio toca o cruza upper Bollinger Band
    - Volumen > promedio 20 períodos
    - Precio empieza a bajar (close < open)

    Exit:
    - Target: Middle Bollinger Band (mean)
    - Stop: ATR * 2 desde entry
    """

    def __init__(
        self,
        symbol: str,
        indicator_manager: IndicatorManager,
        position_sizer: PositionSizer,
        signal_validator: SignalValidator,
        execution_engine = None,
        risk_manager = None,
        # Parámetros optimizados para scalping (Sprint 10)
        rsi_oversold: float = 25.0,  # Optimizado: de 30.0 a 25.0
        rsi_overbought: float = 75.0,  # Optimizado: de 70.0 a 75.0
        rsi_period: int = 7,  # Optimizado: de 14 a 7 para scalping
        bb_period: int = 10,  # Optimizado: de 20 a 10 para scalping
        bb_std: float = 1.5,  # Optimizado: std más conservador
        volume_multiplier: float = 1.5,  # Optimizado: de 1.2 a 1.5
        volume_sma_period: int = 20,
        atr_stop_multiplier: float = 2.0,  # Less aggressive stop
        profit_target_atr_multiplier: float = 3.0,  # More ambitious take profit
        adx_threshold: float = 25.0,
        enabled: bool = True
    ):
        super().__init__(
            name="MeanReversion",
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator,
            execution_engine=execution_engine,
            risk_manager=risk_manager,
            enabled=enabled
        )

        # Parámetros optimizados para scalping
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_multiplier = volume_multiplier
        self.volume_sma_period = volume_sma_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.profit_target_atr_multiplier = profit_target_atr_multiplier
        self.adx_threshold = adx_threshold

        # Minimum periods needed for indicators
        self.min_periods = max(self.rsi_period, self.bb_period, 14)  # ATR period is typically 14

        # Rolling state to reduce redundant signals
        self.volume_history = deque(maxlen=volume_sma_period)
        self.prev_rsi: Optional[float] = None
        self.previous_close: Optional[float] = None
        self.cooldown_bars = max(5, self.rsi_period)
        self.bars_since_last_signal = self.cooldown_bars
        self.last_signal_side: Optional[str] = None

        self.logger.info(
            "mean_reversion_optimized_initialized",
            symbol=symbol,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            rsi_period=rsi_period,
            bb_period=bb_period,
            bb_std=bb_std,
            volume_multiplier=volume_multiplier,
            atr_stop_multiplier=atr_stop_multiplier,
            adx_threshold=adx_threshold
        )

    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Generar señal de trading basada en condiciones actuales"""
        bar = market_data.get('bar', {})
        if not bar:
            return None

        return self.check_setup(bar)

    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular precio de stop loss"""
        if side == "BUY":
            return entry_price - (atr * self.atr_stop_multiplier)
        else:  # SELL
            return entry_price + (atr * self.atr_stop_multiplier)

    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular precio de take profit basado en ATR (optimizado para scalping)"""
        if side == "BUY":
            return entry_price + (atr * self.profit_target_atr_multiplier)
        else:  # SELL
            return entry_price - (atr * self.profit_target_atr_multiplier)

    def check_setup(self, bar: Dict) -> Optional[TradingSignal]:
        """Verificar setup de mean reversion"""

        # Obtener indicadores
        indicators = self.indicator_manager.get_all_values(self.symbol)

        if not indicators:
            return None

        # Indicadores requeridos
        rsi = indicators.get(f'rsi_{self.rsi_period}') or indicators.get('rsi_5')
        bb_upper = indicators.get('bb_upper')
        bb_middle = indicators.get('bb_middle')
        bb_lower = indicators.get('bb_lower')
        bb_bandwidth = indicators.get('bb_bandwidth')
        volume = bar.get('volume', 0)
        atr = indicators.get('atr')
        adx = indicators.get('adx')

        # Validar que todos los indicadores estén disponibles
        if None in [rsi, bb_upper, bb_middle, bb_lower, atr] or atr <= 0:
            self._update_state(rsi, bar.get('close'), None)
            return None

        if adx is not None and adx > self.adx_threshold:
            self._update_state(rsi, bar.get('close'), None)
            return None

        current_price = bar['close']
        previous_close = bar.get('previous_close', current_price)

        self.volume_history.append(volume)
        if len(self.volume_history) < self.volume_sma_period:
            self._update_state(rsi, current_price, None)
            return None

        volume_sma = sum(self.volume_history) / len(self.volume_history)
        volume_threshold = volume_sma * self.volume_multiplier
        if volume < volume_threshold:
            self._update_state(rsi, current_price, None)
            return None

        if self.bars_since_last_signal < self.cooldown_bars:
            self._update_state(rsi, current_price, None)
            return None

        volume_ratio = volume / volume_sma if volume_sma else 0
        volume_context = f"{volume_ratio:.1f}x average"

        # ========== LONG SETUP (Oversold Reversal) ==========
        rsi_turning_up = (
            self.prev_rsi is not None and
            self.prev_rsi < self.rsi_oversold and
            rsi > self.prev_rsi
        )
        rsi_turning_down = (
            self.prev_rsi is not None and
            self.prev_rsi > self.rsi_overbought and
            rsi < self.prev_rsi
        )

        signal: Optional[TradingSignal] = None

        if rsi < self.rsi_oversold and rsi_turning_up:
            # Precio debe estar en o cerca de lower band
            distance_to_lower = abs(current_price - bb_lower) / current_price

            if distance_to_lower < 0.005:  # Dentro del 0.5% de lower band
                # Confirmar inicio de reversión (precio subiendo)
                if bar['close'] > bar['open']:
                    # Calcular stops
                    stop_loss = current_price - (atr * self.atr_stop_multiplier)
                    take_profit = self.get_take_profit(current_price, "BUY", atr)  # Optimizado: usar ATR-based target

                    position_size = self.position_sizer.calculate_quantity(
                        self.symbol,
                        current_price,
                        stop_loss,
                        atr
                    )

                    if position_size <= 0:
                        self._update_state(rsi, current_price, None)
                        return None

                    signal = TradingSignal(
                        strategy_name=self.name,
                        symbol=self.symbol,
                        signal_type=SignalType.MEAN_REVERSION,
                        action=SignalAction.BUY,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=position_size,
                        confidence=self._calculate_confidence(
                            rsi=rsi,
                            distance_to_band=distance_to_lower,
                            volume_ratio=volume_ratio,
                            side="LONG"
                        ),
                        indicators={
                            'rsi': rsi,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'bb_upper': bb_upper,
                            'volume': volume,
                            'volume_sma': volume_sma,
                            'atr': atr,
                            'adx': adx,
                            'bb_bandwidth': bb_bandwidth
                        },
                        reason=f"Mean reversion oversold: RSI {rsi:.1f}, price near lower BB, volume {volume_context}"
                    )

                    self.logger.info(
                        "mean_reversion_long_setup",
                        rsi=rsi,
                        price=current_price,
                        bb_lower=bb_lower,
                        volume_ratio=volume_ratio,
                        confidence=signal.confidence
                    )

                    self._update_state(rsi, current_price, signal)
                    return signal

        # ========== SHORT SETUP (Overbought Reversal) ==========
        elif rsi > self.rsi_overbought and rsi_turning_down:
            # Precio debe estar en o cerca de upper band
            distance_to_upper = abs(current_price - bb_upper) / current_price

            if distance_to_upper < 0.005:  # Dentro del 0.5% de upper band
                # Confirmar inicio de reversión (precio bajando)
                if bar['close'] < bar['open']:
                    # Calcular stops
                    stop_loss = current_price + (atr * self.atr_stop_multiplier)
                    take_profit = self.get_take_profit(current_price, "SELL", atr)  # Optimizado: usar ATR-based target

                    position_size = self.position_sizer.calculate_quantity(
                        self.symbol,
                        current_price,
                        stop_loss,
                        atr
                    )

                    if position_size <= 0:
                        self._update_state(rsi, current_price, None)
                        return None

                    signal = TradingSignal(
                        strategy_name=self.name,
                        symbol=self.symbol,
                        signal_type=SignalType.MEAN_REVERSION,
                        action=SignalAction.SELL,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=position_size,
                        confidence=self._calculate_confidence(
                            rsi=rsi,
                            distance_to_band=distance_to_upper,
                            volume_ratio=volume_ratio,
                            side="SHORT"
                        ),
                        indicators={
                            'rsi': rsi,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'bb_upper': bb_upper,
                            'volume': volume,
                            'volume_sma': volume_sma,
                            'atr': atr,
                            'adx': adx,
                            'bb_bandwidth': bb_bandwidth
                        },
                        reason=f"Mean reversion overbought: RSI {rsi:.1f}, price near upper BB, volume {volume_context}"
                    )

                    self.logger.info(
                        "mean_reversion_short_setup",
                        rsi=rsi,
                        price=current_price,
                        bb_upper=bb_upper,
                        volume_ratio=volume_ratio,
                        confidence=signal.confidence
                    )

                    self._update_state(rsi, current_price, signal)
                    return signal

        self._update_state(rsi, current_price, None)
        return None

    def _calculate_confidence(
        self,
        rsi: float,
        distance_to_band: float,
        volume_ratio: float,
        side: str
    ) -> float:
        """
        Calcular confianza de la señal (0-1)

        Factores:
        1. Extremo de RSI (más extremo = más confianza)
        2. Proximidad a banda (más cerca = más confianza)
        3. Volumen (más alto = más confianza)
        """

        # Factor 1: RSI extremo
        if side == "LONG":
            rsi_score = (self.rsi_oversold - rsi) / self.rsi_oversold  # 0 a 1
        else:  # SHORT
            rsi_score = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)  # 0 a 1

        rsi_score = max(0, min(1, rsi_score))  # Clamp 0-1

        # Factor 2: Proximidad a banda (distance_to_band ya es ratio)
        # Invertir: menor distancia = mayor score
        band_score = 1 - (distance_to_band / 0.005)  # 0.005 es threshold
        band_score = max(0, min(1, band_score))

        # Factor 3: Volumen
        volume_score = min(1, (volume_ratio - 1) / 0.5)  # 1.0 a 1.5 ratio = 0 a 1 score
        volume_score = max(0, volume_score)

        # Combinar (pesos: RSI 40%, Band 30%, Volume 30%)
        confidence = (rsi_score * 0.4) + (band_score * 0.3) + (volume_score * 0.3)

        return round(confidence, 2)

    def get_strategy_info(self) -> Dict:
        """Obtener información de la estrategia"""
        info = super().get_strategy_info()
        info.update({
            "parameters": {
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "volume_multiplier": self.volume_multiplier,
                "atr_stop_multiplier": self.atr_stop_multiplier
            }
        })
        return info

    def _update_state(
        self,
        rsi: Optional[float],
        close_price: Optional[float],
        signal: Optional[TradingSignal]
    ) -> None:
        """Persist rolling metrics and enforce a small cooldown between entries."""

        if rsi is not None:
            self.prev_rsi = rsi

        if close_price is not None:
            self.previous_close = close_price

        if signal is not None:
            self.bars_since_last_signal = 0
            self.last_signal_side = signal.action
        else:
            self.bars_since_last_signal = min(
                self.bars_since_last_signal + 1,
                self.cooldown_bars + 1
            )