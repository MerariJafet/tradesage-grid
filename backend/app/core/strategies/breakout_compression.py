from app.core.strategies.base import BaseStrategy
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from typing import Optional, Dict
from datetime import datetime

class BreakoutCompressionStrategy(BaseStrategy):
    """
    Estrategia de Breakout por Compresión de Bollinger Bands

    Setup:
    1. Bollinger Bands comprimidas (bandwidth < threshold)
    2. Precio rompe banda superior (BUY) o inferior (SELL)
    3. Volumen por encima de promedio (confirmación)
    4. RSI no en extremos (evitar reversiones)

    Gestión:
    - Stop: 0.5-1.0 × ATR por debajo/arriba de entrada
    - TP: 1.2-1.5 × ATR en dirección del breakout
    - Trailing stop opcional después de 0.8× ATR
    """

    def __init__(
        self,
        symbol: str,
        indicator_manager,
        position_sizer,
        signal_validator,
        execution_engine=None,  # ✨ NUEVO
        risk_manager=None,  # ✨ NUEVO
        # Parámetros de la estrategia
        bb_compression_threshold: float = 0.02,  # 2% bandwidth
        volume_multiplier: float = 1.2,  # 120% del promedio
        rsi_min: float = 30,
        rsi_max: float = 70,
        stop_atr_multiplier: float = 0.75,
        tp_atr_multiplier: float = 1.2,
        min_confidence: float = 0.60,
        enabled: bool = True
    ):
        super().__init__(
            name="BreakoutCompression",
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator,
            execution_engine=execution_engine,  # ✨ NUEVO
            risk_manager=risk_manager,  # ✨ NUEVO
            enabled=enabled
        )

        # Parámetros
        self.bb_compression_threshold = bb_compression_threshold
        self.volume_multiplier = volume_multiplier
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.stop_atr_multiplier = stop_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.min_confidence = min_confidence

        # Estado
        self.last_bar_close = None
        self.volume_ema = None
        self.ema_period = 20

    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """Generar señal de breakout"""
        bar = market_data['bar']
        indicators = market_data['indicators']

        # Verificar que indicadores están listos
        required_indicators = ['bb_upper', 'bb_lower', 'bb_middle', 'bb_bandwidth', 'atr', 'rsi_14']
        if not all(ind in indicators for ind in required_indicators):
            return None

        # Extraer indicadores
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        bb_bandwidth = indicators['bb_bandwidth']
        atr = indicators['atr']
        rsi = indicators['rsi_14']

        current_close = bar['close']
        current_volume = bar['volume']

        # Actualizar EMA de volumen
        if self.volume_ema is None:
            self.volume_ema = current_volume
        else:
            alpha = 2 / (self.ema_period + 1)
            self.volume_ema = (current_volume * alpha) + (self.volume_ema * (1 - alpha))

        # === CONDICIONES DE SETUP ===

        # 1. Bollinger Bands comprimidas
        if bb_bandwidth > self.bb_compression_threshold:
            self.logger.debug(
                "bb_not_compressed",
                bandwidth=bb_bandwidth,
                threshold=self.bb_compression_threshold
            )
            return None

        # 2. RSI en zona neutra (no extremos)
        if rsi < self.rsi_min or rsi > self.rsi_max:
            self.logger.debug(
                "rsi_in_extreme",
                rsi=rsi,
                min=self.rsi_min,
                max=self.rsi_max
            )
            return None

        # 3. Volumen por encima de promedio
        if current_volume < self.volume_ema * self.volume_multiplier:
            self.logger.debug(
                "volume_too_low",
                current=current_volume,
                required=self.volume_ema * self.volume_multiplier
            )
            return None

        # === DETECCIÓN DE BREAKOUT ===

        action = None
        breakout_type = None

        # Breakout alcista (arriba de banda superior)
        if current_close > bb_upper:
            # Verificar que barra anterior estaba dentro de bandas
            if self.last_bar_close and self.last_bar_close <= bb_upper:
                action = SignalAction.BUY
                breakout_type = "bullish"

        # Breakout bajista (abajo de banda inferior)
        elif current_close < bb_lower:
            if self.last_bar_close and self.last_bar_close >= bb_lower:
                action = SignalAction.SELL
                breakout_type = "bearish"

        # Actualizar último close
        self.last_bar_close = current_close

        if action is None:
            return None

        # === CALCULAR PRECIOS DE GESTIÓN ===

        entry_price = current_close
        stop_loss = self.get_stop_loss(entry_price, action, atr)
        take_profit = self.get_take_profit(entry_price, action, atr)

        # === CALCULAR TAMAÑO DE POSICIÓN ===

        quantity = self.position_sizer.calculate_quantity(
            symbol=self.symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            atr=atr
        )

        if quantity == 0:
            self.logger.warning("position_sizer_returned_zero_quantity")
            return None

        # === CALCULAR CONFIANZA ===

        confidence = self._calculate_confidence(
            bb_bandwidth=bb_bandwidth,
            volume_ratio=current_volume / self.volume_ema,
            rsi=rsi
        )

        if confidence < self.min_confidence:
            self.logger.debug(
                "confidence_too_low",
                confidence=confidence,
                min_required=self.min_confidence
            )
            return None

        # === CREAR SEÑAL ===

        signal = TradingSignal(
            strategy_name=self.name,
            symbol=self.symbol,
            signal_type=SignalType.BREAKOUT,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            confidence=confidence,
            indicators={
                'bb_bandwidth': bb_bandwidth,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'atr': atr,
                'rsi': rsi,
                'volume_ratio': current_volume / self.volume_ema
            },
            reason=f"{breakout_type.capitalize()} breakout: BB compressed ({bb_bandwidth:.4f}), volume spike ({current_volume / self.volume_ema:.2f}x)",
            expiry_seconds=30  # Señal válida por 30 segundos
        )

        return signal

    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular stop loss"""
        stop_distance = atr * self.stop_atr_multiplier

        if side == SignalAction.BUY:
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance

    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        """Calcular take profit"""
        tp_distance = atr * self.tp_atr_multiplier

        if side == SignalAction.BUY:
            return entry_price + tp_distance
        else:  # SELL
            return entry_price - tp_distance

    def _calculate_confidence(
        self,
        bb_bandwidth: float,
        volume_ratio: float,
        rsi: float
    ) -> float:
        """
        Calcular nivel de confianza de la señal (0-1)

        Factores:
        - BB bandwidth más comprimida = mayor confianza
        - Mayor volumen = mayor confianza
        - RSI más neutral (50) = mayor confianza
        """

        # Score de compresión (0-1)
        # bandwidth < 0.01 = 1.0, bandwidth > 0.02 = 0.0
        compression_score = max(0, min(1, (0.02 - bb_bandwidth) / 0.01))

        # Score de volumen (0-1)
        # ratio > 1.5 = 1.0, ratio < 1.2 = 0.0
        volume_score = max(0, min(1, (volume_ratio - 1.2) / 0.3))

        # Score de RSI (0-1)
        # RSI = 50 = 1.0, RSI en extremos = 0.0
        rsi_distance_from_neutral = abs(50 - rsi)
        rsi_score = max(0, 1 - (rsi_distance_from_neutral / 20))

        # Confianza ponderada
        confidence = (
            compression_score * 0.4 +
            volume_score * 0.4 +
            rsi_score * 0.2
        )

        return confidence

    def get_parameters(self) -> Dict:
        """Obtener parámetros de la estrategia"""
        return {
            "bb_compression_threshold": self.bb_compression_threshold,
            "volume_multiplier": self.volume_multiplier,
            "rsi_min": self.rsi_min,
            "rsi_max": self.rsi_max,
            "stop_atr_multiplier": self.stop_atr_multiplier,
            "tp_atr_multiplier": self.tp_atr_multiplier,
            "min_confidence": self.min_confidence
        }