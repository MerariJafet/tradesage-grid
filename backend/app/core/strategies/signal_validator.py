from app.core.strategies.signal import TradingSignal
from app.core.indicators.indicator_manager import IndicatorManager
from app.utils.logger import get_logger
from typing import Optional, List
import asyncio

logger = get_logger("signal_validator")

class SignalValidator:
    """
    Valida señales antes de ejecución
    Filtros: liquidez, spread, riesgo acumulado, condiciones de mercado
    """

    def __init__(
        self,
        indicator_manager: IndicatorManager,
        max_open_positions: int = 3,
        max_daily_trades: int = 20,
        max_spread_pct: float = 0.10  # 0.10%
    ):
        self.indicator_manager = indicator_manager
        self.max_open_positions = max_open_positions
        self.max_daily_trades = max_daily_trades
        self.max_spread_pct = max_spread_pct

        # Estado
        self.open_positions_count = 0
        self.daily_trades_count = 0
        self.total_risk_allocated = 0.0

    async def validate(
        self,
        signal: TradingSignal,
        current_orderbook: Optional[dict] = None
    ) -> tuple[bool, list[str]]:
        """
        Validar señal completa

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # 1. Validar expiración
        if signal.is_expired():
            errors.append("Signal expired")
            return False, errors

        # 2. Validar límite de posiciones abiertas
        if self.open_positions_count >= self.max_open_positions:
            errors.append(f"Max open positions reached ({self.max_open_positions})")

        # 3. Validar límite de trades diarios
        if self.daily_trades_count >= self.max_daily_trades:
            errors.append(f"Max daily trades reached ({self.max_daily_trades})")

        # 4. Validar ratio riesgo/recompensa mínimo
        rr_ratio = signal.get_risk_reward_ratio()
        if rr_ratio < 1.2:  # Mínimo 1.2:1
            errors.append(f"Risk/Reward ratio too low: {rr_ratio:.2f} (min 1.2)")

        # 5. Validar indicadores actuales
        indicators_valid, indicator_errors = await self._validate_indicators(signal)
        if not indicators_valid:
            errors.extend(indicator_errors)

        # 6. Validar spread (si tenemos orderbook)
        if current_orderbook:
            spread_valid, spread_error = self._validate_spread(signal, current_orderbook)
            if not spread_valid:
                errors.append(spread_error)

        # 7. Validar liquidez mínima
        if current_orderbook:
            liquidity_valid, liquidity_error = self._validate_liquidity(signal, current_orderbook)
            if not liquidity_valid:
                errors.append(liquidity_error)

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                "signal_validation_failed",
                strategy=signal.strategy_name,
                symbol=signal.symbol,
                errors=errors
            )

        # Actualizar señal con resultado
        signal.is_valid = is_valid
        signal.validation_errors = errors

        return is_valid, errors

    async def _validate_indicators(
        self,
        signal: TradingSignal
    ) -> tuple[bool, list[str]]:
        """Validar que indicadores actuales soportan la señal"""
        errors = []

        # Determinar indicadores requeridos según tipo de señal
        required_indicators = self._get_required_indicators(signal.signal_type)

        # Verificar que indicadores requeridos están listos
        if not self.indicator_manager.is_ready(signal.symbol, required_indicators):
            errors.append("Indicators not ready")
            return False, errors

        # Obtener valores actuales
        current_indicators = self.indicator_manager.get_all_values(signal.symbol)

        # Validaciones específicas por tipo de señal
        if signal.signal_type == "breakout":
            # Verificar que BB no esté demasiado expandido
            bb_bandwidth = current_indicators.get('bb_bandwidth', 0)
            if bb_bandwidth > 0.03:  # > 3%
                errors.append(f"BB too expanded for breakout: {bb_bandwidth:.4f}")

        elif signal.signal_type == "mean_reversion":
            # Verificar que RSI esté en extremos
            rsi_2 = current_indicators.get('rsi_2', 50)
            if 20 < rsi_2 < 80:
                errors.append(f"RSI(2) not in extreme zone: {rsi_2:.2f}")

        return len(errors) == 0, errors

    def _get_required_indicators(self, signal_type: str) -> List[str]:
        """Obtener lista de indicadores requeridos para un tipo de señal"""
        if signal_type == "momentum":
            return ['macd', 'rsi_14', 'atr']  # MACD provides macd_line/macd_signal, rsi_14 provides rsi
        elif signal_type == "breakout":
            return ['bb', 'rsi_14', 'atr']
        elif signal_type == "mean_reversion":
            return ['bb', 'rsi_2', 'atr']
        else:
            # Default: check all indicators
            return None

    def _validate_spread(
        self,
        signal: TradingSignal,
        orderbook: dict
    ) -> tuple[bool, Optional[str]]:
        """Validar que el spread no sea excesivo"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            return False, "Orderbook empty"

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        spread_pct = ((best_ask - best_bid) / best_bid) * 100

        if spread_pct > self.max_spread_pct:
            return False, f"Spread too wide: {spread_pct:.4f}% (max {self.max_spread_pct}%)"

        return True, None

    def _validate_liquidity(
        self,
        signal: TradingSignal,
        orderbook: dict
    ) -> tuple[bool, Optional[str]]:
        """Validar liquidez suficiente en el lado relevante"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        # Determinar lado según acción
        if signal.action == "BUY":
            relevant_side = asks
            side_name = "asks"
        else:
            relevant_side = bids
            side_name = "bids"

        # Calcular volumen en top 5 niveles
        volume_top_5 = sum(qty for _, qty in relevant_side[:5])

        # Requerimiento: al menos 2x la cantidad de la señal
        required_volume = signal.quantity * 2

        if volume_top_5 < required_volume:
            return False, f"Insufficient {side_name} liquidity: {volume_top_5:.4f} (need {required_volume:.4f})"

        return True, None

    def register_position_opened(self, risk_amount: float):
        """Registrar que se abrió una posición"""
        self.open_positions_count += 1
        self.daily_trades_count += 1
        self.total_risk_allocated += risk_amount

        logger.info(
            "position_registered",
            open_positions=self.open_positions_count,
            daily_trades=self.daily_trades_count,
            total_risk=self.total_risk_allocated
        )

    def register_position_closed(self, risk_amount: float):
        """Registrar que se cerró una posición"""
        self.open_positions_count = max(0, self.open_positions_count - 1)
        self.total_risk_allocated = max(0, self.total_risk_allocated - risk_amount)

        logger.info(
            "position_closed_registered",
            open_positions=self.open_positions_count,
            total_risk=self.total_risk_allocated
        )

    def reset_daily_counters(self):
        """Resetear contadores diarios (llamar a medianoche)"""
        self.daily_trades_count = 0
        logger.info("daily_counters_reset")