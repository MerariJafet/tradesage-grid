from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from app.core.strategies.signal import TradingSignal, SignalAction
from app.core.strategies.position import Position, PositionStatus
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.utils.logger import get_logger
from datetime import datetime, time
import pandas as pd
import uuid

logger = get_logger("base_strategy")

class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias de trading

    Implementa:
    - Lifecycle hooks (on_bar, on_tick, on_orderbook)
    - Gestión de posiciones
    - Integración con IndicatorManager
    - Signal generation y validación
    """

    def __init__(
        self,
        name: str,
        symbol: str,
        indicator_manager: IndicatorManager,
        position_sizer: PositionSizer,
        signal_validator: SignalValidator,
        execution_engine = None,  # ✨ NUEVO
        risk_manager = None,  # ✨ NUEVO
        enabled: bool = True
    ):
        self.name = name
        self.symbol = symbol
        self.indicator_manager = indicator_manager
        self.position_sizer = position_sizer
        self.signal_validator = signal_validator
        self.execution_engine = execution_engine  # ✨ NUEVO
        self.risk_manager = risk_manager  # ✨ NUEVO
        self.enabled = enabled

        # Estado de posiciones
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.open_position: Optional[Position] = None  # Posición actualmente abierta

        # Estadísticas
        self.total_signals_generated = 0
        self.total_signals_executed = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        self.logger = logger.bind(strategy=self.name, symbol=self.symbol)
        self.logger.info("strategy_initialized")

        # Session filter configuration (disabled by default)
        self.session_filter_enabled = False
        self.session_start_time = time(0, 0)
        self.session_end_time = time(23, 59, 59, 999999)

    @abstractmethod
    async def generate_signal(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Generar señal de trading basada en condiciones actuales

        Args:
            market_data: Dict con 'bar', 'orderbook', 'indicators'

        Returns:
            TradingSignal o None si no hay señal
        """
        pass

    @abstractmethod
    def get_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calcular precio de stop loss

        Args:
            entry_price: Precio de entrada
            side: 'BUY' o 'SELL'
            atr: ATR actual

        Returns:
            Precio de stop loss
        """
        pass

    @abstractmethod
    def get_take_profit(self, entry_price: float, side: str, atr: float) -> float:
        """
        Calcular precio de take profit

        Args:
            entry_price: Precio de entrada
            side: 'BUY' o 'SELL'
            atr: ATR actual

        Returns:
            Precio de take profit
        """
        pass

    async def on_bar(self, bar: Dict):
        """
        Callback cuando llega nueva barra

        Args:
            bar: Dict con open, high, low, close, volume, timestamp
        """
        if not self.enabled:
            return

        # Actualizar posición abierta con precio actual
        if self.open_position:
            await self._update_open_position(bar['close'])

        # Intentar generar señal
        await self._try_generate_signal(bar)

    async def on_tick(self, tick: Dict):
        """
        Callback cuando llega nuevo tick (para estrategias de alta frecuencia)

        Args:
            tick: Dict con price, quantity, timestamp
        """
        if not self.enabled or not self.open_position:
            return

        # Solo actualizar posición abierta
        await self._update_open_position(tick['price'])

    async def on_orderbook(self, orderbook: Dict):
        """
        Callback cuando llega actualización de orderbook

        Args:
            orderbook: Dict con bids, asks, timestamp
        """
        # Implementar en subclases si necesario
        pass

    async def _try_generate_signal(self, bar: Dict):
        """Intentar generar y validar señal"""
        try:
            # No generar señal si ya hay posición abierta
            if self.open_position:
                return

            # Obtener indicadores actuales
            indicators = self.indicator_manager.get_all_values(self.symbol)

            # Preparar market data
            market_data = {
                'bar': bar,
                'indicators': indicators
            }

            if not self._is_session_allowed(bar.get('timestamp')):
                return

            # Generar señal (implementado en subclase)
            signal = await self.generate_signal(market_data)

            if signal:
                self.total_signals_generated += 1

                self.logger.info(
                    "signal_generated",
                    action=signal.action,
                    confidence=signal.confidence,
                    reason=signal.reason
                )

                # Validar señal
                is_valid, errors = await self.signal_validator.validate(signal)

                if is_valid:
                    self.logger.info("signal_valid", signal_id=str(uuid.uuid4())[:8])
                    self.total_signals_executed += 1
                    
                    # ✨ NUEVO: Ejecutar orden si hay execution engine
                    if self.execution_engine:
                        await self._execute_signal(signal, bar)
                else:
                    self.logger.warning("signal_invalid", errors=errors)

        except Exception as e:
            self.logger.error("signal_generation_error", error=str(e), exc_info=True)

    def _is_session_allowed(self, raw_timestamp) -> bool:
        """Enforce optional session window when enabled."""
        if not self.session_filter_enabled:
            return True

        timestamp = self._coerce_timestamp(raw_timestamp)
        if timestamp is None:
            self.logger.debug(
                "session_filter_blocked",
                reason="invalid_timestamp",
                raw=str(raw_timestamp)
            )
            return False

        current_time = timestamp.time()
        if self.session_start_time <= self.session_end_time:
            allowed = self.session_start_time <= current_time < self.session_end_time
        else:
            allowed = (
                current_time >= self.session_start_time or
                current_time < self.session_end_time
            )

        if not allowed:
            self.logger.debug(
                "session_filter_blocked",
                timestamp=timestamp.isoformat(),
                window=f"{self.session_start_time.strftime('%H:%M')}-{self.session_end_time.strftime('%H:%M')} UTC"
            )

        return allowed

    @staticmethod
    def _coerce_timestamp(raw_timestamp) -> Optional[datetime]:
        """Attempt to convert arbitrary timestamp objects into naive UTC datetimes."""
        if raw_timestamp is None:
            return None

        if isinstance(raw_timestamp, datetime):
            return raw_timestamp

        try:
            ts = pd.to_datetime(raw_timestamp, utc=False, errors='coerce')
        except Exception:
            return None

        if ts is None or pd.isna(ts):
            return None

        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()

        return None

    async def _execute_signal(self, signal: TradingSignal, bar: Dict):
        """✨ NUEVO: Ejecutar señal en el execution engine"""
        try:
            from app.core.execution.order import Order, OrderSide, OrderType, TimeInForce
            
            # ✨ NUEVO: Verificar con risk manager antes de ejecutar
            if self.risk_manager:
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    self.logger.warning(
                        "trade_blocked_by_risk_manager",
                        reason=reason
                    )
                    return
                
                # Verificar límites de posición
                risk_amount = signal.get_risk_amount()
                can_open, position_reason = self.risk_manager.can_open_position(
                    signal.symbol,
                    signal.quantity,
                    risk_amount
                )
                
                if not can_open:
                    self.logger.warning(
                        "position_blocked_by_risk_manager",
                        reason=position_reason,
                        risk_amount=risk_amount
                    )
                    return
            
            # Convertir señal a orden
            order = Order(
                strategy_name=self.name,
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=signal.quantity,
                time_in_force=TimeInForce.GTC,
                mode="paper"
            )
            
            # Obtener datos de mercado para ejecución
            current_price = bar['close']
            
            # Obtener orderbook si está disponible
            orderbook = None  # TODO: obtener del ws_manager
            
            # Obtener ATR para slippage
            indicators = self.indicator_manager.get_all_values(signal.symbol)
            atr = indicators.get('atr')
            
            # Ejecutar orden
            filled_order = await self.execution_engine.submit_order(
                order=order,
                current_price=current_price,
                orderbook=orderbook,
                atr=atr,
                spread_pct=0.08  # TODO: calcular spread real
            )
            
            if filled_order.status == "FILLED" or filled_order.status == "PARTIALLY_FILLED":
                self.logger.info(
                    "order_executed",
                    order_id=filled_order.id,
                    filled_price=filled_order.filled_price,
                    filled_quantity=filled_order.filled_quantity,
                    commission=filled_order.commission,
                    slippage=filled_order.slippage
                )
                
                # Crear posición
                from app.core.strategies.position import Position, PositionSide
                
                position = Position(
                    id=filled_order.id,
                    strategy_name=self.name,
                    symbol=signal.symbol,
                    side=PositionSide.LONG if signal.action == "BUY" else PositionSide.SHORT,
                    entry_price=filled_order.filled_price,
                    quantity=filled_order.filled_quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_commission=filled_order.commission,
                    entry_signal=signal.model_dump()
                )
                
                self.open_position = position
                
                # ✨ NUEVO: Registrar en risk manager
                if self.risk_manager:
                    self.risk_manager.register_position_opened(
                        position_id=filled_order.id,
                        symbol=signal.symbol,
                        risk_amount=signal.get_risk_amount()
                    )
                
                # Registrar en validator
                risk_amount = signal.get_risk_amount()
                self.signal_validator.register_position_opened(risk_amount)
                
            else:
                self.logger.warning(
                    "order_rejected",
                    order_id=filled_order.id,
                    reason=filled_order.rejection_reason
                )
        
        except Exception as e:
            self.logger.error("execute_signal_error", error=str(e), exc_info=True)

    async def _update_open_position(self, current_price: float):
        """Actualizar posición abierta y verificar stops"""
        if not self.open_position:
            return

        # Actualizar PnL no realizado
        self.open_position.update_unrealized_pnl(current_price)

        # Verificar stop loss
        if self.open_position.is_stop_hit(current_price):
            self.logger.info(
                "stop_loss_hit",
                position_id=self.open_position.id,
                current_price=current_price,
                stop_loss=self.open_position.stop_loss
            )
            await self._close_position(current_price, "stop_loss")
            return

        # Verificar take profit
        if self.open_position.is_take_profit_hit(current_price):
            self.logger.info(
                "take_profit_hit",
                position_id=self.open_position.id,
                current_price=current_price,
                take_profit=self.take_profit
            )
            await self._close_position(current_price, "take_profit")
            return

    async def _close_position(self, exit_price: float, exit_reason: str):
        """Cerrar posición abierta"""
        if not self.open_position:
            return

        # Cerrar posición
        self.open_position.close(exit_price, exit_reason, exit_commission=0.0)

        # Actualizar estadísticas
        self.total_trades += 1
        if self.open_position.realized_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        pnl = self.open_position.realized_pnl

        self.logger.info(
            "position_closed",
            position_id=self.open_position.id,
            exit_reason=exit_reason,
            exit_price=exit_price,
            pnl=pnl,
            duration=self.open_position.duration_seconds
        )
        
        # ✨ NUEVO: Actualizar risk manager
        if self.risk_manager:
            # Registrar resultado del trade
            self.risk_manager.register_trade_result(
                pnl=pnl,
                symbol=self.symbol,
                strategy=self.name
            )
            
            # Cerrar posición en risk manager
            self.risk_manager.register_position_closed(
                position_id=self.open_position.id,
                symbol=self.symbol,
                risk_amount=self.open_position.get_risk_amount()
            )
            
            # Actualizar balance con PnL
            current_balance = self.risk_manager.current_balance
            new_balance = current_balance + pnl
            risk_events = self.risk_manager.update_balance(new_balance)
            
            # Procesar eventos de riesgo
            for event in risk_events:
                self.logger.warning(
                    "risk_event_from_trade",
                    type=event.type,
                    severity=event.severity,
                    message=event.message
                )

        # Notificar al validator
        risk_amount = self.open_position.get_risk_amount() if hasattr(self.open_position, 'get_risk_amount') else 0
        self.signal_validator.register_position_closed(risk_amount)

        # Limpiar
        self.open_position = None

    def get_statistics(self) -> Dict:
        """Obtener estadísticas de la estrategia"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        return {
            "name": self.name,
            "symbol": self.symbol,
            "enabled": self.enabled,
            "signals_generated": self.total_signals_generated,
            "signals_executed": self.total_signals_executed,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "has_open_position": self.open_position is not None,
            "open_position": self.open_position.to_dict() if self.open_position else None
        }

    def reset_statistics(self):
        """Resetear estadísticas"""
        self.total_signals_generated = 0
        self.total_signals_executed = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.logger.info("statistics_reset")