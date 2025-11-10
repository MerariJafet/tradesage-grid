from typing import Dict, List, Optional
from datetime import datetime, time, timedelta, timezone
import pandas as pd
import numpy as np
from app.core.strategies.signal import TradingSignal, SignalAction, SignalType
from app.core.orders import LimitOrder, StopLimitOrder, TrailingStopOrder, IcebergOrder, OrderManager
from app.core.orders.base_order import OrderSide
from app.utils.logger import get_logger

logger = get_logger("signal_aggregator_v2")

class SignalAggregatorV2:
    """
    Signal Aggregator V2 - Intelligent Weighting and Filters

    Versión avanzada con:
    1. Weighting dinámico basado en métricas históricas (Sharpe, Profit Factor, Win Rate)
    2. Filtros de volatilidad y sesión
    3. Resolución mejorada de conflictos
    4. Capacidades de backtesting
    5. Rolling window recalibration
    """

    def __init__(self, strategies_config: Dict, min_confidence: float = 0.3,
                 require_confirmation: bool = False,
                 single_signal_override: Optional[float] = None):
        # Configuración de estrategias
        self.strategies_config = strategies_config  # Dict con nombres de estrategias
        self.min_confidence = min_confidence
        self.require_confirmation = require_confirmation
        self.single_signal_override = single_signal_override

        # Weights dinámicos por estrategia
        self.dynamic_weights = {}  # strategy_name -> weight (0-1)
        self.performance_history = pd.DataFrame()  # Historial de rendimiento

        # Parámetros de recalibración
        self.recalibration_window_days = 90  # Rolling window para weights
        self.last_recalibration = None

        # Filtros
        self.volatility_filter_enabled = True
        self.session_filter_enabled = True
        self.liquidity_filter_enabled = True

        # Umbrales de filtros
        self.atr_volatility_high_threshold = 1.5  # ATR > 1.5x SMA -> skip
        self.atr_volatility_low_threshold = 0.5   # ATR < 0.5x SMA -> skip
        self.volume_liquidity_threshold = 1.5     # Volume > 1.5x SMA -> ok
        self.session_start_time = time(9, 0)
        self.session_end_time = time(18, 0)

        # Historial de señales
        self.aggregated_signals_history: List[dict] = []
        self.max_history = 1000

        # Estado de señales activas
        self.active_signals: Dict[str, List[TradingSignal]] = {}  # symbol -> list of signals
        self.last_signal_times: Dict[str, pd.Timestamp] = {}
        self.symbol_cooldown_seconds = 900  # enforce 15-minute cooldown between aggregated trades per symbol
        self.single_signal_weight_threshold = 0.5

        logger.info("signal_aggregator_v2_initialized",
                   strategies=list(strategies_config.keys()),
                   min_confidence=min_confidence)

    def update_performance_history(self, strategy_name: str, metrics: Dict):
        """
        Actualizar historial de rendimiento para recalibración de weights

        Args:
            strategy_name: Nombre de la estrategia
            metrics: Dict con 'sharpe', 'profit_factor', 'win_rate', 'timestamp'
        """
        new_row = pd.DataFrame([{
            'strategy': strategy_name,
            'sharpe': metrics.get('sharpe', 0),
            'profit_factor': metrics.get('profit_factor', 1),
            'win_rate': metrics.get('win_rate', 0),
            'timestamp': pd.to_datetime(metrics.get('timestamp', datetime.now(timezone.utc)))
        }])

        self.performance_history = pd.concat([self.performance_history, new_row], ignore_index=True)

        # Mantener solo últimos 365 días
        cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=365)
        self.performance_history = self.performance_history[
            self.performance_history['timestamp'] > cutoff_date
        ]

        logger.debug("performance_history_updated",
                    strategy=strategy_name,
                    total_records=len(self.performance_history))

    def recalibrate_weights(self):
        """
        Recalibrar weights dinámicos basado en rendimiento reciente
        """
        if self.performance_history.empty:
            # Weights por defecto si no hay historial
            default_weight = 1.0 / len(self.strategies_config)
            self.dynamic_weights = {name: default_weight for name in self.strategies_config.keys()}
            logger.info("default_weights_applied", weights=self.dynamic_weights)
            return

        # Filtrar datos recientes
        cutoff_date = datetime.now(timezone.utc) - pd.Timedelta(days=self.recalibration_window_days)
        recent_perf = self.performance_history[
            self.performance_history['timestamp'] > cutoff_date
        ]

        if recent_perf.empty:
            logger.warning("insufficient_recent_performance_data")
            return

        # Calcular métricas promedio por estrategia
        strategy_metrics = {}
        for strategy in self.strategies_config.keys():
            strat_data = recent_perf[recent_perf['strategy'] == strategy]
            if not strat_data.empty:
                strategy_metrics[strategy] = {
                    'sharpe': strat_data['sharpe'].mean(),
                    'profit_factor': strat_data['profit_factor'].mean(),
                    'win_rate': strat_data['win_rate'].mean()
                }

        if not strategy_metrics:
            logger.warning("no_strategy_metrics_available")
            return

        # Normalizar métricas y calcular weights compuestos
        # Weight = 0.4 * norm_sharpe + 0.4 * norm_profit_factor + 0.2 * norm_win_rate
        metrics_df = pd.DataFrame(strategy_metrics).T

        # Normalización min-max
        normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min() + 1e-8)

        # Calcular weights compuestos
        for strategy in strategy_metrics.keys():
            norm_sharpe = normalized.loc[strategy, 'sharpe']
            norm_pf = normalized.loc[strategy, 'profit_factor']
            norm_wr = normalized.loc[strategy, 'win_rate']

            weight = 0.4 * norm_sharpe + 0.4 * norm_pf + 0.2 * norm_wr
            self.dynamic_weights[strategy] = weight

        # Normalizar weights para que sumen 1
        total_weight = sum(self.dynamic_weights.values())
        if total_weight > 0:
            self.dynamic_weights = {k: v/total_weight for k, v in self.dynamic_weights.items()}

        self.last_recalibration = datetime.now(timezone.utc)

        logger.info("weights_recalibrated",
                   weights=self.dynamic_weights,
                   strategies=list(strategy_metrics.keys()))

    def apply_filters(self, market_data: pd.DataFrame, timestamp: pd.Timestamp) -> bool:
        """
        Aplicar filtros de volatilidad, sesión y liquidez

        Args:
            market_data: DataFrame con columnas 'atr', 'volume', etc.
            timestamp: Timestamp actual

        Returns:
            True si pasa todos los filtros, False si no
        """
        if market_data.empty:
            return False

        latest = market_data.iloc[-1]

        # Filtro de volatilidad
        atr_available = 'atr' in market_data.columns and not market_data['atr'].isna().all()
        if self.volatility_filter_enabled and atr_available:
            atr_current = latest.get('atr', 0)
            atr_series = market_data['atr'].dropna()
            atr_sma = atr_series.tail(20).mean() if len(atr_series) >= 20 else atr_series.mean()

            if atr_sma and atr_current > self.atr_volatility_high_threshold * atr_sma:
                logger.debug("volatility_filter_high", atr_current=atr_current, threshold=atr_sma * self.atr_volatility_high_threshold)
                return False

            if atr_sma and atr_current < self.atr_volatility_low_threshold * atr_sma:
                logger.debug("volatility_filter_low", atr_current=atr_current, threshold=atr_sma * self.atr_volatility_low_threshold)
                return False
        elif self.volatility_filter_enabled and not atr_available:
            logger.debug("volatility_filter_skipped", reason="atr_missing", columns=list(market_data.columns))

        # Filtro de liquidez
        volume_available = 'volume' in market_data.columns and not market_data['volume'].isna().all()
        if self.liquidity_filter_enabled and volume_available:
            volume_current = latest.get('volume', 0)
            volume_series = market_data['volume'].dropna()
            volume_sma = volume_series.tail(20).mean() if len(volume_series) >= 20 else volume_series.mean()

            if volume_current < self.volume_liquidity_threshold * volume_sma:
                logger.debug("liquidity_filter", volume_current=volume_current, threshold=volume_sma * self.volume_liquidity_threshold)
                return False
        elif self.liquidity_filter_enabled and not volume_available:
            logger.debug("liquidity_filter_skipped", reason="volume_missing", columns=list(market_data.columns))

        # Filtro de sesión (para BTCUSDT: evitar sesiones asiáticas de baja liquidez)
        if self.session_filter_enabled:
            if timestamp is None or pd.isna(timestamp):
                logger.debug("session_filter", reason="timestamp_missing")
                return False

            if isinstance(timestamp, pd.Timestamp):
                current_time = timestamp.time()
            else:
                current_time = getattr(timestamp, "time", lambda: None)()

            if current_time is None:
                logger.debug("session_filter", reason="invalid_timestamp", value=str(timestamp))
                return False

            session_start = self.session_start_time
            session_end = self.session_end_time

            if current_time < session_start or current_time >= session_end:
                logger.debug(
                    "session_filter",
                    current_time=current_time.isoformat(),
                    allowed_window="09:00-18:00 UTC"
                )
                return False

        return True

    def add_signal(self, signal: TradingSignal):
        """
        Agregar señal al agregador

        Args:
            signal: TradingSignal a agregar
        """
        symbol = signal.symbol

        if symbol not in self.active_signals:
            self.active_signals[symbol] = []

        # Reemplazar señal existente de la misma estrategia
        self.active_signals[symbol] = [
            s for s in self.active_signals[symbol]
            if s.strategy_name != signal.strategy_name
        ]

        # Agregar nueva señal
        self.active_signals[symbol].append(signal)

        logger.debug("signal_added",
                    symbol=symbol,
                    strategy=signal.strategy_name,
                    action=signal.action,
                    confidence=signal.confidence)

    def aggregate_signals(self, symbol: str, market_data: Optional[pd.DataFrame] = None,
                         timestamp: Optional[pd.Timestamp] = None) -> Optional[TradingSignal]:
        """
        Agregar señales para un símbolo con filtros aplicados

        Args:
            symbol: Símbolo
            market_data: DataFrame con datos de mercado para filtros
            timestamp: Timestamp actual

        Returns:
            Señal agregada o None
        """
        if symbol not in self.active_signals:
            return None

        aggregate_timestamp = None
        if timestamp is not None:
            aggregate_timestamp = pd.to_datetime(timestamp, errors="coerce")

        if market_data is not None and not market_data.empty:
            market_data = market_data.copy()

            # Parche de Gabriel: Forzar el índice a datetime64[ns]
            if not np.issubdtype(market_data.index.dtype, np.datetime64):
                market_data.index = pd.to_datetime(market_data.index, unit='ns', errors='coerce')
            market_data = market_data[~market_data.index.isna()]

        signals = self.active_signals[symbol]
        if not signals:
            return None

        signals_df = pd.DataFrame([
            {
                'signal': s,
                'timestamp': getattr(s, 'timestamp', None)
            }
            for s in signals
        ])

        if not signals_df.empty:
            # Asegurar que todas las señales tengan timestamp datetime
            signals_df['timestamp'] = signals_df['timestamp'].apply(self._coerce_signal_timestamp)
            signals_df = signals_df.dropna(subset=['timestamp'])

        last_signal_time = self.last_signal_times.get(symbol)
        if last_signal_time is not None and not signals_df.empty:
            signals_df = signals_df[signals_df['timestamp'] > last_signal_time]

        if signals_df.empty:
            logger.debug("no_valid_signals", symbol=symbol)
            return None

        if aggregate_timestamp is None:
            aggregate_timestamp = signals_df['timestamp'].max()

        if last_signal_time is not None and aggregate_timestamp is not None:
            time_since_last = aggregate_timestamp - last_signal_time
            if time_since_last < timedelta(seconds=self.symbol_cooldown_seconds):
                logger.debug(
                    "symbol_cooldown_active",
                    symbol=symbol,
                    elapsed=time_since_last.total_seconds(),
                    required=self.symbol_cooldown_seconds
                )
                return None

        signals = signals_df['signal'].tolist()
        self.active_signals[symbol] = signals

        # Aplicar filtros si hay datos disponibles
        if market_data is not None and aggregate_timestamp is not None:
            if not self.apply_filters(market_data, aggregate_timestamp):
                logger.debug("filters_failed", symbol=symbol)
                return None

        # Filtrar señales por confianza mínima
        valid_signals = [s for s in signals if s.confidence >= self.min_confidence]

        if not valid_signals:
            logger.debug("no_valid_signals", symbol=symbol)
            return None

        effective_timestamp = aggregate_timestamp if aggregate_timestamp is not None else pd.Timestamp.utcnow()

        # Una sola señal
        if len(valid_signals) == 1:
            single_signal = valid_signals[0]
            if self.require_confirmation:
                if not self._can_accept_single_signal(single_signal):
                    logger.debug(
                        "single_signal_rejected",
                        symbol=symbol,
                        reason="confirmation_required",
                        confidence=single_signal.confidence,
                        override=self.single_signal_override
                    )
                    return None
                logger.debug(
                    "single_signal_override_accept",
                    symbol=symbol,
                    confidence=single_signal.confidence,
                    override=self.single_signal_override
                )
            single_signal.timestamp = effective_timestamp.to_pydatetime()
            self.last_signal_times[symbol] = effective_timestamp
            return single_signal

        # Múltiples señales - agregar
        aggregated_signal = self._aggregate_multiple_signals(symbol, valid_signals, effective_timestamp)
        if aggregated_signal:
            self.last_signal_times[symbol] = effective_timestamp
        return aggregated_signal

    def _can_accept_single_signal(self, signal: TradingSignal) -> bool:
        """Determinar si una señal individual de alta convicción puede operar sin confirmación."""
        if self.single_signal_override is None:
            return False

        if signal.confidence < self.single_signal_override:
            return False

        if not self.dynamic_weights:
            self.recalibrate_weights()

        weight = self.dynamic_weights.get(signal.strategy_name, 0.5)
        return weight >= self.single_signal_weight_threshold

    @staticmethod
    def _coerce_signal_timestamp(value) -> Optional[pd.Timestamp]:
        if value is None:
            return pd.NaT

        if isinstance(value, pd.Timestamp):
            return value.tz_localize(None) if value.tzinfo else value

        if isinstance(value, datetime):
            return pd.Timestamp(value)

        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return pd.NaT

        if isinstance(value, (int, np.integer)):
            return pd.to_datetime(value, unit='ms', errors='coerce')

        try:
            return pd.to_datetime(value, unit='ms', errors='coerce')
        except (TypeError, ValueError, OverflowError):
            return pd.to_datetime(value, errors='coerce')

    def _aggregate_multiple_signals(self, symbol: str, signals: List[TradingSignal],
                                   aggregate_timestamp: pd.Timestamp) -> Optional[TradingSignal]:
        """
        Agregar múltiples señales con weighting inteligente y resolución de conflictos
        """
        # Recalibrar weights si es necesario (cada hora)
        if (self.last_recalibration is None or
            (datetime.now(timezone.utc) - self.last_recalibration).total_seconds() > 3600):
            self.recalibrate_weights()

        # Separar por dirección
        long_signals = [s for s in signals if s.action == SignalAction.BUY]
        short_signals = [s for s in signals if s.action == SignalAction.SELL]

        # Calcular confianza ponderada para cada dirección
        long_confidence = self._calculate_weighted_confidence(long_signals)
        short_confidence = self._calculate_weighted_confidence(short_signals)

        logger.debug("signal_aggregation_analysis",
                    symbol=symbol,
                    long_signals=len(long_signals),
                    short_signals=len(short_signals),
                    long_confidence=long_confidence,
                    short_confidence=short_confidence)

        # Resolución de conflictos mejorada
        confidence_diff = abs(long_confidence - short_confidence)

        # Si direcciones opuestas y diferencia pequeña (< 0.2), cancelar
        if long_signals and short_signals and confidence_diff < 0.2:
            logger.info("signal_conflict_cancelled",
                       symbol=symbol,
                       long_confidence=long_confidence,
                       short_confidence=short_confidence,
                       diff=confidence_diff)
            return None

        # Determinar dirección ganadora
        if long_confidence > short_confidence and long_confidence >= self.min_confidence:
            return self._create_aggregated_signal(symbol, SignalAction.BUY, long_confidence,
                                                  long_signals, aggregate_timestamp)
        elif short_confidence > long_confidence and short_confidence >= self.min_confidence:
            return self._create_aggregated_signal(symbol, SignalAction.SELL, short_confidence,
                                                  short_signals, aggregate_timestamp)

        # Confianza insuficiente
        logger.debug("insufficient_aggregated_confidence",
                    symbol=symbol,
                    long_confidence=long_confidence,
                    short_confidence=short_confidence)
        return None

    def _calculate_weighted_confidence(self, signals: List[TradingSignal]) -> float:
        """
        Calcular confianza ponderada usando weights dinámicos
        """
        if not signals:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for signal in signals:
            # Usar weight dinámico o por defecto
            weight = self.dynamic_weights.get(signal.strategy_name, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _create_aggregated_signal(self, symbol: str, action: SignalAction, confidence: float,
                                 source_signals: List[TradingSignal],
                                 aggregate_timestamp: pd.Timestamp) -> TradingSignal:
        """
        Crear señal agregada con mejores cálculos
        """
        # Usar señal con mayor confianza como base
        base_signal = max(source_signals, key=lambda s: s.confidence)

        # Calcular precios promedio ponderados por confianza y weight dinámico
        total_weighted_confidence = 0.0
        entry_price_sum = 0.0
        stop_loss_sum = 0.0
        take_profit_sum = 0.0
        quantity_sum = 0.0

        for signal in source_signals:
            weight = self.dynamic_weights.get(signal.strategy_name, 0.5)
            weighted_conf = signal.confidence * weight

            entry_price_sum += signal.entry_price * weighted_conf
            stop_loss_sum += signal.stop_loss * weighted_conf
            take_profit_sum += signal.take_profit * weighted_conf
            quantity_sum += signal.quantity * weighted_conf
            total_weighted_confidence += weighted_conf

        # Promedios ponderados
        if total_weighted_confidence > 0:
            entry_price = entry_price_sum / total_weighted_confidence
            stop_loss = stop_loss_sum / total_weighted_confidence
            take_profit = take_profit_sum / total_weighted_confidence
            quantity = quantity_sum / total_weighted_confidence
        else:
            # Fallback
            entry_price = base_signal.entry_price
            stop_loss = base_signal.stop_loss
            take_profit = base_signal.take_profit
            quantity = base_signal.quantity

        indicator_snapshot = {}
        indicator_fields = {
            'atr': [],
            'adx': [],
            'bb_bandwidth': []
        }

        for signal in source_signals:
            signal_indicators = getattr(signal, 'indicators', {}) or {}
            for field in indicator_fields:
                value = signal_indicators.get(field)
                if value is not None:
                    indicator_fields[field].append(value)

        for field, values in indicator_fields.items():
            if values:
                indicator_snapshot[field] = float(np.mean(values))

        # Crear señal agregada
        aggregated_signal = TradingSignal(
            strategy_name="AggregatedV2",
            symbol=symbol,
            signal_type=SignalType.AGGREGATED,
            action=action,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            quantity=round(quantity, 4),
            confidence=round(confidence, 3),
            indicators=indicator_snapshot,
            reason=f"V2 Aggregated: {len(source_signals)} sources, conf {confidence:.3f}",
            source_signals=[s.strategy_name for s in source_signals],
            timestamp=aggregate_timestamp.to_pydatetime()
        )

        # Guardar en historial
        self.aggregated_signals_history.append({
            "timestamp": aggregate_timestamp.isoformat(),
            "signal": aggregated_signal.dict() if hasattr(aggregated_signal, 'dict') else str(aggregated_signal),
            "source_signals": [s.strategy_name for s in source_signals],
            "weights_used": self.dynamic_weights.copy(),
            "filters_applied": {
                "volatility": self.volatility_filter_enabled,
                "session": self.session_filter_enabled,
                "liquidity": self.liquidity_filter_enabled
            }
        })

        # Mantener límite de historial
        if len(self.aggregated_signals_history) > self.max_history:
            self.aggregated_signals_history = self.aggregated_signals_history[-self.max_history:]

        logger.info("aggregated_signal_v2_created",
                   symbol=symbol,
                   action=action,
                   confidence=confidence,
                   sources=len(source_signals),
                   weights=list(self.dynamic_weights.keys()))

        return aggregated_signal

    def create_advanced_orders(self, aggregated_signal: TradingSignal,
                             order_manager: OrderManager,
                             market_data: pd.DataFrame,
                             advanced_orders_params: Dict = None) -> List[str]:
        """
        Crear órdenes avanzadas basadas en señal agregada

        Args:
            aggregated_signal: Señal agregada
            order_manager: OrderManager para gestionar órdenes
            market_data: Datos de mercado actuales
            advanced_orders_params: Parámetros para órdenes avanzadas

        Returns:
            Lista de IDs de órdenes creadas
        """
        if advanced_orders_params is None:
            advanced_orders_params = {
                "limit_offset": 0.001,  # 0.1%
                "iceberg_visible": 0.05,  # 0.05 BTC
                "trailing_atr": 1.2,  # 1.2x ATR
                "stop_adaptive": True
            }

        order_ids = []

        try:
            # Determinar side de la orden
            order_side = OrderSide.BUY if aggregated_signal.action == SignalAction.BUY else OrderSide.SELL

            # Calcular precios con offset
            limit_offset = advanced_orders_params.get("limit_offset", 0.001)
            if order_side == OrderSide.BUY:
                limit_price = aggregated_signal.entry_price * (1 + limit_offset)
            else:
                limit_price = aggregated_signal.entry_price * (1 - limit_offset)

            # 1. Crear orden principal (Limit Order)
            limit_order = LimitOrder(
                symbol=aggregated_signal.symbol,
                side=order_side,
                quantity=aggregated_signal.quantity,
                limit_price=limit_price,
                timestamp=datetime.now(timezone.utc)
            )
            order_ids.append(order_manager.add_order(limit_order))

            # 2. Crear Iceberg Order para entrada stealth
            iceberg_visible = advanced_orders_params.get("iceberg_visible", 0.05)
            iceberg_order = IcebergOrder(
                symbol=aggregated_signal.symbol,
                side=order_side,
                total_quantity=aggregated_signal.quantity,
                display_quantity=iceberg_visible,
                limit_price=limit_price,
                timestamp=datetime.now(timezone.utc)
            )
            order_ids.append(order_manager.add_order(iceberg_order))

            # 3. Crear Stop-Loss (Stop-Limit Order)
            if aggregated_signal.stop_loss:
                stop_price = aggregated_signal.stop_loss
                if advanced_orders_params.get("stop_adaptive", False):
                    # Adaptive stop based on volatility
                    current_atr = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else 0
                    if current_atr > 0:
                        atr_multiplier = advanced_orders_params.get("trailing_atr", 1.2)
                        stop_offset = current_atr * atr_multiplier
                        if order_side == OrderSide.BUY:
                            stop_price = aggregated_signal.entry_price - stop_offset
                        else:
                            stop_price = aggregated_signal.entry_price + stop_offset

                stop_loss_order = StopLimitOrder(
                    symbol=aggregated_signal.symbol,
                    side=OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY,
                    quantity=aggregated_signal.quantity,
                    stop_price=stop_price,
                    limit_price=stop_price * 0.998,  # 0.2% slippage protection
                    timestamp=datetime.now(timezone.utc)
                )
                order_ids.append(order_manager.add_order(stop_loss_order))

            # 4. Crear Take-Profit con Trailing Stop
            if aggregated_signal.take_profit:
                # Orden limit para take profit
                take_profit_order = LimitOrder(
                    symbol=aggregated_signal.symbol,
                    side=OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY,
                    quantity=aggregated_signal.quantity,
                    limit_price=aggregated_signal.take_profit,
                    timestamp=datetime.now(timezone.utc)
                )
                order_ids.append(order_manager.add_order(take_profit_order))

                # Trailing stop para proteger ganancias
                trailing_atr = advanced_orders_params.get("trailing_atr", 1.2)
                trailing_stop = TrailingStopOrder(
                    symbol=aggregated_signal.symbol,
                    side=OrderSide.SELL if order_side == OrderSide.BUY else OrderSide.BUY,
                    quantity=aggregated_signal.quantity,
                    trailing_percent=trailing_atr,  # Use ATR multiplier as percent
                    timestamp=datetime.now(timezone.utc)
                )
                # Set initial price for trailing
                trailing_stop.set_initial_price(aggregated_signal.entry_price)
                order_ids.append(order_manager.add_order(trailing_stop))

            logger.info("advanced_orders_created",
                       signal_id=getattr(aggregated_signal, 'id', 'unknown'),
                       orders_created=len(order_ids),
                       order_types=['limit', 'iceberg', 'stop_limit', 'take_profit', 'trailing_stop'][:len(order_ids)])

            return order_ids

        except Exception as e:
            logger.error("advanced_orders_creation_failed",
                        error=str(e),
                        signal_symbol=aggregated_signal.symbol)
            return []

    def clear_signals(self, symbol: Optional[str] = None):
        """Limpiar señales activas"""
        if symbol:
            self.active_signals.pop(symbol, None)
            logger.debug("signals_cleared_for_symbol", symbol=symbol)
        else:
            self.active_signals.clear()
            logger.debug("all_signals_cleared")

    def get_recent_aggregated_signals(self, limit: int = 50) -> List[dict]:
        """Obtener señales agregadas recientes"""
        return self.aggregated_signals_history[-limit:]

    def configure_filters(self, volatility_enabled: bool = None, session_enabled: bool = None,
                         liquidity_enabled: bool = None, max_volatility: float = None,
                         session_start_time: Optional[time] = None,
                         session_end_time: Optional[time] = None,
                         volume_threshold: Optional[float] = None):
        """Configurar filtros del agregador."""
        if volatility_enabled is not None:
            self.volatility_filter_enabled = volatility_enabled
        if session_enabled is not None:
            self.session_filter_enabled = session_enabled
        if liquidity_enabled is not None:
            self.liquidity_filter_enabled = liquidity_enabled
        if max_volatility is not None:
            self.max_volatility_threshold = max_volatility
        if session_start_time is not None:
            self.session_start_time = session_start_time
        if session_end_time is not None:
            self.session_end_time = session_end_time
        if volume_threshold is not None:
            self.volume_liquidity_threshold = volume_threshold

        logger.info("filters_configured",
                   volatility=self.volatility_filter_enabled,
                   session=self.session_filter_enabled,
                   liquidity=self.liquidity_filter_enabled,
                   max_volatility=self.max_volatility_threshold,
                   session_start=self.session_start_time,
                   session_end=self.session_end_time,
                   volume_threshold=self.volume_liquidity_threshold)