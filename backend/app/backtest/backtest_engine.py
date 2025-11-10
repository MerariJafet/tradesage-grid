# backend/app/backtest/backtest_engine.py

import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from app.backtest.models import BacktestBar, BacktestTrade, BacktestResult
from app.core.indicators.indicator_manager import IndicatorManager
from app.core.strategies.base import BaseStrategy
from app.core.strategies.position_sizer import PositionSizer
from app.core.strategies.signal_validator import SignalValidator
from app.core.position_sizing import calculate_position_size, DEFAULT_STARTING_BALANCE
from app.core.orders import BaseOrder, LimitOrder, StopLimitOrder, IcebergOrder, TrailingStopOrder, OrderManager
from app.core.orders.base_order import OrderSide
from app.utils.logger import get_logger
import uuid
import pandas as pd
from app.core.strategies_edge import manage_exit, detect_regime

logger = get_logger("backtest_engine")

class BacktestEngine:
    """
    Motor de backtesting para simular estrategias de trading

    Flujo:
    1. Cargar datos históricos
    2. Inicializar indicadores
    3. Por cada barra:
       - Actualizar indicadores
       - Ejecutar estrategia
       - Gestionar posición abierta
       - Calcular PnL
    4. Generar reporte de resultados
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0004,  # 0.04% Binance Futures
        slippage_pct: float = 0.02,  # 0.02% slippage
        max_open_positions: int = 3  # 🚀 NUEVO: Límite de posiciones simultáneas
    ):
        self.initial_balance = initial_balance if initial_balance > 0 else DEFAULT_STARTING_BALANCE
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.max_open_positions = max_open_positions  # 🚀 NUEVO
        self.force_fill_simulation = False
        self.dynamic_tp_sl = False
        self.atr_stop_multiplier = 1.8
        self.tp_sl_ratio = 2.0
        self.edge_mode = False

        # Estado del backtest
        if self.initial_balance <= 0:
            self.initial_balance = DEFAULT_STARTING_BALANCE
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Trades - 🚀 CAMBIO CRÍTICO: De single trade a lista
        self.trades: List[BacktestTrade] = []
        self.open_trades: List[BacktestTrade] = []  # 🚀 NUEVO: Lista de trades abiertos
        self.open_trade: Optional[BacktestTrade] = None  # Mantener para compatibilidad

        # Balance history para equity curve
        self.balance_history: List[Dict] = []

        # Aggregated backtest helpers
        self.pending_signals: List[Dict] = []
        self.pnl_history: List[Dict] = []
        self.order_manager: Optional[OrderManager] = None
        
        # 🚀 NUEVO: Estadísticas de señales
        self.total_signals_generated = 0
        self.total_signals_blocked = 0

        logger.info(
            "backtest_engine_initialized",
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            slippage_pct=slippage_pct
        )


    async def run_backtest(
        self,
        strategy: BaseStrategy,
        bars: List[dict],
        symbol: str
    ) -> BacktestResult:
        """
        Ejecutar backtest completo

        Args:
            strategy: Estrategia a testear
            bars: Datos históricos (lista de diccionarios)
            symbol: Símbolo (ej: BTCUSDT)

        Returns:
            BacktestResult con métricas completas
        """

        logger.info(
            "backtest_started",
            strategy=strategy.name,
            symbol=symbol,
            bars_count=len(bars),
            start_date=datetime.fromtimestamp(bars[0]['timestamp'] / 1000),
            end_date=datetime.fromtimestamp(bars[-1]['timestamp'] / 1000)
        )

        # Reset estado
        self._reset()

        # Crear indicator manager para el backtest
        indicator_manager = strategy.indicator_manager

        # Procesar cada barra
        for i, bar_dict in enumerate(bars):
            # Actualizar indicadores con la nueva barra
            indicator_manager.update_with_bar(symbol, bar_dict)

            # 🚀 PASO 1: Verificar TODAS las posiciones abiertas (TP/SL/Time)
            trades_to_close = []
            for trade in self.open_trades[:]:  # Copiar lista para iterar seguro
                should_close, exit_price, exit_reason = self._check_trade_exit(trade, bar_dict)
                if should_close:
                    trades_to_close.append((trade, exit_price, exit_reason))
            
            # Cerrar los trades que alcanzaron TP/SL/Time
            for trade, exit_price, exit_reason in trades_to_close:
                self._close_trade(
                    trade=trade,
                    exit_timestamp=bar_dict['timestamp'],
                    exit_price=exit_price,
                    exit_reason=exit_reason
                )

            # 🚀 PASO 2: Buscar nuevas oportunidades si hay espacio
            if len(self.open_trades) < self.max_open_positions:
                # Generar señal usando strategy.check_setup()
                signal = strategy.check_setup(bar_dict)
                
                if signal:
                    self.total_signals_generated += 1
                    self._open_trade(signal, bar_dict, symbol)
            else:
                # Contar señales bloqueadas por límite de posiciones
                potential_signal = strategy.check_setup(bar_dict) if hasattr(strategy, 'check_setup') else None
                if potential_signal:
                    self.total_signals_blocked += 1
            
            # Mantener open_trade actualizado (para compatibilidad con código legacy)
            self.open_trade = self.open_trades[0] if self.open_trades else None

            # Registrar balance actual
            self.balance_history.append({
                'timestamp': bar_dict['timestamp'],
                'balance': self.current_balance,
                'drawdown': self.current_drawdown
            })

            # Log progreso cada 1000 barras
            if (i + 1) % 1000 == 0:
                logger.info(
                    "backtest_progress",
                    processed=i + 1,
                    total=len(bars),
                    pct=(i + 1) / len(bars) * 100,
                    balance=self.current_balance,
                    trades=len(self.trades),
                    open_positions=len(self.open_trades)
                )

        # 🚀 Cerrar TODAS las posiciones abiertas al final del backtest
        for trade in self.open_trades[:]:  # Copiar lista
            last_bar = bars[-1]
            self._close_trade(
                trade=trade,
                exit_timestamp=last_bar['timestamp'],
                exit_price=last_bar['close'],
                exit_reason="backtest_end"
            )

        # Generar resultado
        result = self._generate_result(
            strategy=strategy,
            symbol=symbol,
            start_date=datetime.fromtimestamp(bars[0]['timestamp'] / 1000),
            end_date=datetime.fromtimestamp(bars[-1]['timestamp'] / 1000)
        )

        logger.info(
            "backtest_completed",
            strategy=strategy.name,
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            total_pnl=result.total_pnl,
            final_balance=result.final_balance
        )

        return result

    def _reset(self):
        """Reset estado del backtest"""
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.trades = []
        self.open_trade = None
        self.balance_history = []

    def _open_trade(
        self,
        signal,
        bar_dict: dict,
        symbol: str
    ):
        """Abrir nuevo trade CON TP/SL dinámico basado en ATR"""

        # Aplicar slippage
        if signal.action == "BUY":
            entry_price = bar_dict['close'] * (1 + self.slippage_pct / 100)
        else:  # SELL
            entry_price = bar_dict['close'] * (1 - self.slippage_pct / 100)

        # NUEVO: Obtener ATR de los indicadores de la señal
        atr_value = None
        if hasattr(signal, 'indicators') and isinstance(signal.indicators, dict):
            atr_value = signal.indicators.get('atr_14')
        
        # Fallback a ATR del bar_dict o 2% del precio
        if atr_value is None or atr_value <= 0:
            atr_value = bar_dict.get('atr', bar_dict['close'] * 0.02)

        # NUEVO: TP/SL basado en ATR con ratios configurables
        # Stop Loss = 1.5x ATR (ajustable vía self.atr_stop_multiplier)
        # Take Profit = 2.0x ATR (ajustable vía self.tp_sl_ratio)
        sl_multiplier = getattr(self, 'atr_stop_multiplier', 1.5)
        tp_multiplier = getattr(self, 'tp_sl_ratio', 2.0)
        
        if signal.action == "BUY":
            stop_loss = entry_price - (atr_value * sl_multiplier)
            take_profit = entry_price + (atr_value * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr_value * sl_multiplier)
            take_profit = entry_price - (atr_value * tp_multiplier)

        # Calcular quantity basado en riesgo del 1%
        risk_amount = self.current_balance * 0.01  # 1% del balance
        distance_to_sl = abs(entry_price - stop_loss)
        
        # Quantity = risk / distance_to_sl
        if distance_to_sl > 0:
            quantity = risk_amount / distance_to_sl
        else:
            quantity = risk_amount / entry_price

        trade = BacktestTrade(
            id=str(uuid.uuid4()),
            strategy_name=signal.strategy_name if hasattr(signal, 'strategy_name') else signal.source,
            symbol=symbol,
            side=signal.action,
            entry_timestamp=bar_dict['timestamp'],
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # 🚀 NUEVO: Agregar a lista de trades abiertos
        self.open_trades.append(trade)
        self.open_trade = trade  # Mantener compatibilidad

        logger.info(
            "trade_opened_with_atr_stops",
            trade_id=trade.id,
            side=trade.side,
            entry_price=entry_price,
            quantity=trade.quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr_value,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier
        )

    def _check_trade_exit(self, trade: BacktestTrade, bar_dict: dict) -> tuple[bool, Optional[float], Optional[str]]:
        """
        🚀 NUEVO: Verificar si un trade específico debe cerrarse (TP/SL/Time)
        
        Returns:
            (should_close, exit_price, exit_reason)
        """
        
        # 1. CHECK STOP LOSS (prioridad máxima)
        if trade.stop_loss:
            if trade.side == "BUY":
                if bar_dict['low'] <= trade.stop_loss:
                    return (True, trade.stop_loss, "stop_loss")
            else:  # SELL
                if bar_dict['high'] >= trade.stop_loss:
                    return (True, trade.stop_loss, "stop_loss")
        
        # 2. CHECK TAKE PROFIT
        if trade.take_profit:
            if trade.side == "BUY":
                if bar_dict['high'] >= trade.take_profit:
                    return (True, trade.take_profit, "take_profit")
            else:  # SELL
                if bar_dict['low'] <= trade.take_profit:
                    return (True, trade.take_profit, "take_profit")
        
        # 3. TIME LIMIT DESACTIVADO (Sprint V2.2.1)
        # V2.1.3 demostró que time limits matan el 84% de trades prematuramente
        # Dejar que TP/SL decidan - el mercado es más sabio
        # time_elapsed_ms = bar_dict['timestamp'] - trade.entry_timestamp
        # if time_elapsed_ms >= 300000:  # 5 minutos
        #     return (True, bar_dict['close'], "time_limit")
        
        # Mantener abierto
        return (False, None, None)

    def _check_stops(self, bar_dict: dict):
        """Verificar si se alcanzó stop loss o take profit"""

        if not self.open_trade:
            return

        should_close = False
        exit_reason = None
        exit_price = None

        if self.open_trade.side == "BUY":
            # Check stop loss (low alcanzó stop)
            if bar_dict['low'] <= self.open_trade.stop_loss:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = self.open_trade.stop_loss

            # Check take profit (high alcanzó target)
            elif bar_dict['high'] >= self.open_trade.take_profit:
                should_close = True
                exit_reason = "take_profit"
                exit_price = self.open_trade.take_profit

        else:  # SELL
            # Check stop loss (high alcanzó stop)
            if bar_dict['high'] >= self.open_trade.stop_loss:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = self.open_trade.stop_loss

            # Check take profit (low alcanzó target)
            elif bar_dict['low'] <= self.open_trade.take_profit:
                should_close = True
                exit_reason = "take_profit"
                exit_price = self.open_trade.take_profit

        if should_close:
            self._close_trade(
                trade=self.open_trade,
                exit_timestamp=bar_dict['timestamp'],
                exit_price=exit_price,
                exit_reason=exit_reason
            )

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_timestamp: int,
        exit_price: float,
        exit_reason: str
    ):
        """Cerrar trade y actualizar balance"""

        # Aplicar slippage al exit
        if trade.side == "BUY":
            exit_price = exit_price * (1 - self.slippage_pct / 100)
        else:  # SELL
            exit_price = exit_price * (1 + self.slippage_pct / 100)

        # Cerrar trade y calcular PnL
        trade.close_trade(
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            commission_rate=self.commission_rate
        )

        # Actualizar balance
        self.current_balance += trade.pnl

        # Actualizar drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = self.peak_balance - self.current_balance
            drawdown_pct = (self.current_drawdown / self.peak_balance) * 100

            if drawdown_pct > self.max_drawdown:
                self.max_drawdown = drawdown_pct

        # Guardar trade
        self.trades.append(trade)
        
        # 🚀 NUEVO: Remover de lista de trades abiertos
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        
        # Actualizar open_trade para compatibilidad
        self.open_trade = self.open_trades[0] if self.open_trades else None

        logger.info(
            "trade_closed",
            trade_id=trade.id,
            exit_reason=exit_reason,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            balance=self.current_balance,
            drawdown=self.current_drawdown
        )

    def _generate_result(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Generar resultado del backtest"""

        # Calcular métricas básicas
        total_trades = len(self.trades)

        if total_trades == 0:
            # No hubo trades
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_balance=self.initial_balance,
                final_balance=self.current_balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0,
                total_pnl_pct=0,
                win_rate=0,
                profit_factor=0,
                max_drawdown=self.max_drawdown,
                max_drawdown_pct=self.max_drawdown,
                sharpe_ratio=0,
                sortino_ratio=0,
                avg_win=0,
                avg_loss=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                largest_win=0,
                largest_loss=0,
                avg_trade_duration_minutes=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                trades=[]
            )

        # Separar winning y losing trades
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        # Total PnL
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        # Win rate
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Avg win/loss
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_win_pct = sum(t.pnl_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss_pct = sum(t.pnl_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Largest win/loss
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)

        # Trade duration
        durations = []
        for t in self.trades:
            if t.exit_timestamp:
                duration_ms = t.exit_timestamp - t.entry_timestamp
                durations.append(duration_ms / (1000 * 60))  # Convert to minutes

        avg_trade_duration = sum(durations) / len(durations) if durations else 0

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0

        for t in self.trades:
            if t.pnl > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))

        # Sharpe ratio (simplificado)
        returns = [t.pnl_pct for t in self.trades]
        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0

        # Sortino ratio (solo considera downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5 if negative_returns else 0
        sortino_ratio = (avg_return / downside_deviation) if downside_deviation > 0 else 0

        # Max drawdown amount
        max_dd_amount = (self.max_drawdown / 100) * self.peak_balance if self.max_drawdown > 0 else 0

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd_amount,
            max_drawdown_pct=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_minutes=avg_trade_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades
        )

    def _generate_simple_signal(self, bar_dict: dict, symbol: str, indicator_manager):
        """
        Generar señal simple para backtest
        Esta es una implementación simplificada para testing
        """
        import random

        # Para testing: generar señal aleatoria cada 100 barras aproximadamente
        if random.random() < 0.01:  # 1% de probabilidad por barra
            action = random.choice(['BUY', 'SELL'])

            # Crear señal
            signal = type('Signal', (), {
                'action': action,
                'strategy_name': 'SimpleBacktest',
                'source': 'SimpleBacktest',
                'quantity': 0.001,  # Fixed quantity for now
                'stop_loss': bar_dict['close'] * (0.98 if action == 'BUY' else 1.02),  # 2% stop
                'take_profit': bar_dict['close'] * (1.04 if action == 'BUY' else 0.96),  # 4% target
                'confidence': 0.6
            })()
            return signal

        return None

    # 🚀 SPRINT 10: Automated Parameter Optimization

    async def grid_search_optimization(
        self,
        strategy_class,
        bars: List[BacktestBar],
        symbol: str,
        param_grid: Dict[str, List],
        max_evaluations: int = 100
    ) -> Dict:
        """
        Grid search optimization para encontrar mejores parámetros

        Args:
            strategy_class: Clase de estrategia a optimizar
            bars: Datos históricos
            symbol: Par de trading
            param_grid: Diccionario con parámetros a probar
            max_evaluations: Máximo número de evaluaciones

        Returns:
            Dict con mejores parámetros y métricas
        """
        from itertools import product
        import random

        logger.info("starting_grid_search", symbol=symbol, max_evaluations=max_evaluations)

        # Generar combinaciones de parámetros
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        # Limitar evaluaciones si hay demasiadas combinaciones
        if len(all_combinations) > max_evaluations:
            all_combinations = random.sample(all_combinations, max_evaluations)
            logger.info("sampled_combinations", total=len(all_combinations))

        best_result = None
        best_sharpe = float('-inf')

        for i, params in enumerate(all_combinations):
            # Crear diccionario de parámetros
            param_dict = dict(zip(param_names, params))

            try:
                logger.info("evaluating_params", iteration=i+1, total=len(all_combinations), params=param_dict)

                # Crear estrategia con parámetros
                strategy = self._create_strategy_with_params(strategy_class, symbol, param_dict)

                # Ejecutar backtest
                result = await self.run_backtest(strategy, bars, symbol)

                # Evaluar resultado (usar Sharpe Ratio como métrica principal)
                if result.sharpe_ratio > best_sharpe and result.total_trades > 0:
                    best_sharpe = result.sharpe_ratio
                    best_result = {
                        'params': param_dict,
                        'result': result,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_trades': result.total_trades,
                        'win_rate': result.win_rate,
                        'total_pnl': result.total_pnl
                    }

            except Exception as e:
                logger.error("grid_search_error", params=param_dict, error=str(e))
                continue

        logger.info("grid_search_completed", best_sharpe=best_sharpe)
        return best_result

    async def walk_forward_analysis(
        self,
        strategy_class,
        bars: List[BacktestBar],
        symbol: str,
        params: Dict,
        window_size: float = 0.6,
        step_size: float = 0.2
    ) -> Dict:
        """
        Walk-forward analysis para validar robustez de parámetros

        Args:
            strategy_class: Clase de estrategia
            bars: Datos históricos completos
            symbol: Par de trading
            params: Parámetros optimizados
            window_size: Tamaño de ventana de optimización (0.6 = 60%)
            step_size: Tamaño de paso para rolling (0.2 = 20%)

        Returns:
            Dict con resultados de walk-forward
        """
        logger.info("starting_walk_forward", symbol=symbol, window_size=window_size, step_size=step_size)

        results = []
        n_bars = len(bars)
        step_bars = int(step_size * n_bars)

        for i in range(0, n_bars - step_bars, step_bars):
            # Definir ventanas
            train_end = i + int(window_size * n_bars)
            test_end = min(train_end + step_bars, n_bars)

            if test_end >= n_bars:
                break

            train_bars = bars[i:train_end]
            test_bars = bars[train_end:test_end]

            try:
                logger.info("walk_forward_step",
                          step=i//step_bars + 1,
                          train_period=f"{i} to {train_end}",
                          test_period=f"{train_end} to {test_end}")

                # Optimizar en ventana de entrenamiento (usar parámetros fijos para simplificar)
                strategy = self._create_strategy_with_params(strategy_class, symbol, params)

                # Evaluar en ventana de test
                result = await self.run_backtest(strategy, test_bars, symbol)

                results.append({
                    'step': i//step_bars + 1,
                    'train_period': f"{i}-{train_end}",
                    'test_period': f"{train_end}-{test_end}",
                    'result': result,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'total_pnl': result.total_pnl
                })

            except Exception as e:
                logger.error("walk_forward_error", step=i//step_bars + 1, error=str(e))
                continue

        # Calcular estadísticas agregadas
        if results:
            sharpe_ratios = [r['sharpe_ratio'] for r in results if r['total_trades'] > 0]
            win_rates = [r['win_rate'] for r in results if r['total_trades'] > 0]
            total_pnls = [r['total_pnl'] for r in results]

            summary = {
                'total_steps': len(results),
                'avg_sharpe': sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0,
                'std_sharpe': (sum((x - (sum(sharpe_ratios)/len(sharpe_ratios)))**2 for x in sharpe_ratios) / len(sharpe_ratios))**0.5 if sharpe_ratios else 0,
                'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else 0,
                'total_pnl': sum(total_pnls),
                'sharpe_stability': (sum(sharpe_ratios) / len(sharpe_ratios)) / ((sum((x - (sum(sharpe_ratios)/len(sharpe_ratios)))**2 for x in sharpe_ratios) / len(sharpe_ratios))**0.5) if sharpe_ratios else 0,
                'steps': results
            }
        else:
            summary = {'error': 'No valid results'}

        logger.info("walk_forward_completed", steps=len(results))
        return summary

    def _create_strategy_with_params(self, strategy_class, symbol: str, params: Dict):
        """
        Crear instancia de estrategia con parámetros específicos
        """
        # Crear componentes necesarios
        indicator_manager = IndicatorManager()
        indicator_manager.initialize_symbol(symbol)

        position_sizer = PositionSizer(account_balance=self.initial_balance)
        signal_validator = SignalValidator(indicator_manager=indicator_manager)

        # Crear estrategia con parámetros
        strategy = strategy_class(
            symbol=symbol,
            indicator_manager=indicator_manager,
            position_sizer=position_sizer,
            signal_validator=signal_validator,
            **params  # Pasar parámetros dinámicos
        )

        return strategy

    def _reset_state(self):
        """Resetear estado interno para simulaciones agregadas."""

        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.trades = []
        self.open_trade = None
        self.balance_history = []
        self.pending_signals = []
        self.pnl_history = []

    async def run_aggregated_signals_backtest(
        self,
        data: pd.DataFrame,
        aggregated_signals: List[Dict],
        symbol: str,
        risk_per_trade_pct: float = 0.02,
        order_manager: Optional[OrderManager] = None,
        force_fill_simulation: bool = False,
        dynamic_tp_sl: bool = False,
        atr_stop_multiplier: float = 1.8,
        tp_sl_ratio: float = 2.0
    ) -> Dict[str, Any]:
        """
        Ejecutar backtest usando señales agregadas

        Args:
            data: DataFrame con datos OHLCV
            aggregated_signals: Lista de señales agregadas
            symbol: Símbolo
            risk_per_trade_pct: Porcentaje de riesgo por trade

        Returns:
            BacktestResult con métricas completas
        """
        logger.info("running_aggregated_signals_backtest",
                   symbol=symbol, signals=len(aggregated_signals))

        # Reset estado
        self._reset_state()
        self.order_manager = order_manager
        self.force_fill_simulation = force_fill_simulation
        self.dynamic_tp_sl = dynamic_tp_sl
        self.atr_stop_multiplier = max(0.1, atr_stop_multiplier)
        self.tp_sl_ratio = max(0.1, tp_sl_ratio)

        # Convertir señales a formato procesable
        processed_signals = self._process_aggregated_signals(aggregated_signals, data)

        # Ejecutar simulación
        await self._simulate_aggregated_trading(data, processed_signals, symbol, risk_per_trade_pct)

        # Calcular métricas finales
        metrics = self._calculate_final_metrics()

        logger.info("aggregated_signals_backtest_completed",
                   symbol=symbol,
                   total_trades=metrics.get("total_trades", 0),
                   final_balance=self.current_balance,
                   sharpe_ratio=metrics.get("sharpe_ratio", 0.0))

        return {
            "metrics": metrics,
            "pnl_history": self.pnl_history,
            "trades": self.trades
        }

    def _process_aggregated_signals(self, aggregated_signals: List[Dict],
                                  data: pd.DataFrame) -> List[Dict]:
        """Procesar y validar señales agregadas"""

        processed_signals = []

        for signal in aggregated_signals:
            try:
                # Asegurar que tenga timestamp
                if 'timestamp' not in signal and hasattr(signal, 'timestamp'):
                    signal['timestamp'] = signal.timestamp
                elif 'timestamp' not in signal:
                    # Intentar inferir del índice de datos
                    signal['timestamp'] = data.index[-1]

                # Validar campos requeridos
                required_fields = ['entry_price', 'stop_loss', 'take_profit', 'quantity', 'action']
                if not all(field in signal for field in required_fields):
                    logger.warning("signal_missing_required_fields", signal=signal)
                    continue

                # Convertir action a string si es enum
                if hasattr(signal['action'], 'value'):
                    signal['action'] = signal['action'].value

                indicators = signal.get('indicators')
                if isinstance(indicators, dict) and 'atr' in indicators and 'atr' not in signal:
                    try:
                        signal['atr'] = float(indicators['atr'])
                    except (TypeError, ValueError):
                        pass

                processed_signals.append(signal)

            except Exception as e:
                logger.error("signal_processing_error", signal=signal, error=str(e))
                continue

        logger.debug("signals_processed",
                    total=len(aggregated_signals),
                    valid=len(processed_signals))

        return processed_signals

    async def _simulate_aggregated_trading(self, data: pd.DataFrame,
                                         signals: List[Dict], symbol: str,
                                         risk_per_trade_pct: float):
        """Simular trading con señales agregadas"""

        signals.sort(key=lambda x: x['timestamp'])
        signal_idx = 0

        for current_time, row in data.iterrows():
            # Registrar balance mark-to-market (sin mark-to-market de posición abierta)
            self.balance_history.append({
                'timestamp': self._timestamp_to_ms(current_time),
                'balance': self.current_balance
            })

            if self.order_manager:
                self._process_advanced_orders(row, symbol)

            while signal_idx < len(signals) and signals[signal_idx]['timestamp'] <= current_time:
                self.pending_signals.append(signals[signal_idx])
                signal_idx += 1

            if not self.open_trade and self.pending_signals:
                for pending_signal in list(self.pending_signals):
                    opened = await self._try_open_trade(
                        signal=pending_signal,
                        bar=row,
                        timestamp=current_time,
                        symbol=symbol,
                        risk_per_trade_pct=risk_per_trade_pct
                    )
                    if opened:
                        self.pending_signals.remove(pending_signal)
                        break

            if self.open_trade:
                await self._manage_open_position(row, current_time)

        # Cerrar posición abierta al final del periodo
        if self.open_trade:
            final_row = data.iloc[-1]
            await self._manage_open_position(final_row, data.index[-1], force_exit=True)

    def _process_advanced_orders(self, bar: pd.Series, symbol: str) -> None:
        """Simular fills para órdenes avanzadas utilizando datos OHLC."""

        if not self.order_manager:
            return

        active_orders = list(self.order_manager.get_active_orders(symbol=symbol))
        if not active_orders:
            return

        for order in active_orders:
            payload = self._order_to_payload(order)
            if not payload:
                continue

            fill_price = self._simulate_order_fill(payload, bar)
            if fill_price is None:
                continue

            remaining_quantity = order.quantity - order.filled_quantity
            if remaining_quantity <= 0:
                continue

            if isinstance(order, IcebergOrder):
                order.fill(fill_price, remaining_quantity)
            else:
                order.fill(fill_price, remaining_quantity)

            if order not in self.order_manager.filled_orders:
                self.order_manager.filled_orders.append(order)

            if order in self.order_manager.orders:
                self.order_manager.orders.remove(order)

    def _order_to_payload(self, order: BaseOrder) -> Optional[Dict[str, Any]]:
        """Convertir una orden avanzada en un payload simple para simulación."""

        side = order.side.value.lower() if isinstance(order.side, OrderSide) else str(order.side).lower()

        if isinstance(order, TrailingStopOrder):
            trail_price = order.get_trigger_price()
            return {
                'type': 'trailing',
                'side': side,
                'trail_price': trail_price
            }

        if isinstance(order, (LimitOrder, IcebergOrder)):
            limit_price = getattr(order, 'limit_price', None)
            if limit_price is None:
                return None
            return {
                'type': 'limit',
                'side': side,
                'price': limit_price
            }

        if isinstance(order, StopLimitOrder):
            limit_price = order.limit_price
            stop_price = order.stop_price
            return {
                'type': 'stop_limit',
                'side': side,
                'price': limit_price,
                'stop_price': stop_price
            }

        trigger_price = None
        if hasattr(order, 'get_trigger_price'):
            trigger_price = order.get_trigger_price()

        if trigger_price is not None:
            return {
                'type': 'limit',
                'side': side,
                'price': trigger_price
            }

        return None

    async def _try_open_trade(self, signal: Dict, bar: pd.Series,
                              timestamp: pd.Timestamp, symbol: str,
                              risk_per_trade_pct: float) -> bool:
        """Intentar abrir un trade basado en la señal y la barra actual."""

        try:
            action = signal['action'].upper()
            entry_price = signal.get('entry_price', bar['close'])

            bar_atr = None
            try:
                bar_atr = float(bar.get('atr')) if 'atr' in bar else None
            except (TypeError, ValueError):
                bar_atr = None

            atr = None
            if isinstance(signal, dict):
                atr = signal.get('atr')
            if atr is not None:
                try:
                    atr = float(atr)
                except (TypeError, ValueError):
                    atr = None

            atr_value = atr if atr and atr > 0 else bar_atr
            if not atr_value or atr_value <= 0:
                atr_value = entry_price * 0.005

            stop_multiplier = self.atr_stop_multiplier or 1.8
            target_multiplier = stop_multiplier * self.tp_sl_ratio if self.tp_sl_ratio > 0 else stop_multiplier * 2.0

            if action == 'BUY':
                stop_loss = entry_price - stop_multiplier * atr_value
                take_profit = entry_price + target_multiplier * atr_value
            else:
                stop_loss = entry_price + stop_multiplier * atr_value
                take_profit = entry_price - target_multiplier * atr_value

            signal_indicators = {}
            if isinstance(signal, dict):
                signal_indicators = signal.get('indicators', {}) or {}
            else:
                signal_indicators = getattr(signal, 'indicators', {}) or {}

            order_payload = {
                'type': signal.get('order_type', 'limit') if isinstance(signal, dict) else 'limit',
                'side': action.lower(),
                'price': entry_price,
                'trail_price': signal.get('trail_price') if isinstance(signal, dict) else None
            }

            execution_price = self._simulate_order_fill(order_payload, bar)
            if execution_price is None:
                return False

            quantity = self._calculate_position_size(
                execution_price=execution_price,
                stop_loss=stop_loss,
                risk_per_trade_pct=risk_per_trade_pct,
                atr=atr_value
            )
            if quantity <= 0:
                logger.debug("position_size_invalid", execution_price=execution_price, stop_loss=stop_loss)
                return False

            trade = BacktestTrade(
                id=str(uuid.uuid4()),
                strategy_name="AggregatedV2",
                symbol=symbol,
                side=action,
                entry_timestamp=self._timestamp_to_ms(timestamp),
                entry_price=execution_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if self.edge_mode:
                adx_value = signal_indicators.get('adx')
                bandwidth_value = signal_indicators.get('bb_bandwidth')
                bar_context = {
                    'timestamp': timestamp,
                    'close': float(bar.get('close', execution_price)),
                    'volume': float(bar.get('volume', 0.0))
                }
                regime_label = detect_regime(
                    bar_context,
                    adx_value if adx_value is not None else bar.get('adx'),
                    bandwidth_value if bandwidth_value is not None else bar.get('bb_bandwidth')
                )
                trade.metadata = {
                    'edge_mode': True,
                    'entry_atr': float(atr_value) if atr_value is not None else None,
                    'entry_adx': float(adx_value) if adx_value is not None else (float(bar.get('adx')) if bar.get('adx') is not None else None),
                    'entry_bandwidth': float(bandwidth_value) if bandwidth_value is not None else (float(bar.get('bb_bandwidth')) if bar.get('bb_bandwidth') is not None else None),
                    'regime': regime_label,
                    'entry_timestamp_iso': timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp)
                }
            else:
                trade.metadata = {}

            entry_commission = execution_price * quantity * self.commission_rate
            trade.commission += entry_commission
            self.current_balance -= entry_commission

            self.open_trade = trade

            logger.debug("aggregated_signal_executed",
                        symbol=symbol,
                        action=action,
                        entry_price=execution_price,
                        quantity=quantity)
            return True

        except Exception as exc:
            logger.error("aggregated_signal_execution_error",
                        signal=signal, error=str(exc))
            return False

    def _simulate_order_fill(self, order: Dict[str, Any], bar: pd.Series) -> Optional[float]:
        """Simular fills contra la vela actual usando prioridad OHLC."""

        try:
            open_price = float(bar['open'])
            high = float(bar['high'])
            low = float(bar['low'])
            close_price = float(bar['close'])
        except (KeyError, TypeError, ValueError):
            return None

        order_type = str(order.get('type', 'limit')).lower()
        side = str(order.get('side', '')).lower()
        if side not in {'buy', 'sell'}:
            return None

        def _fallback_fill() -> Optional[float]:
            return close_price if self.force_fill_simulation else None

        def _limit_fill(limit_price: Optional[float]) -> Optional[float]:
            if limit_price is None:
                return None

            try:
                price = float(limit_price)
            except (TypeError, ValueError):
                return None

            if side == 'buy':
                if open_price <= price:
                    return open_price if self.force_fill_simulation else min(price, open_price)
                if low <= price <= high:
                    return price
                if self.force_fill_simulation and close_price <= price:
                    return close_price
            else:
                if open_price >= price:
                    return open_price if self.force_fill_simulation else max(price, open_price)
                if low <= price <= high:
                    return price
                if self.force_fill_simulation and close_price >= price:
                    return close_price

            return None

        if order_type == 'market':
            return close_price

        if order_type == 'trailing':
            trail_price = order.get('trail_price')
            if trail_price is None:
                return None
            try:
                trail_price = float(trail_price)
            except (TypeError, ValueError):
                return None

            if side == 'buy' and low <= trail_price:
                return trail_price
            if side == 'sell' and high >= trail_price:
                return trail_price
            return _fallback_fill()

        stop_price = order.get('stop_price')
        try:
            stop_price = float(stop_price) if stop_price is not None else None
        except (TypeError, ValueError):
            stop_price = None

        if order_type == 'stop_limit':
            limit_price = order.get('price', stop_price)
            triggered = False

            if stop_price is not None:
                if side == 'buy':
                    if open_price >= stop_price or high >= stop_price:
                        triggered = True
                else:
                    if open_price <= stop_price or low <= stop_price:
                        triggered = True
            else:
                triggered = True

            if not triggered and not self.force_fill_simulation:
                return None

            fill_price = _limit_fill(limit_price)
            if fill_price is not None:
                return fill_price

            return _fallback_fill()

        # Default: limit order semantics
        limit_price = order.get('price')
        fill_price = _limit_fill(limit_price)
        if fill_price is not None:
            return fill_price

        return _fallback_fill()

    def _calculate_position_size(self, execution_price: float, stop_loss: float,
                                  risk_per_trade_pct: float, atr: Optional[float] = None) -> float:
        effective_balance = self.current_balance if self.current_balance > 0 else DEFAULT_STARTING_BALANCE
        return calculate_position_size(
            balance=effective_balance,
            entry_price=execution_price,
            stop_loss_price=stop_loss,
            risk_percent=risk_per_trade_pct,
            atr=atr
        )

    async def _manage_open_position(self, row: pd.Series, timestamp: pd.Timestamp,
                                  force_exit: bool = False):
        """Gestionar o cerrar una posición abierta utilizando OHLC."""

        if not self.open_trade:
            return

        trade = self.open_trade

        if self.edge_mode:
            close_value = row.get('close', trade.entry_price)
            try:
                close_price = float(close_value)
            except (TypeError, ValueError):
                close_price = trade.entry_price

            bar_context = {
                'timestamp': timestamp,
                'close': close_price,
                'volume': float(row.get('volume', 0.0)) if 'volume' in row else 0.0
            }
            atr_value = row.get('atr') if 'atr' in row else None
            adx_value = row.get('adx') if 'adx' in row else None
            manage_exit(trade, bar_context, atr_value, adx_value)

        high = float(row['high'])
        low = float(row['low'])

        exit_price: Optional[float] = None
        exit_reason = ""

        if trade.side == 'BUY':
            stop_hit = low <= trade.stop_loss
            target_hit = high >= trade.take_profit

            if stop_hit:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif target_hit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"
        else:
            stop_hit = high >= trade.stop_loss
            target_hit = low <= trade.take_profit

            if stop_hit:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif target_hit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"

        if force_exit and exit_price is None:
            exit_price = float(row['close'])
            exit_reason = "session_end"

        if exit_price is None:
            return

        self._finalize_trade(trade, exit_price, exit_reason, timestamp)
        self.open_trade = None

    def _finalize_trade(self, trade: BacktestTrade, exit_price: float,
                         exit_reason: str, timestamp: pd.Timestamp) -> None:
        if trade.side == 'BUY':
            adjusted_exit_price = exit_price * (1 - self.slippage_pct / 100)
            pnl = (adjusted_exit_price - trade.entry_price) * trade.quantity
        else:
            adjusted_exit_price = exit_price * (1 + self.slippage_pct / 100)
            pnl = (trade.entry_price - adjusted_exit_price) * trade.quantity

        exit_commission = adjusted_exit_price * trade.quantity * self.commission_rate
        pnl -= exit_commission

        trade.exit_timestamp = self._timestamp_to_ms(timestamp)
        trade.exit_price = adjusted_exit_price
        trade.exit_reason = exit_reason
        trade.pnl = pnl
        trade.commission += exit_commission
        invested = trade.entry_price * trade.quantity
        trade.pnl_pct = (pnl / invested) * 100 if invested > 0 else 0.0

        self.current_balance += pnl
        if self.current_balance < 100.0:
            logger.warning(
                "balance_underflow_detected",
                current_balance=self.current_balance,
                reset_to=DEFAULT_STARTING_BALANCE
            )
            self.current_balance = DEFAULT_STARTING_BALANCE
        self.peak_balance = max(self.peak_balance, self.current_balance)
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.current_drawdown = drawdown
            self.max_drawdown = max(self.max_drawdown, drawdown)

        self.trades.append(trade)

        metadata = trade.metadata if isinstance(getattr(trade, 'metadata', None), dict) else {}
        self.pnl_history.append({
            'trade_id': trade.id,
            'symbol': getattr(trade, 'symbol', ''),
            'side': trade.side,
            'entry_timestamp': trade.entry_timestamp,
            'exit_timestamp': trade.exit_timestamp,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'reason': trade.exit_reason,
            'entry_atr': metadata.get('entry_atr'),
            'entry_adx': metadata.get('entry_adx'),
            'entry_bandwidth': metadata.get('entry_bandwidth'),
            'regime': metadata.get('regime')
        })

        logger.debug("position_closed",
                    trade_id=trade.id,
                    exit_reason=exit_reason,
                    pnl=trade.pnl,
                    balance=self.current_balance)

    def _calculate_final_metrics(self) -> Dict[str, float]:
        """Calcular métricas finales del backtest."""

        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
                "final_balance": self.current_balance
            }

        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        returns = []
        for trade in self.trades:
            invested = trade.entry_price * trade.quantity
            if invested > 0 and trade.pnl is not None:
                returns.append(trade.pnl / invested)

        sharpe_ratio = 0.0
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5
            if std_return > 0:
                sharpe_ratio = avg_return / std_return * (252 ** 0.5)

        total_pnl = sum(t.pnl for t in self.trades if t.pnl is not None)

        return {
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_pnl": total_pnl,
            "final_balance": self.current_balance
        }

    @staticmethod
    def _timestamp_to_ms(timestamp: pd.Timestamp) -> int:
        ts = pd.Timestamp(timestamp)
        return int(ts.value // 1_000_000)